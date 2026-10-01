"""Backend OpenVINO per l'esecuzione SwiGLU di un singolo expert.

Scelte dettate da misure (tools/check_share_weights.py, tools/check_ov_precisions.py):

* **f16, non bf16.** I pesi bf16 passati come ``numpy`` ``uint16`` a una porta
  ``ov.Type.bf16`` vengono reinterpretati come interi: 2048 NaN su GPU, errore
  6e+19 su CPU. Con ``numpy`` ``float16`` nativo l'errore relativo e' 3,8e-04.
* **Un solo request riutilizzato** per shape: crearlo costa 0,018 ms, quindi
  era irrilevante, ma non costa nulla riutilizzarlo.
* **Device di default = CPU.** Con i pesi passati a ogni chiamata il plugin GPU
  gira a 1,05 GB/s contro i 5,25 GB/s del plugin CPU, e finisce 5x piu' lento.
  Su 320 GEMM da 4 MFLOP il trasferimento dei pesi domina il compute, e
  l'upload sul dispositivo e' piu' costoso di una memcpy locale. Su Lunar Lake
  la memoria e' condivisa: la GPU non ha nulla da guadagnare qui.
* I pesi si convertono una volta sola per expert e si riutilizzano dalla cache:
  il backend non deve ricevere array float gia' dequantizzati.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import openvino as ov
import openvino.opset13 as ops

from .key import ExpertKey
from .store import ExpertStore
from .dequant import dequantize_to

# Ordine di preferenza misurato su Lunar Lake per il carico "pesi per call".
# Ogni voce puo' essere sovrascritta con il parametro `device`.
DEVICE_PREFERENCE = ("CPU", "GPU", "NPU")


def pick_device(core: ov.Core, requested: str = "AUTO") -> str:
    available = list(core.available_devices)
    if requested != "AUTO" and requested in available:
        return requested
    for candidate in DEVICE_PREFERENCE:
        if candidate in available:
            return candidate
    return available[0] if available else "CPU"


class OpenVINOExpertBackend:
    """OpenVINO-accelerated SwiGLU expert execution backend."""

    def __init__(self, store: ExpertStore, device: str = "AUTO",
                 weight_dtype: str = "f16", cache_capacity: int = 0,
                 tensor_types: dict[int, dict[str, int]] | None = None,
                 dequant_fn: Any = None) -> None:
        self.store = store
        self.core = ov.Core()
        self.target_device = pick_device(self.core, device)
        # I pesi sono f16 e basta. Il ramo f32 non e' un'opzione: raddoppia
        # la cache degli expert (6 -> 12 MiB per expert) e su questa
        # piattaforma i pesi in float32 su porte a 16 bit non funzionano.
        if weight_dtype != "f16":
            raise ValueError(
                f"weight_dtype={weight_dtype!r}: i pesi devono essere f16"
            )
        self.weight_dtype = np.float16
        self.tensor_types = tensor_types or {}
        self._dequant_fn = dequant_fn
        self.cache_capacity = cache_capacity

        self._compiled_models: dict[tuple[int, int], Any] = {}
        self._requests: dict[tuple[int, int], Any] = {}
        self._weight_cache: dict[ExpertKey, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._weight_order: list[ExpertKey] = []
        # Buffer f16 riciclati, indicizzati per numero di elementi: un expert
        # ha tensori di due grandezze diverse (hidden x inter e inter x hidden).
        # Senza questo pool ogni dequant chiede pagine nuove al sistema operativo
        # e i page fault costano 0,56 ms su 0,96 ms per tensor, cioe' il 59% del
        # tempo di dequantizzazione. Vedi tools/test_weight_pool.py.
        self._pool: dict[int, list[np.ndarray]] = {}

        print(f"[OpenVINOBackend] device={self.target_device} "
              f"weights={weight_dtype} available={self.core.available_devices}")

    # ------------------------------------------------------------------ graph

    def _get_request(self, hidden_size: int, intermediate_size: int):
        key = (hidden_size, intermediate_size)
        request = self._requests.get(key)
        if request is not None:
            return request

        wtype = ov.Type.f16
        x = ops.parameter([1, hidden_size], ov.Type.f32, name="activation")
        gate = ops.parameter([hidden_size, intermediate_size], wtype, name="gate")
        up = ops.parameter([hidden_size, intermediate_size], wtype, name="up")
        down = ops.parameter([intermediate_size, hidden_size], wtype, name="down")

        xc = ops.convert(x, wtype)
        gate_proj = ops.matmul(xc, gate, transpose_a=False, transpose_b=False)
        up_proj = ops.matmul(xc, up, transpose_a=False, transpose_b=False)
        # SwiGLU: silu(gate) * up, scritto come gate * sigmoid(gate) * up.
        act = ops.multiply(gate_proj, ops.sigmoid(gate_proj))
        out = ops.matmul(ops.multiply(act, up_proj), down,
                         transpose_a=False, transpose_b=False)

        model = ov.Model([ops.convert(out, ov.Type.f32)], [x, gate, up, down],
                         name="SwiGLU_Expert")
        compiled = self.core.compile_model(model, self.target_device)
        self._compiled_models[key] = compiled
        request = compiled.create_infer_request()
        self._requests[key] = request
        return request

    # ------------------------------------------------------------------ weights

    def _weights(self, key: ExpertKey, hidden_size: int,
                 intermediate_size: int):
        cached = self._weight_cache.get(key)
        if cached is not None:
            self._weight_order.remove(key)
            self._weight_order.append(key)
            return cached

        layer_types = self.tensor_types.get(key.layer, {})
        gate = self._decode(self.store.read_tensor(key, "gate"),
                            layer_types.get("gate"),
                            (hidden_size, intermediate_size))
        up = self._decode(self.store.read_tensor(key, "up"),
                          layer_types.get("up"),
                          (hidden_size, intermediate_size))
        down = self._decode(self.store.read_tensor(key, "down"),
                            layer_types.get("down"),
                            (intermediate_size, hidden_size))

        weights = (gate, up, down)
        self._weight_cache[key] = weights
        self._weight_order.append(key)
        if self.cache_capacity and len(self._weight_order) > self.cache_capacity:
            oldest = self._weight_order.pop(0)
            self._release(self._weight_cache.pop(oldest, None))
        return weights

    def _release(self, weights) -> None:
        """Ricicla i buffer di un expert evictato.

        Il backend non espone i propri array: ``_weights`` e' privato e
        ``forward`` consuma i pesi in ``infer`` prima di restituire, quindi
        un buffer evictato non e' piu' letto da nessuno. Tenerlo in una lista
        di riuso evita che il prossimo expert debba faultare pagine nuove.

        I pool sono limitati a ``cache_capacity`` elementi: la cache cresce
        oltre perche' il prefetch riempie le prime voci, ma i buffer non
        devono crescere con lei, altrimenti la RAM non tornerebbe piu' al
        valore iniziale.
        """
        if weights is None:
            return
        for array in weights:
            n = array.size
            slot = self._pool.get(n)
            if slot is None:
                self._pool[n] = [array]
            elif len(slot) < max(1, self.cache_capacity):
                slot.append(array)

    def _acquire(self, n_elements: int) -> np.ndarray:
        slot = self._pool.get(n_elements)
        if slot:
            return slot.pop()
        return np.empty(n_elements, dtype=self.weight_dtype)

    def pool_size(self) -> int:
        return sum(len(slot) for slot in self._pool.values())

    def _decode(self, raw: bytes, ggml_type: int | None,
                shape: tuple[int, int]) -> np.ndarray:
        n_elements = shape[0] * shape[1]
        # Solo il percorso dequantizzato scrive nel buffer: nel ramo
        # non-quantizzato l'array e' una vista su `raw` e non va riciclato.
        if ggml_type is not None and self._dequant_fn is not None:
            array = self._dequant_fn(raw, ggml_type, n_elements,
                                      out=self._acquire(n_elements))
        elif ggml_type is not None:
            array = dequantize_to(raw, ggml_type, "f16")
        else:
            array = np.frombuffer(raw, dtype=self.weight_dtype)
        if array.size != n_elements:
            raise ValueError(
                f"expert tensor has {array.size} {self.weight_dtype} elements, "
                f"expected {n_elements} for shape {shape}. "
                "If this tensor is still quantized, pass the ggml type so it "
                "can be dequantized: a raw quantized buffer reinterpreted as "
                "float produces garbage, and for small weights the result is "
                "all zeros."
            )
        return np.ascontiguousarray(array.reshape(shape))

    def preload(self, key: ExpertKey, hidden_size: int, intermediate_size: int) -> None:
        self._weights(key, hidden_size, intermediate_size)

    def cache_size(self) -> int:
        return len(self._weight_cache)

    def clear_cache(self) -> None:
        self._weight_cache.clear()
        self._weight_order.clear()
        for slot in self._pool.values():
            slot.clear()

    # ------------------------------------------------------------------ forward

    def forward(self, key: ExpertKey, activation: list[float] | np.ndarray,
                hidden_size: int, intermediate_size: int) -> np.ndarray:
        gate, up, down = self._weights(key, hidden_size, intermediate_size)
        request = self._get_request(hidden_size, intermediate_size)

        if isinstance(activation, list):
            x = np.ascontiguousarray(activation, dtype=np.float32).reshape(1, hidden_size)
        else:
            x = np.ascontiguousarray(activation, dtype=np.float32).reshape(1, hidden_size)

        result = request.infer({
            "activation": x,
            "gate": gate,
            "up": up,
            "down": down,
        })
        return np.asarray(list(result.values())[0], dtype=np.float32).reshape(hidden_size)
