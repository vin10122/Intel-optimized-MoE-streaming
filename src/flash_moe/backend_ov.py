"""Backend OpenVINO per l'esecuzione SwiGLU di un singolo expert.

Scelte dettate da misure (tools/check_share_weights.py, tools/check_ov_precisions.py):

* **f16, non bf16.** I pesi bf16 passati come ``numpy`` ``uint16`` a una porta
  ``ov.Type.bf16`` vengono reinterpretati come interi: 2048 NaN su GPU, errore
  6e+19 su CPU. Con ``numpy`` ``float16`` nativo l'errore relativo e' 3,8e-04.
* **Un request per thread e shape**: i worker riusano request indipendenti.
  Il prefill usa un grafo con numero di token dinamico e quattro thread CPU;
  il decode OpenVINO usa un thread per expert e quattro worker di default.
* **Device di default = CPU.** Con i pesi passati a ogni chiamata il plugin GPU
  gira a 1,05 GB/s contro i 5,25 GB/s del plugin CPU, e finisce 5x piu' lento.
  Su 320 GEMM da 4 MFLOP il trasferimento dei pesi domina il compute, e
  l'upload sul dispositivo e' piu' costoso di una memcpy locale. Su Lunar Lake
  la memoria e' condivisa: la GPU non ha nulla da guadagnare qui.
* I pesi si convertono una volta sola per expert e si riutilizzano dalla cache:
  il backend non deve ricevere array float gia' dequantizzati.
* **I pesi NON possono diventare Constant.** Misurato: un modello con i tre
  pesi come Constant inferisce in 0,29 ms/expert contro 1,02 ms del percorso
  con i pesi come input, quindi sembra il modo giusto. Ma compilare quel
  modello costa 25-32 ms (misurati su 6 compilazioni, con e senza
  ``INFERENCE_PRECISION_HINT``): un expert ricompilato ogni volta e' 30 volte
  piu' lento di uno che riceve i pesi. Serve un hit rate del 97% per
  ammortizzare, e su 10240 expert con 8 per layer il working set e' 320 chiavi
  su 10240: non e' raggiungibile con una cache che stia in RAM. Mutare il
  buffer del Constant dopo la compilazione non funziona, il plugin lo ha gia'
  copiato (verificato: il risultato non cambia). Il percorso veloce resta
  quindi un solo modello compilato condiviso + ``share_inputs``, che toglie la
  copia senza toccare la compilazione.
* **Il batch degli 8 expert non aiuta**: 8 slot in un solo infer f16 costano
  1,65 ms/expert contro 1,08 ms di 8 infer separati. Il plugin non guadagna
  nulla nel muoversi piu' byte alla volta, e la banda resta il collo.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import threading
from functools import wraps

import numpy as np
import openvino as ov
import openvino.opset13 as ops

from .key import ExpertKey
from .store import ExpertStore
from .dequant import dequantize_to

# Ordine di preferenza misurato su Lunar Lake per il carico "pesi per call".
# Ogni voce puo' essere sovrascritta con il parametro `device`.
DEVICE_PREFERENCE = ("CPU", "GPU", "NPU")


# Buffer di riciclo per classe di grandezza: bastano pochi array
# per evitare i page fault delle allocazioni nuove (vedi
# ``_release``). Indipendente da ``cache_capacity``: con una cache
# grande trattenere un buffer per ogni slot di cache e' memoria
# sprecata che manda in paging.
_POOL_MAX = 16


def pick_device(core: ov.Core, requested: str = "AUTO") -> str:
    available = list(core.available_devices)
    if requested != "AUTO" and requested in available:
        return requested
    for candidate in DEVICE_PREFERENCE:
        if candidate in available:
            return candidate
    return available[0] if available else "CPU"


def _own_weights(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._memory_lock:
            self._active += 1
        try:
            return method(self, *args, **kwargs).copy()
        finally:
            with self._memory_lock:
                self._active -= 1
                if not self._active:
                    retired, self._retired = self._retired, []
                    for weights in retired:
                        self._recycle(weights)
    return call


def _compile_serially(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._compile_lock:
            return method(self, *args, **kwargs)
    return call


class OpenVINOExpertBackend:
    """OpenVINO-accelerated SwiGLU expert execution backend."""

    def __init__(self, store: ExpertStore, device: str = "AUTO",
                 weight_dtype: str = "f16", cache_capacity: int = 0,
                 tensor_types: dict[int, dict[str, int]] | None = None,
                 dequant_fn: Any = None,
                 inference_threads: int = 1, prefill_threads: int = 4,
                 expert_workers: int = 4, native_decode: str = "auto",
                 native_library=None) -> None:
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
        if self.cache_capacity < 0:
            raise ValueError(
                f"cache_capacity must be >= 0, got {self.cache_capacity}"
            )

        if min(inference_threads, prefill_threads, expert_workers) < 1:
            raise ValueError("thread and worker counts must be positive")
        self.inference_threads = inference_threads
        self.prefill_threads = prefill_threads
        self._ov_properties = {"INFERENCE_NUM_THREADS": str(inference_threads)}
        self._memory_lock = threading.RLock()
        self._compile_lock = threading.Lock()
        self._active = 0
        self._retired = []
        from .native_expert import load_native_expert
        self.native_expert = load_native_expert(
            native_decode, native_library, self.target_device, self.tensor_types
        )
        self._native_enabled = False

        self._compiled_models: dict[tuple[int, int], Any] = {}
        self._batch_compiled_models: dict[tuple[int, int], Any] = {}
        self._thread_local = threading.local()
        self._weight_cache: OrderedDict[
            ExpertKey, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = OrderedDict()
        # Buffer f16 riciclati, indicizzati per numero di elementi: un expert
        # ha tensori di due grandezze diverse (hidden x inter e inter x hidden).
        # Senza questo pool ogni dequant chiede pagine nuove al sistema operativo
        # e i page fault costano 0,56 ms su 0,96 ms per tensor, cioe' il 59% del
        # tempo di dequantizzazione. Vedi tools/test_weight_pool.py.
        self._pool: dict[int, list[np.ndarray]] = {}
        # OpenVINO >= 2022.3 accetta share_inputs. Se il parametro non esiste
        # il fallback sopra lo spegne da solo e il backend continua a funzionare.
        self._share_inputs = True
        self._lock = threading.Lock()
        # Reuse a bounded worker pool across all layers and decode steps.
        self._moe_executor = ThreadPoolExecutor(max_workers=expert_workers)

        print(f"[OpenVINOBackend] device={self.target_device} "
              f"weights={weight_dtype} inference_threads={self.inference_threads} "
              f"available={self.core.available_devices}")

    # ------------------------------------------------------------------ graph

    @_compile_serially
    def _get_request(self, hidden_size: int, intermediate_size: int):
        key = (hidden_size, intermediate_size)
        compiled = self._compiled_models.get(key)
        if compiled is None:
            x = ops.parameter([1, hidden_size], ov.Type.f32, name="activation")
            model = self._build_expert_graph(x, hidden_size, intermediate_size)
            compiled = self.core.compile_model(
                model, self.target_device, self._ov_properties
            )
            self._compiled_models[key] = compiled
        tl = getattr(self._thread_local, "requests", None)
        if tl is None:
            tl = {}
            self._thread_local.requests = tl
        request = tl.get(key)
        if request is None:
            request = compiled.create_infer_request()
            tl[key] = request
        return request

    @_compile_serially
    def _get_batch_request(self, hidden_size: int, intermediate_size: int):
        key = (hidden_size, intermediate_size)
        compiled = self._batch_compiled_models.get(key)
        if compiled is None:
            x = ops.parameter([-1, hidden_size], ov.Type.f32, name="activation")
            model = self._build_expert_graph(x, hidden_size, intermediate_size)
            compiled = self.core.compile_model(
                model, self.target_device, dict(self._ov_properties, INFERENCE_NUM_THREADS=str(self.prefill_threads))
            )
            self._batch_compiled_models[key] = compiled
        tl = getattr(self._thread_local, "batch_requests", None)
        if tl is None:
            tl = {}
            self._thread_local.batch_requests = tl
        request = tl.get(key)
        if request is None:
            request = compiled.create_infer_request()
            tl[key] = request
        return request

    @staticmethod
    def _build_expert_graph(x, hidden_size: int, intermediate_size: int):
        wtype = ov.Type.f16
        # I pesi sono gia' nella disposizione in cui stanno nel GGUF: gate e up
        # sono (n_ff, n_embd), down e' (n_embd, n_ff). Il GGUF li descrive come
        # {n_embd, n_ff, n_expert}, cioe' con l'hidden sull'asse veloce, quindi
        # in memoria l'n_ff e' la riga: e' quella che va letta, non la sua
        # trasposta. Dichiarare qui la forma inversa e moltiplicare per
        # transpose_b evita sia di trasporre gli array a ogni expert (2 MiB
        # per expert, 320 expert per token) sia di sbagliare i numeri: con
        # transpose_b=False su un array letto cosi' il prodotto non dava
        # errore, dava la proiezione di un altro tensor (a ffn_gate la somma
        # passava da -893,15 a 7,24).
        gate = ops.parameter([intermediate_size, hidden_size], wtype, name="gate")
        up = ops.parameter([intermediate_size, hidden_size], wtype, name="up")
        down = ops.parameter([hidden_size, intermediate_size], wtype, name="down")

        xc = ops.convert(x, wtype)
        gate_proj = ops.matmul(xc, gate, transpose_a=False, transpose_b=True)
        up_proj = ops.matmul(xc, up, transpose_a=False, transpose_b=True)
        # SwiGLU: silu(gate) * up, scritto come gate * sigmoid(gate) * up.
        act = ops.multiply(gate_proj, ops.sigmoid(gate_proj))
        out = ops.matmul(ops.multiply(act, up_proj), down,
                         transpose_a=False, transpose_b=True)

        model = ov.Model([ops.convert(out, ov.Type.f32)], [x, gate, up, down],
                         name="SwiGLU_Expert")
        return model

    # ------------------------------------------------------------------ weights

    def _weights(self, key: ExpertKey, hidden_size: int,
                 intermediate_size: int):
        with self._lock:
            cached = self._weight_cache.get(key)
            if cached is not None:
                self._weight_cache.move_to_end(key)
                return cached

        # Dequantizzazione fuori lock: gli expert diversi possono dequantizzare
        # in parallelo senza serializzarsi. Il peso gia' in cache vince al rientro.
        layer_types = self.tensor_types.get(key.layer, {})
        gate = self._decode(self.store.read_tensor(key, "gate"),
                            layer_types.get("gate"),
                            (intermediate_size, hidden_size))
        up = self._decode(self.store.read_tensor(key, "up"),
                          layer_types.get("up"),
                          (intermediate_size, hidden_size))
        down = self._decode(self.store.read_tensor(key, "down"),
                            layer_types.get("down"),
                            (hidden_size, intermediate_size))

        weights = (gate, up, down)
        with self._lock:
            existing = self._weight_cache.get(key)
            if existing is not None:
                self._weight_cache.move_to_end(key)
                self._release(weights)
                return existing
            self._weight_cache[key] = weights
            if self.cache_capacity and len(self._weight_cache) > self.cache_capacity:
                _, oldest_weights = self._weight_cache.popitem(last=False)
                self._release(oldest_weights)
            return weights

    def _release(self, weights) -> None:
        with self._memory_lock:
            if self._active:
                self._retired.append(weights)
            else:
                self._recycle(weights)

    def _recycle(self, weights) -> None:
        """Ricicla i buffer di un expert evictato.

        Chiamata solo quando nessun forward puo' leggere i buffer ritirati.
        ``_own_weights`` ne rinvia il riciclo fino all'ultimo worker attivo.

        I pool sono limitati a pochi buffer per classe: servono a
        evitare i page fault delle allocazioni nuove, non a trattenere
        memoria. Il limite non dipende da ``cache_capacity``: con
        capacity=700 trattenere 700 buffer per classe (2 classi da
        ~17 MB) chiede ~23 GB e il sistema va in paging, misurato:
        il decode passava da 7,2 s a 23,7 s a token. Una manciata di
        buffer per classe e' tutto ciò che serve, il riciclo e'
        sequenziale.
        """
        if weights is None:
            return
        for array in weights:
            if not array.flags.writeable:
                continue
            n = array.size
            slot = self._pool.get(n)
            if slot is None:
                self._pool[n] = [array]
            elif len(slot) < _POOL_MAX:
                slot.append(array)

    def _acquire(self, n_elements: int) -> np.ndarray:
        with self._memory_lock:
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
        with self._lock:
            return len(self._weight_cache)

    def clear_cache(self) -> None:
        with self._lock, self._memory_lock:
            if self._active:
                raise RuntimeError("cannot clear weights during inference")
            self._weight_cache.clear()
            self._pool.clear()
            self._retired.clear()

    def prefetch_weights(self, keys):
        with self._lock:
            needed = [key for key in keys if key not in self._weight_cache]
        discard = getattr(self.store, "discard_prefetch_except", None)
        if discard is not None:
            discard(needed)
        return self.store.prefetch_many(needed)

    def begin_decode(self):
        if self.native_expert is not None:
            self.clear_cache()
            self._native_enabled = True

    def begin_prefill(self):
        self._native_enabled = False

    def _native_forward(self, key, activation, hidden_size, intermediate_size):
        return self.native_expert.forward(
            self.store.read_all(key), self.tensor_types[key.layer],
            activation, hidden_size, intermediate_size,
        )

    def close(self):
        self._moe_executor.shutdown(wait=True)
        self.clear_cache()

    # ------------------------------------------------------------------ forward

    @_own_weights
    def forward(self, key: ExpertKey, activation: list[float] | np.ndarray,
                hidden_size: int, intermediate_size: int) -> np.ndarray:
        if self._native_enabled:
            return self._native_forward(key, activation, hidden_size, intermediate_size)
        gate, up, down = self._weights(key, hidden_size, intermediate_size)
        request = self._get_request(hidden_size, intermediate_size)

        if isinstance(activation, list):
            x = np.ascontiguousarray(activation, dtype=np.float32).reshape(1, hidden_size)
        else:
            x = np.ascontiguousarray(activation, dtype=np.float32).reshape(1, hidden_size)

        feed = {
            "activation": x,
            "gate": gate,
            "up": up,
            "down": down,
        }
        result = self._infer(request, feed)
        return np.asarray(list(result.values())[0], dtype=np.float32).reshape(hidden_size)

    @_own_weights
    def forward_batch(self, key: ExpertKey, activations: np.ndarray,
                      hidden_size: int, intermediate_size: int) -> np.ndarray:
        """Run one expert for multiple prompt tokens with a shared weight load."""
        x = np.ascontiguousarray(activations, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != hidden_size:
            raise ValueError(
                f"expert activations must have shape (tokens, {hidden_size}), "
                f"got {x.shape}"
            )
        if x.shape[0] == 0:
            return np.empty((0, hidden_size), dtype=np.float32)

        gate, up, down = self._weights(key, hidden_size, intermediate_size)
        request = self._get_batch_request(hidden_size, intermediate_size)
        feed = {
            "activation": x,
            "gate": gate,
            "up": up,
            "down": down,
        }
        result = self._infer(request, feed)
        return np.asarray(
            list(result.values())[0], dtype=np.float32
        ).reshape(x.shape[0], hidden_size)

    def _infer(self, request, feed: dict):
        """Inferenza senza copia degli input.

        ``share_inputs=True`` dice a OpenVINO di avvolgere gli array numpy
        senza copiarne il contenuto in un tensor proprio. La copia costa
        quanto il calcolo stesso: misurata sul grafo SwiGLU f16 di questo
        modello, 6 MiB di pesi per expert, infer media 1,024 ms con la copia
        e 0,630 ms senza, su 320 esperti per token sono 0,126 s/token.

        I numeri non cambiano: gli stessi byte finiscono nel tensor, cambia
        solo chi li sposta. Gli array restano di sola lettura per la
        durata della chiamata, che e' sincrona, e i pesi sono gia' in cache.
        """
        if self._share_inputs:
            try:
                return request.infer(feed, share_inputs=True)
            except TypeError:
                # OpenVINO troppo vecchio per il parametro: si resta sul
                # percorso con copia, che e' piu' lento ma identico.
                self._share_inputs = False
        return request.infer(feed)
