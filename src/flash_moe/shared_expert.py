"""Shared expert: il contributo che ogni token attraversa sempre.

Perche' serve
-------------
In Qwen3/Qwen3.5 MoE ogni layer non esegue solo gli 8 expert selezionati dal
router. C'e' anche un *shared expert*, un MLP denso che **ogni token**
attraversa, il cui output viene aggiunto a quello degli expert routed:

    out = sum_k(weight_k * expert_k(x)) + sigmoid(W_gate_inp . x) * swiglu_shexp(x)

Ogni token riceve sempre questo contributo, indipendentemente da quali
expert il router ha scelto. Ignorarlo non e' un'approssimazione: cambia
l'output di ogni token, quindi il modello non funziona.

Semantica (concordata da llama.cpp qwen2moe.cpp, TensorRT-LLM
modeling_qwen_moe.py e transformers modular_qwen3_moe.py):

* ``ffn_gate_inp_shexp.weight`` e' un vettore F32 di hidden elementi. Il suo
  prodotto scalare con l'attivazione passa una sigmoid: e' un *gate* che
  scala il contributo dello shared expert, non un router.
* ``ffn_gate_shexp``/``ffn_up_shexp``/``ffn_down_shexp`` sono un SwiGLU come
  quello degli expert, ma con una sola copia di pesi per layer e senza
  routing. In llama.cpp il gate e' ``silu(g)/g``, che e' la sigmoid.

Forme in questo modello (hidden=2048, n_ff_shexp=512, Q6_K):

    ffn_gate_inp_shexp  [2048]            F32
    ffn_gate_shexp      [2048, 512]       Q6_K
    ffn_up_shexp        [2048, 512]       Q6_K
    ffn_down_shexp      [512, 2048]       Q6_K

I pesi sono gli stessi tensor degli expert, quindi si leggono con lo stesso
meccanismo a slice del manifest e si dequantizzano con lo stesso kernel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .dequant import dequantize_to, name_of

# Ruoli logici -> nome del tensor nel GGUF. Lo shared expert ha un solo gate
# di scala (inp) e i tre pesi del SwiGLU.
SHARED_ROLES: dict[str, str] = {
    "gate_inp": "ffn_gate_inp_shexp.weight",
    "gate": "ffn_gate_shexp.weight",
    "up": "ffn_up_shexp.weight",
    "down": "ffn_down_shexp.weight",
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid stabile: exp non deve overfloware su valori negativi grandi."""
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def shared_expert_slices(tensor_map: dict, layer: int) -> dict[str, dict]:
    """Offset/dimensioni dei tensor shared expert di un layer, letti dal GGUF.

    Restituisce solo i tensor presenti: un modello senza shared expert non
    deve fallire qui, perche' l'assenza e' un caso valido.
    """
    out: dict[str, dict] = {}
    for role, name in SHARED_ROLES.items():
        tensor = tensor_map.get(f"blk.{layer}.{name}")
        if tensor is None:
            continue
        out[role] = {
            "offset": int(tensor.data_offset),
            "nbytes": int(tensor.n_bytes),
            "ggml_type": int(tensor.tensor_type),
            # La forma in numpy e' l'inversa di quella descritta dal GGUF,
            # perche' e' la forma in cui i dati stanno nel file. llama.cpp
            # crea questi tensor come {n_embd, n_ff}, quindi con l'hidden
            # sull'asse veloce e l'n_ff come riga: in memoria sono
            # (n_ff, n_embd). Usare la forma del GGUF non da' errore, da'
            # la proiezione di un altro tensore (a ffn_gate la somma passava
            # da -893,15 a 7,24).
            "shape": tuple(int(v) for v in reversed(tensor.shape)),
        }
    return out


def shared_expert_present(tensor_map: dict, layer: int) -> bool:
    """True se il layer ha i tre pesi del SwiGLU dello shared expert.

    Non basta il solo ``gate_inp``: senza i tre pesi non c'e' niente da
    eseguire, e trattarlo come presente produrrebbe output nullo silenzioso.
    """
    names = tensor_map
    return all(f"blk.{layer}.{SHARED_ROLES[r]}" in names
               for r in ("gate", "up", "down"))


class SharedExpert:
    """Shared expert di tutti i layer, con i pesi tenuti in memoria.

    Sono 4 tensor per layer (2,5 MiB per layer in Q6_K, ~100 MiB per i 40
    layer in f16), e servono a ogni token, quindi non c'e' nulla da guadagnare
    tenendoli su disco: si caricano una volta e restano. Si caricano per
    intero e non a slice, perche' sono piccoli e servono sempre.

    **I pesi si materializzano in f32 al caricamento, non a ogni token.** Il
    calcolo e' un BLAS che richiede f32, e ``array.astype(np.float32)`` su 3
    tensor da 1M di elementi costa 2,08 ms per tensor: rifatto a ogni token
    erano 6,8 ms per layer, cioe' 309 ms per i 40 layer, 3,2 tok/s da soli,
    di cui 250 ms/token sono solo il cast. La conversione e' esatta (f16 -> f32
    non perde cifre) e i numeri non cambiano: cambia solo quando avviene.
    Costo: 240 MiB di f16 diventano 480 MiB di f32, e il cast di prima fascia
    finisce nel tempo di caricamento.
    """

    def __init__(self, gguf_path: str | Path, num_layers: int,
                 hidden_size: int, intermediate_size: int,
                 dequant_fn: Any = None) -> None:
        import gguf

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._dequant_fn = dequant_fn
        reader = gguf.GGUFReader(gguf_path)
        tensor_map = {t.name: t for t in reader.tensors}

        self.layers: dict[int, dict] = {}
        self.missing: list[int] = []
        for layer in range(num_layers):
            if not shared_expert_present(tensor_map, layer):
                self.missing.append(layer)
                continue
            weights: dict[str, np.ndarray] = {}
            for role, info in shared_expert_slices(tensor_map, layer).items():
                tensor = tensor_map[f"blk.{layer}.{SHARED_ROLES[role]}"]
                weights[role] = self._load(tensor.data, info["ggml_type"],
                                           info["shape"])
            # Da qui in avanti i pesi sono f32: il BLAS li vuole cosi' e il
            # cast e' gia' stato fatto. Le copie f16 non servono piu'.
            self.layers[layer] = {
                role: np.ascontiguousarray(array, dtype=np.float32)
                for role, array in weights.items()
            }

        if self.missing:
            raise KeyError(
                f"shared expert mancante nei layer {self.missing}: senza i suoi "
                f"pesi l'output di ogni token e' sbagliato, non approssimato"
            )

    def _load(self, data: Any, ggml_type: int,
              shape: tuple[int, ...]) -> np.ndarray:
        name = name_of(ggml_type)

        # F32/F16 non sono quantizzati: gguf li restituisce gia' col loro
        # dtype, e vanno solo convertiti in f16. Passarli dal kernel K-quant
        # darebbe un array di dimensione sbagliata, perche' il kernel
        # ricostruisce blocchi da 256 elementi.
        if name in ("F32", "F16"):
            dtype = np.float32 if name == "F32" else np.float16
            arr = np.ascontiguousarray(data, dtype=dtype)
            if arr.size != int(np.prod(shape)):
                raise ValueError(
                    f"{name} di {arr.size} elementi, attesi "
                    f"{int(np.prod(shape))} per {shape}"
                )
            return np.ascontiguousarray(arr, dtype=np.float16).reshape(*shape)

        raw = data if isinstance(data, (bytes, bytearray)) else \
            np.ascontiguousarray(data, dtype=np.uint8)
        n_elements = int(np.prod(shape))
        # gguf restituisce i tensor quantizzati con la shape del modello
        # ((hidden, n_ff)), quindi va appiattito: il kernel scrive in un
        # buffer piatto, e dequantize_to su un array 2D produce una forma
        # 2D che non entra nel reshape successivo.
        flat_raw = np.frombuffer(raw.tobytes(), dtype=np.uint8) \
            if not isinstance(raw, (bytes, bytearray)) else \
            np.frombuffer(raw, dtype=np.uint8)
        if self._dequant_fn is not None and len(shape) > 1:
            out = self._dequant_fn(flat_raw.tobytes(), ggml_type, n_elements)
        else:
            out = dequantize_to(flat_raw, ggml_type, "f16")
        out = np.asarray(out, dtype=np.float16).reshape(-1)
        if out.size != n_elements:
            raise ValueError(
                f"dequantizzazione ha prodotto {out.size} elementi, attesi "
                f"{n_elements} per {shape}"
            )
        return np.ascontiguousarray(out, dtype=np.float16).reshape(*shape)

    def has(self, layer: int) -> bool:
        return layer in self.layers

    def forward(self, layer: int, activation: list[float] | np.ndarray
                ) -> np.ndarray | None:
        """Contributo dello shared expert al layer, gia' scalato dal gate.

        Restituisce ``None`` se il layer non ha shared expert, cosi' il
        chiamante puo' sommare senza controllare la presenza.
        """
        weights = self.layers.get(layer)
        if weights is None:
            return None

        x = np.asarray(activation, dtype=np.float32).reshape(-1)
        if x.size != self.hidden_size:
            raise ValueError(
                f"attivazione con {x.size} elementi, attesi {self.hidden_size}"
            )

        # gate di scala: prodotto scalare poi sigmoid, come in llama.cpp
        # np.asarray, non astype: su un array gia' f32 non copia niente. Se un
        # test o un caller riassegna un peso in f16, la conversione torna a
        # farlo qui, una volta sola per chiamata e non per token.
        gate = _sigmoid((x @ np.asarray(weights["gate_inp"], dtype=np.float32)).reshape(()))

        # gate/up sono (n_ff, n_embd) e down e' (n_embd, n_ff), cioe' come
        # stanno in memoria: il prodotto contrae l'hidden, quindi si traspone
        # la vista e non l'array, e BLAS lo fa senza copia.
        g = x @ np.asarray(weights["gate"], dtype=np.float32).T
        act = (g * _sigmoid(g)) * (x @ np.asarray(weights["up"], dtype=np.float32).T)
        out = act @ np.asarray(weights["down"], dtype=np.float32).T

        return (gate * out).astype(np.float32)