"""Router MoE: top-k expert e pesi, con i gate veri del GGUF.

Il gate e' ``blk.N.ffn_gate_inp.weight``, ne=(2048, 256) F32, **uno per
layer**: 40 matrici diverse. Un router unico condiviso fra i layer
selezionerebbe lo stesso set di expert in ogni strato, che non e' quello
che fa il modello.

Sul softmax
-----------
llama.cpp calcola ``softmax`` su tutti i 256 expert, prende i top-k e
rinormalizza. Qui si prende prima il top-k e poi si applica il softmax
solo su quelli. E' la stessa cosa, e non e' un'approssimazione: il
denominatore della rinormalizzazione finale coinvolge solo gli expert
seletti, quindi ``exp(z_i) / sum_{j in T} exp(z_j)`` e' lo stesso
numero in entrambi gli ordini. Il vantaggio e' che non si calcola
softmax su 256 numeri per averne 8.

Sul f16
-------
I gate sono 2048x256 per layer, 40 layer: 20.97M parametri. Si preserva
prima l'arrotondamento scelto da gate_dtype, poi si conserva una sola copia
f32 (80 MiB invece di 40 MiB f16). Il prodotto e il confronto degli score
restano f32: la promozione una volta sola evita conversioni per ogni token
senza modificare i valori arrotondati usati dal routing precedente.
"""

from __future__ import annotations

import numpy as np


class ExpertRouter:
    """Top-k expert selection sui gate di un singolo layer.

    Un istanza per layer. ``route`` e' puro e deterministico: stessa
    attivazione, stesso output, senza stato interno.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        gate_weights: np.ndarray | None = None,
        gate_dtype: np.dtype | type = np.float16,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        if num_experts < 1:
            raise ValueError(f"num_experts must be at least 1, got {num_experts}")
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.gate_dtype = np.dtype(gate_dtype)
        # Preserve the configured rounding, then promote once rather than
        # allocating a new FP32 matrix for every routed token.
        self.gate_weights = np.asarray(self._normalize_gate(
            gate_weights, hidden_size, num_experts, self.gate_dtype
        ), dtype=np.float32)

    @staticmethod
    def _normalize_gate(
        gate_weights: np.ndarray | None,
        hidden_size: int,
        num_experts: int,
        gate_dtype: np.dtype,
    ) -> np.ndarray:
        if gate_weights is None:
            raise ValueError(
                "ExpertRouter requires the layer's real ffn_gate_inp weights. "
                "A random gate does not reproduce the model's routing: it "
                "selects experts the model would never have chosen, and it "
                "makes the prefetch target meaningless."
            )
        w = np.asarray(gate_weights, dtype=gate_dtype)
        if w.shape != (hidden_size, num_experts):
            # Il GGUF tiene il gate come (n_expert, n_embd): l'asse degli
            # esperti e' quello veloce, quindi va trasposto per usarlo come
            # hidden x expert.
            if w.shape == (num_experts, hidden_size):
                w = w.T
            else:
                raise ValueError(
                    f"gate weight shape {w.shape} is neither "
                    f"({hidden_size}, {num_experts}) nor "
                    f"({num_experts}, {hidden_size})"
                )
        if not np.isfinite(np.asarray(w, dtype=np.float32)).all():
            raise ValueError("gate weights contain NaN or Inf")
        return np.ascontiguousarray(w, dtype=gate_dtype)

    @classmethod
    def from_gguf_tensor(
        cls,
        tensor,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        gate_dtype: np.dtype | type = np.float16,
    ) -> "ExpertRouter":
        """Costruisce il router di un layer da ``ffn_gate_inp.weight``.

        Il tensor e' F32 nel GGUF; la precisione di storage e' un parametro
        perche' cambia il routing: in f16 una coppia di expert con punteggio
        vicino si ribalta, e con 2048 termini per il prodotto l'errore
        relativo del gate si accumula. Vedere tools/test_router.py, che
        misura la frequenza con cui accade.

        Nei log di llama.cpp il gate appare come
        ``MUL_MAT(blk.0.ffn_gate_inp.weight{2048, 256}, ...) = {256, ...}``:
        l'asse veloce del tensore e' l'hidden, quindi in memoria e'
        ``(n_expert, n_hidden)`` e non ``(n_hidden, n_expert)``. Usare la
        forma del GGUF cosi' com'e' scambia i due assi: il prodotto
        ``x @ gate`` non da' piu' i punteggi degli expert, e perche' i
        punteggi escano sbagliati restano quasi tutti uguali, i pesi di
        routing diventano quasi uniformi (0.127, 0.126, 0.125...) invece
        che distinti. Il reshape mette in fila quello che c'e' davvero, e
        ``_normalize_gate`` fa la trasposizione.
        """
        shape = tuple(int(s) for s in reversed(tensor.shape))
        if shape not in ((num_experts, hidden_size), (hidden_size, num_experts)):
            raise ValueError(
                f"ffn_gate_inp shape {shape} does not match "
                f"{num_experts} experts x {hidden_size} hidden"
            )
        gate = np.asarray(tensor.data)
        if gate.size != num_experts * hidden_size:
            raise ValueError(
                f"ffn_gate_inp has {gate.size} elements, expected "
                f"{num_experts * hidden_size}"
            )
        return cls(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            gate_weights=gate.reshape(shape).astype(gate_dtype, copy=False),
            gate_dtype=gate_dtype,
        )

    def scores(self, activation: list[float] | np.ndarray) -> np.ndarray:
        """Punteggio di tutti gli expert, in f32. Non seleziona."""
        x = np.asarray(activation, dtype=np.float32).reshape(-1)
        if x.size != self.hidden_size:
            raise ValueError(
                f"activation has {x.size} elements, expected {self.hidden_size}"
            )
        # x @ gate in f32; la promozione avviene sull'array del gate solo se
        # non e' gia' f32.
        return (x @ np.asarray(self.gate_weights, dtype=np.float32)).astype(np.float32)

    def route(self, activation: list[float] | np.ndarray) -> tuple[list[int], list[float]]:
        """Restituisce (indici dei top-k, pesi normalizzati che sommano a 1)."""
        scores = self.scores(activation)

        # argsort stabile sui soli top-k: il confronto e' in f32.
        top_k_indices = np.argsort(scores)[-self.top_k:][::-1].astype(np.int64).tolist()
        top_k_scores = scores[top_k_indices].astype(np.float64)

        exp_scores = np.exp(top_k_scores - np.max(top_k_scores))
        total = exp_scores.sum()
        if not np.isfinite(total) or total <= 0.0:
            # Tutti i top-k a -inf: il softmax non ha un modo naturale di
            # ripartire. Si ripartisce uniformemente invece di restituire
            # NaN, che poi finirebbe negli hidden senza che nessuno se ne
            # accorga.
            weights = np.full(len(top_k_indices), 1.0 / len(top_k_indices))
        else:
            weights = exp_scores / total

        return top_k_indices, [float(w) for w in weights]


def softmax_all_experts(scores: np.ndarray) -> np.ndarray:
    """Reference: softmax su tutti gli expert, per il test di equivalenza."""
    s = np.asarray(scores, dtype=np.float64)
    e = np.exp(s - s.max())
    return e / e.sum()


def routers_from_gguf(
    tmap: dict,
    num_layers: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    arch: str = "qwen35moe",
    gate_dtype: np.dtype | type = np.float16,
) -> dict[int, ExpertRouter]:
    """Costruisce il router di ogni layer da ``blk.N.ffn_gate_inp.weight``.

    Punto unico di lettura dei gate: lo usano sia il modello completo sia i
    runner, cosi' il prefetch e la selezione degli expert non possono
    divergere perche' leggono i pesi in modo diverso.
    """
    routers: dict[int, ExpertRouter] = {}
    missing: list[int] = []
    for layer in range(num_layers):
        gate_t = tmap.get(f"blk.{layer}.ffn_gate_inp.weight")
        if gate_t is None:
            missing.append(layer)
            continue
        routers[layer] = ExpertRouter.from_gguf_tensor(
            gate_t, hidden_size, num_experts, top_k, gate_dtype=gate_dtype
        )
    if missing:
        raise KeyError(
            f"manca ffn_gate_inp.weight per i layer {missing}: senza il gate "
            "non si puo' scegliere quali expert eseguire"
        )
    return routers


def synthetic_router(
    hidden_size: int,
    num_experts: int,
    top_k: int,
    seed: int = 0,
    layer: int | None = None,
) -> ExpertRouter:
    """Router con gate finto, deterministico. Solo per test e benchmark.

    Non riproduce il routing del modello: serve a esercitare store, cache e
    prefetch senza dipendere dal GGUF. Ogni layer riceve un seed diverso,
    altrimenti tutti i layer selezionerebbero gli stessi expert e il test
    non coprirebbe nulla sul routing per-layer.
    """
    rng = np.random.default_rng(seed if layer is None else seed + 1000 * layer)
    gate = rng.standard_normal((hidden_size, num_experts)).astype(np.float16)
    return ExpertRouter(hidden_size, num_experts, top_k=top_k, gate_weights=gate)
