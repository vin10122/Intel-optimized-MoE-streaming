from __future__ import annotations

import numpy as np

from .dequant import dequantize_to, name_of


class EmbeddingLookup:
    """Token embedding lookup with GGUF K-quant dequantization.

    ``token_embd.weight`` is stored as a single large quantized tensor
    (Q4_K, shape ``(vocab_size, hidden)``).  Unlike the MoE experts, the
    embedding is read once and cached as a contiguous f16 matrix; it does
    not participate in the expert LRU store.

    508M elementi in f32 sono 1940 MiB, in f16 970 MiB. Su un modello da
    16 GiB e 15 GiB di RAM la differenza decide se il prompt entra.
    """

    def __init__(self, weight: np.ndarray, vocab_size: int, hidden_size: int) -> None:
        if weight.shape != (vocab_size, hidden_size):
            raise ValueError(
                f"embedding weight shape {weight.shape} != "
                f"({vocab_size}, {hidden_size})"
            )
        self.weight = np.ascontiguousarray(weight, dtype=np.float16)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    @classmethod
    def from_gguf_tensor(
        cls,
        tensor,
        tensor_type: int | None = None,
        gguf_reader=None,
    ) -> "EmbeddingLookup":
        """Builds an embedding from a GGUF tensor.

        ``tensor`` can be:
        - A ``gguf.GGUFReaderTensor`` (uses ``tensor.data`` directly)
        - A ``(path, offset, nbytes, tensor_type)`` tuple for sliced access

        For F16 tensors (``tensor_type == 1``), the raw data is used directly.
        For quantized types, ``gguf.quants.dequantize`` is applied.
        """
        if tensor_type is None:
            tensor_type = int(tensor.tensor_type)

        shape = tuple(int(s) for s in reversed(tensor.shape))
        vocab_size, hidden_size = shape

        if tensor_type == 1:  # F16
            weight = tensor.data.astype(np.float16)
        elif tensor_type in (0,):  # F32
            weight = np.asarray(tensor.data, dtype=np.float16)
        else:
            raw = tensor.data
            if hasattr(raw, "tobytes"):
                raw = raw.tobytes()
            weight = dequantize_to(
                np.frombuffer(raw, dtype=np.uint8),
                tensor_type,
                "f16",
            ).reshape(vocab_size, hidden_size)

        return cls(weight, vocab_size, hidden_size)

    def forward(self, input_ids: np.ndarray | list[int] | int) -> np.ndarray:
        """Looks up embeddings for input token IDs.

        Args:
            input_ids: int, list of ints, or ``(batch, seq)`` array.

        Returns:
            ``(batch, seq, hidden)`` or ``(hidden,)`` for single token.
        """
        if isinstance(input_ids, int):
            ids = np.array([input_ids], dtype=np.int64)
        elif isinstance(input_ids, list):
            ids = np.array(input_ids, dtype=np.int64)
        else:
            ids = np.asarray(input_ids, dtype=np.int64)

        out = self.weight[ids]
        return out.squeeze() if out.size == self.hidden_size else out


class LMHead:
    """Language modeling head: hidden → logits.

    ``output.weight`` is stored as a large quantized tensor
    (Q6_K, shape ``(vocab_size, hidden)``).  The forward pass computes
    ``logits = activation @ weight.T``.

    **I pesi stanno in f32 dal caricamento.** Il prodotto è 248320 × 2048,
    cioè 508M di pesi: convertirli da f16 a f32 a ogni token costava 872 ms
    isolati e 1236 ms nel modello, cioè il 21% del token, per un GEMV che in
    BLAS f32 fa 31,5 ms. Con i pesi già in f32 la conversione non la fa
    nessuno: 27,7x, e il risultato è identico bit per bit perché f16 → f32 non
    perde cifre. Costo: 970 MiB di f16 diventano 1940 MiB di f32, e la
    dequantizzazione al caricamento deve ancora passare da f16, che è il
    percorso che il modello usava già: i valori sono quelli di prima, arrotondati
    a f16 nello stesso punto.
    """

    def __init__(self, weight: np.ndarray, vocab_size: int, hidden_size: int) -> None:
        if weight.shape != (vocab_size, hidden_size):
            raise ValueError(
                f"lm_head weight shape {weight.shape} != "
                f"({vocab_size}, {hidden_size})"
            )
        self.weight = np.ascontiguousarray(weight, dtype=np.float32)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # Il ciclo su blocchi resta, ma non serve piu' a contenere la memoria
        # di staging: il blocco e' una vista dei pesi, non una copia. Serve a
        # tenere ogni scrittura dei logits dentro una finestra contigua.
        # Cambiare il blocco cambierebbe il blocking del GEMM e con esso gli
        # ultimi bit del risultato, quindi il valore viene lasciato com'e'.
        self.chunk_rows = 8192

    @classmethod
    def from_gguf_tensor(
        cls,
        tensor,
        tensor_type: int | None = None,
    ) -> "LMHead":
        if tensor_type is None:
            tensor_type = int(tensor.tensor_type)

        shape = tuple(int(s) for s in reversed(tensor.shape))
        vocab_size, hidden_size = shape

        if tensor_type == 1:  # F16
            weight = tensor.data.astype(np.float16)
        elif tensor_type in (0,):  # F32
            weight = np.asarray(tensor.data, dtype=np.float16)
        else:
            raw = tensor.data
            if hasattr(raw, "tobytes"):
                raw = raw.tobytes()
            weight = dequantize_to(
                np.frombuffer(raw, dtype=np.uint8),
                tensor_type,
                "f16",
            ).reshape(vocab_size, hidden_size)

        return cls(weight, vocab_size, hidden_size)

    def forward(self, hidden: np.ndarray) -> np.ndarray:
        """Computes logits.

        Args:
            hidden: ``(batch, seq, hidden)`` or ``(hidden,)`` array.

        Returns:
            ``(batch, seq, vocab_size)`` or ``(vocab_size,)`` for single.
        """
        squeezed = False
        if hidden.ndim == 1:
            hidden = hidden.reshape(1, 1, -1)
            squeezed = True

        h = np.ascontiguousarray(hidden, dtype=np.float32).reshape(-1, self.hidden_size)
        logits = np.empty((h.shape[0], self.vocab_size), dtype=np.float32)
        for start in range(0, self.vocab_size, self.chunk_rows):
            stop = min(start + self.chunk_rows, self.vocab_size)
            # Il blocco e' una vista sui pesi f32: niente staging, niente cast.
            logits[:, start:stop] = h @ self.weight[start:stop].T

        logits = logits.reshape(*hidden.shape[:-1], self.vocab_size)
        if squeezed:
            logits = logits.squeeze()
        return logits


def argmax_sample(logits: np.ndarray, temperature: float = 1.0, top_k: int = 1) -> int:
    """Greedy decoding with optional temperature."""
    if temperature != 1.0:
        logits = logits / temperature
    return int(np.argmax(logits))


def sample(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    rng: np.random.Generator | None = None,
) -> int:
    """Top-k / top-p / temperature sampling.

    Args:
        logits: 1-D array of vocab_size logits.
        temperature: Softmax temperature (1.0 = no scaling, 0.0 = greedy).
        top_k: If > 0, keep only the top-k tokens.
        top_p: If > 0, keep smallest set with cumulative prob >= top_p (nucleus).
        rng: Optional PRNG for sampling; deterministic if None.

    Returns:
        Selected token ID.
    """
    if rng is None:
        rng = np.random.default_rng()

    logits = np.asarray(logits, dtype=np.float64)

    # temperature <= 0 vuol dire greedy, non dividere per zero. La divisione
    # produceva +/-inf, e inf - inf nel softmax e' NaN: il campionamento
    # falliva con "Probabilities contain NaN" invece di restituire il token
    # piu' probabile. E' il caso che si usa di piu' per una verifica.
    if temperature <= 0.0:
        return int(np.argmax(logits))

    if temperature != 1.0:
        logits = logits / temperature

    if not np.all(np.isfinite(logits)):
        raise ValueError(
            "logit non finiti: il modello ha prodotto NaN o Inf, e campionare "
            "su questi valori restituirebbe un token arbitrario senza dire nulla")

    sorted_ids = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_ids]

    keep = len(sorted_ids)
    if top_k > 0:
        keep = min(top_k, keep)
    if top_p > 0.0:
        # Il cumulativo si calcola SUL SET RIMASTO, non su tutta la
        # distribuzione. In llama.cpp top_p riceve i candidati che top_k ha
        # gia' ridotti (gli altri sono -INF) e llama_sampler_softmax_impl li
        # rinormalizza, quindi cum_sum cammina sul set top-k: vedi
        # llama_sampler_top_p_apply in llama-sampler.cpp:1556. Calcolarlo su
        # tutta la distribuzione e poi intersecarlo con top_k dava un insieme
        # piu' largo del richiesto, e si vedeva solo quando entrambi i filtri
        # erano attivi, cioe' esattamente i parametri di default di generate().
        # Con top_k=20 e top_p=0.90 il nucleo era di 15 token invece di 10:
        # non un token sbagliato, un campionamento piu' casuale di quanto
        # chiesto, che su un client streaming non si nota ma cambia l'output.
        head = sorted_logits[:keep]
        probs = np.exp(head - head[0])
        probs = probs / np.sum(probs)
        cumsum = np.cumsum(probs)
        top_p_cut = int(np.searchsorted(cumsum, top_p) + 1)
        keep = min(keep, top_p_cut)

    kept_ids = sorted_ids[:keep]
    kept_logits = sorted_logits[:keep]
    probs = np.exp(kept_logits - np.max(kept_logits))
    probs = probs / np.sum(probs)
    idx = int(rng.choice(len(probs), p=probs))
    return int(kept_ids[idx])
