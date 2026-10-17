from __future__ import annotations

import time

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
        dequant_fn=None,
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
            if dequant_fn is not None:
                weight = dequant_fn(
                    raw, tensor_type, vocab_size * hidden_size
                ).reshape(vocab_size, hidden_size)
            else:
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

    The output matrix is kept in f16 to reduce resident memory by about 1 GiB
    on the tested model. Each chunk is converted to f32 before multiplication,
    preserving the same weights and accumulation precision as the f32 path.
    """

    def __init__(self, weight: np.ndarray, vocab_size: int, hidden_size: int,
                 device: str = "GPU",
                 startup_timings: dict[str, float] | None = None) -> None:
        if weight.shape != (vocab_size, hidden_size):
            raise ValueError(
                f"lm_head weight shape {weight.shape} != "
                f"({vocab_size}, {hidden_size})"
            )
        self.weight = np.ascontiguousarray(weight, dtype=np.float16)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.chunk_rows = 8192
        self.compiled_model = None
        self.infer_request = None

        compile_started = time.perf_counter()
        # Try compiling OpenVINO MatMul for ultra-fast generation (e.g. 14ms vs 1500ms on Arc GPU)
        try:
            import openvino as ov
            import openvino.opset13 as ops
            core = ov.Core()
            dev = device if device in core.available_devices else ("CPU" if "CPU" in core.available_devices else None)
            if dev is not None:
                x = ops.parameter([1, self.hidden_size], ov.Type.f16, name="hidden")
                w = ops.constant(self.weight, ov.Type.f16, name="lm_head_w")
                matmul = ops.matmul(x, w, False, True)
                m = ov.Model([matmul], [x], "lm_head_matmul")
                self.compiled_model = core.compile_model(m, dev)
                self.infer_request = self.compiled_model.create_infer_request()
                self._ov_device = dev
        except Exception as e:
            self.compiled_model = None
            self.infer_request = None
        finally:
            if startup_timings is not None:
                startup_timings["openvino_compile"] = (
                    time.perf_counter() - compile_started
                )

    @classmethod
    def from_gguf_tensor(
        cls,
        tensor,
        tensor_type: int | None = None,
        device: str = "GPU",
        startup_timings: dict[str, float] | None = None,
        dequant_fn=None,
    ) -> "LMHead":
        if tensor_type is None:
            tensor_type = int(tensor.tensor_type)

        shape = tuple(int(s) for s in reversed(tensor.shape))
        vocab_size, hidden_size = shape

        dequant_started = time.perf_counter()
        if tensor_type == 1:  # F16
            weight = tensor.data.astype(np.float16)
        elif tensor_type in (0,):  # F32
            weight = np.asarray(tensor.data, dtype=np.float16)
        else:
            raw = tensor.data
            if hasattr(raw, "tobytes"):
                raw = raw.tobytes()
            if dequant_fn is not None:
                weight = dequant_fn(
                    raw, tensor_type, vocab_size * hidden_size
                ).reshape(vocab_size, hidden_size)
            else:
                weight = dequantize_to(
                    np.frombuffer(raw, dtype=np.uint8),
                    tensor_type,
                    "f16",
                ).reshape(vocab_size, hidden_size)

        if startup_timings is not None:
            startup_timings["dequant"] = time.perf_counter() - dequant_started
        return cls(weight, vocab_size, hidden_size, device=device,
                   startup_timings=startup_timings)

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

        # Fast path with OpenVINO if decoding single token and model available
        if self.infer_request is not None and h.shape[0] == 1:
            try:
                res = self.infer_request.infer({self.compiled_model.input(0): h.astype(np.float16)})
                logits = np.asarray(res[self.compiled_model.output(0)], dtype=np.float32)
                logits = logits.reshape(*hidden.shape[:-1], self.vocab_size)
                if squeezed:
                    logits = logits.squeeze()
                return logits
            except Exception:
                pass

        logits = np.empty((h.shape[0], self.vocab_size), dtype=np.float32)
        for start in range(0, self.vocab_size, self.chunk_rows):
            stop = min(start + self.chunk_rows, self.vocab_size)
            block = self.weight[start:stop].astype(np.float32)
            logits[:, start:stop] = h @ block.T

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
