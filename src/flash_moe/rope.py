from __future__ import annotations

import numpy as np

DEFAULT_HEAD_DIM: int = 128
DEFAULT_HEAD_COUNT: int = 16
DEFAULT_KV_HEAD_COUNT: int = 2
DEFAULT_ROPE_DIM: int = 64
DEFAULT_FREQ_BASE: float = 1000000.0


def _rope_frequencies(rope_dim: int, freq_base: float, device: str = "cpu") -> np.ndarray:
    """Computes the inverse-frequency array for RoPE.

    Returns ``theta[i] = 1 / (freq_base ** (2i / rope_dim))`` for i in
    ``[0, rope_dim)``.  The array shape is ``(rope_dim,)``; even indices are
    the cosine components, odd indices the sine components in the interleaved
    layout used by GPT-NeoX / Qwen.
    """
    positions = np.arange(0, rope_dim, 2, dtype=np.float32)
    freqs = 1.0 / (freq_base ** (positions / rope_dim))
    return freqs


class RoPE:
    """Rotary Positional Embedding (GPT-NeoX interleaved layout).

    The model has ``head_dim = 128`` and ``rope_dimension = 64``, meaning
    only the first ``rope_dim`` entries of each head receive rotation; the
    remaining ``head_dim - rope_dim`` entries pass through unchanged.
    This is the *partial rotary* convention used by Qwen3.
    """

    def __init__(
        self,
        head_dim: int = DEFAULT_HEAD_DIM,
        rope_dim: int = DEFAULT_ROPE_DIM,
        freq_base: float = DEFAULT_FREQ_BASE,
        max_seq_len: int = 4096,
    ) -> None:
        if rope_dim > head_dim:
            raise ValueError(f"rope_dim ({rope_dim}) cannot exceed head_dim ({head_dim})")
        if rope_dim % 2 != 0:
            raise ValueError(f"rope_dim ({rope_dim}) must be even")
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.freq_base = freq_base
        self._freqs = _rope_frequencies(rope_dim, freq_base)
        self._sin_cache: dict[int, np.ndarray] = {}
        self._cos_cache: dict[int, np.ndarray] = {}

    def _compute_cache(self, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        sin = self._sin_cache.get(seq_len)
        cos = self._cos_cache.get(seq_len)
        if sin is not None and cos is not None:
            return cos, sin

        positions = np.arange(seq_len, dtype=np.float32)
        freqs = np.outer(positions, self._freqs)  # (seq_len, rope_dim // 2)
        # GPT-NeoX interleaved layout: sin/cos table has rope_dim entries,
        # where sin[p, 2i] == sin[p, 2i+1] == sin(freq_i * p).
        sin_full = np.repeat(np.sin(freqs), 2, axis=1)       # (seq_len, rope_dim)
        cos_full = np.repeat(np.cos(freqs), 2, axis=1)       # (seq_len, rope_dim)

        sin = np.zeros((seq_len, self.head_dim), dtype=np.float32)
        cos = np.zeros((seq_len, self.head_dim), dtype=np.float32)
        sin[:, : self.rope_dim] = sin_full
        cos[:, : self.rope_dim] = cos_full
        sin[:, self.rope_dim :] = 0.0
        cos[:, self.rope_dim :] = 1.0

        self._sin_cache[seq_len] = sin
        self._cos_cache[seq_len] = cos
        return cos, sin

    def get_cache(self, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns ``(cos, sin)`` arrays of shape ``(seq_len, head_dim)``."""
        return self._compute_cache(seq_len)

    def apply_rotary(
        self,
        x: np.ndarray,
        seq_offset: int = 0,
    ) -> np.ndarray:
        """Applies RoPE to ``x`` of shape ``(..., head_dim)``.

        ``seq_offset`` is the absolute position offset when appending to a
        KV cache (i.e. the current token ordinal *before* this one).
        """
        head_dim = x.shape[-1]
        if head_dim != self.head_dim:
            raise ValueError(f"expected head_dim={self.head_dim}, got {head_dim}")
        seq_len = x.shape[-2] if x.ndim >= 2 else 1
        total_len = seq_offset + seq_len
        cos, sin = self._compute_cache(total_len)
        if seq_offset > 0:
            cos = cos[seq_offset:total_len]
            sin = sin[seq_offset:total_len]

        x1 = x[..., : self.rope_dim]
        x2 = x[..., self.rope_dim :]

        rot = np.stack([-x1[..., 1::2], x1[..., 0::2]], axis=-1)
        rot = rot.reshape(x1.shape[:-1] + (self.rope_dim,))
        rotated = np.concatenate(
            [x1 * cos[..., : self.rope_dim] + rot * sin[..., : self.rope_dim], x2],
            axis=-1,
        )
        return rotated
