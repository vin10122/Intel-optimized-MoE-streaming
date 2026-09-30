from __future__ import annotations

import numpy as np

from .rope import RoPE, DEFAULT_HEAD_DIM, DEFAULT_HEAD_COUNT, DEFAULT_KV_HEAD_COUNT, DEFAULT_ROPE_DIM, DEFAULT_FREQ_BASE
from .norm import RMSNorm

DEFAULT_HIDDEN_SIZE: int = 2048
DEFAULT_NUM_HEADS: int = 16
DEFAULT_NUM_KV_HEADS: int = 2
DEFAULT_MAX_SEQ_LEN: int = 2048


class KVCache:
    """Grouped-query attention KV cache.

    Stores ``(seq_len, num_kv_heads, head_dim)`` arrays per layer, growing
    as new tokens arrive.  Supports multi-query / multi-group attention
    where ``num_heads`` is a multiple of ``num_kv_heads``.
    """

    def __init__(self, num_kv_heads: int, head_dim: int, max_seq_len: int = DEFAULT_MAX_SEQ_LEN) -> None:
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.keys: np.ndarray | None = None
        self.values: np.ndarray | None = None
        self.seq_len: int = 0

    def allocate(self, batch_size: int = 1) -> None:
        # Cache in f16 come tutto il resto: 10 layer full-attention x 2048
        # posizioni x 2 head KV x 256 dim sono 40 MiB in f16, 80 MiB in f32.
        # Il promote a f32 avviene dentro l'attenzione, sui soli prodotti
        # scalari, che sono (seq, num_heads, head_dim).
        self.keys = np.zeros((self.max_seq_len, self.num_kv_heads, self.head_dim), dtype=np.float16)
        self.values = np.zeros((self.max_seq_len, self.num_kv_heads, self.head_dim), dtype=np.float16)
        self.seq_len = 0

    def append(self, k: np.ndarray, v: np.ndarray) -> None:
        """Append key/value tensors to the cache.

        ``k`` and ``v`` should have shape ``(num_kv_heads, head_dim)`` for
        single-token append, or ``(seq, num_kv_heads, head_dim)`` for
        multi-token append (matching the cache layout).
        """
        if self.keys is None:
            self.allocate()
        if k.ndim == 2:
            k = k.reshape(1, self.num_kv_heads, self.head_dim)
            v = v.reshape(1, self.num_kv_heads, self.head_dim)
        n_tokens = k.shape[0]

        if self.seq_len + n_tokens > self.max_seq_len:
            overflow = self.seq_len + n_tokens - self.max_seq_len
            self.keys[:-overflow] = self.keys[overflow:]
            self.values[:-overflow] = self.values[overflow:]
            self.seq_len = self.max_seq_len - n_tokens

        pos = self.seq_len
        self.keys[pos:pos + n_tokens] = k
        self.values[pos:pos + n_tokens] = v
        self.seq_len += n_tokens

    def get(self, start: int = 0) -> tuple[np.ndarray, np.ndarray]:
        if self.keys is None:
            return np.empty((0, self.num_kv_heads, self.head_dim)), np.empty((0, self.num_kv_heads, self.head_dim))
        return self.keys[:self.seq_len], self.values[:self.seq_len]

    def reset(self) -> None:
        self.keys = None
        self.values = None
        self.seq_len = 0


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    rope: RoPE | None = None,
    seq_offset: int = 0,
    causal_mask: bool = True,
) -> np.ndarray:
    """Scaled dot-product attention with optional RoPE.

    Args:
        q: (batch, num_heads, seq, head_dim)
        k: (batch, num_kv_heads, seq, head_dim)
        v: (batch, num_kv_heads, seq, head_dim)
        rope: If provided, applies rotary to q and k before computing attention.
        seq_offset: Position offset for KV cache (for autoregressive generation).
        causal_mask: If True, mask out future positions.

    Returns:
        (batch, num_heads, seq, head_dim)
    """
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
    group_size = num_heads // num_kv_heads

    # Q/K/V arrivano dalla cache in f16: gli score e il softmax vanno in f32.
    # In f16 einsum e' anche molto piu' lento, numpy non ha BLAS a mezza
    # precisione.
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    if rope is not None:
        q_shape = q.shape
        k_shape = k.shape
        q = rope.apply_rotary(q.reshape(-1, q.shape[-2], q.shape[-1]), seq_offset)
        q = q.reshape(q_shape)
        k = rope.apply_rotary(k.reshape(-1, k.shape[-2], k.shape[-1]), seq_offset)
        k = k.reshape(k_shape)

    # Repeat K/V for grouped query attention
    if group_size > 1:
        k = np.repeat(k, group_size, axis=1)
        v = np.repeat(v, group_size, axis=1)

    # Q: (batch, heads, seq_q, head_dim)
    # K: (batch, heads, seq_k, head_dim)
    # Scores: (batch, heads, seq_q, seq_k)
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = np.einsum("bhqd,bhkd->bhqk", q, k) * scale

    if causal_mask:
        seq_q = scores.shape[-2]
        seq_k = scores.shape[-1]
        mask = np.triu(np.ones((seq_q, seq_k), dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)

    scores = softmax(scores, axis=-1)

    out = np.einsum("bhqk,bhkd->bhqd", scores, v)
    return out


class AttentionLayer:
    """Transformer attention layer supporting two Qwen3 MoE variants.

    Variant A (combined QKV, layers without QK-norm):
        - ``attn_qkv``: projected weight of shape ``(4 * hidden, hidden)``
          containing Q, K, V, and an extra gate in sequence.
        - ``attn_gate``: weight of shape ``(2 * hidden, hidden)``.

    Variant B (split Q/K/V, layers with QK-norm):
        - ``attn_q``, ``attn_k``, ``attn_v``: separate projections.
        - ``attn_q_norm``, ``attn_k_norm``: RMSNorm on Q and K.
        - ``attn_output``: output projection.

    Both variants use grouped-query attention (GQA).
    """

    def __init__(
        self,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_heads: int = DEFAULT_NUM_HEADS,
        num_kv_heads: int = DEFAULT_NUM_KV_HEADS,
        head_dim: int | None = None,
        kv_head_dim: int | None = None,
        rope_dim: int = DEFAULT_ROPE_DIM,
        freq_base: float = DEFAULT_FREQ_BASE,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.kv_head_dim = kv_head_dim or self.head_dim
        self.rope = RoPE(
            head_dim=max(self.head_dim, self.kv_head_dim),
            rope_dim=rope_dim,
            freq_base=freq_base,
            max_seq_len=max_seq_len,
        )
        self.q_norm: RMSNorm | None = None
        self.k_norm: RMSNorm | None = None

    def set_qk_norm(self, q_norm: RMSNorm, k_norm: RMSNorm) -> None:
        """Configure QK normalization (used by split QKV variant)."""
        self.q_norm = q_norm
        self.k_norm = k_norm

    def _reshape_qkv(self, x: np.ndarray, num_heads: int, head_dim: int) -> np.ndarray:
        """Reshape (batch, seq, num_heads * head_dim) → (batch, num_heads, seq, head_dim)."""
        b, s, _ = x.shape
        return x.reshape(b, s, num_heads, head_dim).transpose(0, 2, 1, 3)

    def forward_combined(
        self,
        x: np.ndarray,
        qkv_weight: np.ndarray,
        output_weight: np.ndarray,
    ) -> np.ndarray:
        """Forward pass for the combined QKV variant.

        ``qkv_weight`` has shape ``(4 * hidden, hidden)`` containing
        [Q, K, V, gate] concatenated.  ``output_weight`` has shape
        ``(hidden, 2 * hidden)``.
        """
        b, s, _ = x.shape
        qkv = x @ qkv_weight.T  # (b, s, 4*hidden)
        q, k, v, gate = np.split(qkv, 4, axis=-1)  # each (b, s, hidden)

        q = self._reshape_qkv(q, self.num_heads, self.head_dim)
        k = self._reshape_qkv(k, self.num_kv_heads, self.kv_head_dim)
        v = self._reshape_qkv(v, self.num_kv_heads, self.kv_head_dim)

        attn_out = scaled_dot_product_attention(q, k, v, rope=self.rope)
        b, h, s, d = attn_out.shape
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(b, s, h * d)

        # Gate is SiLU(gate) * attn_out (gated attention)
        gate = gate * (1.0 / (1.0 + np.exp(-gate)))
        attn_out = attn_out * gate

        out = attn_out @ output_weight.T  # (b, s, hidden)
        return out

    def forward_split(
        self,
        x: np.ndarray,
        q_weight: np.ndarray,
        k_weight: np.ndarray,
        v_weight: np.ndarray,
        output_weight: np.ndarray,
    ) -> np.ndarray:
        """Forward pass for the split Q/K/V variant with QK-norm.

        ``q_weight``: (num_heads * head_dim, hidden)
        ``k_weight``: (num_kv_heads * kv_head_dim, hidden)
        ``v_weight``: (num_kv_heads * kv_head_dim, hidden)
        ``output_weight``: (hidden, num_heads * head_dim)
        """
        b, s, _ = x.shape

        q = self._reshape_qkv(x @ q_weight.T, self.num_heads, self.head_dim)
        k = self._reshape_qkv(x @ k_weight.T, self.num_kv_heads, self.kv_head_dim)
        v = self._reshape_qkv(x @ v_weight.T, self.num_kv_heads, self.kv_head_dim)

        # Apply QK-norm if configured
        if self.q_norm is not None:
            b, h, s, d = q.shape
            q = self.q_norm.forward(q.reshape(-1, d)).reshape(b, h, s, d)
        if self.k_norm is not None:
            b, nh, s, d = k.shape
            k = self.k_norm.forward(k.reshape(-1, d)).reshape(b, nh, s, d)

        attn_out = scaled_dot_product_attention(q, k, v, rope=self.rope)
        b, h, s, d = attn_out.shape
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(b, s, h * d)

        out = attn_out @ output_weight.T  # (b, s, hidden)
        return out

    def forward(
        self,
        x: np.ndarray,
        *,
        qkv_weight: np.ndarray | None = None,
        q_weight: np.ndarray | None = None,
        k_weight: np.ndarray | None = None,
        v_weight: np.ndarray | None = None,
        output_weight: np.ndarray | None = None,
        gate_weight: np.ndarray | None = None,
        cache: KVCache | None = None,
        position_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """Unified forward — dispatches to combined or split based on provided weights.

        For the combined variant, pass ``qkv_weight`` and ``gate_weight`` (which
        acts as the output projection).  For the split variant, pass
        ``q_weight``, ``k_weight``, ``v_weight``, and ``output_weight``.
        """
        if qkv_weight is not None:
            return self.forward_combined(x, qkv_weight, gate_weight or output_weight)
        return self.forward_split(x, q_weight, k_weight, v_weight, output_weight)
