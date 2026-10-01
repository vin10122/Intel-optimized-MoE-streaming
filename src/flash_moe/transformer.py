from __future__ import annotations

import numpy as np

from typing import Any
from .key import ExpertKey
from .store import AsyncExpertStore
from .dequant import dequantize_to, name_of
from .norm import RMSNorm
from .rope import RoPE, DEFAULT_ROPE_DIM, DEFAULT_FREQ_BASE
from .attention import AttentionLayer, KVCache, scaled_dot_product_attention, softmax
from .backend_ov import OpenVINOExpertBackend
from .router import ExpertRouter
from .moe_layer import run_moe_layer


class TransformerBlock:
    """Full transformer block: attention + MoE feed-forward.

    Supports both attention variants found in the Qwen3 MoE model:
    - Combined QKV (30 layers): ``attn_qkv`` + ``attn_gate`` as output projection
    - Split Q/K/V (10 layers): ``attn_q/k/v/output`` + QK-norm
    """

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        kv_head_dim: int,
        rope_dim: int,
        freq_base: float,
        max_seq_len: int,
        attn_norm_weight: np.ndarray,
        post_attn_norm_weight: np.ndarray,
        qkv_weight: np.ndarray | None = None,
        attn_gate_weight: np.ndarray | None = None,
        q_weight: np.ndarray | None = None,
        k_weight: np.ndarray | None = None,
        v_weight: np.ndarray | None = None,
        attn_output_weight: np.ndarray | None = None,
        q_k_norm: np.ndarray | None = None,
        k_k_norm: np.ndarray | None = None,
        backend: OpenVINOExpertBackend | None = None,
        router: ExpertRouter | None = None,
        intermediate_size: int = 512,
        num_experts: int = 256,
        shared_expert: Any = None,
    ) -> None:
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts

        self.is_combined = qkv_weight is not None
        self.qkv_weight = qkv_weight
        self.attn_gate_weight = attn_gate_weight
        self.q_weight = q_weight
        self.k_weight = k_weight
        self.v_weight = v_weight
        self.attn_output_weight = attn_output_weight

        # Separate RoPE for attention heads
        # Combined variant: 128-dim heads, split variant: 256-dim heads
        attn_head_dim = 128 if self.is_combined else 256
        self.rope = RoPE(
            head_dim=attn_head_dim,
            rope_dim=rope_dim,
            freq_base=freq_base,
            max_seq_len=max_seq_len,
        )

        self.attn = AttentionLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_head_dim=kv_head_dim,
            rope_dim=rope_dim,
            freq_base=freq_base,
            max_seq_len=max_seq_len,
        )

        self.attn_norm = RMSNorm(attn_norm_weight, eps=1e-6)
        self.post_attn_norm = RMSNorm(post_attn_norm_weight, eps=1e-6)
        self.eps = 1e-6
        self.kv_cache: KVCache | None = None

        if q_k_norm is not None and k_k_norm is not None:
            self.q_norm = RMSNorm(q_k_norm, eps=1e-6)
            self.k_norm = RMSNorm(k_k_norm, eps=1e-6)
            self.attn.set_qk_norm(self.q_norm, self.k_norm)

        self.backend = backend
        self.router = router
        # Ogni token attraversa anche lo shared expert: senza, l'output del
        # layer e' sbagliato. Passato dal modello, che lo carica una volta.
        self.shared_expert = shared_expert

    def init_kv_cache(self, max_seq_len: int) -> None:
        if self.is_combined:
            # MHA: cache 16 heads × 128 head_dim
            self.kv_cache = KVCache(self.attn.num_heads, 128, max_seq_len)
        else:
            # GQA: cache 2 kv_heads × 256 kv_head_dim
            self.kv_cache = KVCache(self.attn.num_kv_heads, 256, max_seq_len)

    def _attention_forward(self, x: np.ndarray, seq_offset: int = 0) -> np.ndarray:
        """Runs attention forward, returning (batch, seq, hidden).

        x: (batch, seq, hidden)
        """
        b, s, h = x.shape

        if self.is_combined:
            qkv = x @ self.qkv_weight.T  # (b, s, 8192) = 4*hidden
            q, k, v, extra = np.split(qkv, 4, axis=-1)

            gate_proj = x @ self.attn_gate_weight.T  # (b, s, 4096) = 2*hidden
            gate1, gate2 = np.split(gate_proj, 2, axis=-1)  # each (b, s, hidden)

            q = self.attn._reshape_qkv(q, self.attn.num_heads, self.attn.head_dim)
            k = self.attn._reshape_qkv(k, self.attn.num_heads, self.attn.head_dim)
            v = self.attn._reshape_qkv(v, self.attn.num_heads, self.attn.head_dim)

            if self.kv_cache is not None and seq_offset > 0:
                new_k = k[:, :, -1:, :]
                new_v = v[:, :, -1:, :]
                self.kv_cache.append(new_k[0], new_v[0])
                k_full, v_full = self.kv_cache.get()
                k = k_full.reshape(1, self.attn.num_heads, -1, self.attn.head_dim)
                v = v_full.reshape(1, self.attn.num_heads, -1, self.attn.head_dim)

            attn_out = scaled_dot_product_attention(q, k, v, rope=self.rope, seq_offset=seq_offset)
            b, nh, s2, d = attn_out.shape
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(b, s2, nh * d)

            gate1_act = gate1 * (1.0 / (1.0 + np.exp(-gate1)))
            gate2_act = gate2 * (1.0 / (1.0 + np.exp(-gate2)))
            out = attn_out * gate1_act + extra * gate2_act
        else:
            # Split variant: Q head_dim=512, K/V head_dim=256, QK-norm=(256,)
            # Q is split into [Q_attn(256 per head), Q_gate(256 per head)]
            q_full = x @ self.q_weight.T  # (b, s, 8192)
            k_raw = x @ self.k_weight.T   # (b, s, 512)
            v_raw = x @ self.v_weight.T   # (b, s, 512)

            # Split Q into attention part (first 4096) and gate part (last 4096)
            q_attn, q_gate = np.split(q_full, 2, axis=-1)  # each (b, s, 4096)
            
            # Reshape for attention: 16 heads × 256
            q_attn = self.attn._reshape_qkv(q_attn, self.attn.num_heads, 256)
            k = self.attn._reshape_qkv(k_raw, self.attn.num_kv_heads, 256)
            v = self.attn._reshape_qkv(v_raw, self.attn.num_kv_heads, 256)

            # Apply QK-norm (RMSNorm with weight=256)
            if hasattr(self, 'q_norm') and self.q_norm is not None:
                b, nh, s3, d = q_attn.shape
                q_attn = self.q_norm.forward(q_attn.reshape(-1, d)).reshape(b, nh, s3, d)
            if hasattr(self, 'k_norm') and self.k_norm is not None:
                b, nkv, s3, d = k.shape
                k = self.k_norm.forward(k.reshape(-1, d)).reshape(b, nkv, s3, d)

            if self.kv_cache is not None and seq_offset > 0:
                new_k = k[:, :, -1:, :]
                new_v = v[:, :, -1:, :]
                self.kv_cache.append(new_k[0], new_v[0])
                k_full, v_full = self.kv_cache.get()
                k = k_full.reshape(1, self.attn.num_kv_heads, -1, 256)
                v = v_full.reshape(1, self.attn.num_kv_heads, -1, 256)

            # Use this block's RoPE (256-dim for split variant)
            attn_out = scaled_dot_product_attention(q_attn, k, v, rope=self.rope, seq_offset=seq_offset)
            b, nh, s2, d = attn_out.shape
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(b, s2, nh * d)  # (b, s, 4096)

            # Apply Q gate (SiLU)
            q_gate_act = q_gate * (1.0 / (1.0 + np.exp(-q_gate)))
            gated = attn_out * q_gate_act  # (b, s, 4096)

            # Output projection: (4096, 2048) → 2048
            out = gated @ self.attn_output_weight.T  # (b, s, 2048)

        return out

    def _moe_forward(self, x: np.ndarray) -> np.ndarray:
        """Output MoE del layer su tutta la sequenza.

        Chiama flash_moe.moe_layer.run_moe_layer, la stessa funzione usata da
        MoEEngine: il calcolo del layer MoE e' definito in un punto solo, con
        la somma dei top-k e il contributo dello shared expert. Qui non ci
        era nessuna copia da mantenere allineata.
        """
        if self.backend is None or self.router is None:
            return x  # identity if no MoE

        b, s, h = x.shape
        result = np.zeros_like(x)
        for si in range(s):
            activation = x[0, si, :]
            result[0, si, :] = run_moe_layer(
                layer=self.layer_idx,
                activation=activation,
                router=self.router,
                backend=self.backend,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                shared_expert=self.shared_expert,
            )
        return result

    def forward(
        self,
        x: np.ndarray,
        seq_offset: int = 0,
        moe_activation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Full transformer block forward.

        Architecture (Qwen3):
        1. attn_norm(x) → attention → residual
        2. post_attn_norm(h) → MoE MLP → residual

        Args:
            x: (batch, seq, hidden) — pre-norm input
            seq_offset: Position in KV cache for autoregressive decoding
            moe_activation: Optional pre-computed activation for the MoE layer
        """
        # 1. Self-attention
        normed = self.attn_norm.forward(x)
        attn_out = self._attention_forward(normed, seq_offset)
        h = x + attn_out  # residual

        # 2. MoE feed-forward
        normed2 = self.post_attn_norm.forward(h)

        # _moe_forward itera gia' su tutte le posizioni della sequenza, quindi
        # prefill e decode passano dallo stesso ramo.
        moe_out = self._moe_forward(normed2)

        return h + moe_out
