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
        # pesi QK-norm grezzi: li applica AttentionLayer, non questo blocco
        self.q_k_norm = q_k_norm
        self.k_k_norm = k_k_norm
        self._prepared_weights: dict[str, np.ndarray] | None = None
        if q_k_norm is not None and k_k_norm is not None:
            self.q_norm = RMSNorm(q_k_norm, eps=1e-6)
            self.k_norm = RMSNorm(k_k_norm, eps=1e-6)

        self.backend = backend
        self.router = router
        # Ogni token attraversa anche lo shared expert: senza, l'output del
        # layer e' sbagliato. Passato dal modello, che lo carica una volta.
        self.shared_expert = shared_expert

    def init_kv_cache(self, max_seq_len: int) -> None:
        self.attn.init_cache(max_seq_len)
        self.kv_cache = self.attn.cache

    def _attn_weights(self):
        """Pesi dell'attenzione nel formato ggml, preparati una volta sola.

        Il modello li tiene in convenzione numpy (out, in) perche' e' cosi' che
        li carica dalla GGUF; AttentionLayer li vuole (in, out), cioe' la
        disposizione con cui si scrive ``x @ w`` senza trasporre. La
        trasposizione e' una vista: non copia niente.
        """
        if self._prepared_weights is None:
            self._prepared_weights = self.attn.prepare_weights(
                np.ascontiguousarray(self.q_weight.T),
                np.ascontiguousarray(self.k_weight.T),
                np.ascontiguousarray(self.v_weight.T),
                self.q_k_norm, self.k_k_norm,
                np.ascontiguousarray(self.attn_output_weight.T),
                keep_f16=True,
            )
        return self._prepared_weights

    def _attention_forward(self, x: np.ndarray, seq_offset: int = 0) -> np.ndarray:
        """Attenzione del blocco su (batch, seq, hidden).

        Non implementa l'attenzione: chiama AttentionLayer.forward, che e' il
        percorso verificato. Qui c'era una seconda implementazione con lo
        stesso calcolo ma orientamenti diversi (pesi in (out, in), gate Q
        applicata in modo diverso, prefillo che non popolava la cache), e due
        copie di un calcolo che si sbaglia in modi diversi sono la peggio
        combinazione possibile.
        """
        if self.is_combined:
            raise TypeError(
                f"blk.{self.layer_idx} e' un layer SSM (Gated DeltaNet): "
                "l'attenzione non si esegue su questo blocco, vedi "
                "flash_moe.gated_deltanet. I layer full attention sono quelli "
                "con full_attention_interval = 4.")
        b, s, h = x.shape
        flat = np.ascontiguousarray(x.reshape(b * s, h))
        out = self.attn.forward(flat, self._attn_weights(), cache=self.attn.cache)
        return out.reshape(b, s, h)


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
        1. attn_norm(x) â†’ attention â†’ residual
        2. post_attn_norm(h) â†’ MoE MLP â†’ residual

        Args:
            x: (batch, seq, hidden) â€” pre-norm input
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
