from __future__ import annotations

import numpy as np

from typing import Any
from pathlib import Path

from .key import ExpertKey
from .store import AsyncExpertStore
from .dequant import dequantize_to, name_of
from .norm import RMSNorm
from .rope import RoPE, DEFAULT_ROPE_DIM, DEFAULT_FREQ_BASE
from .attention import AttentionLayer, KVCache, scaled_dot_product_attention, softmax
from .backend_ov import OpenVINOExpertBackend
from .router import ExpertRouter, routers_from_gguf
from .shared_expert import SharedExpert
from .embedding import EmbeddingLookup, LMHead, sample
from .transformer import TransformerBlock

import gguf


class Qwen3MoEModel:
    """Full Qwen3 MoE model for autoregressive generation.

    Integrates:
    - Token embedding (dequantized on-load, GPU-f16 compatible)
    - 40 transformer blocks (2 attention variants × quantization types)
    - MoE routing + OpenVINO SwiGLU backend with async prefetch
    - KV cache for GQA
    - LM head for token logits
    """

    def __init__(
        self,
        gguf_path: Path,
        device: str = "GPU",
        max_seq_len: int = 2048,
        cache_capacity: int = 700,
        dequant_fn: Any = None,
    ) -> None:
        self.gguv_path = gguf_path
        self.device = device
        self.max_seq_len = max_seq_len
        self.cache_capacity = cache_capacity

        reader = gguf.GGUFReader(gguf_path)
        self.tmap = {t.name: t for t in reader.tensors}
        self.reader = reader

        arch = str(reader.fields["general.architecture"].parts[-1].tobytes().decode())
        self.arch = arch

        self.block_count = int(reader.fields[f"{arch}.block_count"].parts[-1][0])
        self.hidden_size = int(reader.fields[f"{arch}.embedding_length"].parts[-1][0])
        self.intermediate_size = int(reader.fields[f"{arch}.expert_feed_forward_length"].parts[-1][0])
        self.num_experts = int(reader.fields[f"{arch}.expert_count"].parts[-1][0])
        self.num_heads = int(reader.fields[f"{arch}.attention.head_count"].parts[-1][0])
        self.num_kv_heads = int(reader.fields[f"{arch}.attention.head_count_kv"].parts[-1][0])
        self.rope_dim = int(reader.fields[f"{arch}.rope.dimension_count"].parts[-1][0]) if f"{arch}.rope.dimension_count" in reader.fields else 64
        self.freq_base = float(reader.fields[f"{arch}.rope.freq_base"].parts[-1][0]) if f"{arch}.rope.freq_base" in reader.fields else 1000000.0
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_head_dim = int(reader.fields[f"{arch}.attention.key_length"].parts[-1][0]) if f"{arch}.attention.key_length" in reader.fields else 256
        self.eps = float(reader.fields[f"{arch}.attention.layer_norm_rms_epsilon"].parts[-1][0])

        self.bos_token_id = int(reader.fields["tokenizer.ggml.bos_token_id"].parts[-1][0])
        self.eos_token_id = int(reader.fields["tokenizer.ggml.eos_token_id"].parts[-1][0])

        self.tensor_types = self._build_tensor_types()

        # Build embedding
        embd_t = self.tmap["token_embd.weight"]
        self.embedding = EmbeddingLookup.from_gguf_tensor(embd_t)

        # Build LM head
        out_t = self.tmap["output.weight"]
        self.lm_head = LMHead.from_gguf_tensor(out_t)

        # Build output norm
        out_norm_t = self.tmap["output_norm.weight"]
        out_norm_w = np.asarray(out_norm_t.data, dtype=np.float16)
        self.output_norm = RMSNorm(out_norm_w, self.eps)

        # Build store + manifest
        from run_q4k_f16_openvino import build_manifest_from_gguf
        self.manifest = build_manifest_from_gguf(gguf_path)
        self.store = AsyncExpertStore(
            gguf_path.parent, capacity=cache_capacity, manifest=self.manifest
        )

        # Build backend
        self.backend = OpenVINOExpertBackend(
            self.store,
            device=device,
            weight_dtype="f16",
            cache_capacity=cache_capacity,
            tensor_types=self.tensor_types,
            dequant_fn=dequant_fn,
        )

        # Un router per layer: ffn_gate_inp.weight e' (2048, 256) e cambia
        # in ogni strato. Un router unico condiviso farebbe selezionare lo
        # stesso set di expert in tutti i 40 layer.
        self.top_k = int(reader.fields[f"{arch}.expert_used_count"].parts[-1][0])
        self.routers = routers_from_gguf(
            self.tmap, self.block_count, self.hidden_size, self.num_experts,
            self.top_k, arch=arch,
        )
        # Compatibilita': i call site che leggono model.router continuano a
        # funzionare e ottengono il router del layer 0.
        self.router = self.routers[0]

        # Shared expert: MLP denso che OGNI token attraversa, oltre ai top-k
        # selezionati dal router. Senza, l'output di ogni layer e' sbagliato.
        self.shared_expert = None
        if "blk.0.ffn_gate_shexp.weight" in self.tmap:
            self.shared_expert = SharedExpert(
                gguf_path, num_layers=self.block_count,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                dequant_fn=dequant_fn,
            )

        # Build transformer blocks
        self.blocks: list[TransformerBlock] = []
        for layer in range(self.block_count):
            block = self._build_block(layer)
            self.blocks.append(block)

    def _build_tensor_types(self) -> dict[int, dict[str, int]]:
        """Reads ggml types per-layer for expert tensors (gate/up/down)."""
        out: dict[int, dict[str, int]] = {}
        for layer in range(self.block_count):
            layer_types: dict[str, int] = {}
            for role in ("gate", "up", "down"):
                name = f"blk.{layer}.ffn_{role}_exps.weight"
                if name in self.tmap:
                    layer_types[role] = int(self.tmap[name].tensor_type)
            if layer_types:
                out[layer] = layer_types
        return out

    def _get_tensor(self, name: str) -> np.ndarray:
        """Reads a tensor from GGUF as float16, dequantizing if needed.

        Tutto quello che resta in RAM e' f16. Restituire f32 raddoppiava i
        1028M parametri di attenzione (da 1,91 a 3,83 GiB) e i 116M dei
        gate SSM senza guadagnare nulla: OpenVINO e i pesi degli expert
        lavorano gia' in f16.
        """
        t = self.tmap[name]
        ttype = int(t.tensor_type)
        shape = tuple(int(s) for s in reversed(t.shape))

        if ttype == 1:  # F16
            return t.data.astype(np.float16).reshape(shape)
        elif ttype == 0:  # F32
            return np.asarray(t.data, dtype=np.float16).reshape(shape)
        else:
            raw = t.data.tobytes() if hasattr(t.data, "tobytes") else t.data
            return dequantize_to(
                np.frombuffer(raw, dtype=np.uint8), ttype, "f16"
            ).reshape(shape)

    def _build_block(self, layer: int) -> TransformerBlock:
        """Build a transformer block for the given layer index."""
        attn_norm_t = self.tmap[f"blk.{layer}.attn_norm.weight"]
        attn_norm_w = np.asarray(attn_norm_t.data, dtype=np.float16)

        post_attn_norm_t = self.tmap.get(f"blk.{layer}.post_attention_norm.weight")
        if post_attn_norm_t is not None:
            post_attn_norm_w = np.asarray(post_attn_norm_t.data, dtype=np.float16)
        else:
            post_attn_norm_w = np.ones(self.hidden_size, dtype=np.float16)

        is_combined = f"blk.{layer}.attn_qkv.weight" in self.tmap

        if is_combined:
            qkv_w = self._get_tensor(f"blk.{layer}.attn_qkv.weight")
            gate_w = self._get_tensor(f"blk.{layer}.attn_gate.weight")
            return TransformerBlock(
                layer_idx=layer,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_heads,
                head_dim=128,
                kv_head_dim=128,
                rope_dim=self.rope_dim,
                freq_base=self.freq_base,
                max_seq_len=self.max_seq_len,
                attn_norm_weight=attn_norm_w,
                post_attn_norm_weight=post_attn_norm_w,
                qkv_weight=qkv_w,
                attn_gate_weight=gate_w,
                backend=self.backend,
                router=self.routers[layer],
                intermediate_size=self.intermediate_size,
                num_experts=self.num_experts,
                shared_expert=self.shared_expert,
            )
        else:
            q_w = self._get_tensor(f"blk.{layer}.attn_q.weight")
            k_w = self._get_tensor(f"blk.{layer}.attn_k.weight")
            v_w = self._get_tensor(f"blk.{layer}.attn_v.weight")
            out_w = self._get_tensor(f"blk.{layer}.attn_output.weight")

            qk_norm_t = self.tmap.get(f"blk.{layer}.attn_q_norm.weight")
            k_norm_t = self.tmap.get(f"blk.{layer}.attn_k_norm.weight")
            qk_norm_w = np.asarray(qk_norm_t.data, dtype=np.float16) if qk_norm_t else None
            k_norm_w = np.asarray(k_norm_t.data, dtype=np.float16) if k_norm_t else None

            return TransformerBlock(
                layer_idx=layer,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=512,
                kv_head_dim=self.kv_head_dim,
                rope_dim=self.rope_dim,
                freq_base=self.freq_base,
                max_seq_len=self.max_seq_len,
                attn_norm_weight=attn_norm_w,
                post_attn_norm_weight=post_attn_norm_w,
                q_weight=q_w,
                k_weight=k_w,
                v_weight=v_w,
                attn_output_weight=out_w,
                q_k_norm=qk_norm_w,
                k_k_norm=k_norm_w,
                backend=self.backend,
                router=self.routers[layer],
                intermediate_size=self.intermediate_size,
                num_experts=self.num_experts,
                shared_expert=self.shared_expert,
            )

    def init_caches(self) -> None:
        for block in self.blocks:
            block.init_kv_cache(self.max_seq_len)

    def forward_layer(
        self,
        layer_idx: int,
        x: np.ndarray,
        seq_offset: int = 0,
    ) -> np.ndarray:
        """Forward pass through a single transformer block.

        Args:
            layer_idx: Layer index (0 to block_count-1)
            x: (batch, seq, hidden) input
            seq_offset: Position offset in KV cache

        Returns:
            (batch, seq, hidden) output
        """
        return self.blocks[layer_idx].forward(x, seq_offset)

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 1,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 0.95,
        seed: int = 42,
    ) -> list[int]:
        """Autoregressive generation.

        Args:
            input_ids: List of token IDs for the prompt
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            seed: PRNG seed

        Returns:
            List of generated token IDs (input + generated)
        """
        self.init_caches()
        rng = np.random.default_rng(seed)
        generated = list(input_ids)

        for step in range(max_new_tokens):
            seq_offset = len(generated) - 1

            # Embed last token
            if step == 0:
                # Prefill: process entire sequence
                emb = self.embedding.forward(generated)  # (seq, hidden)
                x = emb.reshape(1, -1, self.hidden_size)
            else:
                # Decode: only last token
                emb = self.embedding.forward(generated[-1:])  # (1, hidden)
                x = emb.reshape(1, len(generated[-1:]), self.hidden_size)

            # Forward through all layers
            for layer_idx in range(self.block_count):
                x = self.blocks[layer_idx].forward(x, seq_offset=max(0, seq_offset))

            # Output norm + LM head
            logits = self.output_norm.forward(x)  # (1, seq, hidden)
            logits = self.lm_head.forward(logits)  # (1, seq, vocab)

            # Get last token logits
            last_logits = logits[-1] if logits.ndim > 1 else logits

            # Sample
            next_token = sample(
                last_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                rng=rng,
            )
            generated.append(next_token)

            if next_token == self.eos_token_id:
                break

        return generated

    def close(self) -> None:
        self.store.close()
