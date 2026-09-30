"""Test model loading + single layer forward (attention only, no MoE)."""
from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flash_moe import Qwen3MoEModel, RMSNorm, RoPE, scaled_dot_product_attention  # noqa: E402
from flash_moe.dequant import dequantize_to, name_of  # noqa: E402
from flash_moe.paths import model_path  # noqa: E402

PASSED = []
FAILED = []

def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name if ok else (name, detail))
    tag = "  PASS  " if ok else "  FAIL  "
    print(f"{tag}{name}" + (f"  ({detail})" if detail else ""), flush=True)


def numpy_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def main() -> int:
    gguf_path = model_path()
    print("=" * 78)
    print(" TEST MODEL: load + forward attention layer")
    print("=" * 78)

    model = Qwen3MoEModel(
        gguf_path,
        device="GPU",
        max_seq_len=128,
        cache_capacity=700,
    )
    print(f"Model loaded: {model.block_count} layers, hidden={model.hidden_size}")
    print(f"  heads={model.num_heads}, kv_heads={model.num_kv_heads}")
    print(f"  rope_dim={model.rope_dim}, freq_base={model.freq_base}")
    print(f"  head_dim={model.head_dim}, kv_head_dim={model.kv_head_dim}")
    print(f"  eps={model.eps}")
    print(f"  BOS={model.bos_token_id}, EOS={model.eos_token_id}")

    # =====================================================================
    # Test layer 0 (combined QKV, F16)
    print("\n--- Layer 0 (combined QKV, F16) ---")
    layer0 = model.blocks[0]
    x = np.random.randn(1, 4, model.hidden_size).astype(np.float32) * 0.01

    # Manual forward: attn_norm -> attention -> residual
    normed = layer0.attn_norm.forward(x)
    attn_out = layer0._attention_forward(normed, seq_offset=0)
    residual = x + attn_out

    print(f"  x shape: {x.shape}")
    print(f"  normed shape: {normed.shape}")
    print(f"  attn_out shape: {attn_out.shape}")
    print(f"  residual shape: {residual.shape}")
    check("combined layer output shape", residual.shape == (1, 4, model.hidden_size))
    check("no NaN in combined attention", not np.isnan(attn_out).any())

    # =====================================================================
    # Test layer 3 (split QKV, Q4_K)
    print("\n--- Layer 3 (split QKV, Q4_K) ---")
    layer3 = model.blocks[3]
    x3 = np.random.randn(1, 4, model.hidden_size).astype(np.float32) * 0.01

    normed3 = layer3.attn_norm.forward(x3)
    attn_out3 = layer3._attention_forward(normed3, seq_offset=0)
    residual3 = x3 + attn_out3

    print(f"  attn_out shape: {attn_out3.shape}")
    check("split layer output shape", residual3.shape == (1, 4, model.hidden_size))
    check("no NaN in split attention", not np.isnan(attn_out3).any())

    # =====================================================================
    # Test layer 39 (split QKV, Q4_K/Q6_K)
    print("\n--- Layer 39 (split QKV, Q4_K/Q6_K/Q4_K) ---")
    layer39 = model.blocks[39]
    x39 = np.random.randn(1, 1, model.hidden_size).astype(np.float32) * 0.01  # single token

    normed39 = layer39.attn_norm.forward(x39)
    attn_out39 = layer39._attention_forward(normed39, seq_offset=0)
    residual39 = x39 + attn_out39

    print(f"  attn_out shape: {attn_out39.shape}")
    check("layer 39 output shape", residual39.shape == (1, 1, model.hidden_size))
    check("no NaN in layer 39 attention", not np.isnan(attn_out39).any())

    model.close()

    print("\n" + "=" * 78)
    print(f" PASSED: {len(PASSED)}   FAILED: {len(FAILED)}")
    for name, detail in FAILED:
        print(f"   - {name}: {detail}")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
