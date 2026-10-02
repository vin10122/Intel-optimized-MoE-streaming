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
    # Layer 0 e' un layer SSM: l'attenzione non si esegue, e deve dirlo
    print("\n--- Layer 0 (SSM / Gated DeltaNet, non attenzione) ---")
    layer0 = model.blocks[0]
    x = np.random.randn(1, 4, model.hidden_size).astype(np.float32) * 0.01
    normed = layer0.attn_norm.forward(x)
    check("il layer 0 e' combinato (qkv fused, quindi SSM)", layer0.is_combined, "")
    try:
        layer0._attention_forward(normed, seq_offset=0)
        check("l'attenzione su un layer SSM viene rifiutata", False, "accettata")
    except TypeError as err:
        check("l'attenzione su un layer SSM viene rifiutata", True,
              str(err)[:52])

    # =====================================================================
    # I layer full attention sono 3, 7, 11, ... (full_attention_interval = 4)
    print("\n--- Layer 3 (full attention) ---")
    layer3 = model.blocks[3]
    check("il layer 3 e' split (full attention)", not layer3.is_combined, "")
    x3 = np.random.randn(1, 4, model.hidden_size).astype(np.float32) * 0.01
    normed3 = layer3.attn_norm.forward(x3)
    attn_out3 = layer3._attention_forward(normed3, seq_offset=0)
    residual3 = x3 + attn_out3
    print(f"  attn_out shape: {attn_out3.shape}")
    check("layer 3 output shape", residual3.shape == (1, 4, model.hidden_size))
    check("no NaN nell'attenzione del layer 3", not np.isnan(attn_out3).any())

    # =====================================================================
    # La proprieta' che conta: lo stesso blocco, con cache, deve dare lo stesso
    # risultato prossimo a passo e in un colpo solo. Un errore nella cache
    # (maschera causale, offset di posizione, ordine dei token) lo rompe.
    print("\n--- equivalenza prefill / decode sul layer 3 ---")
    layer3.init_kv_cache(64)
    full = layer3._attention_forward(normed3, seq_offset=0)

    layer3.init_kv_cache(64)
    step = np.concatenate(
        [layer3._attention_forward(normed3[:, t:t + 1], seq_offset=0)
         for t in range(normed3.shape[1])], axis=1)
    rel = float(np.abs(step - full).max() / np.abs(full).max())
    check("token per token == prefillo", rel < 1e-4,
          f"rel={rel:.2e}, seq_len={layer3.attn.cache.seq_len}")

    # la cache deve contenere i 4 token del prefillo
    check("la cache contiene i token del prefillo",
          layer3.attn.cache.seq_len == 4,
          f"seq_len={layer3.attn.cache.seq_len}")

    # =====================================================================
    # Layer 39: ultimo blocco, percorso identico
    print("\n--- Layer 39 (full attention) ---")
    layer39 = model.blocks[39]
    x39 = np.random.randn(1, 1, model.hidden_size).astype(np.float32) * 0.01
    normed39 = layer39.attn_norm.forward(x39)
    attn_out39 = layer39._attention_forward(normed39, seq_offset=0)
    residual39 = x39 + attn_out39
    print(f"  attn_out shape: {attn_out39.shape}")
    check("layer 39 output shape", residual39.shape == (1, 1, model.hidden_size))
    check("no NaN nell'attenzione del layer 39", not np.isnan(attn_out39).any())

    # =====================================================================
    # L'ordine dei layer: 10 full attention, 30 SSM, ai posti giusti
    print("\n--- mappa dei 40 layer ---")
    attn_layers = [i for i, b in enumerate(model.blocks) if not b.is_combined]
    ssm_layers = [i for i, b in enumerate(model.blocks) if b.is_combined]
    check("i layer full attention sono 10, ai multipli di 4 meno 1",
          attn_layers == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39],
          f"{attn_layers}")
    check("gli altri 30 sono SSM", len(ssm_layers) == 30, f"{len(ssm_layers)}")

    model.close()

    print("\n" + "=" * 78)
    print(f" PASSED: {len(PASSED)}   FAILED: {len(FAILED)}")
    for name, detail in FAILED:
        print(f"   - {name}: {detail}")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
