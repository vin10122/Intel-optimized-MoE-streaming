"""Test embedding lookup + LM head vs gguf.quants reference."""
from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

import gguf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flash_moe import EmbeddingLookup, LMHead  # noqa: E402
from flash_moe.dequant import dequantize_to, name_of  # noqa: E402
from flash_moe.paths import model_path  # noqa: E402
from flash_moe.rope import RoPE, DEFAULT_ROPE_DIM, DEFAULT_FREQ_BASE  # noqa: E402

PASSED = []
FAILED = []

def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name if ok else (name, detail))
    tag = "  PASS  " if ok else "  FAIL  "
    print(f"{tag}{name}" + (f"  ({detail})" if detail else ""), flush=True)


def main() -> int:
    gguf_path = model_path()
    print("=" * 78)
    print(" TEST EMBEDDING + LM HEAD: dequant vs gguf.quants reference")
    print("=" * 78)

    reader = gguf.GGUFReader(gguf_path)
    tmap = {t.name: t for t in reader.tensors}

    # =====================================================================
    # Test 1: Token embedding lookup
    print("\n--- Token Embedding (token_embd.weight) ---")
    embd_t = tmap["token_embd.weight"]
    embd_shape = tuple(int(s) for s in reversed(embd_t.shape))
    embd_type = name_of(embd_t.tensor_type)
    print(f"  shape: {embd_shape}, type: {embd_type}, n_bytes: {embd_t.n_bytes}")

    # Build embedding from GGUF tensor
    embedding = EmbeddingLookup.from_gguf_tensor(embd_t)
    print(f"  Built embedding: vocab={embedding.vocab_size}, hidden={embedding.hidden_size}")

    # Reference dequant
    raw = embd_t.data
    if embd_t.tensor_type == 1:  # F16
        ref_weight = embd_t.data.astype(np.float32)
    else:
        raw_bytes = raw.tobytes() if hasattr(raw, "tobytes") else raw
        ref_weight = dequantize_to(
            np.frombuffer(raw_bytes, dtype=np.uint8),
            int(embd_t.tensor_type),
            "f32",
        ).reshape(embd_shape)

    # Confronto. L'embedding e' tenuto in f16, quindi il reference va
    # riportato alla stessa precisione: il confronto e' bit esatto, piu'
    # stretto del 1e-6 che valeva quando la storage era f32. La perdita
    # f32 -> f16 viene misurata separatamente, non nascosta dietro una
    # tolleranza.
    ref_f16 = ref_weight.astype(np.float16)
    diff = np.abs(embedding.weight - ref_f16).max()
    print(f"  Max diff vs reference (both f16): {diff:.3e}")
    check("embedding weights bit-exact vs reference", diff == 0.0, f"diff={diff:.3e}")

    rel = np.abs(
        embedding.weight.astype(np.float32) - ref_weight
    ).max() / max(float(np.abs(ref_weight).max()), 1e-9)
    print(f"  Relative error f32 -> f16 storage: {rel:.3e}")
    check("f16 storage within f16 resolution", rel < 1e-3, f"rel={rel:.3e}")

    # Test lookup
    token_ids = [0, 1, 100, 1000, 10000]
    emb_out = embedding.forward(token_ids)
    ref_out = ref_weight[token_ids].astype(np.float16)
    diff = np.abs(emb_out - ref_out).max()
    print(f"  Lookup shape: {emb_out.shape}")
    check("embedding lookup correct", diff == 0.0, f"diff={diff:.3e}")

    # =====================================================================
    # Test 2: LM head
    print("\n--- LM Head (output.weight) ---")
    out_t = tmap["output.weight"]
    out_shape = tuple(int(s) for s in reversed(out_t.shape))
    out_type = name_of(out_t.tensor_type)
    print(f"  shape: {out_shape}, type: {out_type}, n_bytes: {out_t.n_bytes}")

    lm_head = LMHead.from_gguf_tensor(out_t)
    print(f"  Built LM head: vocab={lm_head.vocab_size}, hidden={lm_head.hidden_size}")

    # Reference
    raw = out_t.data
    if out_t.tensor_type == 1:
        ref_out_weight = out_t.data.astype(np.float32)
    else:
        raw_bytes = raw.tobytes() if hasattr(raw, "tobytes") else raw
        ref_out_weight = dequantize_to(
            np.frombuffer(raw_bytes, dtype=np.uint8),
            int(out_t.tensor_type),
            "f32",
        ).reshape(out_shape)

    diff = np.abs(lm_head.weight - ref_out_weight.astype(np.float16)).max()
    print(f"  Max diff vs reference (f16 rounding): {diff:.3e}")
    check("LM head weights bit-exact vs reference", diff == 0.0, f"diff={diff:.3e}")
    check("LM head weights tenuti in f16", lm_head.weight.dtype == np.float16,
          f"{lm_head.weight.dtype}")

    # Test forward: hidden -> logits
    hidden = np.random.randn(2048).astype(np.float32) * 0.01
    logits = lm_head.forward(hidden)
    ref_logits = hidden @ ref_out_weight.T
    diff = np.abs(logits - ref_logits).max() / max(float(np.abs(ref_logits).max()), 1e-9)
    print(f"  Logits shape: {logits.shape}")
    check("LM head forward matches NumPy", diff < 1e-2, f"rel={diff:.3e}")
    check("No NaN in LM head output", not np.isnan(logits).any())

    # I pesi in f32 non devono cambiare il risultato rispetto al percorso
    # precedente, che li upcastava da f16 a ogni blocco. Stessi numeri in
    # entrata, stesso blocking, quindi la differenza deve essere zero esatta:
    # un confronto "abbastanza vicino" qui lascerebbe passare un doppio
    # arrotondamento, che e' esattamente il difetto che il test deve chiudere.
    staged = hidden @ lm_head.weight.astype(np.float16).astype(np.float32).T
    diff_exact = np.abs(logits - staged).max()
    check("logits identici al percorso con staging f16", diff_exact == 0.0,
          f"max|diff|={diff_exact:.3e}")

    # =====================================================================
    # Test 3: Single token embedding + LM head round-trip
    print("\n--- Embedding round-trip (single token) ---")
    token_id = 1234
    emb = embedding.forward(token_id)  # (hidden,)
    logits = lm_head.forward(emb)      # (vocab,)
    print(f"  Token {token_id} -> emb shape {emb.shape} -> logits shape {logits.shape}")
    check("single token embedding+lmhead roundtrip", 
          emb.shape == (2048,) and logits.shape == (embedding.vocab_size,))

    print("\n" + "=" * 78)
    print(f" PASSED: {len(PASSED)}   FAILED: {len(FAILED)}")
    for name, detail in FAILED:
        print(f"   - {name}: {detail}")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
