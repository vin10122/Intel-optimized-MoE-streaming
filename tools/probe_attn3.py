"""Il layer 3 e' il primo di attenzione piena: ogni tappa contro llama.cpp.

Il delta net (layer 0-2) e il MoE (layer 0) sono gia' verificati. Qui la
domanda e' se l'attenzione piena e' corretta o se il layer sbaglia dopo le
proiezioni, e' per questo che si guarda ogni tappa: le proiezioni sono gia'
verificate (Qcur_full entro 0,03%), quindi la prima tappa che si stacca
dice se il problema e' nel cuore dell'attenzione o nel MoE.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\valer\Desktop\moe-client-prot\Nuova cartella")
sys.path.insert(0, str(ROOT / "src"))

import gguf  # noqa: E402
from run_q4k_f16_openvino import dequant_to_f16  # noqa: E402
from flash_moe import Qwen3MoEModel  # noqa: E402
from flash_moe.paths import model_path  # noqa: E402
from flash_moe.tokenizer import GGUFTokenizer  # noqa: E402

PROMPT = "La capital of France is"
LAYER = 3

REF = [
    ("attn_norm", 203.636353),
    ("attn_output", -4.033005),
    ("attn_residual", 4.483288),
    ("attn_post_norm", -44.121193),
    ("ffn_moe_gate", -7826.492676),
    ("ffn_moe_swiglu", -14.762588),
    ("ffn_moe_out", -0.563628),
    ("ffn_gate", -704.023682),
    ("ffn_shexp", -7.353246),
    ("ffn_shexp_gated", -1.095660),
    ("ffn_out", -1.659286),
    ("l_out", 2.823996),
]

path = model_path()
reader = gguf.GGUFReader(path)
tok = GGUFTokenizer(reader)
ids = tok.encode(PROMPT, add_bos=False)

model = Qwen3MoEModel(path, device="CPU", max_seq_len=256, cache_capacity=64,
                      dequant_fn=dequant_to_f16)
model.init_caches()

x = model.embedding.forward(np.asarray(ids, dtype=np.int64)).reshape(
    1, len(ids), model.hidden_size)
for layer in range(LAYER):
    x = model.blocks[layer].forward(x)

block = model.blocks[LAYER]
seq = len(ids)
normed = block.attn_norm.forward(x)[0].astype(np.float32)

attn_out = block._attention_forward(normed.reshape(1, seq, model.hidden_size))
attn_out = np.asarray(attn_out).reshape(seq, model.hidden_size)
resid = x.reshape(seq, model.hidden_size) + attn_out
post = block.post_attn_norm.forward(resid.reshape(1, seq, model.hidden_size))
post = np.asarray(post).reshape(seq, model.hidden_size)
moe = np.asarray(block._moe_forward(post.reshape(1, seq, model.hidden_size)))
moe = moe.reshape(seq, model.hidden_size)
l_out = resid + moe

ours = [
    ("attn_norm", float(normed.sum())),
    ("attn_output", float(attn_out.sum())),
    ("attn_residual", float(resid.sum())),
    ("attn_post_norm", float(post.sum())),
    ("ffn_out", float(moe.sum())),
    ("l_out", float(l_out.sum())),
]

print(f"{'tappa':18s} {'noi':>14s} {'llama.cpp':>14s} {'errore':>9s}")
for (name, ref), (_, v) in zip(REF, ours):
    err = abs(v - ref) / max(abs(ref), 1e-9)
    flag = "" if err < 0.03 else "   <-- si stacca"
    print(f"{name:18s} {v:14.5f} {ref:14.5f} {err * 100:8.2f}%{flag}")

print(f"\n|norma l_out| noi {np.linalg.norm(l_out):.4f}")
print(f"|norma l_out| llama.cpp 1.8333 (dal confronto per layer)")
model.close()
