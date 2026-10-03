"""Dove si rompe il layer 0: confronto tappa per tappa con llama.cpp.

llama.cpp stampa la somma di ogni tensore del grafo. La somma e' un
impronta debole (le cancellazioni la possono tenere vicina anche quando i
numeri sono sbagliati) ma e' sufficiente per localizzare: e' la prima
tappa dove la nostra somma si stacca, e quella e' la tappa da guardare.
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

# somme stampate da llama-eval-callback sullo stesso prompt
LCPP = [
    ("embedding (GET_ROWS)",            "model.input_embed",   -0.229735),
    ("attn_norm",                       "attn_norm-0",         -58.350986),
    ("delta net dopo ssm_out",          "linear_attn_out-0",    2.822281),
    ("residuo h",                       "attn_residual-0",      2.592548),
    ("output MoE",                      "ffn_out-0",            0.063000),
    ("uscita del layer",                "l_out-0",              2.655544),
]

path = model_path()
tok = GGUFTokenizer(gguf.GGUFReader(path))
ids = tok.encode(PROMPT, add_bos=False)
print(f"prompt {PROMPT!r} -> {len(ids)} token {ids}")
print("SENZA BOS, come llama.cpp\n")

model = Qwen3MoEModel(path, device="CPU", max_seq_len=256, cache_capacity=64,
                      dequant_fn=dequant_to_f16)
model.init_caches()
x = model.embedding.forward(np.asarray(ids, dtype=np.int64))
x = x.reshape(1, len(ids), model.hidden_size)
block = model.blocks[0]

got: list[tuple[str, float]] = [("embedding (GET_ROWS)", float(x.sum()))]

normed = block.attn_norm.forward(x)
got.append(("attn_norm", float(normed.sum())))

mixed = block.ssm_runner(normed)
got.append(("delta net dopo ssm_out", float(mixed.sum())))

h = x + mixed
got.append(("residuo h", float(h.sum())))

normed2 = block.post_attn_norm.forward(h)
moe_out = block._moe_forward(normed2)
got.append(("output MoE", float(moe_out.sum())))

out = h + moe_out
got.append(("uscita del layer", float(out.sum())))

print(f"{'tappa':28s} {'noi':>12} {'llama.cpp':>12} {'errore':>10}")
for (label, tensor, ref), (_, ours) in zip(LCPP, got):
    err = abs(ours - ref) / max(abs(ref), 1e-9)
    flag = "" if err < 0.05 else "   <-- si stacca qui"
    print(f"{label:28s} {ours:12.5f} {ref:12.5f} {err * 100:9.2f}%{flag}")

# Il router e' il sospetto piu' economico da verificare: se i token scelti
# non sono quelli di llama.cpp, tutto il resto del layer e'rumore.
ids_sel, weights = model.routers[0].route(x[0, 0, :])
print(f"\nnoi scegliamo gli expert {ids_sel} con pesi "
      f"{weights}")
route_sum = float(np.asarray(model.routers[0].route(x[0, 0, :])[1]).sum())
print(f"somma dei pesi di routing: {route_sum:.5f} "
      f"(le somme di routing in llama.cpp: ffn_gate-0 = -893.148376)")
model.close()
