"""Confronto end-to-end: la somma dei logits e il token successivo.

E' il controllo che conta, perche' attraversa tutto: embedding, 40 layer
(delta net e attenzione), MoE, norm finale e lm_head. La somma dei logits
e' il numero piu' fragile che si possa usare su una distribuzione (quasi
tutti i valori si cancellano), ma qui serve proprio perche' e' sensibile:
se i due runtime dessero lo stesso risultato solo per caso, la somma
cambierebbe. Il token greedy e' il controllo leggibile.
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
REF_LOGIT_SUM = -328174.531250

path = model_path()
reader = gguf.GGUFReader(path)
tok = GGUFTokenizer(reader)
ids = tok.encode(PROMPT, add_bos=False)
print(f"prompt {PROMPT!r} -> {ids}")

model = Qwen3MoEModel(path, device="CPU", max_seq_len=256, cache_capacity=64,
                      dequant_fn=dequant_to_f16)
model.init_caches()

x = model.embedding.forward(np.asarray(ids, dtype=np.int64)).reshape(
    1, len(ids), model.hidden_size)
for layer in range(model.block_count):
    x = model.blocks[layer].forward(x)

last = x[0, -1, :].astype(np.float32)
final_norm = model.final_norm.forward(last.reshape(1, 1, -1)) if hasattr(
    model, "final_norm") else None
if final_norm is None:
    w = np.asarray(model.tmap["output_norm.weight"].data,
                   dtype=np.float32).reshape(-1)
    final_norm = (last / np.sqrt(np.mean(last * last) + 1e-6)) * w
final_norm = np.asarray(final_norm).reshape(-1)

logits = model.lm_head.forward(final_norm.reshape(1, -1))
logits = np.asarray(logits, dtype=np.float64).reshape(-1)

ours = float(logits.sum())
err = abs(ours - REF_LOGIT_SUM) / abs(REF_LOGIT_SUM)
print(f"\nsomma logits  noi {ours:.3f}   llama.cpp {REF_LOGIT_SUM:.3f}   "
      f"errore {err * 100:.2f}%")

top = np.argsort(logits)[-5:][::-1]
print("\ntop-5 nostri (indice, logit, testo):")
for i in top:
    print(f"  {int(i):8d}  {logits[i]:9.4f}  {tok.decode([int(i)])!r}")
best = int(top[0])
print(f"\ngreedy: {best} {tok.decode([best])!r}")
p = np.exp(logits - logits.max())
p /= p.sum()
print(f"entropia {float(-(p * np.log(p)).sum()):.3f} nats")
model.close()