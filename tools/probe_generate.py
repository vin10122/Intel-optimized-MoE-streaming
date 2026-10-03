"""Generazione greedy, token per token: il percorso con cache KV.

Il confronto per somma copre il solo prefill. Il decode e' un codice
diverso: la cache KV si riempie di un token alla volta e i layer di
attenzione leggono da li'. Se il prefill e' corretto e il decode no, si
vede dai token generati, quindi questa e' la verifica che chiude.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\valer\Desktop\moe-client-prot\Nuova cartella")
sys.path.insert(0, str(ROOT / "src"))

import gguf  # noqa: E402
from run_q4k_f16_openvino import dequant_to_f16  # noqa: E402
from flash_moe import Qwen3MoEModel  # noqa: E402
from flash_moe.paths import model_path  # noqa: E402
from flash_moe.tokenizer import GGUFTokenizer  # noqa: E402

PROMPT = "Il sole splende sopra"
N_NEW = 12

path = model_path()
reader = gguf.GGUFReader(path)
tok = GGUFTokenizer(reader)
ids = tok.encode(PROMPT, add_bos=False)

model = Qwen3MoEModel(path, device="CPU", max_seq_len=256, cache_capacity=64,
                      dequant_fn=dequant_to_f16)
model.init_caches()

out: list[int] = []
t0 = time.perf_counter()
x = model.embedding.forward(np.asarray(ids, dtype=np.int64)).reshape(
    1, len(ids), model.hidden_size)
for layer in range(model.block_count):
    x = model.blocks[layer].forward(x)
t1 = time.perf_counter()
print(f"prefillo: {len(ids)} token in {t1 - t0:.2f} s")

w = np.asarray(model.tmap["output_norm.weight"].data,
               dtype=np.float32).reshape(-1)


def next_token(last: np.ndarray) -> int:
    f = last / np.sqrt(np.mean(last * last) + 1e-6) * w
    logits = np.asarray(model.lm_head.forward(
        f.astype(np.float32).reshape(1, -1)), dtype=np.float64).reshape(-1)
    return int(np.argmax(logits))


cur = next_token(x[0, -1, :])
out.append(cur)
for _ in range(N_NEW - 1):
    step = model.embedding.forward(np.asarray([cur], dtype=np.int64)).reshape(
        1, 1, model.hidden_size)
    for layer in range(model.block_count):
        step = model.blocks[layer].forward(step)
    cur = next_token(step[0, -1, :])
    out.append(cur)

t2 = time.perf_counter()
print(f"decode: {len(out)} token in {t2 - t1:.2f} s "
      f"({(t2 - t1) / len(out) * 1000:.0f} ms/token, "
      f"{len(out) / (t2 - t1):.2f} tok/s)")
print(f"\n{prompt!r} -> {tok.decode(out)!r}" if (prompt := PROMPT) else "")
model.close()