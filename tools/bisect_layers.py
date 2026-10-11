"""Bisect: il primo layer dove la nostra uscita si separa da llama.cpp.

llama-eval-callback stampa per ogni `l_out-N` i primi 3 e gli ultimi 3
valori di ogni posizione, piu' la somma. Sono pochi numeri, ma sono
l'impronta di un layer: se coincidono, il layer e' giusto. Il primo layer
dove non coincidono e' quello da guardare.

Confronto solo la posizione 0 (il token dopo BOS), cosi' l'allineamento
con llama.cpp non dipende da come i due tokenizer spezzano il prompt oltre
al BOS.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gguf  # noqa: E402
from run_q4k_f16_openvino import dequant_to_f16  # noqa: E402
from flash_moe import Qwen3MoEModel  # noqa: E402
from flash_moe.paths import model_path  # noqa: E402
from flash_moe.tokenizer import GGUFTokenizer  # noqa: E402

PROMPT = "La capital of France is"
LCPP = Path(os.environ.get("FLASH_MOE_LCPP_EVAL", ROOT / "lcpp_eval.txt")).expanduser()

NUM = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?")
HEAD = re.compile(r"^\s*common_debug_cb_eval:\s+(l_out-(\d+)) = ")


def parse_lcpp(path: Path) -> dict[int, dict]:
    """Estrae per ogni layer i 6 valori stampati e la somma."""
    out: dict[int, dict] = {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    while i < len(lines):
        m = HEAD.match(lines[i])
        if not m:
            i += 1
            continue
        layer = int(m.group(2))
        vals: list[float] = []
        total = None
        j = i + 1
        while j < len(lines) and not lines[j].lstrip().startswith("common_debug_cb_eval"):
            nums = NUM.findall(lines[j])
            if "sum" in lines[j]:
                total = float(nums[0]) if nums else None
                break
            vals.extend(float(v) for v in nums)
            j += 1
        # 6 numeri per posizione, 5 posizioni: la prima posizione sono i primi 6
        out[layer] = {"head": vals[:6], "sum": total}
        i = j
    return out


def main():
    ref = parse_lcpp(LCPP)
    print(f"llama.cpp: {len(ref)} layer con l_out stampato "
          f"(0..{max(ref)})")

    path = model_path()
    tok = GGUFTokenizer(gguf.GGUFReader(path))
    ids = tok.encode(PROMPT, add_bos=False)
    print(f"prompt {PROMPT!r}: {len(ids)} token {ids}")

    model = Qwen3MoEModel(path, device="CPU", max_seq_len=256, cache_capacity=64,
                         dequant_fn=dequant_to_f16)
    model.init_caches()
    x = model.embedding.forward(np.asarray(ids, dtype=np.int64))
    x = x.reshape(1, len(ids), model.hidden_size)

    print(f"\n{'L':>2} {'|ours|':>9} {'|lcpp|':>9} {'somma nostra':>13} "
          f"{'somma lcpp':>12} {'3 valori iniziali':>34}")
    first_bad = None
    for layer in range(model.block_count):
        x = model.blocks[layer].forward(x)
        ours = x[0, 0, :].astype(np.float32)
        head = np.concatenate([ours[:3], ours[-3:]])
        r = ref.get(layer)
        norm = float(np.linalg.norm(ours))
        if r is None:
            print(f"{layer:2d} {norm:9.4f} {'-':>9} {'(nessun ref)':>13}")
            continue
        rsum = r["sum"]
        delta = abs(float(ours.sum()) - rsum) / max(abs(rsum), 1e-9)
        same = "SI" if np.allclose(head, r["head"], atol=2e-3, rtol=2e-3) else "NO"
        if same == "NO" and first_bad is None:
            first_bad = layer
        print(f"{layer:2d} {norm:9.4f} {'':>9} {float(ours.sum()):13.4f} "
              f"{rsum:12.4f} {same:>6}  {np.array2string(head, precision=4)}")
        if layer == 0:
            print(f"     lcpp:                              "
                  f"{'':>12} {'':>12} {'':>6}  "
                  f"{np.array2string(np.asarray(r['head']), precision=4)}")

    print(f"\nprimo layer che non coincide: {first_bad}")

    logits = model.lm_head.forward(model.output_norm.forward(x))[0, -1]
    p = np.exp(logits - logits.max())
    p /= p.sum()
    order = np.argsort(p)[::-1][:10]
    print("\nI nostri top-10 sull'ultima posizione")
    for i in order:
        print(f"  {int(i):7d}  {p[i]:.5f}  {tok.decode([int(i)])!r}")
    print(f"  entropia {float(-(p * np.log(p)).sum()):.3f} nats")
    model.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
