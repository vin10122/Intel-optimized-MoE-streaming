"""Dove si rompe il delta net, elemento per elemento.

llama-eval-callback stampa per ogni tensore i primi 3 e gli ultimi 3
valori di ogni posizione. Sono sei numeri per posizione: non bastano per
un confronto statistico, ma bastano per decidere se una tappa e' corretta
o no, e le somme non bastano, perche' con 2048 termini che si cancellano
la somma resta vicina anche quando i valori sono sbagliati.

Confronto le stesse sei posizioni della nostra implementazione, per ogni
tappa della catena del delta net e per ognuna delle 5 posizioni del prompt.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\valer\Desktop\moe-client-prot\Nuova cartella")
sys.path.insert(0, str(ROOT / "src"))

import gguf  # noqa: E402
from run_q4k_f16_openvino import dequant_to_f16  # noqa: E402
from flash_moe import Qwen3MoEModel  # noqa: E402
from flash_moe.gated_deltanet import _softplus  # noqa: E402
from flash_moe.paths import model_path  # noqa: E402
from flash_moe.tokenizer import GGUFTokenizer  # noqa: E402

PROMPT = "La capital of France is"
LCPP = Path(r"C:\Users\valer\AppData\Local\Temp\kilo\lcpp_eval.txt")
NUM = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?")
TENSOR = re.compile(r"^\s*common_debug_cb_eval:\s+(\S+) = \(f32\)")

STAGES = [
    ("attn_norm",            "attn_norm-0"),
    ("qkv proiettato",       "linear_attn_qkv_mixed-0"),
    ("conv dopo il SiLU",    "conv_output_silu-0"),
    ("alpha",                "alpha-0"),
    ("gate",                 "gate-0"),
    ("z",                    "z-0"),
    ("dopo ssm_out",         "linear_attn_out-0"),
]


def parse(path: Path) -> dict[str, list[list[float]]]:
    """Per ogni tensore, i 6 valori stampati di ogni posizione."""
    out: dict[str, list[list[float]]] = {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    while i < len(lines):
        m = TENSOR.match(lines[i])
        if not m:
            i += 1
            continue
        name = m.group(1)
        rows: list[list[float]] = []
        j = i + 1
        while j < len(lines) and not TENSOR.match(lines[j]):
            stripped = lines[j].strip()
            if stripped.startswith("sum"):
                break
            if stripped.startswith("["):
                nums = [float(v) for v in NUM.findall(stripped)]
                if nums:
                    rows.append(nums)
            j += 1
        if rows:
            out.setdefault(name, rows)
        i = j
    return out


def headtail(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    return np.concatenate([v[:3], v[-3:]])


def main() -> None:
    ref = parse(LCPP)
    path = model_path()
    tok = GGUFTokenizer(gguf.GGUFReader(path))
    ids = tok.encode(PROMPT, add_bos=False)
    model = Qwen3MoEModel(path, device="CPU", max_seq_len=256, cache_capacity=64,
                         dequant_fn=dequant_to_f16)
    model.init_caches()
    x = model.embedding.forward(np.asarray(ids, dtype=np.int64))
    x = x.reshape(1, len(ids), model.hidden_size)
    block = model.blocks[0]
    lyr = model.ssm[0]
    normed = block.attn_norm.forward(x)[0]

    ours: dict[str, list[np.ndarray]] = {k: [] for k, _ in STAGES}
    inner_conv = lyr._conv
    conv_seen: list[np.ndarray] = []

    def conv_spy(qkv: np.ndarray) -> np.ndarray:
        out = inner_conv(qkv)
        conv_seen.append(out)
        return out

    lyr._conv = conv_spy

    for si in range(len(ids)):
        qkv, z, beta, alpha = lyr._projections(normed[si])
        ours["attn_norm"].append(headtail(normed[si]))
        ours["qkv proiettato"].append(headtail(qkv))
        ours["alpha"].append(headtail(alpha))
        ours["gate"].append(headtail(_softplus(alpha + lyr.small["dt_bias"])
                                     * lyr.small["a"]))
        ours["z"].append(headtail(z))
        ours["dopo ssm_out"].append(headtail(lyr.forward(normed[si])))
        # la conv e' dentro forward: si legge dopo, non prima
        ours["conv dopo il SiLU"].append(headtail(conv_seen[si]))

    print(f"{'tappa':22s} " + " ".join(f"{'pos' + str(i):>9}" for i in range(len(ids))))
    for key, tensor in STAGES:
        rows = ref.get(tensor)
        if rows is None:
            print(f"{key:22s} (nessun riferimento)")
            continue
        cells = []
        for si in range(min(len(rows), len(ids))):
            r = np.asarray(rows[si][:6], dtype=np.float32)
            o = ours[key][si][:6]
            scale = max(float(np.abs(r).max()), 1e-6)
            rel = float(np.abs(o - r).max()) / scale
            cells.append(f"{rel * 100:8.2f}%" + ("*" if rel < 0.02 else " "))
        print(f"{key:22s} " + " ".join(cells))
    print("\n* entro il 2%: la tappa coincide con llama.cpp sugli stessi elementi")
    model.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()