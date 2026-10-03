"""Confronto tappa per tappa del MoE del layer 0 con llama.cpp.

Il router e' gia' verificato (logits entro 0,03%), quindi qui si guarda
solo il cammino dei pesi: i tre prodotti di ogni expert, il contributo
dello shared expert, e la somma finale. Le somme di riferimento sono
quelle stampate da llama-eval-callback sullo stesso prompt.
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
HID, INTER, NEXP, TOPK = 2048, 512, 256, 8

REF = [
    ("ffn_moe_gate-0", -3767.470215),
    ("ffn_moe_up-0", 31.123177),
    ("ffn_moe_swiglu-0", -5.973990),
    ("ffn_moe_down-0", -0.167642),
    ("ffn_moe_weighted-0", -0.160594),
    ("ffn_moe_out-0", -0.160593),
    ("ffn_gate-0", -893.148376),
    ("ffn_up-0", 73.933220),
    ("ffn_swiglu-0", 3.016987),
    ("ffn_shexp-0", 2.780174),
    ("ffn_shexp_gated-0", 0.223592),
    ("ffn_out-0", 0.063000),
]

path = model_path()
reader = gguf.GGUFReader(path)
tmap = {t.name: t for t in reader.tensors}
tok = GGUFTokenizer(reader)
ids = tok.encode(PROMPT, add_bos=False)

model = Qwen3MoEModel(path, device="CPU", max_seq_len=256, cache_capacity=64,
                      dequant_fn=dequant_to_f16)
model.init_caches()
x = model.embedding.forward(np.asarray(ids, dtype=np.int64))
x = x.reshape(1, len(ids), model.hidden_size)
block = model.blocks[0]
mixed = block.ssm_runner(block.attn_norm.forward(x))
h = x + mixed
acts = block.post_attn_norm.forward(h)[0].astype(np.float32)

def expert_matrix(role: str, eid: int) -> np.ndarray:
    """Un expert del layer 0, con la stessa forma che usa il backend."""
    t = tmap[f"blk.0.ffn_{role}_exps.weight"]
    per = t.n_bytes // NEXP
    with open(path, "rb") as fh:
        fh.seek(t.data_offset + eid * per)
        raw = fh.read(per)
    n = HID * INTER
    shape = (INTER, HID) if role in ("gate", "up") else (HID, INTER)
    w = dequant_to_f16(raw, int(t.tensor_type), n)
    return np.asarray(w, dtype=np.float32).reshape(shape)

gate_sum = up_sum = swiglu_sum = down_sum = weighted_sum = 0.0
routed = np.zeros(HID, dtype=np.float32)
router = model.routers[0]

for i, a in enumerate(acts):
    eids, weights = router.route(a)
    total = np.zeros(HID, dtype=np.float32)
    for eid, w in zip(eids, weights):
        xa = a.reshape(1, HID)
        g = (xa @ expert_matrix("gate", eid).T)[0]
        u = (xa @ expert_matrix("up", eid).T)[0]
        s = (g / (1.0 + np.exp(-g))) * u
        d = s @ expert_matrix("down", eid).T
        gate_sum += float(g.sum())
        up_sum += float(u.sum())
        swiglu_sum += float(s.sum())
        down_sum += float(d.sum())
        weighted_sum += float((w * d).sum())
        total += w * d
    routed += total

shared = model.shared_expert.layers[0]
w_gate = shared["gate"]
w_up = shared["up"]
w_down = shared["down"]
w_gate_inp = shared["gate_inp"]

sg = su = ssilu = sdown = sgated = 0.0
shared_out = np.zeros(HID, dtype=np.float32)
for a in acts:
    xa = a.reshape(1, HID)
    g = (xa @ w_gate.T)[0]
    u = (xa @ w_up.T)[0]
    s = (g / (1.0 + np.exp(-g))) * u
    d = s @ w_down.T
    gate_sigmoid = 1.0 / (1.0 + np.exp(-float((xa @ w_gate_inp.reshape(-1, 1))[0, 0])))
    sg += float(g.sum())
    su += float(u.sum())
    ssilu += float(s.sum())
    sdown += float(d.sum())
    sgated += float((d * gate_sigmoid).sum())
    shared_out += d * gate_sigmoid

ours = {
    "ffn_moe_gate-0": gate_sum,
    "ffn_moe_up-0": up_sum,
    "ffn_moe_swiglu-0": swiglu_sum,
    "ffn_moe_down-0": down_sum,
    "ffn_moe_weighted-0": weighted_sum,
    "ffn_moe_out-0": float(routed.sum()),
    "ffn_gate-0": sg,
    "ffn_up-0": su,
    "ffn_swiglu-0": ssilu,
    "ffn_shexp-0": sdown,
    "ffn_shexp_gated-0": sgated,
    "ffn_out-0": float((routed + shared_out).sum()),
}

print(f"forme: att {acts.shape}, gate {w_gate.shape}, up {w_up.shape}, "
      f"down {w_down.shape}, gate_inp {w_gate_inp.shape}")
print(f"\n{'tappa':24s} {'noi':>15s} {'llama.cpp':>15s} {'errore':>9s}")
for name, ref in REF:
    v = ours[name]
    err = abs(v - ref) / max(abs(ref), 1e-9)
    flag = "" if err < 0.02 else "   <-- si stacca"
    print(f"{name:24s} {v:15.6f} {ref:15.6f} {err * 100:8.2f}%{flag}")

model.close()
