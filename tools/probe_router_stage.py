import sys, numpy as np
sys.path.insert(0, "src")
import gguf
from run_q4k_f16_openvino import dequant_to_f16
from flash_moe import Qwen3MoEModel
from flash_moe.tokenizer import GGUFTokenizer
from flash_moe.paths import model_path

REF = {"logits": -7842.232910, "probs": 5.000006, "weights": 1.436570}
p = model_path()
tok = GGUFTokenizer(gguf.GGUFReader(p))
ids = tok.encode("La capital of France is", add_bos=False)
m = Qwen3MoEModel(p, device="CPU", max_seq_len=256, cache_capacity=64, dequant_fn=dequant_to_f16)
m.init_caches()
x = m.embedding.forward(np.asarray(ids, np.int64)).reshape(1, len(ids), m.hidden_size)
b = m.blocks[0]
normed = b.attn_norm.forward(x)
mixed = b.ssm_runner(normed)
h = x + mixed
a = b.post_attn_norm.forward(h)[0]
r = m.routers[0]
logits = np.stack([r.scores(a[i]) for i in range(len(ids))])
probs = np.stack([np.exp(l - l.max()) / np.exp(l - l.max()).sum() for l in logits])
w = []
sel = []
for i in range(len(ids)):
    k = r.top_k
    idx = np.argsort(logits[i])[-k:][::-1]
    sel.append(idx)
    w.append(probs[i][idx])
w = np.stack(w)
print(f"{'tappa':10s} {'noi':>16s} {'llama.cpp':>16s}  errore")
for name, ours, ref in (("logits", logits.sum(), REF["logits"]),
                        ("probs", probs.sum(), REF["probs"]),
                        ("weights", w.sum(), REF["weights"])):
    err = abs(ours - ref) / abs(ref)
    print(f"{name:10s} {ours:16.6f} {ref:16.6f}  {err*100:7.2f}%")
print("expert scelti per posizione:", [list(s) for s in sel])
print("pesi pos0:", (w[0] / w[0].sum()).round(4).tolist())
m.close()

