"""Lo shared expert cambia davvero l'output di ogni token.

Il rischio di questo componente non e' che sbagli silenziosamente i numeri,
e' che sembri funzionare: e' un contributo in piu' a una somma, quindi se lo
si dimentica non compare nessun errore, solo un modello peggiore. I controlli
qui sotto verificano che il contributo ci sia, abbia la forma giusta, e sia
grande abbastanza da non essere trascurabile rispetto agli expert routed.

Verifica anche la forma del gate sigmoid e la coerenza del layout dei pesi
con tensor.shape del GGUF, che e' dove un errore di reshape passerebbe
inosservato: i valori resterebbero plausibili.
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import gguf

from flash_moe.paths import model_path
from flash_moe.shared_expert import (
    SharedExpert, shared_expert_present, shared_expert_slices, _sigmoid,
)
from flash_moe.dequant import name_of

H, I, N_LAYER = 2048, 512, 40
path = model_path()
r = gguf.GGUFReader(path)
tmap = {t.name: t for t in r.tensors}

ok = bad = 0
def check(label, cond, detail=""):
    global ok, bad
    if cond:
        ok += 1; print(f"  PASS  {label}  ({detail})")
    else:
        bad += 1; print(f"  FAIL  {label}  ({detail})")

print("1. presenza nel GGUF")
check("layer 0 ha shared expert", shared_expert_present(tmap, 0))
n_with = sum(1 for l in range(N_LAYER) if shared_expert_present(tmap, l))
check("tutti i 40 layer lo hanno", n_with == N_LAYER, f"({n_with}/{N_LAYER})")

print("\n2. forme dei tensor")
sl = shared_expert_slices(tmap, 0)
for role, want in (("gate_inp", (H,)), ("gate", (H, I)),
                   ("up", (H, I)), ("down", (I, H))):
    got = sl[role]["shape"]
    check(f"{role} shape {got}", got == want, f"(atteso {want})")

print("\n3. sigmoid numericamente")
v = np.array([-800.0, -1.0, 0.0, 1.0, 800.0], dtype=np.float32)
s = _sigmoid(v)
check("nessun overflow/NaN", bool(np.all(np.isfinite(s))), f"{s}")
check("s(0) = 0.5", abs(float(s[2]) - 0.5) < 1e-6, f"({s[2]})")
check("s(800) = 1", abs(float(s[4]) - 1.0) < 1e-6, f"({s[4]})")
check("s(-800) = 0", abs(float(s[0])) < 1e-6, f"({s[0]})")
check("monotona", bool(np.all(np.diff(s) >= 0)))

print("\n4. caricamento e forma degli output")
se = SharedExpert(path, num_layers=2, hidden_size=H, intermediate_size=I)
check("2 layer caricati", len(se.layers) == 2, f"({sorted(se.layers)})")
for role, want in (("gate_inp", (H,)), ("gate", (H, I)),
                   ("up", (H, I)), ("down", (I, H))):
    got = se.layers[0][role].shape
    check(f"pesi {role} {got}", got == want, f"atteso {want} dtype f32")

rng = np.random.default_rng(0)
x = rng.standard_normal(H).astype(np.float32)
out = se.forward(0, x)
check("output hidden-shaped", out.shape == (H,), f"{out.shape} {out.dtype}")
check("output finito", bool(np.all(np.isfinite(out))))
check("output non tutto zero", float(np.abs(out).max()) > 0,
      f"|max|={float(np.abs(out).max()):.5f}")

print("\n5. il gate sigmoid scala davvero")
# sigmoid(0) = 0.5, non 0: azzerare i pesi NON azzera il contributo. Il test
# non puo' assumere il segno di x @ w, perche' con x casuale il prodotto
# scalare segue il segno di sum(x), non quello dei pesi. Per verificare
# l'estremi si satura il gate nel verso giusto, calcolato dal prodotto.
x0 = rng.standard_normal(H).astype(np.float32)
ref = se.forward(0, x0)
check("gate a zero -> sigmoid 0.5 (non nullo)", float(np.abs(ref).max()) > 0,
      f"sigmoid(0)=0.5, |max|={float(np.abs(ref).max()):.5f}")

saved = se.layers[0]["gate_inp"].copy()
sign = -1.0 if float(x0.sum()) > 0 else 1.0   # cosi' x@w e' saturo nel verso noto
se.layers[0]["gate_inp"][:] = np.float16(sign * 1e4)
off = se.forward(0, x0)
se.layers[0]["gate_inp"][:] = np.float16(-sign * 1e4)
on = se.forward(0, x0)
se.layers[0]["gate_inp"] = saved
restored = se.forward(0, x0)

check("gate saturato a sigmoid~0 -> contributo nullo",
      float(np.abs(off).max()) == 0.0, f"|max|={float(np.abs(off).max()):.3e}")
check("gate saturato a sigmoid~1 -> contributo massimo",
      float(np.abs(on).max()) > float(np.abs(restored).max()),
      f"|max|saturo={float(np.abs(on).max()):.5f} vs normale="
      f"{float(np.abs(restored).max()):.5f}")
check("ripristinato -> contributo intermedio",
      0 < float(np.abs(restored).max()) < float(np.abs(on).max()))

# il gate e' in (0,1): il contributo deve essere <= quello non scalato
w = se.layers[0]
x1 = rng.standard_normal(H).astype(np.float32)
g = x1 @ w["gate_inp"].astype(np.float32)
gg = x1 @ w["gate"].astype(np.float32)
act = (gg * _sigmoid(gg)) * (x1 @ w["up"].astype(np.float32))
unscaled = act @ w["down"].astype(np.float32)
scaled = se.forward(0, x1)
ratio = float(np.linalg.norm(scaled) / (np.linalg.norm(unscaled) + 1e-30))
check("rapporto = sigmoid(gate_inp)", abs(ratio - float(_sigmoid(g.reshape(())))) < 1e-4,
      f"ratio={ratio:.6f} sigmoid={float(_sigmoid(g.reshape(()))):.6f}")

print("\n6. impatto sull'output del layer (il punto che conta)")
from flash_moe.key import ExpertKey
from flash_moe.manifest import ExpertManifest, ExpertRecord, ExpertTensorSlice
from flash_moe.store import AsyncExpertStore
from flash_moe.backend_ov import OpenVINOExpertBackend
from flash_moe.router import routers_from_gguf
from flash_moe.engine import MoEEngine
from run_q4k_f16_openvino import build_manifest_from_gguf, dequant_to_f16, expert_tensor_types

manifest = build_manifest_from_gguf(path)
ttypes = expert_tensor_types(path)
store = AsyncExpertStore(path.parent, capacity=64, manifest=manifest)
backend = OpenVINOExpertBackend(store, device="CPU", cache_capacity=64,
                                tensor_types=ttypes, dequant_fn=dequant_to_f16)
routers = routers_from_gguf(tmap, num_layers=1, hidden_size=H,
                            num_experts=256, top_k=8, gate_dtype=np.float32)

eng_no = MoEEngine(store, backend, routers, H, I, shared_expert=None)
eng_si = MoEEngine(store, backend, routers, H, I, shared_expert=se)

xa = rng.standard_normal(H).astype(np.float32)
a = np.asarray(eng_no.process_token(0, xa), dtype=np.float32)
b = np.asarray(eng_si.process_token(0, xa), dtype=np.float32)
rel = float(np.abs(a - b).max() / max(1e-12, float(np.abs(a).max())))
ratio = float(np.linalg.norm(b) / (np.linalg.norm(a) + 1e-30))
check("l'output cambia", not np.array_equal(a, b), f"|max diff|={float(np.abs(a-b).max()):.5f}")
check("cambia in modo misurabile (>1%)", rel > 0.01,
      f"rel={rel:.3f} norma shared/routed={ratio:.3f}")
check("stesso hidden", a.shape == b.shape == (H,))

print(f"\n{'=' * 64}\n PASSATI: {ok}   FALLITI: {bad}\n{'=' * 64}")
sys.exit(1 if bad else 0)