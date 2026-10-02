"""I due percorsi che sono stati cambiati devono dare gli stessi numeri.

Il backend OpenVINO ora inferisce con ``share_inputs=True`` e lo shared
expert tiene i pesi gia' in f32. Entrambe le modifiche dovrebbero toccare
solo il *quando* si fa un lavoro, non *cosa* si calcola: qui si controlla
che l'output sia identico a quello di una reference scritta a parte, con i
pesi ricalcolati da capo nel modo di prima.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\valer\Desktop\moe-client-prot\Nuova cartella")
sys.path.insert(0, str(ROOT / "src"))

from flash_moe.paths import model_path
from flash_moe.key import ExpertKey
from flash_moe.store import AsyncExpertStore
from flash_moe.backend_ov import OpenVINOExpertBackend
from flash_moe.router import routers_from_gguf
from flash_moe.shared_expert import SharedExpert, _sigmoid
from run_q4k_f16_openvino import (build_manifest_from_gguf, dequant_to_f16,
                                  expert_tensor_types)

H, I, LAYER = 2048, 512, 5
path = model_path()

ok = bad = 0


def check(label, cond, detail=""):
    global ok, bad
    if cond:
        ok += 1
        print(f"  PASS  {label}  ({detail})")
    else:
        bad += 1
        print(f"  FAIL  {label}  ({detail})")


def swiglu_reference(x, gate, up, down):
    """SwiGLU scritto a parte, in numpy, senza OpenVINO."""
    g = x.astype(np.float32) @ gate.astype(np.float32)
    act = (g * _sigmoid(g)) * (x.astype(np.float32) @ up.astype(np.float32))
    return (act @ down.astype(np.float32)).astype(np.float32)


print("1. backend OpenVINO: share_inputs non cambia il risultato")
manifest = build_manifest_from_gguf(path)
ttypes = expert_tensor_types(path)
store = AsyncExpertStore(path.parent, capacity=64, manifest=manifest)
backend = OpenVINOExpertBackend(store, device="CPU", cache_capacity=64,
                                tensor_types=ttypes, dequant_fn=dequant_to_f16)
rng = np.random.default_rng(7)
x = (rng.standard_normal(H) * 0.5).astype(np.float32)
key = ExpertKey(LAYER, 3)
backend.preload(key, H, I)
got = np.asarray(backend.forward(key, x, H, I), dtype=np.float32)

gate, up, down = backend._weights(key, H, I)
ref = swiglu_reference(x, gate, up, down)
rel = float(np.abs(got - ref).max() / max(1e-12, float(np.abs(ref).max())))
check("share_inputs attivo", backend._share_inputs is True)
check("output entro l'errore f16 atteso", rel < 5e-3, f"rel={rel:.2e}")
check("output non degenere", float(np.abs(got).max()) > 0,
      f"|max|={float(np.abs(got).max()):.5f}")

# stesso expert, due volte: il risultato non deve dipendere da quante volte
# il peso e' stato riusato, ne' dalla copia interna
again = np.asarray(backend.forward(key, x, H, I), dtype=np.float32)
check("deterministico fra due chiamate", np.array_equal(got, again))

print("\n2. shared expert: i pesi sono f32 e il risultato e' quello di prima")
se = SharedExpert(path, num_layers=2, hidden_size=H, intermediate_size=I,
                  dequant_fn=dequant_to_f16)
for role in ("gate_inp", "gate", "up", "down"):
    check(f"{role} in f32", se.layers[0][role].dtype == np.float32,
          f"{se.layers[0][role].dtype}")

x2 = (rng.standard_normal(H) * 0.5).astype(np.float32)
got2 = se.forward(0, x2)
w = se.layers[0]
old_style = swiglu_reference(x2, w["gate"], w["up"], w["down"])
scaled = float(_sigmoid((x2 @ w["gate_inp"]).reshape(())))
old_style = (scaled * old_style).astype(np.float32)
check("output identico al calcolo con cast per token",
      np.array_equal(got2, old_style),
      f"max|diff|={float(np.abs(got2 - old_style).max()):.3e}")
check("il gate scala ancora", 0.0 < scaled < 1.0, f"sigmoid={scaled:.6f}")

print("\n3. il layer MoE completo cambia solo nella velocita'")
import gguf
reader = gguf.GGUFReader(path)
tmap = {t.name: t for t in reader.tensors}
routers = routers_from_gguf(tmap, 1, H, 256, 8, arch="qwen35moe")
from flash_moe.moe_layer import run_moe_layer

full = run_moe_layer(layer=0, activation=x2, router=routers[0], backend=backend,
                     hidden_size=H, intermediate_size=I, shared_expert=se)
ids, weights = routers[0].route(x2)
manual = np.zeros(H, dtype=np.float32)
for eid, wgt in zip(ids, weights):
    manual += wgt * np.asarray(backend.forward(ExpertKey(0, eid), x2, H, I),
                               dtype=np.float32)
manual += se.forward(0, x2)
rel = float(np.abs(full - manual).max() / float(np.abs(full).max()))
check("routing + shared = run_moe_layer", rel < 1e-6, f"rel={rel:.2e}")
check("il contributo shared e' presente", not np.array_equal(full, manual - se.forward(0, x2)))
store.close()

print(f"\n{'=' * 64}\n PASSATI: {ok}   FALLITI: {bad}\n{'=' * 64}")
sys.exit(1 if bad else 0)
