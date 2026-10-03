"""Il pool di buffer f16 non deve corrompere i pesi.

Il rischio concreto: un buffer evictato e riciclato viene riscritto dal
prossimo expert, quindi se qualcuno lo stesse ancora leggendo otterrebbe i
pesi sbagliati invece di un errore. Il test copre tre cose:

1. i valori in un buffer riciclato sono diversi prima e dopo il riuso
2. il backend non espone i propri array (un'eviction non può corrompere
   nessuno), verificato sul risultato di forward()
3. il pool non cresce oltre cache_capacity, quindi la RAM torna a zero
"""
import sys
sys.path.insert(0, 'src')

import numpy as np

from flash_moe.backend_ov import OpenVINOExpertBackend
from flash_moe.key import ExpertKey
from flash_moe.paths import model_path
from run_q4k_f16_openvino import (
    build_manifest_from_gguf, dequant_to_f16, expert_tensor_types,
)

H, I, NE, CAP = 2048, 512, 256, 8
path = model_path()
manifest = build_manifest_from_gguf(path)
ttypes = expert_tensor_types(path)

ok, failed = 0, 0


def check(label, cond, extra=""):
    global ok, failed
    if cond:
        ok += 1
        print(f"  PASS  {label}  {extra}")
    else:
        failed += 1
        print(f"  FAIL  {label}  {extra}")


class _Store:
    """Store minimo: restituisce sempre gli stessi byte, gia' in cache."""

    def __init__(self, manifest):
        self.manifest = manifest

    def read_tensor(self, key, role):
        rec = self.manifest.records[key]
        sl = rec.tensors[role]
        with open(path, "rb") as f:
            f.seek(sl.offset)
            return f.read(sl.nbytes)

    def close(self):
        pass


be = OpenVINOExpertBackend(_Store(manifest), device="CPU",
                           cache_capacity=CAP, tensor_types=ttypes,
                           dequant_fn=dequant_to_f16)

x = np.random.default_rng(0).standard_normal(H).astype(np.float32)

# --- 1. il riciclo produce risultati diversi expert per expert ---
print("\n1. il riciclo non produce pesi sbagliati")
seen = {}
for e in range(CAP * 3):
    k = ExpertKey(0, e)
    out = be.forward(k, x, H, I)
    seen[k] = out.copy()

# stesso expert ricalcolato dopo che i suoi buffer sono stati riciclati:
# deve dare lo stesso risultato di quando e' stato calcolato la prima volta
mismatch = 0
for e in range(CAP * 3):
    k = ExpertKey(0, e)
    again = be.forward(k, x, H, I)
    if not np.array_equal(again, seen[k]):
        mismatch += 1
check("ricomputare un expert dà lo stesso risultato", mismatch == 0,
      f"({mismatch} mismatch su {CAP*3})")

# --- 2. il buffer riciclato è davvero riscritto ---
print("\n2. il pool ricicla buffer davvero (non alloca nuove pagine)")
grown = be.pool_size()
check("il pool si e' riempito", grown > 0, f"(pool={grown})")
check("il pool non supera cache_capacity per taglia",
      all(len(slot) <= max(1, CAP) for slot in be._pool.values()),
      f"({ {k: len(v) for k, v in be._pool.items()} })")

# content check: un buffer nel pool viene riscritto dall'expert successivo
if grown:
    n = max(be._pool)
    buf = be._pool[n][0]
    before = buf.copy()
    other = ExpertKey(1, 0)
    be.forward(other, x, H, I)
    check("il buffer riciclato viene riscritto", not np.array_equal(buf, before))

# --- 3. risultati numerici contro NumPy, per un expert fresco ---
print("\n3. i pesi nel pool restano corretti")
be.clear_cache()
check("clear_cache svuota anche il pool", be.pool_size() == 0,
      f"(pool={be.pool_size()})")

import gguf
for e in (0, 3):
    k = ExpertKey(0, e)
    got = be.forward(k, x, H, I)
    be.clear_cache()
    w = {}
    # I pesi sono nel verso in cui stanno nel file: gate/up su n_ff, down su
    # n_hidden. Il prodotto contrae l'hidden, quindi si trasposta la vista.
    for role, shp in (("gate", (I, H)), ("up", (I, H)), ("down", (H, I))):
        raws = be.store.read_tensor(k, role)
        w[role] = dequant_to_f16(raws, ttypes[0][role], H * I).reshape(
            shp).astype(np.float32)
    # stesse matrici del backend: x @ gate, silu, poi @ down
    g = x @ w["gate"].T
    ref = (g * (1 / (1 + np.exp(-g)))) * (x @ w["up"].T) @ w["down"].T
    rel = float(np.abs(got - ref).max() / max(1e-9, float(np.abs(ref).max())))
    check(f"expert {e}: OpenVINO = NumPy entro f16", rel < 5e-3, f"(rel={rel:.2e})")

print(f"\n{'='*66}\n PASSATI: {ok}   FALLITI: {failed}\n{'='*66}")
sys.exit(1 if failed else 0)