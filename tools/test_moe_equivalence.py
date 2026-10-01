"""I due percorsi del layer MoE devono dare lo stesso risultato.

Il layer MoE era calcolato in due punti: MoEEngine.process_token e
TransformerBlock._moe_forward. Erano due implementazioni, e lo shared expert
era finito solo nella prima: TransformerBlock continuava a produrre output
sbagliati senza alcun errore. E' esattamente il difetto che questo test
vuole impedire che torni.

Il test verifica:
  1. i due percorsi coincidono bit per bit a parita' di layer e attivazione
  2. TransformerBlock include il contributo dello shared expert
  3. senza shared expert i due coincidono ancora (la differenza deve venire
     solo dallo shared expert, non da altro)
  4. il percorso del block non e' piu' una copia: usa la funzione condivisa
"""
import sys
sys.path.insert(0, 'src')

import numpy as np

from flash_moe.paths import model_path
from flash_moe.key import ExpertKey
from flash_moe.engine import MoEEngine
from flash_moe.moe_layer import run_moe_layer
from flash_moe.store import AsyncExpertStore
from flash_moe.manifest import ExpertManifest, ExpertRecord, ExpertTensorSlice
from flash_moe.backend_ov import OpenVINOExpertBackend
from flash_moe.router import routers_from_gguf
from flash_moe.shared_expert import SharedExpert
from flash_moe.transformer import TransformerBlock
from flash_moe.norm import RMSNorm
from run_q4k_f16_openvino import build_manifest_from_gguf, dequant_to_f16, expert_tensor_types

import gguf

H, I, NE, TOP_K = 2048, 512, 256, 8
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

manifest = build_manifest_from_gguf(path)
ttypes = expert_tensor_types(path)
store = AsyncExpertStore(path.parent, capacity=64, manifest=manifest)
backend = OpenVINOExpertBackend(store, device="CPU", cache_capacity=64,
                                tensor_types=ttypes, dequant_fn=dequant_to_f16)
routers = routers_from_gguf(tmap, num_layers=1, hidden_size=H,
                            num_experts=NE, top_k=TOP_K, gate_dtype=np.float32)
shared = SharedExpert(path, num_layers=1, hidden_size=H,
                      intermediate_size=I, dequant_fn=dequant_to_f16)

rng = np.random.default_rng(7)
x = rng.standard_normal(H).astype(np.float32)

# un block MoE-like: i pesi di attenzione non servono per il confronto, che
# guarda solo al ramo MoE. Si costruisce comunque un TransformerBlock vero,
# perche' il punto e' che _moe_forward sia quello del progetto.
def make_block(with_shared):
    ones = np.ones(H, dtype=np.float16)
    return TransformerBlock(
        layer_idx=0, hidden_size=H, num_heads=16, num_kv_heads=16,
        head_dim=128, kv_head_dim=128, rope_dim=128, freq_base=10000.0,
        max_seq_len=64, attn_norm_weight=ones, post_attn_norm_weight=ones,
        qkv_weight=np.zeros((H, H * 3), dtype=np.float16),
        attn_gate_weight=np.zeros((H, H), dtype=np.float16),
        backend=backend, router=routers[0], intermediate_size=I,
        num_experts=NE, shared_expert=(shared if with_shared else None),
    )

print("1. i due percorsi coincidono con shared expert")
eng = MoEEngine(store, backend, routers, H, I, shared_expert=shared)
a = np.asarray(eng.process_token(0, x), dtype=np.float32)
b = make_block(True)._moe_forward(x.reshape(1, 1, H))[0, 0].astype(np.float32)
check("engine == TransformerBlock", np.allclose(a, b, rtol=0, atol=0),
      f"max|d|={float(np.abs(a - b).max()):.3e}")
check("output non triviale", float(np.abs(a).max()) > 0,
      f"|max|={float(np.abs(a).max()):.5f}")

print("\n2. TransformerBlock include lo shared expert")
c = make_block(False)._moe_forward(x.reshape(1, 1, H))[0, 0].astype(np.float32)
check("con e senza shared cambiano", not np.allclose(a, c, rtol=0, atol=0),
      f"max|d|={float(np.abs(a - c).max()):.5f}")
eng_ns = MoEEngine(store, backend, routers, H, I, shared_expert=None)
d = np.asarray(eng_ns.process_token(0, x), dtype=np.float32)
check("senza shared i due percorsi coincidono", np.allclose(c, d, rtol=0, atol=0),
      f"max|d|={float(np.abs(c - d).max()):.3e}")
check("engine senza shared == block senza shared",
      np.allclose(c, d, rtol=0, atol=0))
check("block senza shared == solo expert routed",
      float(np.abs(a - c).max()) > 0,
      f"lo shared vale {float(np.linalg.norm(a - c)):.5f} in norma")

print("\n3. la funzione condivisa e' un'unica definizione")
import inspect
from flash_moe import transformer as tmod
from flash_moe import engine as emod
src_t = inspect.getsource(tmod.TransformerBlock._moe_forward)
src_e = inspect.getsource(emod.MoEEngine.process_token)
check("_moe_forward delega a run_moe_layer", "run_moe_layer(" in src_t)
check("process_token delega a run_moe_layer", "run_moe_layer(" in src_e)
check("nessuna copia del calcolo nel block",
      "prefetch_many" not in src_t and "backend.forward" not in src_t,
      "il block non piu' esegue gli expert per conto suo")
check("nessuna copia del calcolo nell'engine",
      "prefetch_many" not in src_e and "backend.forward" not in src_e)

print("\n4. la funzione condivisa usata direttamente coincide")
e = run_moe_layer(0, x, routers[0], backend, H, I, shared_expert=shared)
check("run_moe_layer == engine", np.array_equal(e, a))
f = run_moe_layer(0, x, routers[0], backend, H, I, shared_expert=None)
check("run_moe_layer senza shared == engine senza shared", np.array_equal(f, d))

print(f"\n{'=' * 66}\n PASSATI: {ok}   FALLITI: {bad}\n{'=' * 66}")
sys.exit(1 if bad else 0)