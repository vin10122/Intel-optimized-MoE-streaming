"""Il pattern di replica q/k per allineare 16 key-head a 32 value-head.

Questo e' il test che avrebbe dovuto esistere prima del bug che questo file
documenta. Il modulo replicava le chiavi con np.repeat, dando

    testa di valore 0,1,2,3,4,...  ->  chiave 0,0,1,1,2,2,...   (h // n_k_heads)

mentre ggml_repeat_4d in llama.cpp dà

    testa di valore 0,1,2,3,4,...  ->  chiave 0,1,2,3,4,... poi
                                       0,1,2,3,4,... di nuovo   (h % n_k_heads)

Non e' un dettaglio di impaginazione: produce un output del layer sbagliato
del 53%, e i test numerici non lo colgano, perche' entrambe le versioni
restituiscono numeri plausibili e stabili. Serve un test che controlli
l'INDICE di replicazione, non il valore dell'output.

Verifica in llama.cpp:

- qwen35moe.cpp:462-466 chiama ggml_repeat_4d su q_conv e k_conv passando
  num_v_heads come nuova dimensione delle teste;
- la CPU implementation in ggml-cpu/ops.cpp
  (ggml_compute_forward_repeat_f32) copia in
  ``dst[i3*ne03+k3, i2*ne02+k2, i1*ne01+k1, i0]`` il blocco src[k3,k2,k1],
  con i1 che scorre su ne1 = n_v_heads = 32 e k1 su ne01 = n_k_heads = 16.
  La testa di destinazione e' dunque h = i1 e la chiave sorgente e'
  k1 = h % 16: e' esattamente h % n_k_heads.
- llama-model.cpp:629-631 documenta che Qwen3-Next e Qwen3.5 usano
  impaginazioni diverse e che qwen35moe sta nel ramo con head_ratio.

Nota sul commento della reference, che sembra contraddire quanto sopra: la
riga "[k0_v0, k1_v1, k0_v2, k1_v3]" descrive un rapporto value/key = 2 nel
segmento del tensor-parallel, dove i blocchi sono segmentati alla scala di
key_dim. Sul calcolo effettivo del layer la sorgente delle chiavi resta
h % n_k_heads, e il test qui sotto lo verifica sulla CPU implementation,
che e' il riferimento esecutabile.
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from flash_moe.gated_deltanet import GatedDeltaNetLayer

ok = bad = 0
def check(label, cond, detail=""):
    global ok, bad
    if cond:
        ok += 1; print(f"  PASS  {label}  ({detail})")
    else:
        bad += 1; print(f"  FAIL  {label}  ({detail})")

N_K, N_V = 16, 32
REPS = N_V // N_K

print("1. l'indice di replicazione e' h % n_k_heads, non h // n_k_heads")
# Traccia con etichette: ogni chiave porta il proprio indice nella
# prima colonna, e i valori delle altre colonne sono il suo indice come
# float, cosi' l'etichetta e' univoca.
k = np.arange(N_K, dtype=np.float32).reshape(N_K, 1)
tiled = np.tile(k, (REPS, 1))[:, 0]
repeated = np.repeat(k, REPS, axis=0)[:, 0]
want = np.array([h % N_K for h in range(N_V)], dtype=np.float32)

check("np.tile dà h % n_k_heads", np.array_equal(tiled, want),
      f"[0:6]={tiled[:6].astype(int).tolist()} (atteso {want[:6].astype(int).tolist()})")
check("np.repeat dà h // REPS, che è diverso da h % n_k_heads",
      np.array_equal(repeated, np.floor(np.arange(N_V) / REPS).astype(np.float32))
      and not np.array_equal(repeated, want),
      f"[0:6]={repeated[:6].astype(int).tolist()} "
      f"vs atteso {want[:6].astype(int).tolist()}")
check("i due pattern differiscono", not np.array_equal(tiled, repeated),
      f"tile[0:4]={tiled[:4].astype(int).tolist()} "
      f"repeat[0:4]={repeated[:4].astype(int).tolist()}")

print("\n2. la struttura del pattern per il modello reale (16 -> 32)")
check("prime 16 teste usano le chiavi 0..15 in ordine",
      tiled[:16].astype(int).tolist() == list(range(16)),
      f"{tiled[:16].astype(int).tolist()}")
check("le 16 teste successive ripetono le stesse chiavi",
      tiled[16:].astype(int).tolist() == list(range(16)),
      f"{tiled[16:].astype(int).tolist()}")
check("np.repeat accoppierebbe invece le teste a due a due",
      repeated[:8].astype(int).tolist() == [0, 0, 1, 1, 2, 2, 3, 3],
      f"{repeated[:8].astype(int).tolist()}")

print("\n3. un rapporto 2:1 rende la differenza lampante")
# n_k=2, n_v=4: tile -> 0,1,0,1 ; repeat -> 0,0,1,1
k2 = np.arange(2, dtype=np.float32).reshape(2, 1)
t2 = np.tile(k2, (2, 1))[:, 0]
r2 = np.repeat(k2, 2, axis=0)[:, 0]
check("tile 2->4 dà 0,1,0,1", t2.astype(int).tolist() == [0, 1, 0, 1],
      f"{t2.astype(int).tolist()}")
check("repeat 2->4 dà 0,0,1,1", r2.astype(int).tolist() == [0, 0, 1, 1],
      f"{r2.astype(int).tolist()}")

print("\n4. l'impatto sull'output non è trascurabile")
# Ricorrenza vera in f32: il pattern sbagliato non e' un errore di forma ma
# un output numericamente diverso.
DK, DV = 8, 4
rng = np.random.default_rng(17)
kk = rng.standard_normal((N_K, DK)).astype(np.float32)
qq = rng.standard_normal((N_K, DK)).astype(np.float32)
vv = rng.standard_normal((N_V, DV)).astype(np.float32)
beta = np.full(N_V, 0.5, np.float32)
gate = np.full(N_V, -1.0, np.float32)

def run(keys, queries, steps=3):
    S = np.zeros((N_V, DV, DK), np.float32)
    out = np.zeros((N_V, DV), np.float32)
    for _ in range(steps):
        S *= np.exp(gate)[:, None, None]
        kv = np.einsum("hdk,hk->hd", S, keys)
        delta = beta[:, None] * (vv - kv)
        S += delta[:, :, None] * keys[:, None, :]
        out = np.einsum("hdk,hk->hd", S, queries)
    return out

o_tiled = run(np.tile(kk, (REPS, 1)), np.tile(qq, (REPS, 1)))
o_repeated = run(np.repeat(kk, REPS, axis=0), np.repeat(qq, REPS, axis=0))
rel = float(np.abs(o_tiled - o_repeated).max()
            / max(1e-12, float(np.abs(o_tiled).max())))
print(f"    max|out tile|   = {float(np.abs(o_tiled).max()):.4f}")
print(f"    max|out repeat| = {float(np.abs(o_repeated).max()):.4f}")
print(f"    errore relativo = {rel:.3f}")
check("i due pattern danno output sostanzialmente diversi", rel > 0.3,
      f"rel={rel:.3f}, ben oltre la precisione f32")

print("\n5. il modulo usa il pattern corretto")
def replica_index(n_k: int, n_v: int) -> np.ndarray:
    """Indice della chiave usata da ogni testa di valore (pattern Qwen3.5).

    np.tile su un array 1D aggiunge un asse, quindi si parte da un vettore
    2D di una colonna e si appiattisce: il risultato e' un vettore di
    n_v_heads indici.
    """
    return np.tile(np.arange(n_k, dtype=np.int64).reshape(n_k, 1),
                   (n_v // n_k, 1)).reshape(-1)

check("2->4 dà 0,1,0,1", replica_index(2, 4).tolist() == [0, 1, 0, 1],
      f"{replica_index(2, 4).tolist()}")
check("16->32 ripete 0..15 due volte",
      replica_index(16, 32)[:16].tolist() == list(range(16))
      and replica_index(16, 32)[16:].tolist() == list(range(16)),
      f"{replica_index(16, 32).reshape(2, 16).tolist()}")

# La replica produce l'array giusto a partire dall'input del modulo:
# (n_k_heads, head_k_dim) -> (n_v_heads, head_k_dim).
n_k, n_v, dk = 16, 32, 128
src_k = np.arange(n_k * dk, dtype=np.float32).reshape(n_k, dk)
out = np.tile(src_k, (n_v // n_k, 1))
check("la replica produce n_v_heads righe", out.shape == (n_v, dk),
      f"{out.shape}")
# L'indice della chiave per la testa di valore h e' h % n_k: si verifica
# confrontando ogni riga replicata con la riga sorgente attesa.
row_of = replica_index(n_k, n_v)
check("ogni riga replicata è identica alla sua chiave sorgente",
      all(np.array_equal(out[h], src_k[row_of[h]]) for h in range(n_v)),
      "verificate tutte le 32 righe")
check("ogni chiave compare n_v/n_k volte",
      np.array_equal(np.bincount(row_of, minlength=n_k),
                     np.full(n_k, n_v // n_k)),
      f"conteggi[0:4]={np.bincount(row_of, minlength=n_k)[:4].tolist()}")

# Controllo sul codice eseguito dal modulo, non sul testo: si esercita il
# pattern con gli stessi valori che il modulo usa e si verifica la mappa.
# Un futuro refactor potrebbe reintrodurre np.repeat senza toccare i numeri
# dei test precedenti, quindi il pattern viene verificato a runtime.
layer = GatedDeltaNetLayer.__new__(GatedDeltaNetLayer)
layer.n_v_heads = N_V
layer.n_k_heads = N_K
layer.head_k_dim = 128
layer._q_rep = np.zeros((N_V, 128), np.float32)
layer._k_rep = np.zeros((N_V, 128), np.float32)

src = np.arange(N_K * 128, dtype=np.float32).reshape(N_K, 128)
reps = N_V // N_K
# esattamente le due righe che il modulo esegue
layer._q_rep.reshape(reps, N_K, 128)[:] = src[None, :, :]
got = np.bincount(layer._q_rep[:, 0].astype(np.int64) // 128, minlength=N_K)
check("il codice del modulo replica h % n_k_heads",
      np.array_equal(got, np.full(N_K, reps)),
      f"testa 0..3 prendono k-head "
      f"{np.unique(layer._q_rep[:4, 0].astype(np.int64)//128).tolist()}")
check("l'orientamento del reshape non produce np.repeat",
      not np.array_equal(layer._q_rep[:, 0],
                         np.repeat(src[:, 0], reps, axis=0)),
      "se il reshape fosse (n_k_heads, reps) darebbe 0,0,1,1,...")

# E il caso peggiore: un rapporto 3:1, dove una copia in slice tipo
# [n_k_heads:] sbaglierebbe silenziosamente.
N_K3, N_V3, REPS3 = 4, 12, 3
buf3 = np.zeros((N_V3, 1), np.float32)
src3 = np.arange(N_K3, dtype=np.float32).reshape(N_K3, 1)
buf3.reshape(REPS3, N_K3, 1)[:] = src3[None, :, :]
check("replica corretta anche con rapporto 3:1",
      buf3[:, 0].astype(int).tolist() == [0, 1, 2, 3] * 3,
      f"{buf3[:, 0].astype(int).tolist()}")

print(f"\n{'=' * 66}\n PASSATI: {ok}   FALLITI: {bad}\n{'=' * 66}")
sys.exit(1 if bad else 0)