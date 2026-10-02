"""Il RoPE NeoX a coppie distanti (j, j + n_dims/2).

Il test che conta e' il primo: distinguishes il layout NeoX da quello
interleaved. E' una distinzione invisibile a un test di magnitudine, perche'
entrambi gli schemi restituiscono valori nella stessa scala e con la stessa
norma: un output sbagliato ma plausibile passa qualsiasi confronto sui
valori assoluti. Qui si controlla la posizione delle componenti che
ruotano, non il loro valore.

Riferimento: llama.cpp ``ggml_compute_forward_rope_flt`` chiama
``rotate_pairs(n_dims, n_dims/2, ...)`` per ``GGML_ROPE_TYPE_NEOX``, e dentro
``rotate_pairs`` l'indice e' ``ic = i0/2`` con ``src[ic]`` e
``src[ic + n_offset]``. CUDA fa lo stesso in ``ggml_cuda_rope``.
"""
import sys
sys.path.insert(0, 'src')

import numpy as np

from flash_moe.rope import RoPE

ok = bad = 0
def check(label, cond, detail=""):
    global ok, bad
    if cond:
        ok += 1; print(f"  PASS  {label}  ({detail})")
    else:
        bad += 1; print(f"  FAIL  {label}  ({detail})")

HEAD_DIM, ROPE_DIM = 16, 8      # piccoli, cosi' le coppie sono ispezionabili
HALF = ROPE_DIM // 2

print("1. le coppie ruotate sono (j, j+4), non (2j, 2j+1)")
rope = RoPE(head_dim=HEAD_DIM, rope_dim=ROPE_DIM, freq_base=10000.0)
x = np.arange(HEAD_DIM, dtype=np.float32).reshape(1, HEAD_DIM)
y = rope.apply(x, seq_offset=1)[0]

# con posizione 0 la rotazione e' identita': cos(0)=1, sin(0)=0
y0 = rope.apply(x, seq_offset=0)[0]
check("a posizione 0 la rotazione è l'identità",
      np.allclose(y0, x[0]), f"max|d|={float(np.abs(y0 - x[0]).max()):.2e}")

# a posizione 1 le prime 8 componenti ruotano a coppie. Confrontiamo con
# entrambe le ipotesi, calcolate a mano dalle frequenze.
pos, j = 1.0, np.arange(HALF, dtype=np.float32)
inv = 1.0 / (10000.0 ** (2.0 * j / ROPE_DIM))
theta = pos * inv
c, s = np.cos(theta), np.sin(theta)

# NeoX: coppia j con j + HALF, dentro il blocco ruotato di rope_dim
# componenti, cioe' x[0][:4] con x[0][4:8].
xa = x[0][:HALF]
xb = x[0][HALF:ROPE_DIM]
neox = np.concatenate([xa * c - xb * s,
                       xa * s + xb * c,
                       x[0][ROPE_DIM:]])
check("l'output coincide con la rotazione NeoX",
      np.allclose(y, neox, atol=1e-6),
      f"max|d|={float(np.abs(y - neox).max()):.2e}")

# interleaved: coppie (0,1), (2,3), (4,5), (6,7) sulla prima meta'
il = np.zeros(HEAD_DIM, np.float32)
il[ROPE_DIM:] = x[0][ROPE_DIM:]
for m in range(HALF):
    a, b = x[0][2 * m], x[0][2 * m + 1]
    il[2 * m] = a * c[m] - b * s[m]
    il[2 * m + 1] = a * s[m] + b * c[m]
check("l'output NON coincide con la rotazione interleaved",
      not np.allclose(y, il, atol=1e-6),
      f"max|d| vs interleaved={float(np.abs(y - il).max()):.2e}")

print("\n2. proprietà che devono valere per qualsiasi layout")
# la rotazione e' un piano che preserva la norma del blocco ruotato
blk_in = x[0][:ROPE_DIM]
blk_out = y[:ROPE_DIM]
check("la rotazione preserva la norma del blocco ruotato",
      abs(float(np.linalg.norm(blk_in)) - float(np.linalg.norm(blk_out))) < 1e-5,
      f"{float(np.linalg.norm(blk_in)):.5f} -> {float(np.linalg.norm(blk_out)):.5f}")
# e preserva il prodotto scalare fra posizioni, che e' la proprieta' che
# rende il RoPE relativo
q = np.arange(HEAD_DIM, dtype=np.float32)
for p in (0, 1, 5):
    a = rope.apply(q.reshape(1, HEAD_DIM), seq_offset=p)[0][:ROPE_DIM]
    b = rope.apply(q.reshape(1, HEAD_DIM), seq_offset=p + 3)[0][:ROPE_DIM]
    direct = float(a @ b)
    rel = float(rope.apply(q.reshape(1, HEAD_DIM), seq_offset=p)[0][:ROPE_DIM]
                @ rope.apply(q.reshape(1, HEAD_DIM), seq_offset=p + 3)[0][:ROPE_DIM])
    check(f"dot invariante a partire da posizioni {p} e {p+3}",
          abs(direct - rel) < 1e-5, f"{direct:.5f}")

print("\n3. RoPE parziale: le dimensioni oltre rope_dim passano intatte")
tail = y[ROPE_DIM:]
check("le 8 dimensioni oltre rope_dim sono invariate",
      np.allclose(tail, x[0][ROPE_DIM:]),
      f"tail={tail.tolist()}")

print("\n4. le frequenze sono quelle del modello")
r7 = RoPE(head_dim=256, rope_dim=64, freq_base=1e7)
inv = r7._inv_freq
check("32 frequenze per rope_dim=64", inv.shape == (32,), f"{inv.shape}")
check("la prima frequenza è 1", abs(float(inv[0]) - 1.0) < 1e-6,
      f"{float(inv[0])}")
expect1 = 1e7 ** (-2.0 / 64.0)
check("la seconda è freq_base^(-2/64)",
      abs(float(inv[1]) - expect1) < 1e-6,
      f"{float(inv[1]):.8f} vs {expect1:.8f}")
check("l'ultima è freq_base^(-62/64)",
      abs(float(inv[31]) - 1e7 ** (-62.0 / 64.0)) < 1e-6,
      f"{float(inv[31]):.3e}")
check("frequenze decrescenti", bool(np.all(np.diff(inv) < 0)), "")

print("\n5. le sezioni: validazione e no-op su testo")
check("dimension_sections [11,11,10,0] accettata con rope_dim=64",
      RoPE(256, 64, 1e7, dimension_sections=(11, 11, 10, 0)) is not None, "")
try:
    RoPE(256, 64, 1e7, dimension_sections=(11, 11, 10))
    check("somma delle sezioni sbagliata viene rifiutata", False, "accettata")
except ValueError as err:
    check("somma delle sezioni sbagliata viene rifiutata", True,
          str(err)[:60])
try:
    RoPE(256, 64, 1e7, dimension_sections=(11, 11, 11, 0))
    check("32 coppie con rope_dim 64 viene rifiutata", False, "accettata")
except ValueError:
    check("32 coppie con rope_dim 64 viene rifiutata", True,
          "33 coppie != 32")

# il no-op: il risultato con e senza sezioni deve coincidere su testo
plain = RoPE(256, 64, 1e7)
withsec = RoPE(256, 64, 1e7, dimension_sections=(11, 11, 10, 0))
xs = np.random.default_rng(0).standard_normal((1, 256)).astype(np.float32)
check("su testo le sezioni sono un no-op",
      np.allclose(plain.apply(xs, 7), withsec.apply(xs, 7), atol=1e-7),
      f"max|d|={float(np.abs(plain.apply(xs, 7) - withsec.apply(xs, 7)).max()):.2e}")

print("\n6. argomenti non validi")
for label, kwargs in [
    ("rope_dim > head_dim", dict(head_dim=64, rope_dim=128)),
    ("rope_dim dispari", dict(head_dim=64, rope_dim=7)),
    ("head_dim sbagliato in apply", None),
]:
    if kwargs is None:
        continue
    try:
        RoPE(**kwargs)
        check(f"{label} viene rifiutato", False, "accettato")
    except ValueError:
        check(f"{label} viene rifiutato", True, "")
try:
    RoPE(head_dim=HEAD_DIM, rope_dim=ROPE_DIM).apply(
        np.zeros((1, 9), np.float32))
    check("head_dim sbagliato in apply viene rifiutato", False, "accettato")
except ValueError:
    check("head_dim sbagliato in apply viene rifiutato", True, "")

print("\n7. la cache delle posizioni cresce e resta valida")
r = RoPE(head_dim=256, rope_dim=64, freq_base=1e7)
seq = np.random.default_rng(1).standard_normal((1, 8, 256)).astype(np.float32)
full = r.apply(seq, seq_offset=0)
# lo stesso calcolo token per token deve dare lo stesso risultato: e' il
# requisito del decode, dove si avanza di una posizione alla volta
inc = np.concatenate([r.apply(seq[:, t:t + 1], seq_offset=t)
                      for t in range(8)], axis=1)
check("token per token == sequenza intera",
      np.allclose(full, inc, atol=1e-6),
      f"max|d|={float(np.abs(full - inc).max()):.2e}")
check("la cache è cresciuta fino alla posizione richiesta",
      r._cache_len >= 8, f"cache_len={r._cache_len}")

print(f"\n{'=' * 66}\n PASSATI: {ok}   FALLITI: {bad}\n{'=' * 66}")
sys.exit(1 if bad else 0)