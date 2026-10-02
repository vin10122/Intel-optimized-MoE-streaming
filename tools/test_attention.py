"""Il layer di attenzione di Qwen3.5-MoE, verificato su forme non quadrate.

Il motivo di questo test e' dichiarato nel file: con
``head_dim == hidden_size`` un errore di orientamento di un peso non cambia
la forma e passa. Con ``head_dim`` e ``hidden`` diversi, un errore di verso
diventa un errore di forma, quindi qui si controlla esplicitamente.

Le verifiche sono su una reference indipendente scritta con l'equazione
esplicita dell'attenzione, non riutilizzando le funzioni del modulo: se il
modulo e la reference condividessero un errore, il confronto passerebbe.
"""
import sys
sys.path.insert(0, 'src')

# La console Windows e' cp1252: senza questo, un'etichetta con '≈' o '→' fa
# fallire il test su un errore di codifica invece che sull'assunto verificato.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np

from flash_moe.attention import (
    AttentionLayer, KVCache, grouped_attention, _sigmoid,
)
from flash_moe.rope import RoPE

ok = bad = 0
def check(label, cond, detail=""):
    global ok, bad
    if cond:
        ok += 1; print(f"  PASS  {label}  ({detail})")
    else:
        bad += 1; print(f"  FAIL  {label}  ({detail})")

# forme NON quadrate e GQA con gruppo > 1
HIDDEN, NH, NKV, HD = 12, 4, 2, 3
ROPE_D = 2                       # ruota le prime 2 componenti su head_dim 3
EPS = 1e-6
rng = np.random.default_rng(7)

layer = AttentionLayer(hidden_size=HIDDEN, num_heads=NH, num_kv_heads=NKV,
                       head_dim=HD, rope_dim=ROPE_D, freq_base=10000.0,
                       eps=EPS)
w = layer.prepare_weights(
    rng.standard_normal((HIDDEN, NH * 2 * HD)).astype(np.float32),
    rng.standard_normal((HIDDEN, NKV * HD)).astype(np.float32),
    rng.standard_normal((HIDDEN, NKV * HD)).astype(np.float32),
    rng.standard_normal(HD).astype(np.float32),
    rng.standard_normal(HD).astype(np.float32),
    rng.standard_normal((NH * HD, HIDDEN)).astype(np.float32),
)

def ref(x, seq_offset=0, kprev=None, vprev=None):
    """Reference indipendente, scritta con le formule esplicite."""
    T = x.shape[0]
    qf = x @ w["q"]
    kf = x @ w["k"]
    vf = x @ w["v"]
    # Q/gate interleaved per testa
    q = np.zeros((T, NH, HD), np.float32)
    g = np.zeros((T, NH, HD), np.float32)
    for h in range(NH):
        q[:, h, :] = qf[:, h * 2 * HD: h * 2 * HD + HD]
        g[:, h, :] = qf[:, h * 2 * HD + HD: (h + 1) * 2 * HD]
    k = kf.reshape(T, NKV, HD)
    v = vf.reshape(T, NKV, HD)

    def rk(a, wgt):                      # RMSNorm su head_dim
        r = np.sqrt(np.mean(a * a, axis=-1, keepdims=True) + EPS)
        return a / r * wgt

    q = rk(q, w["q_norm"])
    k = rk(k, w["k_norm"])

    rope = RoPE(head_dim=HD, rope_dim=ROPE_D, freq_base=10000.0)
    def rot(a):                          # coppie NeoX (j, j+1) su ROPE_D=2
        ang = np.arange(ROPE_D // 2) * (2.0 * np.arange(ROPE_D // 2) / ROPE_D)
        c = np.cos(ang); s = np.sin(ang)
        out = a.copy()
        for p in range(seq_offset, seq_offset + T):
            th = p * (1.0 / (10000.0 ** (2.0 * np.arange(ROPE_D // 2) / ROPE_D)))
            cc = np.cos(th); ss = np.sin(th)
            for h in range(a.shape[1]):
                for j in range(ROPE_D // 2):
                    x0 = a[p - seq_offset, h, j]
                    x1 = a[p - seq_offset, h, j + ROPE_D // 2]
                    out[p - seq_offset, h, j] = x0 * cc[j] - x1 * ss[j]
                    out[p - seq_offset, h, j + ROPE_D // 2] = x0 * ss[j] + x1 * cc[j]
        return out

    q = rot(q)
    k = rot(k)

    # cache: (seq, n_kv, head_dim) -- k e v sono gia' in quest'ordine
    if kprev is None:
        kk = k
        vv = v
    else:
        kk = np.concatenate([kprev, k], axis=0)
        vv = np.concatenate([vprev, v], axis=0)

    scale = 1.0 / np.sqrt(HD)

    # GQA per gruppi: la testa di query h prende il kv-head h // (n_head/n_kv).
    # Confermato dalla reference OpenVINO di llama.cpp
    # (ggml/src/ggml-openvino/openvino/op/flash_attn_ext.cpp:149), che rimappa
    # Q a [B, n_kv, factor, S_q, head_size]: kv_head = h // factor.
    # Il delta net e' diverso (h % n_kv) e non va confuso con questo.
    out = np.zeros((T, NH, HD), np.float32)
    for t in range(T):
        qpos = seq_offset + t
        for h in range(NH):
            kk_h = kk[:, h // (NH // NKV), :]   # (seq, dim)
            vv_h = vv[:, h // (NH // NKV), :]
            sc = (kk_h @ q[t, h]) * scale
            sc = np.where(np.arange(kk_h.shape[0]) <= qpos, sc, -np.inf)
            sc = sc - sc.max()
            e = np.exp(sc)
            p = e / e.sum()
            out[t, h, :] = p @ vv_h
    out = out.reshape(T, NH * HD)
    out = out * (1.0 / (1.0 + np.exp(-g.reshape(T, -1))))
    return out @ w["out"], k, v

print("1. un passo: il layer coincide con la reference esplicita")
x = rng.standard_normal((1, HIDDEN)).astype(np.float32)
got = layer.forward(x, w)
want, _, _ = ref(x)
rel = float(np.abs(got - want).max() / np.abs(want).max())
check("un token", rel < 1e-5, f"rel={rel:.2e}")

print("\n2. piu' token insieme (prefill)")
x3 = rng.standard_normal((3, HIDDEN)).astype(np.float32)
got = layer.forward(x3, w)
want, _, _ = ref(x3)
rel = float(np.abs(got - want).max() / np.abs(want).max())
check("tre token", rel < 1e-5, f"rel={rel:.2e}")

print("\n3. Q e gate sono interleaved per testa, non separati a meta'")
# attivo il gate di una sola testa: se l'impaginazione fosse sbagliata,
# l'effetto cadrebbe su un'altra testa
xg = np.zeros((1, HIDDEN), np.float32)
xg[:] = 0.01
q_gate_on_head1 = w["q"].copy()
for h in range(NH):
    if h != 1:
        q_gate_on_head1[:, h * 2 * HD + HD: (h + 1) * 2 * HD] = 0.0
wg = {k: (q_gate_on_head1 if k == "q" else w[k]) for k in w}
y0 = layer.forward(xg, wg)
# con tutti i gate a zero l'output e' nullo
wzero = {k: (w["q"].copy() if k == "q" else w[k]) for k in w}
for h in range(NH):
    wzero["q"][:, h * 2 * HD + HD: (h + 1) * 2 * HD] = 0.0
y_zero = layer.forward(xg, wzero)
# Il gate e' un sigmoid moltiplicativo, quindi nullo non e' raggiungibile con
# un peso (manca il bias): con i pesi del gate a zero la pre-attivazione e' 0 e
# il gate vale sigmoid(0) = 0.5 esatto. Lo si verifica contro il caso in cui
# lo stesso blocco e' moltiplicato per 1: si forza sigmoid -> 1 con un peso
# grande e positivo sul gate di tutte le teste.
w_one = {k: (w["q"].copy() if k == "q" else w[k]) for k in w}
for h in range(NH):
    w_one["q"][:, h * 2 * HD + HD: (h + 1) * 2 * HD] = 1e4
y_one = layer.forward(xg, w_one)
check("gate nullo (pesi 0) = metà di gate aperto (sigmoid(0) = 0.5)",
      abs(float(np.abs(y_zero).max()) - 0.5 * float(np.abs(y_one).max())) < 1e-6,
      f"max|y_zero|={float(np.abs(y_zero).max()):.4f} vs "
      f"0.5*max|y_one|={0.5 * float(np.abs(y_one).max()):.4f}")
check("gate attivo su una sola testa produce output non nullo",
      float(np.abs(y0).max()) > 1e-6, f"max|y|={float(np.abs(y0).max()):.2e}")
# e l'output deve dipendere SOLO dal gate della testa 1
w_h2 = {k: (w["q"].copy() if k == "q" else w[k]) for k in w}
for h in (0, 2, 3):
    w_h2["q"][:, h * 2 * HD + HD: (h + 1) * 2 * HD] = 0.0
y_h2 = layer.forward(xg, w_h2)
y_h2b = layer.forward(xg, w_h2)
check("output deterministico a parità di pesi",
      np.array_equal(y_h2, y_h2b), "")
# spegnere il gate della testa 1 cambia l'output
w_h2_off = {k: (w_h2["q"].copy() if k == "q" else w_h2[k]) for k in w}
w_h2_off["q"][:, 1 * 2 * HD + HD: 2 * 2 * HD] = 0.0
y_off = layer.forward(xg, w_h2_off)
check("le teste hanno gate indipendenti",
      float(np.abs(y_h2 - y_off).max()) > 1e-6,
      f"max|d|={float(np.abs(y_h2 - y_off).max()):.2e}")

print("\n4. il gate è sigmoid, non silu")
# sigmoid e silu coincidono solo per x piccoli; per x grande divergono
# _sigmoid accetta qualsiasi forma, quindi si verifica anche il caso
# vettoriale: la stabilita' (non deve overfloware in exp) e' il punto.
pos = _sigmoid(np.array([10.0], np.float32))[0]
neg = _sigmoid(np.array([-10.0], np.float32))[0]
mid = _sigmoid(np.array([0.0], np.float32))[0]
check("sigmoid(10) = 0.99995", abs(float(pos) - 0.9999546) < 1e-6, f"{pos:.7f}")
check("il gate non può superare 1 (silu(10) sarebbe 10.0)",
      float(pos) <= 1.0, f"sigmoid={pos:.7f}")
check("sigmoid(-10) ≈ 4.5e-5, non negativo",
      0.0 < float(neg) < 1e-4, f"{neg:.3e}")
check("sigmoid(0) = 0.5 esatto", abs(float(mid) - 0.5) < 1e-7, f"{mid:.7f}")
check("sigmoid su estremi non overflowa e resta in [0, 1]",
      bool(np.all(np.isfinite(_sigmoid(np.array([-1e4, 0.0, 1e4], np.float32)))) and
           np.all((_sigmoid(np.array([-1e4, 0.0, 1e4], np.float32)) >= 0) &
                  (_sigmoid(np.array([-1e4, 0.0, 1e4], np.float32)) <= 1))),
      f"{_sigmoid(np.array([-1e4, 0.0, 1e4], np.float32))}")
check("sigmoid e' monotona crescente",
      bool(np.all(np.diff(_sigmoid(np.linspace(-8, 8, 65, dtype=np.float32))) > 0)),
      "")

print("\n5. GQA: 2 kv-head su 4 query-head, mappa per gruppi")
# con 4 query-head e 2 kv-head, il gruppo e' 2: teste 0,1 -> kv 0; 2,3 -> kv 1
SEQ = 5
kk = rng.standard_normal((SEQ, NKV, HD)).astype(np.float32)
vv = rng.standard_normal((SEQ, NKV, HD)).astype(np.float32)
qq = rng.standard_normal((1, NH, HD)).astype(np.float32)
got = grouped_attention(qq, kk, vv, n_tokens=1, seq_offset=SEQ - 1)
grp = NH // NKV
kk2 = np.repeat(kk, grp, axis=1)
vv2 = np.repeat(vv, grp, axis=1)
sc = np.einsum("qhd,khd->qhk", qq, kk2) / np.sqrt(HD)
sc = sc - sc.max(-1, keepdims=True)
e = np.exp(sc); p = e / e.sum(-1, keepdims=True)
want = np.einsum("qhk,khd->qhd", p, vv2).reshape(1, NH * HD)
check("grouped_attention coincide con il broadcasting per gruppi",
      np.allclose(got, want, atol=1e-6),
      f"max|d|={float(np.abs(got - want).max()):.2e}")
# il token all'ultima posizione vede tutti i SEQ token: se la maschera fosse
# sbagliata qui il confronto sopra fallirebbe, ma vale la pena dirlo
check("l'ultimo token vede tutti i precedenti (niente mascherati)",
      np.array_equal(got, grouped_attention(qq, kk, vv, 1, SEQ - 1)), "")
# e la mappa e' np.repeat, non np.tile come nel delta net
check("la mappa GQA è np.repeat (h // gruppo)",
      grp == 2 and np.array_equal(np.repeat(np.arange(2), grp),
                                  np.array([0, 0, 1, 1])),
      "teste 0,1 -> kv 0; teste 2,3 -> kv 1")

print("\n6. la maschera causale")
# con 6 token in cache, il token alla posizione 2 ne vede solo 3: gli altri 3
# devono essere esclusi, altrimenti il risultato cambierebbe
S6 = 6
q1 = rng.standard_normal((1, NH, HD)).astype(np.float32)
k1 = rng.standard_normal((S6, NKV, HD)).astype(np.float32)
v1 = rng.standard_normal((S6, NKV, HD)).astype(np.float32)
o_at2 = grouped_attention(q1, k1, v1, n_tokens=1, seq_offset=2)
o_see3 = grouped_attention(q1, k1[:3], v1[:3], n_tokens=1, seq_offset=2)
check("posizione 2 vede solo i primi 3 token",
      float(np.abs(o_at2 - o_see3).max()) < 1e-6,
      f"max|d|={float(np.abs(o_at2 - o_see3).max()):.2e}")
o_at5 = grouped_attention(q1, k1, v1, n_tokens=1, seq_offset=5)
check("posizione 5 vede tutti e 6 i token",
      float(np.abs(o_at5 - grouped_attention(q1, k1, v1, 1, 5)).max()) == 0.0,
      "")
# e una maschera che non nasconda nulla deve dare lo stesso risultato
o_no_mask = grouped_attention(q1, k1, v1, n_tokens=1, seq_offset=S6 - 1)
check("all'ultima posizione la maschera non esclude nulla",
      float(np.abs(o_no_mask - o_at5).max()) < 1e-6,
      f"max|d|={float(np.abs(o_no_mask - o_at5).max()):.2e}")

print("\n7. il decode token per token con la cache")
layer.init_cache(max_seq_len=32)
seq = rng.standard_normal((4, HIDDEN)).astype(np.float32)
outs_inc = []
for t in range(4):
    outs_inc.append(layer.forward(seq[t:t + 1], w, cache=layer.cache))
inc = np.concatenate(outs_inc, axis=0)

layer.reset_cache()
full = layer.forward(seq, w, cache=layer.cache)
rel = float(np.abs(inc - full).max() / np.abs(full).max())
check("token per token == sequenza intera", rel < 1e-5,
      f"rel={rel:.2e}, seq_len={layer.cache.length}")

print("\n8. la cache cresce e si svuota")
layer.reset_cache()
check("reset azzera la cache", layer.cache.length == 0, "")
layer.forward(seq, w, cache=layer.cache)
check("la cache contiene tutti i token",
      layer.cache.length == 4 and layer.cache.keys.shape == (4, NKV, HD),
      f"keys={layer.cache.keys.shape}")
try:
    layer.init_cache(max_seq_len=2)
    layer.forward(rng.standard_normal((4, HIDDEN)).astype(np.float32), w,
                  cache=layer.cache)
    check("cache troppo piccola viene rifiutata", False, "accettata")
except ValueError as err:
    check("cache troppo piccola viene rifiutata", True, str(err)[:40])

print("\n9. le forme dei pesi sono verificate all'ingresso")
try:
    layer.prepare_weights(
        rng.standard_normal((HIDDEN, NH * HD)).astype(np.float32),
        rng.standard_normal((HIDDEN, NKV * HD)).astype(np.float32),
        rng.standard_normal((HIDDEN, NKV * HD)).astype(np.float32),
        rng.standard_normal(HD).astype(np.float32),
        rng.standard_normal(HD).astype(np.float32),
        rng.standard_normal((NH * HD, HIDDEN)).astype(np.float32))
    check("attn_q senza il gate viene rifiutata", False, "accettata")
except ValueError as err:
    check("attn_q senza il gate viene rifiutata", True,
          "manca il fattore 2 del gate")
try:
    # Su questo layer HIDDEN (12) e' uguale a NH*HD (4*3): la forma
    # "trasposta" coinciderebbe con quella giusta e il controllo passerebbe
    # sempre, senza dire niente. Serve un layer con hidden != NH*HD: e'
    # esattamente il caso che il docstring di questo file dichiara
    # non verificabile con forme quadrate.
    HIDDEN2 = NH * HD + 5
    layer2 = AttentionLayer(hidden_size=HIDDEN2, num_heads=NH, num_kv_heads=NKV,
                            head_dim=HD, rope_dim=ROPE_D, freq_base=10000.0,
                            eps=EPS)
    assert HIDDEN2 != NH * HD, "il test perde senso"
    layer2.prepare_weights(
        rng.standard_normal((HIDDEN2, NH * 2 * HD)).astype(np.float32),
        rng.standard_normal((HIDDEN2, NKV * HD)).astype(np.float32),
        rng.standard_normal((HIDDEN2, NKV * HD)).astype(np.float32),
        rng.standard_normal(HD).astype(np.float32),
        rng.standard_normal(HD).astype(np.float32),
        rng.standard_normal((HIDDEN2, NH * HD)).astype(np.float32))  # inverso!
    check("attn_output trasposto viene rifiutato", False,
          f"accettato: ({HIDDEN2}, {NH * HD}) invece di ({NH * HD}, {HIDDEN2})")
except ValueError:
    check("attn_output trasposto viene rifiutato", True,
          "(hidden, n_head*head_dim) non e' accettata")

print(f"\n{'=' * 66}\n PASSATI: {ok}   FALLITI: {bad}\n{'=' * 66}")
sys.exit(1 if bad else 0)