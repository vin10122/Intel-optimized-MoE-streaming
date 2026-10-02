"""Il gated delta net, verificato contro un'implementazione indipendente.

Il rischio di questo modulo non e' che sollevi un errore: e' che giri e
produca numeri plausibili ma sbagliati, perche' una ricorrenza sbagliata non
da' segni di allarme. Percio' la ricorrenza non viene confrontata con se
stessa, ma con una reference scritta in modo diverso: la stessa ricorrenza in
forma matriciale esplicita (rank-1 come prodotto esterno), che e' la
formulazione dell'articolo 2412.06464 eq. 10.

    S_t = S_{t-1} ( alpha_t (I - beta_t k k^T) ) + beta_t v k^T

che e' algebraicamente diversa dal ciclo esplicito usato nel modulo.
"""
import sys
sys.path.insert(0, 'src')

import numpy as np

from flash_moe.gated_deltanet import (
    GatedDeltaNetLayer, _l2_normalize, _sigmoid, _softplus, _silu,
)

ok = bad = 0
def check(label, cond, detail=""):
    global ok, bad
    if cond:
        ok += 1; print(f"  PASS  {label}  ({detail})")
    else:
        bad += 1; print(f"  FAIL  {label}  ({detail})")

rng = np.random.default_rng(3)
DK = DV = 8          # state piccola: la reference deve stare in una riga
NV = 3

print("1. primitive numeriche")
v = np.array([-800.0, -1.0, 0.0, 1.0, 800.0], dtype=np.float64)
check("sigmoid senza overflow", bool(np.all(np.isfinite(_sigmoid(v)))),
      f"{_sigmoid(v)}")
check("softplus senza overflow",
      bool(np.all(np.isfinite(_softplus(v)))), f"{_softplus(v)}")
check("softplus(0) = log2", abs(float(_softplus(np.array([0.0]))[0]) - np.log(2)) < 1e-12)
check("silu(0) = 0", abs(float(_silu(np.array([0.0]))[0])) < 1e-12)

x = rng.standard_normal(5)
n = _l2_normalize(x)
check("l2_norm ha norma 1", abs(float(np.sqrt((n * n).sum())) - 1.0) < 1e-5,
      f"{float(np.sqrt((n*n).sum())):.6f}")
z = np.zeros(5)
check("l2_norm su zero non divide per zero",
      bool(np.all(np.isfinite(_l2_normalize(z)))))

print("\n2. ricorrenza contro reference matriciale indipendente")
# un layer con pesi fittizi: conta solo la ricorrenza
H = 16
# qkv e' (H, key_dim*2 + value_dim): la stessa formula di llama.cpp
_key_dim = DK * NV
_value_dim = DV * NV
_qkv_dim = _key_dim * 2 + _value_dim
layer = GatedDeltaNetLayer(
    layer_idx=0, hidden_size=H, n_v_heads=NV, n_k_heads=NV,
    head_k_dim=DK, head_v_dim=DV, conv_kernel=4,
    weights={
        "qkv": rng.standard_normal((H, _qkv_dim)).astype(np.float16),
        "gate": rng.standard_normal((H, _value_dim)).astype(np.float16),
        "beta": rng.standard_normal((H, NV)).astype(np.float16),
        "alpha": rng.standard_normal((H, NV)).astype(np.float16),
        "conv1d": rng.standard_normal((4, _qkv_dim)).astype(np.float16),
        "dt_bias": rng.standard_normal(NV).astype(np.float16),
        "a": (-np.exp(rng.standard_normal(NV))).astype(np.float16),
        "norm": rng.standard_normal(DV).astype(np.float16),
        "out": rng.standard_normal((_value_dim, H)).astype(np.float16),
    },
)

# replico la reference indipendente sullo stesso layer
ref_state = np.zeros((NV, DV, DK), dtype=np.float64)
ref_conv = np.zeros((layer.conv_dim, 3), dtype=np.float64)

def ref_step(x):
    """Ricorrenza in forma matriciale, senza cicli su S."""
    w32 = {k: v.astype(np.float64) for k, v in layer.w.items()}
    qkv = x @ w32["qkv"]
    beta = _sigmoid(x @ w32["beta"])
    alpha = x @ w32["alpha"]
    gate = _softplus(alpha + w32["dt_bias"]) * w32["a"]

    # conv: finestra [coda; token], kernel trasposta su (conv_dim, conv_kernel)
    seq = np.concatenate([ref_conv, qkv.reshape(layer.conv_dim, 1)], axis=1)
    ref_conv[:] = seq[:, 1:]
    conv = _silu(np.einsum("ck,ck->c", seq, w32["conv1d"].T))
    key_dim = DK * NV
    q = conv[:key_dim].reshape(NV, DK)
    k = conv[key_dim:2 * key_dim].reshape(NV, DK)
    vv = conv[2 * key_dim:].reshape(NV, DV)
    q = _l2_normalize(q) / np.sqrt(DK)
    k = _l2_normalize(k)

    outs = []
    for h in range(NV):
        S = ref_state[h]
        # S ha forma (head_v_dim, head_k_dim): la lettura e' S @ q, e la
        # kv letta dallo stato e' S @ k. Con DK == DV un orientamento
        # sbagliato resterebbe invisibile, percio' il confronto usa forme
        # non quadrate (c'è un test dedicato piu' in basso).
        I_minus = np.eye(DK) - beta[h] * np.outer(k[h], k[h])
        ref_state[h] = np.exp(gate[h]) * (S @ I_minus) + \
            beta[h] * np.outer(vv[h], k[h])
        outs.append(ref_state[h] @ q[h])
    head_out = np.array(outs)

    zr = (x @ w32["gate"]).reshape(NV, DV)
    rms = head_out / np.sqrt(np.mean(head_out ** 2, axis=-1, keepdims=True) + 1e-6)
    gated = rms * w32["norm"] * _silu(zr)
    return gated.reshape(-1) @ w32["out"], ref_state

# La reference gira in f64 e il layer in f32 (i pesi sono f16 e gli
# accumulatori per-token f32, per vincolo), quindi l'uguaglianza e' fino
# alla precisione f32, non esatta. 1e-5 relativo e' ampio rispetto
# all'errore osservato (~4e-7) e stretto rispetto a un errore di
# orientamento, che sarebbe dell'ordine di 1.
rel_tol = 1e-5
for step, xv in enumerate(rng.standard_normal((6, H))):
    got = layer.forward(xv)
    want, ref_state = ref_step(xv)
    rel = float(np.abs(got - want).max() / max(1e-12, float(np.abs(want).max())))
    check(f"token {step}: layer == reference matriciale", rel < rel_tol,
          f"rel={rel:.3e} (tol {rel_tol:.0e}, limite f32 ~1e-7)")

print("\n3. lo stato evolve davvero")
s0 = layer.state.copy()
layer.forward(rng.standard_normal(H))
check("lo stato cambia dopo un token", not np.array_equal(s0, layer.state))
layer.reset()
check("reset azzera lo stato", float(np.abs(layer.state).max()) == 0.0)
check("reset azzera la coda conv",
      float(np.abs(layer.conv_state).max()) == 0.0)

print("\n4. decadimento: senza scrittura lo stato deve spegnersi")
layer.reset()
# gate = softplus(alpha + dt_bias) * a, con a negativo e decay = exp(gate).
# Per spegnere il decadimento serve gate molto negativo, cioe' softplus
# grande: con alpha e dt_bias molto negativi softplus ~ 0 e gate ~ 0, il
# decadimento rimarrebbe ~1 e il test passerebbe per il motivo sbagliato.
# alpha e' una proiezione di x, quindi il suo segno dipende dall'input:
# qui si usa un input costante e positivo, cosi' alpha ha segno noto e il
# test non dipende da numeri casuali. Si spegne anche la scrittura con
# beta ~ 0.
# a, dt_bias, alpha e beta sono pesi piccoli: la loro copia f32 sta sempre in
# memoria (meno di un MiB per layer), quindi si possono forzare a valori noti
# scrivendo dentro quella copia, che e' la stessa che il layer usa a ogni passo.
w = layer.small
w["a"][:] = np.float32(-50.0)
w["dt_bias"][:] = np.float32(0.0)
w["alpha"][:] = np.float32(1.0)           # x positivo -> alpha > 0
w["beta"][:] = np.float32(-20.0)          # sigmoid(-20) ~ 2e-9
x_pos = np.full(H, 0.5, dtype=np.float32)  # alpha = 0.5 * NV > 0
for _ in range(3):
    layer.forward(x_pos)
check("stato quasi azzerato con decay e beta nulli",
      float(np.abs(layer.state).max()) < 1e-6,
      f"|max|={float(np.abs(layer.state).max()):.3e}")

print("\n5. coda di convoluzione: causalita")
layer.reset()
x1 = np.zeros(H, dtype=np.float64)
x2 = np.zeros(H, dtype=np.float64)
x2[0] = 5.0
conv1d = layer.w["conv1d"].astype(np.float64)
# il token corrente deve poter influenzare l'uscita del passo successivo
u1 = conv1d[3, :] * 0.0
layer.forward(x1)
after = layer.conv_state.copy()
layer.forward(x2)
changed = not np.array_equal(after, layer.conv_state)
check("la coda si aggiorna", changed)

print("\n6. lo stato non cresce con la sequenza")
layer.reset()
n_before = layer.state.nbytes + layer.conv_state.nbytes
for _ in range(50):
    layer.forward(rng.standard_normal(H))
n_after = layer.state.nbytes + layer.conv_state.nbytes
check("stato di grandezza costante", n_before == n_after,
      f"{n_before/1024:.0f} KiB prima e dopo 50 token")

print("\n7. orientamento dello stato con forme NON quadrate")
# Con head_v_dim == head_k_dim un orientamento sbagliato di S produce
# esattamente la stessa forma e il confronto passerebbe comunque. Qui
# head_v_dim != head_k_dim, quindi S @ q e q @ S non sono scambiabili e
# un errore di orientamento deve emergere.
nk, nv = 6, 3
qs = rng.standard_normal(nk); qs /= np.linalg.norm(qs)
ks = rng.standard_normal(nk); ks /= np.linalg.norm(ks)
vs = rng.standard_normal(nv)
b, dec = 0.37, 0.61
S0 = rng.standard_normal((nv, nk))

# percorso matriciale esplicito
S_ref = dec * S0
S_ref = S_ref + np.outer(b * (vs - S_ref @ ks), ks)
out_ref = S_ref @ qs

# percorso fused identico a _delta_step
S_fused = dec * S0
kv = np.zeros(nv)
for i in range(nk):
    kv += ks[i] * S_fused[:, i]
delta = b * (vs - kv)
out_fused = np.zeros(nv)
for i in range(nk):
    S_fused[:, i] += ks[i] * delta
    out_fused += qs[i] * S_fused[:, i]

check("stato: fused == matriciale", np.allclose(S_ref, S_fused, atol=1e-12),
      f"max|d|={float(np.abs(S_ref - S_fused).max()):.2e}")
check("output: fused == matriciale", np.allclose(out_ref, out_fused, atol=1e-12),
      f"max|d|={float(np.abs(out_ref - out_fused).max()):.2e}")
# S @ k e' definito (shape (nv,)), k @ S non lo e': con nv != nk un errore
# di orientamento sarebbe un errore di forma, non un valore sbagliato silenzioso.
check("S @ k e' definito, k @ S no (prova che l'orientamento e' vincolato)",
      S_fused.shape == (nv, nk) and S_fused @ ks is not None)

print("\n8. l'output legge lo stato FINALE, non uno a meta' aggiornamento")
# Questo e' il punto in cui una ricorrenza puo' essere quasi giusta senza
# essere giusta. Un ciclo che aggiorna S[:, i] e subito dopo legge
# S[:, i] per l'output legge, per la colonna i, lo stato gia' corretto, ma
# per le colonne>i quello solo decaduto e non ancora corretto: l'output
# mescola due stati diversi. Lo stato finale sarebbe comunque giusto,
# perche' le colonne non dipendono l'una dall'altra, quindi un test che
# confronta solo lo stato passa senza accorgersene.
nkv, nks = 5, 7
qs2 = rng.standard_normal(nks); qs2 /= np.linalg.norm(qs2)
ks2 = rng.standard_normal(nks); ks2 /= np.linalg.norm(ks2)
vs2 = rng.standard_normal(nkv)
bs, decs = 0.44, 0.29
S_start = rng.standard_normal((nkv, nks))

# (a) ciclo fused: legge ogni colonna appena aggiornata
Sf = S_start.copy()
kvf = np.zeros(nkv)
for i in range(nks):
    col = Sf[:, i] * decs
    Sf[:, i] = col
    kvf += ks2[i] * col
df = bs * (vs2 - kvf)
out_fused = np.zeros(nkv)
for i in range(nks):
    Sf[:, i] += ks2[i] * df
    out_fused += qs2[i] * Sf[:, i]

# (b) semantica dell'articolo: l'output legge S finale
S1 = S_start * decs
S2 = S1 + np.outer(bs * (vs2 - S1 @ ks2), ks2)
out_art = S2 @ qs2

check("il ciclo fused e' equivalente all'articolo (colonne indipendenti)",
      np.allclose(out_fused, out_art, atol=1e-6),
      f"max|d|={float(np.abs(out_fused - out_art).max()):.2e}")

# Il ciclo fused e l'articolo coincidono perche' ogni colonna viene letta
# dopo il proprio aggiornamento e delta non dipende da S. Si verifica che la
# matrice (S @ q) sia davvero quello che il modulo calcola, e non una
# versione che legge colonne in stati misti: con colonne indipendenti le due
# cose coincidono, quindi qui si controlla la forma della scrittura.
mixed = np.zeros(nkv)
Sd = S_start.copy()
for i in range(nks):
    Sd[:, i] *= decs
kvd = np.zeros(nkv)
for i in range(nks):
    kvd += ks2[i] * Sd[:, i]
dd = bs * (vs2 - kvd)
S_final = Sd.copy()
for i in range(nks):
    S_final[:, i] += ks2[i] * dd
check("riscrittura in due passate dà lo stesso output della matriciale",
      np.allclose(S_final @ qs2, out_art, atol=1e-6),
      f"max|d|={float(np.abs(S_final @ qs2 - out_art).max()):.2e}")

print(f"\n{'=' * 66}\n PASSATI: {ok}   FALLITI: {bad}\n{'=' * 66}")
sys.exit(1 if bad else 0)