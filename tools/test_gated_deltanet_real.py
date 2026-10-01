"""Il gated delta net con i pesi e le dimensioni reali del modello.

Il test precedente usa pesi piccoli e forme che semplificano il confronto.
Qui si carica il GGUF vero: le dimensioni sono head_v_dim = 128 e
head_k_dim = 128, quindi quadrate, ed e' proprio questa la situazione
pericolosa: un errore di orientamento di S passerebbe inosservato. Per
compensare, il confronto non usa soltanto l'uguaglianza numerica ma anche
un controllo di simmetria: S non e' simmetrica, quindi trasporla cambia i
risultati, e il test verifica che la reference trasposta dia un risultato
DIVERSO. Se le due cose coincidessero, il test non starebbe testando
l'orientamento.

Verifica anche che i pesi si carichino dai tensori giusti del GGUF e che
lo stato abbia la dimensione attesa (32 x 128 x 128 f32 = 2 MiB per layer).
"""
import sys
sys.path.insert(0, 'src')
import numpy as np

from flash_moe.gated_deltanet import (
    GatedDeltaNetLayer, is_recurrent_layer, _l2_normalize, _sigmoid,
    _softplus, _silu,
)

ok = bad = 0
def check(label, cond, detail=""):
    global ok, bad
    if cond:
        ok += 1; print(f"  PASS  {label}  ({detail})")
    else:
        bad += 1; print(f"  FAIL  {label}  ({detail})")

print("1. hyperparametri reali del modello")
import gguf
from flash_moe.paths import model_path

reader = gguf.GGUFReader(model_path())
tensors = {t.name: t for t in reader.tensors}

def meta_of(k, default=None):
    for m in reader.fields.get("qwen35moe", []):
        for f in m.data:
            if f.key == k:
                return f.parts[-1][0] if f.parts else None
    return default

CONV_K = meta_of("ssm.conv_kernel", 4)
GROUPS = meta_of("ssm.group_count", 16)
INNER = meta_of("ssm.inner_size", 4096)
STATE = meta_of("ssm.state_size", 128)
TSR = meta_of("ssm.time_step_rank", 32)
N_K = meta_of("ssm.n_k_heads", 16)
N_V = meta_of("ssm.n_v_heads", 32)
D_K = meta_of("ssm.head_k_dim", 128)
D_V = meta_of("ssm.head_v_dim", 128)
EMBD = meta_of("ssm.embedding_length", 2048)

print(f"  conv_kernel={CONV_K} group_count={GROUPS} inner={INNER} "
      f"state={STATE} ts_rank={TSR}")
print(f"  n_k_heads={N_K} n_v_heads={N_V} head_k_dim={D_K} head_v_dim={D_V} "
      f"embd={EMBD}")

check("n_v_heads multiplo di n_k_heads (replicazione delle chiavi)",
      N_V % N_K == 0, f"{N_V} / {N_K} = {N_V // N_K}")
key_dim = D_K * N_K
value_dim = D_V * N_V
conv_dim = key_dim * 2 + value_dim
print(f"  key_dim={key_dim} value_dim={value_dim} conv_dim={conv_dim}")

# 2. i tensori devono avere le dimensioni dichiarate dai metadata
print("\n2. forma dei tensori reali nel GGUF")
for name, want in [
    ("ssm_a", (TSR,)),
    ("ssm_alpha.weight", (EMBD, TSR)),
    ("ssm_beta.weight", (EMBD, TSR)),
    ("ssm_conv1d.weight", (CONV_K, conv_dim)),
    ("ssm_dt.bias", (TSR,)),
    ("ssm_norm.weight", (D_K,)),
    ("ssm_out.weight", (value_dim, EMBD)),
    ("attn_qkv.weight", (EMBD, conv_dim)),
    ("attn_gate.weight", (EMBD, value_dim)),
]:
    t = tensors[f"blk.0.{name}"]
    got = tuple(int(x) for x in t.shape)
    check(f"{name} {got}", got == want, f"atteso {want}")

# 3. il layer reale
print("\n3. layer reale: stato e coda conv")
from run_q4k_f16_openvino import dequant_to_f16

GGML_F32 = 0
GGML_F16 = 1

def load_f16(name, shape):
    """Porta un tensor del blocco 0 in f16 con la forma richiesta.

    I tensori F32/F16 sono gia' non quantizzati: niente da dequantizzare,
    basta reinterpretare i byte e convertire. Per quelli quantizzati si usa
    dequant_to_f16, che usa il kernel C++ quando e' disponibile. Il numero di
    elementi e' quello dichiarato dalla forma, non len(t.data): per un F32
    i byte sono 4 per elemento e reshape sul conteggio dei byte sbaglierebbe
    per un fattore 4.
    """
    t = tensors[f"blk.0.{name}"]
    want = int(np.prod(shape))
    if t.tensor_type == GGML_F32:
        flat = np.frombuffer(t.data, dtype=np.float32)
        check(f"{name} contiene {want} elementi F32", flat.size == want,
              f"byte={len(t.data)} elem={flat.size}")
        return flat.astype(np.float16).reshape(shape)
    if t.tensor_type == GGML_F16:
        flat = np.frombuffer(t.data, dtype=np.float16)
        return flat.reshape(shape)
    return dequant_to_f16(t.data, t.tensor_type, want).reshape(shape)

w = {
    "qkv": load_f16("attn_qkv.weight", (EMBD, conv_dim)),
    "gate": load_f16("attn_gate.weight", (EMBD, value_dim)),
    "beta": load_f16("ssm_beta.weight", (EMBD, TSR)),
    "alpha": load_f16("ssm_alpha.weight", (EMBD, TSR)),
    "conv1d": load_f16("ssm_conv1d.weight", (CONV_K, conv_dim)),
    "dt_bias": load_f16("ssm_dt.bias", (TSR,)),
    "a": load_f16("ssm_a", (TSR,)),
    "norm": load_f16("ssm_norm.weight", (D_K,)),
    "out": load_f16("ssm_out.weight", (value_dim, EMBD)),
}
check("ssm_a negativity (contiene -exp(A_log))",
      bool(np.all(w["a"].astype(np.float32) < 0)),
      f"max={float(w['a'].max()):.4f}")

layer = GatedDeltaNetLayer(
    layer_idx=0, hidden_size=EMBD, n_v_heads=N_V, n_k_heads=N_K,
    head_k_dim=D_K, head_v_dim=D_V, conv_kernel=CONV_K, weights=w)

st_bytes = N_V * D_V * D_K * 4
check("stato 32x128x128 f32 = 2 MiB",
      layer.state.shape == (N_V, D_V, D_K) and st_bytes == 2 * 1024 * 1024,
      f"{tuple(layer.state.shape)} = {st_bytes/1024**2:.2f} MiB")
check("coda conv (conv_dim, conv_kernel-1)",
      layer.conv_state.shape == (conv_dim, CONV_K - 1),
      f"{tuple(layer.conv_state.shape)} = "
      f"{layer.conv_state.nbytes/1024:.0f} KiB")

# 4. un passo reale: nessun NaN/Inf, valori di magnitudine sanabile
print("\n4. un passo con pesi reali")
rng = np.random.default_rng(11)
# Scala 1.0 e non 0.02: con input artificialmente piccolo l'output sarebbe
# ~1e-5 per costruzione e un test di magnitudine passerebbe (o fallirebbe)
# per il motivo sbagliato. Qui si controlla che il layer produca un
# contributo di grandezza confrontabile con hidden_size, non che sia vivo
# per caso.
x = (rng.standard_normal(EMBD) * 1.0).astype(np.float32)
y = layer.forward(x)
check("output f32 di hidden_size", y.shape == (EMBD,) and y.dtype == np.float32)
check("output senza NaN/Inf", bool(np.all(np.isfinite(y))),
      f"max|y|={float(np.abs(y).max()):.4f}")
check("magnitudine non nulla", float(np.abs(y).max()) > 1e-3,
      f"max|y|={float(np.abs(y).max()):.4f}")
check("lo stato e' finito", bool(np.all(np.isfinite(layer.state))),
      f"max|S|={float(np.abs(layer.state).max()):.4f}")

# 5. reference indipendente sulle dimensioni reali, con controllo di
#    simmetria per non essere cieco all'orientamento
print("\n5. reference matriciale su 3 token, con test di simmetria")
layer.reset()
ref_state = np.zeros((N_V, D_V, D_K), dtype=np.float64)
ref_conv = np.zeros((conv_dim, CONV_K - 1), dtype=np.float64)
w64 = {k: v.astype(np.float64) for k, v in w.items()}

def ref_step(x, state, conv, transpose=False):
    qkv = x @ w64["qkv"]
    beta = _sigmoid(x @ w64["beta"])
    alpha = x @ w64["alpha"]
    gate = _softplus(alpha + w64["dt_bias"]) * w64["a"]
    seq = np.concatenate([conv, qkv.reshape(conv_dim, 1)], axis=1)
    conv = seq[:, 1:].copy()
    c = _silu(np.einsum("ck,ck->c", seq, w64["conv1d"].T))
    q = _l2_normalize(c[:key_dim].reshape(N_K, D_K)) / np.sqrt(D_K)
    k = _l2_normalize(c[key_dim:2*key_dim].reshape(N_K, D_K))
    vv = c[2*key_dim:].reshape(N_V, D_V)
    reps = N_V // N_K
    k = np.repeat(k, reps, axis=0); q = np.repeat(q, reps, axis=0)
    outs = []
    for h in range(N_V):
        S = state[h]
        I_minus = np.eye(D_K) - beta[h] * np.outer(k[h], k[h])
        S = np.exp(gate[h]) * (S @ I_minus) + beta[h] * np.outer(vv[h], k[h])
        state[h] = S
        outs.append((S.T @ q[h]) if transpose else (S @ q[h]))
    head = np.array(outs)
    zr = (x @ w64["gate"]).reshape(N_V, D_V)
    rms = head / np.sqrt(np.mean(head ** 2, axis=-1, keepdims=True) + 1e-6)
    gated = (rms * w64["norm"] * _silu(zr)).reshape(-1)
    # ssm_out e' (value_dim, n_embd): il verso e' gia' quello giusto.
    return gated @ w64["out"], state, conv

for step in range(3):
    x = (rng.standard_normal(EMBD) * 1.0).astype(np.float32)
    got = layer.forward(x)
    want, ref_state, ref_conv = ref_step(x.astype(np.float64), ref_state, ref_conv)
    rel = float(np.abs(got - want).max() / max(1e-12, float(np.abs(want).max())))
    check(f"token {step}: layer == reference", rel < 5e-4,
          f"rel={rel:.3e} (f32 su hidden {EMBD})")

# il test ha davvero sensibilita' all'orientamento?
alt_state = np.zeros((N_V, D_V, D_K))
alt_conv = np.zeros((conv_dim, CONV_K - 1))
xs = (rng.standard_normal(EMBD) * 1.0)
okv, alt_state, alt_conv = ref_step(xs, alt_state, alt_conv)
badv, _, _ = ref_step(xs, np.zeros((N_V, D_V, D_K)),
                      np.zeros((conv_dim, CONV_K - 1)), transpose=True)
rel_bad = float(np.abs(okv - badv).max() / max(1e-12, float(np.abs(okv).max())))
check("la reference trasposta da un risultato DIVERSO (test non cieco)",
      rel_bad > 1e-3, f"|differenza|={rel_bad:.3e}")

# 6. lo stato si azzera e non cresce
print("\n6. reset e stabilita' su sequenza lunga")
layer.reset()
check("reset azzera lo stato", float(np.abs(layer.state).max()) == 0.0)
check("reset azzera la coda", float(np.abs(layer.conv_state).max()) == 0.0)
mx = 0.0
for _ in range(24):
    y = layer.forward((rng.standard_normal(EMBD) * 1.0).astype(np.float32))
    if not np.all(np.isfinite(y)):
        break
    mx = max(mx, float(np.abs(y).max()))
check("24 token reali restano finiti", bool(np.all(np.isfinite(y))),
      f"max|y|={mx:.4f}")
check("lo stato non esplode", float(np.abs(layer.state).max()) < 1e4,
      f"max|S|={float(np.abs(layer.state).max()):.3f}")

# 7. classificazione dei layer
print("\n7. classificazione dei 40 layer")
attn = [i for i in range(40) if not is_recurrent_layer(tensors, i)]
rec = [i for i in range(40) if is_recurrent_layer(tensors, i)]
check("10 layer di attenzione", len(attn) == 10, f"{attn}")
check("30 layer SSM", len(rec) == 30, f"primi 5: {rec[:5]}")

print(f"\n{'=' * 66}\n PASSATI: {ok}   FALLITI: {bad}\n{'=' * 66}")
sys.exit(1 if bad else 0)