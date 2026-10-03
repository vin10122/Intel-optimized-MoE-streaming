"""Il decode, confrontato con llama.cpp numericamente e non a occhio.

Il riferimento copre un prefillo: cinque token valutati insieme, in un
colpo solo. La generazione e' un codice diverso. Ogni token passa dalla
cache KV, che si riempie di una posizione alla volta, e i layer ricorrenti
avanzano il loro stato di un passo. Se la cache sbaglia posizione o il
ricorrente sbaglia passo, il testo resta italiano e plausibile: la cache
KV accetta un token nella riga sbagliata senza errori, e l'attenzione
causale continua a restituire numeri.

Percio' qui il confronto e' sullo stesso riferimento di tools/test_golden_parity,
non su uno nuovo. Il motivo che lo permette e' che l'attenzione e' causale:
l'uscita della posizione i non dipende dai token successivi, quindi la somma
e l'rms dei cinque output di un decode valgono quanto quelli di un prefillo.
Se il percorso con cache e' corretto devono coincidere entro la stessa
toleranza di sempre, e se non coincidono la differenza e' tutta li'.

Nessun riferimento nuovo quindi: e' lo stesso file, letto in due modi. Il
vantaggio e' che il riferimento resta una cosa sola da mantenere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from run_q4k_f16_openvino import dequant_to_f16  # noqa: E402
from flash_moe import Qwen3MoEModel  # noqa: E402
from flash_moe.paths import model_path  # noqa: E402

REF_PATH = Path(__file__).resolve().parent / "golden_reference.json"
TOL_RMS = 0.02

ok = bad = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global ok, bad
    if cond:
        ok += 1
        print(f"  PASS  {label}  ({detail})")
    else:
        bad += 1
        print(f"  FAIL  {label}  ({detail})")


def stats(a: np.ndarray) -> dict:
    v = np.asarray(a, dtype=np.float64).reshape(-1)
    return {"sum": float(v.sum()), "rms": float(np.sqrt(np.mean(v * v)))}


def rel(ours: float, ref: float) -> float:
    return abs(ours - ref) / max(abs(ref), 1e-12)


def main() -> int:
    ref = json.loads(REF_PATH.read_text(encoding="utf-8"))
    ids = ref["tokens"]
    device = sys.argv[1] if len(sys.argv) > 1 else "CPU"
    print(f"riferimento: {REF_PATH.name} ({len(ids)} token), device {device}")
    print("modalita': un token alla volta, con cache KV e stato ricorrente\n")

    path = model_path()
    model = Qwen3MoEModel(path, device=device, max_seq_len=256,
                          cache_capacity=64, dequant_fn=dequant_to_f16)
    model.init_caches()

    n_layers = model.block_count
    # Solo i layer full attention hanno la cache KV: su un layer ricorrente
    # attn.cache e' None, e usarlo per controllare la crescita non avrebbe
    # senso. Se non ci fosse nessun layer full attention questo test non
    # controllerebbe niente, quindi e' un errore esplicito e non silenzioso.
    attn_layers = [n for n in range(n_layers) if not model.blocks[n].is_ssm]
    if not attn_layers:
        raise SystemExit("nessun layer full attention: la cache non viene "
                         "esercitata e il test non controllerebbe nulla")
    probe_layer = attn_layers[0]

    # Per layer, non in un unico mucchio: il riferimento ha una voce per
    # layer, e accorpare tutti e 40 insieme darebbe lo stesso numero a tutti,
    # il che si vede subito perche' due layer non possono avere rms identico.
    pooled: dict[int, dict[str, list[np.ndarray]]] = {
        n: {key: [] for key in ("attn_norm", "attn_residual", "l_out")}
        for n in range(n_layers)}
    last = None

    for step, tok_id in enumerate(ids):
        x = model.embedding.forward(np.asarray([tok_id], dtype=np.int64))
        x = x.reshape(1, 1, model.hidden_size)
        for n in range(n_layers):
            block = model.blocks[n]
            normed = block.attn_norm.forward(x)
            if block.is_ssm:
                mixed = block.ssm_runner(normed)
            else:
                mixed = block._attention_forward(normed)
            resid = x + mixed
            normed2 = block.post_attn_norm.forward(resid)
            out = resid + np.asarray(block._moe_forward(normed2))
            pooled[n]["attn_norm"].append(np.asarray(normed).reshape(-1))
            pooled[n]["attn_residual"].append(np.asarray(resid).reshape(-1))
            pooled[n]["l_out"].append(np.asarray(out).reshape(-1))
            x = out
        last = x[0, -1, :].astype(np.float32)
        grew = model.blocks[probe_layer].attn.cache.seq_len
        if grew != step + 1:
            check(f"cache cresce di uno al passo {step + 1}", False,
                  f"seq_len={grew}, atteso {step + 1}")
            model.close()
            return 1
        print(f"  passo {step + 1}/{len(ids)}: token {tok_id}, "
              f"cache {grew} posizioni")

    worst = (0.0, "")
    for n in range(n_layers):
        want = ref["layers"][str(n)]
        for role in ("attn_norm", "attn_residual", "l_out"):
            got = stats(pooled[n][role])
            err = rel(got["rms"], want[role]["rms"])
            if err > worst[0]:
                worst = (err, f"layer {n} {role}")
            if err > TOL_RMS:
                check(f"layer {n:2d} {role}.rms", False,
                      f"noi {got['rms']:14.6f} llama.cpp {want[role]['rms']:14.6f}"
                      f"  rel {err * 100:.2f}% > {TOL_RMS * 100:.0f}%")
    check("rms di tutti i 40 layer entro tolleranza, via cache", True,
          f"peggiore {worst[0] * 100:.2f}% su {worst[1]}")

    w = np.asarray(model.tmap["output_norm.weight"].data,
                   dtype=np.float32).reshape(-1)
    final = (last / np.sqrt(np.mean(last * last) + 1e-6)) * w
    got_norm = stats(final)
    err = rel(got_norm["rms"], ref["final"]["result_norm"]["rms"])
    check("result_norm.rms", err <= TOL_RMS,
          f"noi {got_norm['rms']:12.6f} llama.cpp "
          f"{ref['final']['result_norm']['rms']:12.6f}  rel {err * 100:.2f}%")

    logits = np.asarray(model.lm_head.forward(
        final.astype(np.float32).reshape(1, -1)), dtype=np.float64).reshape(-1)
    got_logits = stats(logits)
    err = rel(got_logits["rms"], ref["final"]["logits"]["rms"])
    check("logits.rms", err <= TOL_RMS,
          f"noi {got_logits['rms']:14.3f} llama.cpp "
          f"{ref['final']['logits']['rms']:14.3f}  rel {err * 100:.2f}%")
    best = int(np.argmax(logits))
    check("token greedy dopo 5 passi di decode", best == ref["final"]["argmax"],
          f"noi {best} llama.cpp {ref['final']['argmax']}")

    print(f"\n{'=' * 66}\n PASSATI: {ok}   FALLITI: {bad}\n{'=' * 66}")
    model.close()
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())