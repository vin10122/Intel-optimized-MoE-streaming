"""Parita' numerica con llama.cpp, layer per layer, in automatico.

Questo test e' la rete che mancava. Prima la parita' si verificava a mano,
con un build di llama.cpp vivo e un dump da 4,6 MB: nessuno dei due
sopravvive a un clone, quindi nessuno poteva beccare una regressione. Il
riferimento che questo test confronta sono 9 KiB committati nel repo.

Perche' si confronta l'rms e non la somma: su un tensore quasi a media zero la
somma si cancella da sola, e due tensori diversi possono darla uguale. E'
esattamente cosi' che otto test erano passati con i pesi degli expert letti
trasposti. L'rms non si cancella.

Le tolleranze non sono arrotondate a occhio: sono la deriva f16 che si
vede su un layer gia' corretto, piu' un margine per l'accumulo nei 40 layer.
Se un test fallisce, la domanda da fare non e' "quanto e' la tolleranza"
ma "cosa e' cambiato".

Il test e' stato validato per mutazione: rimettendo la convenzione veccola
sui pesi dello shared expert, fallisce dal layer 2 con rms fuori soglia. Serve
dirlo, perche' un test di parita' che passa anche quando il runtime e'rotto e'
il difetto che questo file esiste per non ripetere.
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

# Tolleranza relativa sull'rms, che e' l'unica statistica confrontata come
# soglia. Su un layer corretto la deriva e' la sola f16; il margine copre
# l'accumulo sui 40 layer.
TOL_RMS = 0.02

# La somma non e' una soglia, e' una diagnostica. Su un tensore a media zero
# le cancellazioni la rendono poco informativa: peggio che altrove, prima di
# rinunciarci, il 40% dei confronti sulla somma falliva mentre nessun rms
# superava la soglia. Peggio ancora sui logits, che sono polarizzati: la somma
# li' misura la media, cioe' la parte del vettore che dice meno su chi e'
# corretto. L'rms e l'argmax dicono molto di piu', e sono le due cose che
# vengono confrontate come soglia.
#
# SUM_MEANINGFUL serve solo a non stampare come diagnostica le somme che sono
# rumore puro: sotto quella soglia la somma di riferimento e' piu' piccola di
# quello che il caso darebbe, quindi non la si guarda nemmeno.
SUM_MEANINGFUL = 0.15

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
    return {"sum": float(v.sum()),
            "rms": float(np.sqrt(np.mean(v * v))),
            "n": int(v.size)}


def rel(ours: float, ref: float) -> float:
    return abs(ours - ref) / max(abs(ref), 1e-12)


def sum_is_meaningful(want: dict) -> bool:
    """True se la somma di riferimento non e' una cancellazione."""
    scale = want["rms"] * (want["n"] ** 0.5)
    return abs(want["sum"]) >= SUM_MEANINGFUL * scale


def main() -> int:
    ref = json.loads(REF_PATH.read_text(encoding="utf-8"))
    ids = ref["tokens"]
    devices = sys.argv[1:] or ["CPU"]
    print(f"riferimento: {REF_PATH.name}, prompt {ref['prompt']!r}, "
          f"token {ids}")
    print(f"fonte dei valori: {ref['source']}")
    print(f"dispositivi da verificare: {', '.join(devices)}\n")

    path = model_path()
    overall_bad = 0
    for device in devices:
        print(f"--- {device} " + "-" * (58 - len(device)))
        rc = run_device(ref, ids, path, device)
        overall_bad += rc

    print(f"\n{'=' * 66}")
    print(f" PASSATI: {ok}   FALLITI: {overall_bad}")
    print(f"{'=' * 66}")
    return 1 if overall_bad else 0


def run_device(ref: dict, ids: list[int], path, device: str) -> int:
    global ok
    ok = 0
    model = Qwen3MoEModel(path, device=device, max_seq_len=256,
                          cache_capacity=64, dequant_fn=dequant_to_f16)
    try:
        model.init_caches()
    except Exception as exc:  # il plugin puo' non esistere su questa macchina
        check(f"{device}: il backend si inizializza", False, repr(exc))
        model.close()
        return 1

    x = model.embedding.forward(np.asarray(ids, dtype=np.int64))
    x = x.reshape(1, len(ids), model.hidden_size)

    worst_rms = (0.0, "")
    worst_sum = (0.0, "")
    for n in range(model.block_count):
        block = model.blocks[n]
        normed = block.attn_norm.forward(x)
        if block.is_ssm:
            mixed = block.ssm_runner(normed)
        else:
            mixed = block._attention_forward(normed)
        resid = x + mixed
        normed2 = block.post_attn_norm.forward(resid)
        out = resid + np.asarray(block._moe_forward(normed2))

        want = ref["layers"][str(n)]
        for role, arr in (("attn_norm", normed), ("attn_residual", resid),
                          ("l_out", out)):
            got = stats(arr)
            err = rel(got["rms"], want[role]["rms"])
            if err > worst_rms[0]:
                worst_rms = (err, f"layer {n} {role}")
            if err > TOL_RMS:
                check(f"layer {n:2d} {role}.rms", False,
                      f"noi {got['rms']:14.6f} llama.cpp {want[role]['rms']:14.6f}"
                      f"  rel {err * 100:.2f}% > {TOL_RMS * 100:.0f}%")
            if sum_is_meaningful(want[role]):
                sum_err = rel(got["sum"], want[role]["sum"])
                if sum_err > worst_sum[0]:
                    worst_sum = (sum_err, f"layer {n} {role}")
        x = out

    check("rms di tutti i 40 layer entro tolleranza", True,
          f"peggiore {worst_rms[0] * 100:.2f}% su {worst_rms[1]}")
    print(f"  info  somma, diagnostica non soglia: peggiore "
          f"{worst_sum[0] * 100:.2f}% su {worst_sum[1]}")

    w = np.asarray(model.tmap["output_norm.weight"].data,
                   dtype=np.float32).reshape(-1)
    last = x[0, -1, :].astype(np.float32)
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
    print(f"  info  somma dei logits, diagnostica: noi "
          f"{got_logits['sum']:.3f} llama.cpp "
          f"{ref['final']['logits']['sum']:.3f}  "
          f"rel {rel(got_logits['sum'], ref['final']['logits']['sum']) * 100:.2f}%")

    best = int(np.argmax(logits))
    check("token greedy", best == ref["final"]["argmax"],
          f"noi {best} llama.cpp {ref['final']['argmax']}")

    model.close()
    return bad


if __name__ == "__main__":
    raise SystemExit(main())