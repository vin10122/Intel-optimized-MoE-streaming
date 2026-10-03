"""Costruisce il riferimento di parita' da un dump di llama-eval-callback.

Il dump e' di 4,6 MB e contiene 3787 tensori, perche' llama-eval-callback
stampa tutto il grafo. Non e' committabile e non serve: quello che serve
e' un'impronta di ogni layer che fallisca se il runtime si allontana.

Cosa finisce nel riferimento, e perche' queste cose:

* ``rms`` per ogni layer. La somma da sola non basta: su un tensore quasi a
  media zero le cancellazioni possono tenere il valore vicino anche quando i
  numeri sono diversi, ed e' cosi' che un peso letto trasposto e' passato.
  L'rms non si cancella.
* tre quantita' per layer (attn_norm, attn_residual, l_out). La prima dice se
  l'ingresso del layer e' gia' sbagliato, la seconda se lo sbaglio e' nato
  nell'attenzione o nel delta net, la terza e' l'uscita. Con tre numeri si
  localizza il layer senza aprire nient'altro.
* logits finali con somma, rms e argmax. L'argmax e' il token generato: e'
  l'asserzione leggibile, e un errore di orientamento lo cambia.

Non si tengono le stampe dei dati elementari: sono 3 valori per riga e il
loro campione e' troppo piccolo per servire da riferimento.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_eval_dump import parse  # noqa: E402

PER_LAYER = ("attn_norm", "attn_residual", "l_out")


def entry(rec: dict) -> dict | None:
    if rec["sum"] is None or rec["rms"] is None or rec.get("n") is None:
        return None
    return {"sum": rec["sum"], "rms": rec["rms"], "n": int(rec["n"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--tokens", default="8482 6511 314 9338 369")
    ap.add_argument("--prompt", default="La capital of France is")
    args = ap.parse_args()

    data = parse(args.dump)
    layers: dict[str, dict] = {}
    missing: list[str] = []
    for n in range(args.layers):
        per: dict[str, dict] = {}
        for role in PER_LAYER:
            rec = data.get(f"{role}-{n}")
            got = entry(rec) if rec else None
            if got is None:
                missing.append(f"{role}-{n}")
            else:
                per[role] = got
        layers[str(n)] = per
    if missing:
        raise SystemExit(f"nel dump mancano: {missing}")

    tail: dict[str, dict] = {}
    for role in ("result_norm", "result_output"):
        rec = data.get(role)
        got = entry(rec) if rec else None
        if got is None:
            raise SystemExit(f"nel dump manca {role}")
        tail[role] = got
    if data["result_output"]["argmax"] is None:
        raise SystemExit("nel dump manca l'argmax dei logits")

    ref = {
        "_comment": "Generato da tools/make_golden_reference.py. Non editare a mano.",
        "prompt": args.prompt,
        "tokens": [int(t) for t in args.tokens.split()],
        "source": "llama.cpp llama-eval-callback, prefillo senza BOS, -n 1",
        "model": "Hermes3.6-35B-A3B-Uncensored-Genesis-Final-APEX-Compact.gguf",
        "layers": layers,
        "final": {
            "result_norm": tail["result_norm"],
            "logits": tail["result_output"],
            "argmax": int(data["result_output"]["argmax"]),
        },
    }
    args.out.write_text(json.dumps(ref, indent=1) + "\n", encoding="utf-8")
    print(f"{args.out}: {args.out.stat().st_size / 1024:.1f} KiB, "
          f"{args.layers} layer, argmax {ref['final']['argmax']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())