"""Estrae nome -> somma e dati dal dump di llama-eval-callback.

Il dump e' testo: ogni tensore e' una riga con la descrizione
dell'op, poi ``sum = ...``, poi il tensor stampato con i soli primi e ultimi
tre elementi di ogni riga. I numeri sono utili come impronta: una somma
diversa localizza la tappa, gli elementi dicono di quanto.

Non serve tutto il dump: ``--layer`` tiene i tensor di un layer e
``--prefix`` quelli che iniziano con una stringa.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HEAD = re.compile(r"common_debug_cb_eval:\s+(.+?) = \(\w+\)")
SUM = re.compile(r"^\s+sum = (\S+)")
RMS = re.compile(r"^\s+rms = (\S+)")
ARGMAX = re.compile(r"^\s+argmax = (\d+)")
# La forma di uscita sta in coda alla riga di descrizione: "= {2048, 5, 1, 1}".
# Serve al riferimento per sapere quanti elementi ha un tensore: senza, non si
# puo' distinguere una somma significativa da una cancellazione.
SHAPE = re.compile(r"=\s*\{([\d]+),\s*([\d]+),\s*([\d]+),\s*([\d]+)\}\s*$")
NUM = re.compile(r"-?\d+\.\d+(?:e[+-]\d+)?")


DATA = re.compile(r"^[\s\[\],0-9.eE+-]*$")


def parse(path: Path) -> dict[str, dict]:
    """Raccoglie somma e righe stampate per ogni tensore.

    L'ordine nel file e' descrizione, blocco dati, poi ``sum``: la somma
    arriva DOPO i numeri, quindi leggerla solo mentre si e' dentro il
    blocco la perde per tutti. Qui si tiene il nome corrente e ogni riga
    viene attribuita a lui, finche' non ne arriva un altro.
    """
    out: dict[str, dict] = {}
    name: str | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = HEAD.search(line)
        if m:
            name = m.group(1)
            out.setdefault(name, {"sum": None, "rms": None, "argmax": None,
                                  "ne": None, "rows": []})
            sm = SHAPE.search(line)
            if sm:
                ne = [int(v) for v in sm.groups()]
                n = 1
                for v in ne:
                    n *= v
                out[name]["ne"] = ne
                out[name]["n"] = n
            continue
        if name is None:
            continue
        for regex, key in ((SUM, "sum"), (RMS, "rms"), (ARGMAX, "argmax")):
            m = regex.match(line)
            if m:
                out[name][key] = float(m.group(1))
                break
        else:
            if line.strip().startswith("["):
                out[name]["rows"] = [[float(v) for v in NUM.findall(line)]]
            elif out[name]["rows"] and DATA.match(line) and NUM.search(line):
                out[name]["rows"].append([float(v) for v in NUM.findall(line)])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", type=Path)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()

    data = parse(args.dump)
    for name, rec in data.items():
        if args.layer is not None and not name.endswith(f"-{args.layer}"):
            continue
        if args.prefix and not name.startswith(args.prefix):
            continue
        if rec["sum"] is None and rec["rms"] is None:
            continue
        head = (f"{name:44s} sum={rec['sum']:14.6f}"
                if rec["sum"] is not None else f"{name:44s} {'-':>20s}")
        head += f" rms={rec['rms']:12.6f}" if rec["rms"] is not None else ""
        if rec["argmax"] is not None:
            head += f" argmax={int(rec['argmax'])}"
        rows = rec["rows"]
        if rows:
            head += "  " + " ".join(f"{v:9.4f}" for v in rows[-1][:6])
        print(head)
    print(f"\ntensori letti: {len(data)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())