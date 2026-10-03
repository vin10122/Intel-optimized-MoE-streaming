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
            out.setdefault(name, {"sum": None, "rows": []})
            continue
        if name is None:
            continue
        m = SUM.match(line)
        if m:
            out[name]["sum"] = float(m.group(1))
            continue
        if line.strip().startswith("["):
            out[name]["rows"] = []
            for v in NUM.findall(line):
                out[name]["rows"].append([float(v)])
            continue
        if out[name]["rows"] and DATA.match(line) and NUM.search(line):
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
        if rec["sum"] is None:
            continue
        head = f"{name:40s} sum={rec['sum']:14.6f}"
        rows = rec["rows"]
        if rows:
            head += "  " + " ".join(f"{v:9.4f}" for v in rows[-1][:6])
        print(head)
    print(f"\ntensori letti: {len(data)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())