"""Il fallback NumPy della dequantizzazione, confrontato con la DLL C++.

La DLL e' il percorso normale: ``run_q4k_f16_openvino.dequant_to_f16`` la
chiama se c'e'. Il fallback in ``dequantize_to`` serve quando non c'e', cioe'
su una macchina diversa da questa, ed e' esattamente il caso in cui nessuno
ci passa: nessuno lo esercita e nessuno lo vede.

Era rotto. ``gguf.quants.dequantize`` restituisce una colonna (n, 1) e non
un vettore, quindi l'assegnazione nel buffer 1-D falliva con "could not
broadcast input array from shape (8192,1) into shape (8192,)". Non un
numerismo sbagliato: un'eccezione, su ogni chiamata.

Il confronto e' sui tensori quantizzati veri del modello, perche' un test
su valori sintetici non avrebbe coperto i tipi che qui si incontrano: Q4_K,
Q3_K, Q6_K e Q8_0 hanno block size e byte per blocco diversi.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gguf  # noqa: E402
from run_q4k_f16_openvino import dequant_to_f16  # noqa: E402
from gguf.quants import GGML_QUANT_SIZES  # noqa: E402
from flash_moe.dequant import dequantize_to  # noqa: E402
from flash_moe.paths import model_path  # noqa: E402

# Lo stesso blocco di pesi su tutti e tre i tipi quantizzati che si
# incontrano su questo modello, piu' il primo blocco dell'embedding.
CASES = [
    ("blk.0.ffn_gate_exps.weight", 1048576),
    ("blk.5.ffn_gate_exps.weight", 1048576),
    ("blk.39.ffn_down_exps.weight", 1048576),
    ("blk.0.ffn_gate_shexp.weight", 1048576),
    ("token_embd.weight", 1048576),
]

ok = bad = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global ok, bad
    if cond:
        ok += 1
        print(f"  PASS  {label}  ({detail})")
    else:
        bad += 1
        print(f"  FAIL  {label}  ({detail})")


def main() -> int:
    path = model_path()
    reader = gguf.GGUFReader(path)
    tmap = {t.name: t for t in reader.tensors}

    for name, n_elements in CASES:
        t = tmap[name]
        ttype = int(t.tensor_type)
        # Si prende solo il pezzo di tensor che contiene n_elements. Per un
        # tensor di expert sono tutti e 256 dentro lo stesso blocco di byte:
        # passare tutto e pretendere un expert solo non fallirebbe, ma
        # confronterebbe 256 expert contro uno e chiamerebbe fallimento un
        # numero che e' corretto.
        block_size, bytes_per_block = GGML_QUANT_SIZES[ttype]
        n_bytes = (n_elements // block_size) * bytes_per_block
        with path.open("rb") as fh:
            fh.seek(t.data_offset)
            raw = fh.read(n_bytes)
        if len(raw) != n_bytes:
            check(f"{name}: lettura", False,
                  f"{len(raw)} byte letti, attesi {n_bytes}")
            continue

        try:
            ref = dequant_to_f16(raw, ttype, n_elements)
        except Exception as exc:
            check(f"{name}: percorso C++", False, repr(exc))
            continue

        try:
            got = dequantize_to(raw, ttype, "f16")
        except Exception as exc:
            check(f"{name}: fallback NumPy", False, repr(exc))
            continue

        if got.shape != ref.shape:
            check(f"{name}: fallback NumPy", False,
                  f"forma {got.shape} contro {ref.shape}")
            continue

        a = np.asarray(got, dtype=np.float64)
        b = np.asarray(ref, dtype=np.float64)
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            check(f"{name}: fallback NumPy", False, "i NaN non coincidono")
            continue
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.any():
            err = float(np.abs(a[finite] - b[finite]).max()
                        / max(float(np.abs(b[finite]).max()), 1e-12))
        else:
            err = 0.0
        nans = int((~np.isfinite(a)).sum())
        check(f"{name}: fallback NumPy", err <= 1e-3 and nans == 0,
              f"tipo {t.tensor_type.name}, errore relativo {err:.3e}, "
              f"nan {nans}/{a.size}")

    print(f"\n{'=' * 66}\n PASSATI: {ok}   FALLITI: {bad}\n{'=' * 66}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())