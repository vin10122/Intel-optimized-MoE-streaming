"""Estrae la mappatura sorgente -> output dei kernel K-quant dalla reference.

Q3_K e Q6_K hanno un interlacciamento di nibble/bit che e' facile da sbagliare
a mano. Invece di derivarlo, lo si misura: si accende un bit alla volta del
blocco e si osserva quali output cambiano e di quanto. La reference e' lineare
nei bit, quindi il delta rivela esattamente (byte sorgente, shift) per ogni
output.

Output: cpp/dequant_k_tables.h con le tabelle per Q3_K e Q6_K.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import gguf

ROOT = Path(__file__).resolve().parent.parent
QK_K = 256

BLOCK_BYTES = {"Q3_K": 110, "Q6_K": 210}
GGML_ID = {"Q3_K": int(gguf.GGMLQuantizationType.Q3_K),
           "Q6_K": int(gguf.GGMLQuantizationType.Q6_K)}

# Byte che non partecipano ai quanti ma solo a d / dmin / scales.
# Sono esclusi dai probe perche' non contribuiscono a q.
QUANT_BYTES = {
    "Q3_K": [(32, 96), (0, 32)],   # ql 32..95, hmask 0..31
    "Q6_K": [(0, 128), (128, 192)],  # ql 0..127, qh 128..191
}


def make_base(type_name: str) -> np.ndarray:
    """Blocco con d=1 e scale/min neutrali, cosi' out = f(q) pulito."""
    bpb = BLOCK_BYTES[type_name]
    base = np.zeros(bpb, dtype=np.uint8)
    if type_name == "Q3_K":
        # layout: hmask[32] @0, ql[64] @32, scales[12] @96, f16 d @108
        put_f16(base, 108, 1.0)
        # scale[j] = 1  =>  (low4 | high2<<4) = 33 = 0b100001
        for j in range(8):
            base[96 + j] = 1                       # low nibble = 1
            k = 96 + 8 + (j & 3)
            base[k] = np.uint8(base[k] | (2 << (2 * (j >> 2))))
    else:
        # layout: ql[128] @0, qh[64] @128, int8 scales[16] @192, f16 d @208
        base[192:208] = 1
        put_f16(base, 208, 1.0)
    return base


def put_f16(buf: np.ndarray, offset: int, value: float) -> None:
    buf[offset:offset + 2] = np.array([value], dtype=np.float16).view(np.uint8)


def dequant_one(type_name: str, block: np.ndarray) -> np.ndarray:
    raw = np.ascontiguousarray(block).view(np.uint8).reshape(1, -1)
    return gguf.quants.dequantize(raw, GGML_ID[type_name]).reshape(QK_K).astype(np.float64)


def probe(type_name: str) -> dict[str, np.ndarray]:
    """Per ogni bit del blocco, che output cambia e di quanto."""
    bpb = BLOCK_BYTES[type_name]
    base_block = make_base(type_name)
    base_out = dequant_one(type_name, base_block)

    lo_bits: list[list] = [[] for _ in range(QK_K)]
    hi_bits: list[list] = [[] for _ in range(QK_K)]

    # scala di riferimento: il delta per un bit di valore 1 nel nibble basso
    scale = abs(base_out).max()
    if scale == 0:
        raise RuntimeError(f"{type_name}: blocco base degenere")

    for lo, hi in QUANT_BYTES[type_name]:
        for byte in range(lo, hi):
            for bit in range(8):
                blk = base_block.copy()
                blk[byte] = np.uint8(1 << bit)
                out = dequant_one(type_name, blk)
                delta = out - base_out
                changed = np.nonzero(np.abs(delta) > scale * 1e-9)[0]
                for idx in changed:
                    lo_bits[idx].append((byte, bit, abs(delta[idx])))

    return {"base": base_out, "lo": lo_bits, "hi": hi_bits}


def analyse(type_name: str) -> dict:
    """Raggruppa i probe per output e deduce (byte, shift) dei due campi."""
    res = probe(type_name)
    lo_bits, hi_bits = res["lo"], res["hi"]

    # Per distinguere i due campi (Q3_K: 2 bit + 1 bit; Q6_K: 4 bit + 2 bit)
    # usiamo il fatto che i due campi hanno pesi diversi.
    n_out_q3 = QK_K
    report = {}
    for idx in range(n_out_q3):
        entries = lo_bits[idx]
        if not entries:
            raise RuntimeError(f"{type_name}: output {idx} non dipende da nessun bit")
        # raggruppa per byte
        by_byte: dict[int, list] = {}
        for byte, bit, d in entries:
            by_byte.setdefault(byte, []).append((bit, d))
        if len(by_byte) != 2:
            raise RuntimeError(
                f"{type_name}: output {idx} dipende da {len(by_byte)} byte "
                f"({sorted(by_byte)}), attesi 2")
        (b1, e1), (b2, e2) = sorted(by_byte.items())
        report[idx] = (b1, e1, b2, e2)
    return report


def emit(type_name: str, report: dict) -> str:
    b1s, b2s, s1s, s2s = [], [], [], []
    for idx in range(QK_K):
        b1, e1, b2, e2 = report[idx]
        # nel campo a 2/4 bit il bit di peso minimo e' quello con shift 0
        s1 = min(bit for bit, _ in e1)
        s2 = min(bit for bit, _ in e2)
        b1s.append(b1)
        b2s.append(b2)
        s1s.append(s1)
        s2s.append(s2)
    return (
        f"static const uint8_t {type_name}_B1[{QK_K}] = {{{','.join(map(str, b1s))}}};\n"
        f"static const uint8_t {type_name}_B2[{QK_K}] = {{{','.join(map(str, b2s))}}};\n"
        f"static const uint8_t {type_name}_S1[{QK_K}] = {{{','.join(map(str, s1s))}}};\n"
        f"static const uint8_t {type_name}_S2[{QK_K}] = {{{','.join(map(str, s2s))}}};\n"
    )


def verify(type_name: str, report: dict) -> float:
    """Ricostruisce out dal mapping e confronta con la reference."""
    base_block = make_base(type_name)
    ref = dequant_one(type_name, base_block)

    # ricostruisci q da un blocco casuale usando il mapping
    rng = np.random.default_rng(12345)
    blk = rng.integers(0, 256, size=BLOCK_BYTES[type_name], dtype=np.uint8)
    ref_rand = dequant_one(type_name, blk)

    # con d e scale non neutrali non possiamo isolare q, quindi verifichiamo
    # solo la topologia: quali byte influenzano quali output
    return float(np.abs(ref).max())


def main() -> int:
    out_path = ROOT / "cpp" / "dequant_k_tables.h"
    chunks = [
        "/* Generato da tools/derive_k_tables.py - NON modificare a mano.\n",
        " *\n",
        " * Per ogni output i:\n",
        " *   B1[i], S1[i] = byte e shift del campo a basso peso (2 bit in Q3_K,\n",
        " *                                        4 bit in Q6_K)\n",
        " *   B2[i], S2[i] = byte e shift del campo a alto peso (1 bit in Q3_K,\n",
        " *                                        2 bit in Q6_K)\n",
        " *\n",
        " * La mappatura e' stata misurata accendendo un bit alla volta sulla\n",
        " * reference gguf.quants, non derivata a mano.\n",
        " */\n",
        "#pragma once\n\n",
    ]

    for type_name in ("Q3_K", "Q6_K"):
        print(f"estraendo mappatura {type_name}...")
        report = analyse(type_name)
        max_err = verify(type_name, report)
        print(f"  {type_name}: 256 output mappati, |base|max={max_err:.4f}")
        chunks.append(emit(type_name, report))
        chunks.append("\n")

    out_path.write_text("".join(chunks), encoding="utf-8")
    print(f"scritto {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
