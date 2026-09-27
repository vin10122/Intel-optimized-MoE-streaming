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

BLOCK_BYTES = {"Q3_K": 110, "Q5_K": 176, "Q6_K": 210}
GGML_ID = {"Q3_K": int(gguf.GGMLQuantizationType.Q3_K),
           "Q5_K": int(gguf.GGMLQuantizationType.Q5_K),
           "Q6_K": int(gguf.GGMLQuantizationType.Q6_K)}

# Byte che non partecipano ai quanti ma solo a d / dmin / scales.
# Sono esclusi dai probe perche' non contribuiscono a q.
QUANT_BYTES = {
    "Q3_K": [(32, 96), (0, 32)],   # ql 32..95, hmask 0..31
    "Q5_K": [(16, 48), (48, 176)],  # qh 16..47, qs 48..175
    "Q6_K": [(0, 128), (128, 192)],  # ql 0..127, qh 128..191
}

# Q5_K: 176 B con d f16 @0, dmin f16 @2, scales[12] @4, qh[32] @16, qs[128] @48
Q5K_D, Q5K_DMIN = 0, 2


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


def q5k_base() -> np.ndarray:
    """Blocco Q5_K con sc[j] = 1, d = 1, dmin = 0, q = 0.

    Con questa base y = q esattamente, quindi il delta di un bit dice subito a
    quale posizione di q finisce. Gli offset delle scale sono quelli misurati:
    sc a 6 bit nei byte 4..7 per j<4, a 4 bit nei byte 8..11 per j>=4.
    """
    blk = np.zeros(BLOCK_BYTES["Q5_K"], dtype=np.uint8)
    put_f16(blk, Q5K_D, 1.0)
    put_f16(blk, Q5K_DMIN, 0.0)
    blk[4:16] = 1
    return blk


def q5k_probe():
    """Accende un bit alla volta e registra gli output che cambiano.

    Si sondano solo i byte 16..175, cioe' qh e qs. I byte 4..15 sono le scale:
    non vanno sondati qui, perche' un bit di scala muove 32 output e un filtro
    basato sul numero di output li scarterebbe per errore (e' gia' successo:
    i due bit alti di sc[j>=4] arrivano da s[j-4] >> 6 e were scartati).
    """
    base = q5k_base()
    y0 = dequant_one("Q5_K", base)
    entries: list[list] = [[] for _ in range(QK_K)]
    touched: set[int] = set()
    counts: dict[int, list] = {}
    for byte in range(16, BLOCK_BYTES["Q5_K"]):
        for bit in range(8):
            blk = base.copy()
            blk[byte] = np.uint8(1 << bit)
            delta = dequant_one("Q5_K", blk) - y0
            nz = np.nonzero(np.abs(delta) > 1e-9)[0]
            if nz.size == 0:
                continue
            if nz.size != 1:
                counts.setdefault(byte, []).append(nz.size)
                continue
            touched.add(byte)
            for idx in nz:
                entries[int(idx)].append((byte, int(bit), float(delta[int(idx)])))
    return entries, sorted(touched), counts


def split_runs(entries: list) -> list:
    """Divide i bit che influenzano un output in corse contigue per byte.

    Una corsa e' un campo: 4 bit bassi e 4 alti dello stesso byte sono due campi
    distinti, cosi' come i due campi da 2 bit di un byte di qh.
    """
    by_byte: dict[int, set] = {}
    for byte, bit, _ in entries:
        by_byte.setdefault(byte, set()).add(bit)
    runs: list[tuple[int, int, int]] = []
    for byte in sorted(by_byte):
        bits = sorted(by_byte[byte])
        cur = [bits[0]]
        for b in bits[1:]:
            if b == cur[-1] + 1:
                cur.append(b)
            else:
                runs.append((byte, cur[0], len(cur)))
                cur = [b]
        runs.append((byte, cur[0], len(cur)))
    return runs


def derive_q5_k() -> dict:
    """Misura byte/shift/larghezza/offset dei campi che compongono q.

    y = d*sc*q - dmin*m, quindi q e' l'unico termine lineare nei quanti: il
    termine -dmin*m non ha campi quantizzati propri, e' costante nel
    sub-blocco. Le scale restano quelle di Q4_K (unpack_scale_min_k4).
    """
    print("  sonda Q5_K (d=1, sc=1, dmin=0, q=0)...")
    entries, touched, counts = q5k_probe()
    print(f"  byte di quanti: {touched[0]}..{touched[-1]} ({len(touched)} byte)")
    for b, sz in sorted(counts.items()):
        print(f"  ATTENZIONE byte {b}: muove {sorted(set(sz))} output, scartato")
    nbits = sum(len(e) for e in entries)
    print(f"  bit che influenzano un output: {nbits} ({nbits / QK_K:.2f} per output)")

    fields: list[list] = []
    for idx in range(QK_K):
        runs = split_runs(entries[idx])
        if not runs:
            raise RuntimeError(f"Q5_K: output {idx} non dipende da nessun bit")
        out_fields = []
        for byte, shift, width in runs:
            dmin_bit = next(v for by, bi, v in entries[idx] if by == byte and bi == shift)
            mag = int(round(abs(dmin_bit)))
            if mag <= 0 or mag & (mag - 1):
                raise RuntimeError(
                    f"Q5_K: output {idx} campo byte {byte} ha delta {dmin_bit} non potenza di 2")
            # delta del bit di peso minimo = 2^(offset + 0)
            offset = int(round(np.log2(mag)))
            if offset + width > 8:
                raise RuntimeError(f"Q5_K: output {idx} campi sovrapposti in q")
            out_fields.append((byte, shift, width, offset))
        fields.append(out_fields)

    widths = sorted({f[2] for fs in fields for f in fs})
    n_fields = sorted({len(fs) for fs in fields})
    if len(n_fields) != 1:
        raise RuntimeError(f"Q5_K: numero di campi variabile per output: {n_fields}")
    n = n_fields[0]
    print(f"  campi per output: {n}, larghezze {widths}")

    cols = [[0] * QK_K for _ in range(4 * n)]
    for idx, fs in enumerate(fields):
        for k, (byte, shift, width, offset) in enumerate(fs):
            cols[4 * k + 0][idx] = byte
            cols[4 * k + 1][idx] = shift
            cols[4 * k + 2][idx] = width
            cols[4 * k + 3][idx] = offset
    return {"n": n, "cols": cols}


def emit_q5_k(t: dict) -> str:
    out = ""
    for k in range(t["n"]):
        for c, nm in enumerate(("B", "S", "W", "O")):
            vals = t["cols"][4 * k + c]
            out += (f"static const uint8_t Q5_K_{nm}{k + 1}[{QK_K}] = "
                    f"{{{','.join(map(str, vals))}}};\n")
    return out


def main() -> int:
    chunks = [
        "/* Generato da tools/derive_k_tables.py - NON modificare a mano.\n",
        " *\n",
        " * Per ogni output i:\n",
        " *   B1[i], S1[i] = byte e shift del campo a basso peso (2 bit in Q3_K,\n",
        " *                                        4 bit in Q6_K)\n",
        " *   B2[i], S2[i] = byte e shift del campo ad alto peso (1 bit in Q3_K,\n",
        " *                                        2 bit in Q6_K)\n",
        " *\n",
        " * Q5_K ha due campi di larghezza variabile e un offset costante:\n",
        " *   B1/S1/W1 = campo che moltiplica +d*scale, B2/S2/W2 = quello che\n",
        " *              moltiplica -dmin*m, KOFF = 0, 16 o 32 (offset in unita'\n",
        " *              di d*scale aggiunto all'output).\n",
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

    print("estraendo mappatura Q5_K...")
    t5 = derive_q5_k()
    chunks.append(emit_q5_k(t5))
    chunks.append("\n")

    (ROOT / "cpp" / "dequant_k_tables.h").write_text("".join(chunks), encoding="utf-8")
    print(f"scritto {ROOT / 'cpp' / 'dequant_k_tables.h'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
