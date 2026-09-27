"""Ispeziona i metadata reali del GGUF: quant type per tensor, shape, blocchi.

Serve a verificare se il modello e davvero Q4_K_M e a capire se la divisione
"n_bytes // num_experts" dello store produce slice valide.
"""

from __future__ import annotations

import os
import platform
from collections import Counter, defaultdict
from pathlib import Path

import gguf
import numpy as np

GGUF_PATH = Path(
    r"C:\Users\valer\Desktop\moe-client-prot\Nuova cartella"
    r"\Hermes3.6-35B-A3B-Uncensored-Genesis-Final-APEX-Compact.gguf"
)

# ggml type -> (block_size, bytes_per_block)
TYPE_INFO = {
    "F32": (1, 4),
    "F16": (1, 2),
    "BF16": (1, 2),
    "Q4_0": (32, 18),
    "Q4_1": (32, 20),
    "Q5_0": (32, 22),
    "Q5_1": (32, 24),
    "Q8_0": (32, 34),
    "Q2_K": (256, 84),
    "Q3_K": (256, 110),
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
    "Q6_K": (256, 210),
    "Q8_K": (256, 292),
    "IQ4_NL": (32, 18),
    "IQ4_XS": (256, 136),
    "IQ3_XXS": (256, 98),
    "IQ2_XXS": (256, 66),
    "IQ1_S": (256, 50),
}


def build_qt_name_map() -> dict[int, str]:
    names: dict[int, str] = {}
    for holder in (gguf.GGMLQuantizationType,):
        for key, value in vars(holder).items():
            if isinstance(value, int) and not key.startswith("_"):
                names.setdefault(value, key)
    return names


QT_NAME = build_qt_name_map()


def fmt_size(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"


def field_value(reader, key):
    part = reader.fields[key].parts[-1]
    if part.dtype == np.uint8:
        return bytes(part).decode("utf-8", errors="replace")
    return int(part[0])


def main() -> None:
    print("=" * 78)
    print(" HOST")
    print("=" * 78)
    print(f" platform         : {platform.platform()}")
    print(f" python           : {platform.python_version()}")
    print(f" logical cpus     : {os.cpu_count()}")

    print()
    print("=" * 78)
    print(" GGUF")
    print("=" * 78)
    size_bytes = GGUF_PATH.stat().st_size
    print(f" file             : {GGUF_PATH.name}")
    print(f" size             : {fmt_size(size_bytes)} ({size_bytes:,} bytes)")

    reader = gguf.GGUFReader(GGUF_PATH)
    print(f" reader.data_offset: {reader.data_offset:,}")

    arch = str(field_value(reader, "general.architecture"))
    print(f" architecture      : {arch}")

    meta_keys = [
        "general.name",
        "general.file_type",
        f"{arch}.block_count",
        f"{arch}.embedding_length",
        f"{arch}.feed_forward_length",
        f"{arch}.expert_count",
        f"{arch}.expert_used_count",
        f"{arch}.moe_intermediate_size",
        f"{arch}.attention.head_count",
        f"{arch}.attention.head_dim",
        f"{arch}.rope.dimension_count",
        f"{arch}.context_length",
        f"{arch}.key_length",
        f"{arch}.value_length",
    ]
    for key in meta_keys:
        if key in reader.fields:
            print(f" {key:34s}: {field_value(reader, key)}")

    print()
    print("=" * 78)
    print(" TENSOR: quant type distribution")
    print("=" * 78)
    by_type: Counter[str] = Counter()
    type_bytes: defaultdict[str, int] = defaultdict(int)
    type_elems: defaultdict[str, int] = defaultdict(int)
    for t in reader.tensors:
        name = QT_NAME.get(t.tensor_type, str(t.tensor_type))
        by_type[name] += 1
        type_bytes[name] += t.n_bytes
        type_elems[name] += int(t.n_elements)

    total_bytes = sum(type_bytes.values())
    total_elems = sum(type_elems.values())
    print(f" {'type':10s} {'ntens':>6s} {'size':>12s} {'%':>6s} {'bpw':>6s}")
    for name, count in by_type.most_common():
        tsize = type_bytes[name]
        elems = type_elems[name]
        bpw = tsize * 8 / elems if elems else 0.0
        print(
            f" {name:10s} {count:6d} {fmt_size(tsize):>12s} "
            f"{tsize / total_bytes * 100:5.1f}% {bpw:6.2f}"
        )
    print(
        f" {'TOTAL':10s} {sum(by_type.values()):6d} {fmt_size(total_bytes):>12s} "
        f" {100.0:5.1f}% {total_bytes * 8 / total_elems:6.2f}"
    )

    print()
    print("=" * 78)
    print(" data_offset semantics check")
    print("=" * 78)
    for t in reader.tensors[:6]:
        shape = "x".join(str(int(s)) for s in reversed(t.shape))
        print(
            f" {t.name:40s} {QT_NAME.get(t.tensor_type,'?'):8s} {shape:>18s} "
            f"data_offset={t.data_offset:>12,d} n_bytes={t.n_bytes:>12,d} "
            f"data_len={len(t.data):>12,d}"
        )
    offsets = {t.data_offset for t in reader.tensors}
    print(f" distinct data_offset values: {len(offsets)}")
    if len(offsets) == 1:
        print(" -> ATTENZIONE: data_offset e' identico per tutti i tensor,")
        print("    NON e' l'offset del singolo tensor (vedi t.data per la slice).")

    print()
    print("=" * 78)
    print(" MoE EXPERT TENSORS (blk.0)")
    print("=" * 78)
    print(f" {'name':36s} {'type':8s} {'shape':>20s} {'n_bytes':>12s}")
    for t in reader.tensors:
        if not t.name.startswith("blk.0.ffn"):
            continue
        name = QT_NAME.get(t.tensor_type, str(t.tensor_type))
        shape = "x".join(str(int(s)) for s in reversed(t.shape))
        print(f" {t.name:36s} {name:8s} {shape:>20s} {fmt_size(t.n_bytes):>12s}")

    print()
    print("=" * 78)
    print(" ALL TENSOR NAMES (first 40)")
    print("=" * 78)
    for t in reader.tensors[:40]:
        name = QT_NAME.get(t.tensor_type, str(t.tensor_type))
        shape = "x".join(str(int(s)) for s in reversed(t.shape))
        print(f" {t.name:44s} {name:8s} {shape:>18s}")

    print()
    print("=" * 78)
    print(" EXPERT SLICE VALIDITY CHECK")
    print("=" * 78)
    n_expert = int(field_value(reader, f"{arch}.expert_count"))
    print(f" expert_count = {n_expert}")

    for t in reader.tensors:
        if not t.name.startswith("blk.0.ffn"):
            continue
        name = QT_NAME.get(t.tensor_type, str(t.tensor_type))
        if name not in TYPE_INFO:
            print(f" {t.name:36s} {name:8s} -> tipo non mappato in TYPE_INFO")
            continue
        bs, bpb = TYPE_INFO[name]
        n_elem = int(t.n_elements)
        per_expert_elems = n_elem / n_expert if n_expert else 0
        divisible = n_elem % n_expert == 0 and per_expert_elems % bs == 0
        per_expert_bytes = t.n_bytes // n_expert if n_expert else 0
        block_ok = per_expert_bytes % bpb == 0
        print(
            f" {t.name:36s} {name:8s} n_elem={n_elem:>10,d} "
            f"elem/expert={int(per_expert_elems):>9,d} per_expert={per_expert_bytes:>10,d} B  "
            f"elem_div={divisible}  block_aligned={block_ok}"
        )

    print()
    print("=" * 78)
    print(" LAYER -> QUANT TYPE MATRIX (MoE + attn)")
    print("=" * 78)
    patterns: dict[tuple[str, str], list[int]] = {}
    for t in reader.tensors:
        if not t.name.startswith("blk."):
            continue
        parts = t.name.split(".")
        if len(parts) < 3:
            continue
        try:
            layer = int(parts[1])
        except ValueError:
            continue
        role = parts[2]
        name = QT_NAME.get(t.tensor_type, str(t.tensor_type))
        patterns.setdefault((role, name), []).append(layer)

    for (role, name), layers in sorted(patterns.items()):
        uniq = sorted(set(layers))
        if len(uniq) == 1:
            span = f"L{uniq[0]}"
        else:
            # elenca i cambi
            runs: list[str] = []
            start = prev = uniq[0]
            for value in uniq[1:]:
                if value == prev + 1:
                    prev = value
                    continue
                runs.append(f"L{start}" if start == prev else f"L{start}-{prev}")
                start = prev = value
            runs.append(f"L{start}" if start == prev else f"L{start}-{prev}")
            span = ", ".join(runs)
        print(f" {role:24s} {name:8s} ntens={len(layers):3d}  {span}")


if __name__ == "__main__":
    main()
