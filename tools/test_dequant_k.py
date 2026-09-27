"""Gate numerico dei kernel C++ K-quant.

Ogni kernel viene confrontato, bit per bit sul primo blocco e con tolleranza
sull'intero, contro la reference gguf.quants (la stessa di llama.cpp). Un
kernel che sbaglia un indice fallisce qui, prima di entrare nel path di
inference. E' esattamente il gate che mancava a fast_dequant_q8_0.dll, che era
"2x piu' veloce e produceva NaN".

Uso:
    python tools/test_dequant_k.py                 # sintetico, se la DLL c'e'
    python tools/test_dequant_k.py --gguf <path>   # + tensor reali
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gguf  # noqa: E402

from flash_moe.dequant import GGML_BLOCK_INFO, dequantize, name_of  # noqa: E402

CANDIDATES = [
    ROOT / "src" / "flash_moe" / "fast_dequant_k.dll",
    ROOT / "flash_moe" / "fast_dequant_k.dll",
    ROOT / "build" / "fast_dequant_k.dll",
]

TYPES = ["Q8_0", "Q3_K", "Q4_K", "Q5_K", "Q6_K"]

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAILED.append((name, detail))
        print(f"  FAIL  {name}  {detail}")


def load_dll() -> ctypes.CDLL | None:
    for path in CANDIDATES:
        if not path.exists():
            continue
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as err:
            print(f"  [DLL non caricabile {path.name}: {err}]")
            continue
        for name in ("dequant_k", "dequant_k_supported", "dequant_k_block_size"):
            getattr(lib, name)
        lib.dequant_k.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int,
        ]
        lib.dequant_k.restype = ctypes.c_int
        lib.dequant_k_supported.argtypes = [ctypes.c_int]
        lib.dequant_k_supported.restype = ctypes.c_int
        lib.dequant_k_block_size.argtypes = [ctypes.c_int]
        lib.dequant_k_block_size.restype = ctypes.c_int
        print(f"  DLL: {path}")
        return lib
    return None


def synth_blocks(type_name: str, n_blocks: int) -> np.ndarray:
    """Byte pseudo-casuali ma con la struttura giusta per il tipo.

    Non e' una quantizzazione valida, e va bene: lo scopo e' esercitare la
    decodifica degli indici, non la distribuzione dei valori.
    """
    bs, bpb = GGML_BLOCK_INFO[type_name]
    rng = np.random.default_rng(abs(hash(type_name)) % (2**32))
    data = rng.integers(0, 256, size=n_blocks * bpb, dtype=np.uint8)

    def put_f16(data: np.ndarray, offset: int, value: float) -> None:
        data[offset:offset + 2] = np.array([value], dtype=np.float16).view(np.uint8)

    if type_name == "Q8_0":
        for b in range(n_blocks):
            put_f16(data, b * bpb, float(rng.uniform(0.001, 0.05)))
    elif type_name in ("Q3_K", "Q4_K", "Q5_K"):
        for b in range(n_blocks):
            blk = b * bpb
            put_f16(data, blk, float(rng.uniform(0.01, 0.5)))
            put_f16(data, blk + 2, float(rng.uniform(0.0, 0.2)))
    elif type_name == "Q6_K":
        for b in range(n_blocks):
            put_f16(data, b * bpb + 208, float(rng.uniform(0.01, 0.5)))
    return data


def compare(lib: ctypes.CDLL, type_name: str, data: np.ndarray, label: str) -> None:
    type_id = int(getattr(gguf.GGMLQuantizationType, type_name))
    bs, bpb = GGML_BLOCK_INFO[type_name]
    n = data.nbytes // bpb * bs

    check(f"{label}: {type_name} supportato", bool(lib.dequant_k_supported(type_id)))
    check(f"{label}: {type_name} block size", lib.dequant_k_block_size(type_id) == bs,
          f"dichiarato {lib.dequant_k_block_size(type_id)}, atteso {bs}")

    reference = dequantize(data.tobytes(), type_id)

    out32 = np.empty(n, dtype=np.float32)
    rc = lib.dequant_k(type_id, data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                       out32.ctypes.data, n, 0)
    check(f"{label}: {type_name} dequant_k f32 ritorna 0", rc == 0, f"rc={rc}")

    got32 = out32.astype(np.float64)
    ref32 = reference.astype(np.float64)
    nan_count = int(np.isnan(got32).sum())
    check(f"{label}: {type_name} f32 senza NaN", nan_count == 0, f"nan={nan_count}")

    if nan_count == 0:
        # il primo blocco deve combaciare esattamente: se un indice e' sbagliato
        # lo sbaglio si vede subito e localizza il problema
        first_ok = np.array_equal(got32[:bs], ref32[:bs])
        check(f"{label}: {type_name} f32 primo blocco identico", first_ok,
              "" if first_ok else
              f"max diff {np.abs(got32[:bs] - ref32[:bs]).max():.3e}")

        scale = float(np.abs(ref32).max())
        rel = float(np.abs(got32 - ref32).max() / scale) if scale > 0 else 0.0
        check(f"{label}: {type_name} f32 entro 1e-6", rel < 1e-6, f"rel={rel:.3e}")

    out16 = np.empty(n, dtype=np.uint16)
    rc = lib.dequant_k(type_id, data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                       out16.ctypes.data, n, 1)
    check(f"{label}: {type_name} dequant_k f16 ritorna 0", rc == 0, f"rc={rc}")
    back = out16.view(np.float16).astype(np.float32)
    scale = float(np.abs(reference).max())
    rel16 = float(np.abs(back.astype(np.float64) - ref32).max() / scale) if scale > 0 else 0.0
    check(f"{label}: {type_name} f16 entro 1e-3", rel16 < 1e-3, f"rel={rel16:.3e}")

    rc = lib.dequant_k(type_id, data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                       out32.ctypes.data, n + 1, 0)
    check(f"{label}: {type_name} rifiuta n non multiplo di blocco", rc == -1, f"rc={rc}")


def test_synthetic(lib: ctypes.CDLL) -> None:
    print("\n1. kernel su blocchi sintetici (8 blocchi per tipo)")
    for type_name in TYPES:
        data = synth_blocks(type_name, 8)
        compare(lib, type_name, data, "sintetico")


def test_roundtrip(lib: ctypes.CDLL) -> None:
    print("\n2. kernel contro la reference su tensor quantizzati veri")
    import gguf as _g

    rng = np.random.default_rng(1)
    for type_name in TYPES:
        type_id = int(getattr(_g.GGMLQuantizationType, type_name))
        quants = getattr(_g.quants, type_name)
        bs, _ = GGML_BLOCK_INFO[type_name]
        values = (rng.standard_normal(bs * 4) * 0.05).astype(np.float32)
        try:
            blocks = quants.quantize(values)
        except Exception as err:
            print(f"  SKIP  {type_name}: quantize non disponibile ({err})")
            continue
        data = np.ascontiguousarray(blocks).view(np.uint8).reshape(-1)
        compare(lib, type_name, data, f"quantizzato/{type_name}")


def test_gguf(lib: ctypes.CDLL, gguf_path: Path) -> None:
    print(f"\n3. kernel su tensor reali di {gguf_path.name}")

    reader = gguf.GGUFReader(gguf_path)
    tmap = {t.name: t for t in reader.tensors}
    arch = str(reader.fields["general.architecture"].parts[-1].tobytes().decode())

    def field(name):
        part = reader.fields[name].parts[-1]
        if part.dtype == np.uint8:
            return part.tobytes().decode()
        return int(part[0])

    n_expert = int(field(f"{arch}.expert_count"))
    checked = 0
    with gguf_path.open("rb") as fh:
        for name, t in tmap.items():
            type_name = name_of(t.tensor_type)
            if type_name not in TYPES:
                continue
            if "ffn" not in name and "ssm" not in name and "attn" not in name:
                continue
            per = max(1, t.n_bytes // max(1, n_expert)) if "_exps" in name else t.n_bytes
            if per > 8 * 1024 * 1024:
                per = 8 * 1024 * 1024
            fh.seek(t.data_offset)
            raw = fh.read(per)
            data = np.frombuffer(raw, dtype=np.uint8)
            bs, bpb = GGML_BLOCK_INFO[type_name]
            n = data.nbytes // bpb * bpb
            if n == 0:
                continue
            compare(lib, type_name, data[:n], f"reale/{name}")
            checked += 1
            if checked >= 12:
                break
    print(f"  tensor reali verificati: {checked}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, default=None)
    args = parser.parse_args()

    print("=" * 78)
    print(" GATE NUMERICO KERNEL K-QUANT")
    print("=" * 78)

    lib = load_dll()
    if lib is None:
        print("\nNessuna DLL trovata. Compila con:")
        print("  scripts\\compile_dequant_k.bat")
        return 2

    for fn in (test_synthetic, test_roundtrip):
        try:
            fn(lib)
        except Exception:
            import traceback
            FAILED.append((fn.__name__, "eccezione"))
            traceback.print_exc()

    gguf_path = args.gguf
    if gguf_path is None:
        candidates = sorted(ROOT.glob("*.gguf"))
        if candidates:
            gguf_path = candidates[0]
    if gguf_path and gguf_path.exists():
        try:
            test_gguf(lib, gguf_path)
        except Exception:
            import traceback
            FAILED.append(("test_gguf", "eccezione"))
            traceback.print_exc()
    else:
        print("\n nessun GGUF: salta i test su tensor reali")

    print()
    print("=" * 78)
    print(f" PASSATI: {len(PASSED)}   FALLITI: {len(FAILED)}")
    for name, detail in FAILED:
        print(f"   - {name}: {detail}")
    print("=" * 78)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
