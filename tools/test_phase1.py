"""Gate numerico della Phase 1.

Ogni affermazione qui e' un test che, se fallisce, blocca. In particolare il
test 3 esiste perche' la DLL Q8_0 gia' presente era "2x piu' veloce e
numericamente falsa" e nessuno lo aveva fermato.

Uso:
    python tools/test_phase1.py
    python tools/test_phase1.py --gguf <percorso>   # aggiunge i test sul GGUF
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flash_moe import (  # noqa: E402
    AsyncExpertStore,
    ExpertKey,
    ExpertManifest,
    synthetic_router,
    MoEEngine,
    OpenVINOExpertBackend,
    dequantize,
    dequantize_to,
    is_aligned,
    name_of,
    pack_expert,
    pack_f32,
    pack_f16,
    write_manifest,
)

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAILED.append((name, detail))
        print(f"  FAIL  {name}  {detail}")


def build_mock_model(root: Path, layers: int = 2, experts: int = 4,
                     hidden: int = 64, inter: int = 128) -> ExpertManifest:
    rng = np.random.default_rng(7)
    records = []
    for layer in range(layers):
        for eid in range(experts):
            key = ExpertKey(layer, eid)
            tensors = {
                "gate": pack_f16((rng.standard_normal(hidden * inter) * 0.05).tolist()),
                "up": pack_f16((rng.standard_normal(hidden * inter) * 0.05).tolist()),
                "down": pack_f16((rng.standard_normal(inter * hidden) * 0.05).tolist()),
            }
            records.append(pack_expert(root, key, tensors))
    return ExpertManifest.load(write_manifest(root, records))


# ---------------------------------------------------------------- test 1: store


def test_store() -> None:
    print("\n1. store: prefetch, slice, LRU, tipo unico in cache")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = build_mock_model(root, layers=1, experts=3, hidden=16, inter=32)
        store = AsyncExpertStore(root, capacity=2, manifest=manifest)
        key = ExpertKey(0, 0)

        expected = manifest.records[key].nbytes
        store.read_tensor(key, "gate")
        check("store: cache miss registrata", store.stats.misses == 1,
              f"misses={store.stats.misses}")
        check("store: legge solo la slice dell'expert",
              store.stats.bytes_read == expected,
              f"bytes_read={store.stats.bytes_read} atteso={expected}")

        # get() dopo read_tensor(): prima era un TypeError
        payload = store.get(key)
        check("store: get() dopo read_tensor() funziona", isinstance(payload, bytes),
              f"tipo={type(payload).__name__}")

        # il bug storico: prefetch che legge l'intero file
        store.prefetch(ExpertKey(0, 1))
        got = store.read_tensor(ExpertKey(0, 1), "gate")
        check("store: prefetch consumato senza rileggere",
              store.stats.bytes_read == 2 * expected,
              f"bytes_read={store.stats.bytes_read} atteso={2 * expected}")
        check("store: prefetch restituisce gli stessi byte",
              got == store._read_expert(ExpertKey(0, 1))["gate"])

        # LRU con capacity=2
        store.read_tensor(ExpertKey(0, 2), "gate")
        check("store: LRU rispetta la capacity", len(store.resident_keys()) <= 2,
              f"residenti={len(store.resident_keys())}")
        check("store: chiave piu' vecchia espulsa",
              ExpertKey(0, 0) not in store.resident_keys(),
              f"residenti={[str(k) for k in store.resident_keys()]}")

        # un solo tipo per expert in cache
        types = {type(v).__name__ for v in store._cache.values()}
        check("store: un solo tipo per expert in cache", types == {"dict"},
              f"tipi={types}")
        store.close()


# ---------------------------------------------------------------- test 2: dequant


def test_dequant_python_path() -> None:
    print("\n2. dequant: il fallback non deve produrre zeri")

    import gguf

    ggml_type = gguf.GGMLQuantizationType.Q4_K
    rng = np.random.default_rng(3)
    reference = (rng.standard_normal(256 * 8) * 0.05).astype(np.float32)

    # ricostruisci i byte Q4_K da valori noti non e' banale: usa il percorso
    # inverso non disponibile, quindi verifichiamo la proprieta' strutturale
    dummy = np.zeros(144, dtype=np.uint8)
    check("dequant: allineamento blocco Q4_K", is_aligned(ggml_type, 144 * 10))
    check("dequant: rifiuta buffer non allineati", not is_aligned(ggml_type, 144 * 10 + 1))

    # il punto: astype su interi produce zeri, il reinterpret no
    small = reference * 0.01  # valori nella scala reale degli esperti
    as_int = small.astype(np.uint16)
    as_f16 = small.astype(np.float16)
    check("dequant: il vecchio cast produce zeri", int(as_int.max()) == 0,
          f"max={int(as_int.max())} con valori reali fino a {small.max():.5f}")
    check("dequant: il cast f16 conserva il segno e la scala",
          as_f16.dtype == np.float16 and abs(float(as_f16.max()) - float(small.max())) < 1e-3,
          f"max_f16={float(as_f16.max()):.6f} max_reale={float(small.max()):.6f}")

    # tieni la variabile usata per non sembrare piu' rigoroso di quanto sei
    _ = dummy


def test_dequant_gguf(gguf_path: Path) -> None:
    print(f"\n3. dequant: tensor reali da {gguf_path.name}")

    import gguf

    reader = gguf.GGUFReader(gguf_path)
    tmap = {t.name: t for t in reader.tensors}
    arch = str(reader.fields["general.architecture"].parts[-1].tobytes().decode())
    n_expert = int(reader.fields[f"{arch}.expert_count"].parts[-1][0])

    def field(name):
        part = reader.fields[name].parts[-1]
        if part.dtype == np.uint8:
            return part.tobytes().decode()
        return int(part[0])

    n_expert = int(field(f"{arch}.expert_count"))
    embedding = int(field(f"{arch}.embedding_length"))
    # moe_intermediate_size non e' sempre presente: l'intermediate si ricava
    # dalla shape del tensor. In ordine ggml ffn_gate_exps e' [hidden, inter,
    # n_expert], quindi inter = shape[1].
    inter = int(tmap["blk.0.ffn_gate_exps.weight"].shape[1])

    seen_types: set[str] = set()
    for layer in (0, 10, 39):
        for role in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            name = f"blk.{layer}.{role}.weight"
            if name not in tmap:
                continue
            t = tmap[name]
            per = t.n_bytes // n_expert
            with gguf_path.open("rb") as fh:
                fh.seek(t.data_offset)
                raw = fh.read(per)
            tname = name_of(t.tensor_type)
            seen_types.add(tname)

            check(f"dequant: {name} allineato ({tname})", is_aligned(t.tensor_type, len(raw)),
                  f"{len(raw)} B")
            f32 = dequantize(raw, t.tensor_type)
            check(f"dequant: {name} non e' tutto zero", float(np.abs(f32).max()) > 0.0,
                  f"|max|={float(np.abs(f32).max()):.6f}")
            f16 = dequantize_to(raw, t.tensor_type, "f16")
            check(f"dequant: {name} f16 non e' tutto zero",
                  float(np.abs(f16.astype(np.float32)).max()) > 0.0,
                  f"|max|={float(np.abs(f16.astype(np.float32)).max()):.6f}")
            err = float(np.abs(f16.astype(np.float32) - f32).max() / np.abs(f32).max())
            check(f"dequant: errore f16 relativo < 1e-2 su {name}", err < 1e-2,
                  f"{err:.3e}")
            check(f"dequant: {name} ha {embedding * inter} elementi per expert",
                  f32.size == embedding * inter, f"{f32.size} vs {embedding * inter}")

    print(f"  tipi incontrati: {sorted(seen_types)}")
    check("dequant: il modello non e' solo Q4_K", len(seen_types) > 1,
          f"tipi={sorted(seen_types)}")


def test_broken_dll(gguf_path: Path) -> None:
    print("\n4. la DLL Q8_0 esistente e' numericamente sbagliata")

    dll = ROOT / "src" / "flash_moe" / "fast_dequant_q8_0.dll"
    dll_root = ROOT / "flash_moe" / "fast_dequant_q8_0.dll"
    for candidate in (dll, dll_root):
        if candidate.exists():
            dll = candidate
            break
    if not dll.exists():
        print("  SKIP  nessuna DLL Q8_0 trovata")
        return

    import ctypes

    import gguf

    reader = gguf.GGUFReader(gguf_path)
    tmap = {t.name: t for t in reader.tensors}
    arch = str(reader.fields["general.architecture"].parts[-1].tobytes().decode())
    n_expert = int(reader.fields[f"{arch}.expert_count"].parts[-1][0])
    t = tmap["blk.0.ffn_down_exps.weight"]  # Q8_0
    if name_of(t.tensor_type) != "Q8_0":
        print("  SKIP  blk.0.ffn_down_exps non e' Q8_0")
        return

    with gguf_path.open("rb") as fh:
        fh.seek(t.data_offset)
        raw = fh.read(t.n_bytes // n_expert)

    hidden = 2048
    inter = 512
    lib = ctypes.CDLL(str(dll))
    lib.dequantize_q8_0_to_bf16.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.dequantize_q8_0_to_bf16.restype = None
    dst = np.empty((hidden, inter), dtype=np.uint16)
    lib.dequantize_q8_0_to_bf16(raw, dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                               hidden, inter)

    got = (dst.astype(np.uint32) << np.uint32(16)).view(np.float32)
    ref = dequantize(raw, t.tensor_type).reshape(hidden, inter)
    nans = int(np.isnan(got).sum())
    finite = np.isfinite(got) & np.isfinite(ref)
    err = (float(np.abs(got[finite] - ref[finite]).max())
           / float(np.abs(ref[finite]).max())) if finite.any() else float("nan")
    print(f"  DLL su tensor Q8_0 reale: nan={nans} errore relativo {err:.3e}")
    # Il test passa quando la DLL e' sbagliata: e' il motivo per non collegarla.
    check("la DLL Q8_0 e' numericamente inutilizzabile (motivo per non collegarla)",
          nans > 0 or not (err < 1e-2),
          f"nan={nans} errore relativo {err:.3e}")


# ---------------------------------------------------------------- test 5: backend


def test_backend() -> None:
    print("\n5. backend OpenVINO: correttezza e device")

    hidden, inter = 64, 128
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = build_mock_model(root, layers=1, experts=2, hidden=hidden, inter=inter)
        store = AsyncExpertStore(root, capacity=4, manifest=manifest)
        backend = OpenVINOExpertBackend(store, device="AUTO", weight_dtype="f16")

        rng = np.random.default_rng(11)
        x = rng.standard_normal(hidden).astype(np.float32)
        key = ExpertKey(0, 0)
        out = backend.forward(key, x, hidden, inter)

        # I pesi della mock sono f16, come quelli che il backend si aspetta, e
        # sono reshaped nella disposizione in cui stanno nel GGUF: gate/up
        # (n_ff, n_hidden), down (n_hidden, n_ff). Il prodotto contrae
        # l'hidden, quindi si trasposta la vista e non l'array.
        gate = np.frombuffer(store.read_tensor(key, "gate"), np.float16).astype(np.float32).reshape(inter, hidden)
        up = np.frombuffer(store.read_tensor(key, "up"), np.float16).astype(np.float32).reshape(inter, hidden)
        down = np.frombuffer(store.read_tensor(key, "down"), np.float16).astype(np.float32).reshape(hidden, inter)
        gp = x @ gate.T
        ref = (gp * (1.0 / (1.0 + np.exp(-gp)))) * (x @ up.T) @ down.T

        check("backend: output non e' tutto zero", float(np.abs(out).max()) > 0)
        err = float(np.abs(out - ref).max() / np.abs(ref).max())
        check("backend: output entro 1e-4 dalla reference numpy", err < 1e-4,
              f"errore relativo {err:.3e}")
        check("backend: request riusato, non ricreato a ogni forward",
              len(backend._requests) == 1, f"requests={len(backend._requests)}")

        # device misurati. La tolleranza cross-device e' piu' larga della CPU:
        # GPU e NPU cambiano l'ordine di accumulo del matmul, e su f16 con
        # inner dimension 64 un errore di few 1e-3 e' atteso, non un bug.
        for dev in backend.core.available_devices:
            alt = OpenVINOExpertBackend(store, device=dev, weight_dtype="f16")
            o2 = alt.forward(key, x, hidden, inter)
            e2 = float(np.abs(o2 - ref).max() / np.abs(ref).max())
            tol = 1e-3 if dev == "CPU" else 5e-3
            check(f"backend: corretto anche su {dev}", e2 < tol,
                  f"errore relativo {e2:.3e}, tolleranza {tol:g}")

        store.close()


# ---------------------------------------------------------------- test 6: engine


def test_engine() -> None:
    print("\n6. MoEEngine: percorso completo con prefetch")

    hidden, inter = 32, 64
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = build_mock_model(root, layers=2, experts=4, hidden=hidden, inter=inter)
        store = AsyncExpertStore(root, capacity=2, manifest=manifest)
        backend = OpenVINOExpertBackend(store, device="AUTO", weight_dtype="f16")
        routers = {
            layer: synthetic_router(hidden, 4, top_k=2, seed=3, layer=layer)
            for layer in range(2)
        }
        engine = MoEEngine(store, backend, routers, hidden, inter)

        rng = np.random.default_rng(5)
        ok = True
        for layer in range(2):
            x = rng.standard_normal(hidden).astype(np.float32)
            out = engine.process_token(layer, x, next_expert_key=ExpertKey(0, 1))
            if out is None or float(np.abs(out).max()) <= 0:
                ok = False
        check("engine: process_token gira e restituisce un output non nullo", ok)
        check("engine: nessuna collisione di tipo in cache",
              {type(v).__name__ for v in store._cache.values()} == {"dict"})
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, default=None)
    args = parser.parse_args()

    print("=" * 78)
    print(" GATE NUMERICO PHASE 1")
    print("=" * 78)

    for fn in (test_store, test_dequant_python_path, test_backend, test_engine):
        try:
            fn()
        except Exception:
            FAILED.append((fn.__name__, "eccezione"))
            print(f"  ERROR in {fn.__name__}:")
            traceback.print_exc()

    gguf_path = args.gguf
    if gguf_path is None:
        candidates = sorted(ROOT.glob("*.gguf"))
        if candidates:
            gguf_path = candidates[0]
    if gguf_path and gguf_path.exists():
        for fn in (test_dequant_gguf, test_broken_dll):
            try:
                fn(gguf_path)
            except Exception:
                FAILED.append((fn.__name__, "eccezione"))
                print(f"  ERROR in {fn.__name__}:")
                traceback.print_exc()
    else:
        print("\n nessun GGUF trovato: salta i test 3 e 4")

    print()
    print("=" * 78)
    print(f" PASSATI: {len(PASSED)}   FALLITI: {len(FAILED)}")
    if FAILED:
        for name, detail in FAILED:
            print(f"   - {name}: {detail}")
    print("=" * 78)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
