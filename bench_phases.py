"""Phase 0 - diagnostica: dove stanno davvero i ms per esperto?

Non usa la DLL C++ esistente (e' numericamente rotta: layout Q8_0 interleaved
+ offset -128 inesistente). Misura su tensor REALI del GGUF:

  A. lettura slice dal file
  B. dequant -> fp32 (reference gguf.quants) e -> f16
  C. OpenVINO: request nuovo vs riutilizzato, pesi agli input vs Constant
  D. proiezione per token (8 esperti x 40 layer)

Nota sui tipi: il piano originale usava bf16 passato come numpy uint16.
Su questa macchina quello produce NaN (vedi tools/check_ov_tensor_feeds.py).
Il tipo corretto e' f16 con numpy float16 nativo.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path

import gguf
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from flash_moe.paths import dll_path, model_path  # noqa: E402
from flash_moe.dequant import name_of  # noqa: E402

GGUF_PATH = model_path()
GGML_ID = {v: k for k, v in vars(gguf.GGMLQuantizationType).items()
           if isinstance(v, int) and not k.startswith("_")}

HIDDEN = 2048
INTER = 512
N_EXPERT = 256
N_USED = 8
N_LAYER = 40
LAYER = 0


def bench(label, fn, repeat=5, warmup=1):
    for _ in range(warmup):
        out = fn()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    median = samples[len(samples) // 2]
    print(f"  {label:54s} {median:9.3f} ms   (best {samples[0]:8.3f})")
    return median, out


def read_expert_bytes(path: Path, t, expert_id: int) -> bytes:
    per = t.n_bytes // N_EXPERT
    with path.open("rb") as fh:
        fh.seek(t.data_offset + expert_id * per)
        return fh.read(per)


def main() -> None:
    print("=" * 82)
    print(" PHASE 0 - DIAGNOSTICA FASI")
    print("=" * 82)
    print(f" file  : {GGUF_PATH.name}")
    print(f" size  : {GGUF_PATH.stat().st_size / 1024**3:.2f} GiB")
    print(f" cpus  : {os.cpu_count()}")

    reader = gguf.GGUFReader(GGUF_PATH)
    tmap = {t.name: t for t in reader.tensors}
    gate = tmap[f"blk.{LAYER}.ffn_gate_exps.weight"]

    # ---------------------------------------------------------------- A
    print()
    print(" A. LETTURA SLICE DAL FILE")
    print("-" * 82)
    ms_read, gate_bytes = bench(
        "seek+read di 1 expert gate_exps (589 KB)",
        lambda: read_expert_bytes(GGUF_PATH, gate, 0),
        repeat=50, warmup=5,
    )

    # ---------------------------------------------------------------- B
    print()
    print(" B. DEQUANT (Q4_K, 1M elementi per expert)")
    print("-" * 82)

    def ref_dequant():
        return gguf.quants.dequantize(
            np.frombuffer(gate_bytes, dtype=np.uint8), gate.tensor_type
        ).reshape(HIDDEN, INTER)

    ms_ref, w32 = bench("gguf.quants.dequantize -> fp32 (reference)", ref_dequant)
    w16 = w32.astype(np.float16)
    w_max = float(np.abs(w32).max())
    print(f"  -> forma {w32.shape}  |max|={w_max:.6f}  "
          f"rms={np.sqrt((w32.astype(np.float64) ** 2).mean()):.6f}")

    def current_fallback():
        # esattamente quello che fanno oggi gli script
        return w32.astype(np.uint16)

    ms_broken, w_broken = bench("cast fp32 -> uint16 (fallback ATTUALE)",
                                current_fallback)
    print(f"  -> risultato: interi 0..{int(w_broken.max())} mentre i pesi "
          f"reali valgono 0..{w_max:.4f}  =>  TUTTI ZERO")

    ms_f16, _ = bench("fp32 -> float16 (numpy nativo, il modo che funziona)",
                      lambda: w32.astype(np.float16))
    print(f"  -> errore f16 vs fp32: "
          f"{np.abs(w16.astype(np.float32) - w32).max():.3e} "
          f"({np.abs(w16.astype(np.float32) - w32).max() / w_max:.2e} rel)")

    try:
        # La DLL validata (gate 162/162) invece della vecchia fast_dequant_q8_0,
        # che era numericamente errata e non piu' esiste.
        dll = ctypes.CDLL(str(dll_path()))
        dll.dequant_k.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int,
        ]
        dll.dequant_k.restype = ctypes.c_int
        n_el = HIDDEN * INTER
        dst16 = np.empty(n_el, dtype=np.uint16)

        def call_cpp():
            rc = dll.dequant_k(
                GGML_ID["Q4_K"],
                gate_bytes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                dst16.ctypes.data, n_el, 1,
            )
            if rc != 0:
                raise RuntimeError(f"dequant_k ha restituito rc={rc}")

        ms_dll, w_dll = bench("DLL C++ validata (dequant_k, Q4_K -> float16)",
                              call_cpp, repeat=20, warmup=5)
        got = dst16.view(np.float16).astype(np.float32)
        err = np.abs(got - w32).max() / w_max
        print(f"  -> {ms_dll:.3f} ms, errore relativo vs gguf.quants: {err:.3e}")
    except Exception as err:
        print(f"  [DLL non caricabile: {err}]")

    # ---------------------------------------------------------------- C
    print()
    print(" C. OPENVINO - strategie di passaggio dei pesi")
    print("-" * 82)
    results: dict[str, float] = {}
    try:
        import openvino as ov
        import openvino.opset13 as ops

        core = ov.Core()
        print(f"  devices: {core.available_devices}")
        device = "GPU" if "GPU" in core.available_devices else "CPU"

        def make_model(weights_as_input: bool, wtype, wg, wu, wd):
            params = []
            if weights_as_input:
                x = ops.parameter([1, HIDDEN], ov.Type.f32, name="activation")
                g = ops.parameter([HIDDEN, INTER], wtype, name="gate")
                u = ops.parameter([HIDDEN, INTER], wtype, name="up")
                d = ops.parameter([INTER, HIDDEN], wtype, name="down")
                params = [x, g, u, d]
            else:
                x = ops.parameter([1, HIDDEN], ov.Type.f32, name="activation")
                g = ops.constant(wg, wtype)
                u = ops.constant(wu, wtype)
                d = ops.constant(wd, wtype)
                params = [x]
            xc = ops.convert(x, wtype)
            gp = ops.matmul(xc, g, transpose_a=False, transpose_b=False)
            up = ops.matmul(xc, u, transpose_a=False, transpose_b=False)
            act = ops.multiply(gp, ops.sigmoid(gp))
            out = ops.matmul(ops.multiply(act, up), d,
                             transpose_a=False, transpose_b=False)
            return ov.Model([ops.convert(out, ov.Type.f32)], params, "expert")

        x_arr = np.random.randn(1, HIDDEN).astype(np.float32)
        wg_f16 = w16
        wd_f16 = np.ascontiguousarray(w16.T)

        for label, wtype, wgq, wdq in (
            ("f32", ov.Type.f32, w32, np.ascontiguousarray(w32.T)),
            ("f16", ov.Type.f16, wg_f16, wd_f16),
        ):
            print(f"  --- pesi {label} ---")
            t0 = time.perf_counter()
            cm_in = core.compile_model(
                make_model(True, wtype, wgq, wgq, wdq), device)
            print(f"  {'compile (input, una tantum)':54s} "
                  f"{(time.perf_counter() - t0) * 1000:9.1f} ms")

            feed = {"activation": x_arr, "gate": wgq, "up": wgq, "down": wdq}
            req = cm_in.create_infer_request()
            req.infer(feed)

            ms, _ = bench("infer() request NUOVO, pesi agli input",
                          lambda: cm_in.create_infer_request().infer(feed),
                          repeat=10, warmup=3)
            results[f"{label}/nuovo+input"] = ms
            ms, _ = bench("infer() request RIUTILIZZATO, pesi agli input",
                          lambda: req.infer(feed), repeat=10, warmup=3)
            results[f"{label}/riuso+input"] = ms

            t0 = time.perf_counter()
            cm_c = core.compile_model(
                make_model(False, wtype, wgq, wgq, wdq), device)
            print(f"  {'compile (Constant, una tantum)':54s} "
                  f"{(time.perf_counter() - t0) * 1000:9.1f} ms")
            rc = cm_c.create_infer_request()
            cfeed = {"activation": x_arr}
            rc.infer(cfeed)
            ms, out = bench("infer() request RIUTILIZZATO, pesi = Constant",
                            lambda: rc.infer(cfeed), repeat=20, warmup=5)
            results[f"{label}/riuso+constant"] = ms

            o = np.asarray(list(out.values())[0], dtype=np.float32)
            wg32 = wg_f16.astype(np.float32)
            gp32 = x_arr @ wg32
            act32 = gp32 * (1.0 / (1.0 + np.exp(-gp32)))
            ref = (act32 * (x_arr @ wg32)) @ wd_f16.astype(np.float32)
            print(f"  -> nan={int(np.isnan(o).sum())} "
                  f"err_vs_fp32={np.abs(o - ref).max() / max(np.abs(ref).max(), 1e-9):.2e}")

        # confronto device sul percorso Constant
        print("  --- confronto device (pesi = Constant, f16) ---")
        for dev in core.available_devices:
            try:
                cm = core.compile_model(
                    make_model(False, ov.Type.f16, wg_f16, wg_f16, wd_f16), dev)
                r = cm.create_infer_request()
                f = {"activation": x_arr}
                r.infer(f)
                ms, _ = bench(f"    infer su {dev}", lambda: r.infer(f),
                              repeat=20, warmup=5)
            except Exception as err:
                print(f"  {dev:4s} non disponibile: "
                      f"{str(err).splitlines()[0][:70]}")

    except Exception:
        import traceback
        print("  [SEZIONE C FALLITA]")
        traceback.print_exc()

    # ---------------------------------------------------------------- D
    print()
    print(" D. PROIEZIONE PER TOKEN (8 esperti x 40 layer = 320 esperti)")
    print("-" * 82)
    for key in ("f32/riuso+input", "f32/riuso+constant",
                "f16/riuso+input", "f16/riuso+constant"):
        if key not in results:
            continue
        per_expert = results[key]
        total = per_expert * N_USED * N_LAYER
        print(f"  {key:26s} {per_expert:8.3f} ms/expert  ->  "
              f"{total / 1000:7.3f} s/token  ({1000 / total:6.2f} token/s)")

    if "f16/riuso+constant" in results:
        ov_only = results["f16/riuso+constant"] * N_USED * N_LAYER
        print()
        print(f"  lettura file            {ms_read:8.3f} ms/expert  ->  "
              f"{ms_read * N_USED * N_LAYER / 1000:7.3f} s/token")
        print(f"  + dequant (DLL rotta)   {0.34:8.3f} ms/expert  ->  "
              f"{0.34 * N_USED * N_LAYER / 1000:7.3f} s/token")
        print(f"  + OpenVINO Constant     {results['f16/riuso+constant']:8.3f} ms/expert  ->  "
              f"{ov_only / 1000:7.3f} s/token")
        est = (ms_read + 0.34 + results["f16/riuso+constant"]) * N_USED * N_LAYER
        print(f"  {'STIMA TOTALE (pesi in cache)':54s} {est / 1000:7.3f} s/token  "
              f"({1000 / est:6.2f} token/s)")

    print()
    print("=" * 82)
    print(" FINE PHASE 0")
    print("=" * 82)


if __name__ == "__main__":
    main()
