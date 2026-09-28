"""Verifica se OpenVINO su questa macchina gestisce davvero bf16/f16 nei matmul,
e quanto errore introduce rispetto a fp32.

Il bench Phase 0 ha prodotto output NaN con pesi bf16: se bf16 e' rotto sul
plugin GPU di Lunar Lake, l'intero piano "dequant -> BF16" va rifatto in f16.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import openvino as ov
import openvino.opset13 as ops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from flash_moe.paths import model_path  # noqa: E402

HIDDEN = 2048
INTER = 512
N_EXPERT = 256
LAYER = 0
GGUF_PATH = model_path()


def build(core, device, wtype):
    x = ops.parameter([1, HIDDEN], ov.Type.f32, name="activation")
    g = ops.parameter([HIDDEN, INTER], wtype, name="gate")
    u = ops.parameter([HIDDEN, INTER], wtype, name="up")
    d = ops.parameter([INTER, HIDDEN], wtype, name="down")
    xc = ops.convert(x, wtype)
    gp = ops.matmul(xc, g, transpose_a=False, transpose_b=False)
    up = ops.matmul(xc, u, transpose_a=False, transpose_b=False)
    act = ops.multiply(gp, ops.sigmoid(gp))
    out = ops.matmul(ops.multiply(act, up), d, transpose_a=False, transpose_b=False)
    model = ov.Model([ops.convert(out, ov.Type.f32)], [x, g, u, d], name="e")
    return core.compile_model(model, device)


def dequant_real_gate() -> np.ndarray:
    import gguf

    reader = gguf.GGUFReader(GGUF_PATH)
    tmap = {t.name: t for t in reader.tensors}
    t = tmap[f"blk.{LAYER}.ffn_gate_exps.weight"]
    per = t.n_bytes // N_EXPERT
    with open(GGUF_PATH, "rb") as fh:
        fh.seek(t.data_offset)
        raw = fh.read(per)
    return gguf.quants.dequantize(np.frombuffer(raw, np.uint8), t.tensor_type).reshape(
        HIDDEN, INTER
    )


def f32_to_bf16(a: np.ndarray) -> np.ndarray:
    u = a.view(np.uint32)
    r = ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((u + r) >> np.uint32(16)).astype(np.uint16)


def f32_to_f16(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float16).view(np.uint16)


def ref_swiglu(x, wg, wu, wd):
    gp = x @ wg
    up = x @ wu
    return (gp * (1.0 / (1.0 + np.exp(-gp)))) * up @ wd


def main() -> None:
    core = ov.Core()
    print(f"devices: {core.available_devices}")

    wg = dequant_real_gate()
    wu = wg
    wd = np.ascontiguousarray(wg.T)
    x = np.random.randn(1, HIDDEN).astype(np.float32)

    reference = ref_swiglu(x, wg, wu, wd)
    print(f"riferimento numpy fp32: |max|={np.abs(reference).max():.6f} "
          f"|rms|={np.sqrt((reference ** 2).mean()):.6f}")

    variants = {
        "f32": (ov.Type.f32, wg.astype(np.float32), wd.astype(np.float32)),
        "f16": (ov.Type.f16, f32_to_f16(wg), f32_to_f16(wd)),
        "bf16": (ov.Type.bf16, f32_to_bf16(wg), f32_to_bf16(wd)),
    }

    for device in core.available_devices:
        for name, (wtype, wgq, wdq) in variants.items():
            try:
                cm = build(core, device, wtype)
                req = cm.create_infer_request()
                feed = {
                    "activation": x,
                    "gate": wgq,
                    "up": wgq,
                    "down": wdq,
                }
                res = req.infer(feed)
                out = np.asarray(list(res.values())[0], dtype=np.float32)
                nans = int(np.isnan(out).sum())
                diff = np.abs(out - reference)
                rel = diff.max() / max(np.abs(reference).max(), 1e-9)
                print(
                    f" {device:4s} {name:5s}: nan={nans:5d} "
                    f"max_abs_err={diff.max():.3e} rel={rel:.3e} "
                    f"|out|max={np.abs(out).max():.6f}"
                )
            except Exception as err:
                msg = str(err).splitlines()[0][:110]
                print(f" {device:4s} {name:5s}: FALLITO -> {msg}")

    print()
    print("prova con pesi come Constant (share weights) - niente re-feed:")
    for device in core.available_devices:
        for name, (wtype, wgq, wdq) in variants.items():
            try:
                xp = ops.parameter([1, HIDDEN], ov.Type.f32, name="activation")
                g_c = ops.constant(wgq, wtype)
                u_c = ops.constant(wgq, wtype)
                d_c = ops.constant(wdq, wtype)
                xc = ops.convert(xp, wtype)
                gp = ops.matmul(xc, g_c, transpose_a=False, transpose_b=False)
                up = ops.matmul(xc, u_c, transpose_a=False, transpose_b=False)
                act = ops.multiply(gp, ops.sigmoid(gp))
                out = ops.matmul(ops.multiply(act, up), d_c,
                                 transpose_a=False, transpose_b=False)
                model = ov.Model([ops.convert(out, ov.Type.f32)], [xp], name="ec")
                cm = core.compile_model(model, device)
                req = cm.create_infer_request()
                res = req.infer({"activation": x})
                o = np.asarray(list(res.values())[0], dtype=np.float32)
                nans = int(np.isnan(o).sum())
                diff = np.abs(o - reference)
                rel = diff.max() / max(np.abs(reference).max(), 1e-9)
                print(
                    f" {device:4s} {name:5s}: nan={nans:5d} "
                    f"max_abs_err={diff.max():.3e} rel={rel:.3e}"
                )
            except Exception as err:
                msg = str(err).splitlines()[0][:110]
                print(f" {device:4s} {name:5s}: FALLITO -> {msg}")


if __name__ == "__main__":
    main()
