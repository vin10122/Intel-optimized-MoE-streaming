"""Il passaggio di tensor f16/bf16 a OpenVINO via numpy uint16 e' rotto.
Qual e' il modo corretto? Prova:
  1. numpy float16 nativo
  2. numpy float16 visto come uint16
  3. ov.Tensor costruito da float16
  4. ov.Tensor costruito da uint16 (bf16)
  5. ml_dtypes.bfloat16 se disponibile
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import openvino as ov
import openvino.opset13 as ops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from flash_moe.paths import model_path  # noqa: E402

HIDDEN = 512
INTER = 256
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
    return core.compile_model(ov.Model([ops.convert(out, ov.Type.f32)], [x, g, u, d], "e"), device)


def main() -> None:
    core = ov.Core()
    rng = np.random.default_rng(0)
    wg = (rng.standard_normal((HIDDEN, INTER)) * 0.05).astype(np.float32)
    wd = np.ascontiguousarray((rng.standard_normal((INTER, HIDDEN)) * 0.05)).astype(np.float32)
    x = rng.standard_normal((1, HIDDEN)).astype(np.float32)

    gp = x @ wg
    up = x @ wg
    ref = ((gp * (1 / (1 + np.exp(-gp)))) * up) @ wd
    ref_max = np.abs(ref).max()

    def f32_to_bf16_bits(a):
        u = a.view(np.uint32)
        r = ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
        return ((u + r) >> np.uint32(16)).astype(np.uint16)

    bf16_bits = f32_to_bf16_bits(wg)
    f16_arr = wg.astype(np.float16)

    def report(tag, out):
        o = np.asarray(out, dtype=np.float32)
        nans = int(np.isnan(o).sum())
        err = np.nanmax(np.abs(o - ref)) if nans == 0 else float("nan")
        rel = err / ref_max if nans == 0 else float("nan")
        flag = "OK " if nans == 0 and rel < 1e-2 else "BAD"
        print(f"   {flag} {tag:56s} nan={nans:5d} rel={rel:.3e}")

    for device in core.available_devices:
        print(f"\n {device}")

        # ---- f16
        try:
            cm = build(core, device, ov.Type.f16)
            req = cm.create_infer_request()
            out = req.infer(
                {"activation": x,
                 "gate": f16_arr, "up": f16_arr,
                 "down": wd.astype(np.float16)}
            )
            report("f16 numpy float16 nativo", list(out.values())[0])
        except Exception as err:
            print(f"   ERR f16 float16: {str(err).splitlines()[0][:100]}")

        try:
            cm = build(core, device, ov.Type.f16)
            req = cm.create_infer_request()
            out = req.infer(
                {"activation": x,
                 "gate": f16_arr.view(np.uint16), "up": f16_arr.view(np.uint16),
                 "down": wd.astype(np.float16).view(np.uint16)}
            )
            report("f16 numpy float16 .view(uint16)", list(out.values())[0])
        except Exception as err:
            print(f"   ERR f16 uint16: {str(err).splitlines()[0][:100]}")

        try:
            cm = build(core, device, ov.Type.f16)
            req = cm.create_infer_request()
            tg = ov.Tensor(f16_arr, np.float16)
            out = req.infer(
                {"activation": x, "gate": tg, "up": tg,
                 "down": ov.Tensor(wd.astype(np.float16), np.float16)}
            )
            report("f16 via ov.Tensor(float16)", list(out.values())[0])
        except Exception as err:
            print(f"   ERR f16 ov.Tensor: {str(err).splitlines()[0][:100]}")

        # ---- bf16
        for tag, mk in (
            ("bf16 numpy uint16 (modo attuale)", lambda: bf16_bits),
            ("bf16 via ov.Tensor(uint16)", lambda: ov.Tensor(bf16_bits, np.uint16)),
        ):
            try:
                cm = build(core, device, ov.Type.bf16)
                req = cm.create_infer_request()
                gv, dv = mk(), ov.Tensor(bf16_bits, np.uint16)
                out = req.infer({"activation": x, "gate": gv, "up": gv, "down": dv})
                report(tag, list(out.values())[0])
            except Exception as err:
                print(f"   ERR {tag}: {str(err).splitlines()[0][:100]}")

        try:
            import ml_dtypes

            bf = wg.astype(ml_dtypes.bfloat16)
            cm = build(core, device, ov.Type.bf16)
            req = cm.create_infer_request()
            tb = ov.Tensor(bf)
            out = req.infer({"activation": x, "gate": tb, "up": tb,
                             "down": ov.Tensor(wd.astype(ml_dtypes.bfloat16))})
            report("bf16 numpy ml_dtypes.bfloat16", list(out.values())[0])
        except ImportError:
            print("   -- ml_dtypes non installato (serve per bf16 nativo)")
        except Exception as err:
            print(f"   ERR bf16 ml_dtypes: {str(err).splitlines()[0][:100]}")


if __name__ == "__main__":
    main()
