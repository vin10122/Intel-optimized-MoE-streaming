"""Il passaggio dei pesi per expert e' il collo di bottiglia. Ogni expert ha
pesi diversi, quindi 'Constant' non e' riusabile e bisogna capire quale
strategia di aggiornamento dei pesi costa meno.

Strategie confrontate, tutte sullo stesso grafo SwiGLU f16:
  A. infer(dict) con array nuovi   - cio' che fa il codice oggi
  A2. infer(dict) ricreato ogni volta
  D. batch: k=8 esperti impilati, 1 solo infer per layer
  E. D + dequant/copy fuso (stima: un solo passaggio di memoria)

Le misure sono alternate A/D/A/D per separare il segnale dal drift.
"""

from __future__ import annotations

import time

import numpy as np
import openvino as ov
import openvino.opset13 as ops

HIDDEN = 2048
INTER = 512
K = 8
ROUNDS = 12

WEIGHT_BYTES = (HIDDEN * INTER * 2) * 2 + (INTER * HIDDEN * 2)


def timeit(fn, rounds):
    samples = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return samples[len(samples) // 2], samples[0], samples[-1]


def build_single(core, device):
    x = ops.parameter([1, HIDDEN], ov.Type.f32, name="activation")
    g = ops.parameter([HIDDEN, INTER], ov.Type.f16, name="gate")
    u = ops.parameter([HIDDEN, INTER], ov.Type.f16, name="up")
    d = ops.parameter([INTER, HIDDEN], ov.Type.f16, name="down")
    xc = ops.convert(x, ov.Type.f16)
    gp = ops.matmul(xc, g, transpose_a=False, transpose_b=False)
    up = ops.matmul(xc, u, transpose_a=False, transpose_b=False)
    act = ops.multiply(gp, ops.sigmoid(gp))
    out = ops.matmul(act, d, transpose_a=False, transpose_b=False)
    return ov.Model([ops.convert(out, ov.Type.f32)], [x, g, u, d], "expert")


def build_batched(core, device):
    # k esperti impilati: [1,H] @ [H, k*I] -> [1, k*I], poi @ [k*I, H] -> [1, H]
    x = ops.parameter([1, HIDDEN], ov.Type.f32, name="activation")
    g = ops.parameter([HIDDEN, K * INTER], ov.Type.f16, name="gate")
    u = ops.parameter([HIDDEN, K * INTER], ov.Type.f16, name="up")
    d = ops.parameter([K * INTER, HIDDEN], ov.Type.f16, name="down")
    xc = ops.convert(x, ov.Type.f16)
    gp = ops.matmul(xc, g, transpose_a=False, transpose_b=False)
    up = ops.matmul(xc, u, transpose_a=False, transpose_b=False)
    act = ops.multiply(gp, ops.sigmoid(gp))
    out = ops.matmul(act, d, transpose_a=False, transpose_b=False)
    return ov.Model([ops.convert(out, ov.Type.f32)], [x, g, u, d], "expert_batched")


def main() -> None:
    core = ov.Core()
    device = "GPU" if "GPU" in core.available_devices else "CPU"
    print(f"device: {device}")
    print(f"pesi f16 per expert : {WEIGHT_BYTES:,} B ({WEIGHT_BYTES / 1024**2:.2f} MiB)")
    print(f"pesi f16 per layer  : {K * WEIGHT_BYTES:,} B "
          f"({K * WEIGHT_BYTES / 1024**2:.2f} MiB)")
    print(f"esperti per token   : 8 x 40 = {K * 40}")
    print()

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, HIDDEN)).astype(np.float32)

    # Pool pre-generato: NON generare i pesi dentro la sezione misurata,
    # altrimenti si misura standard_normal e non la copia.
    POOL = 4
    pool_g = [(rng.standard_normal((HIDDEN, INTER)) * 0.05).astype(np.float16)
              for _ in range(POOL)]
    pool_d = [np.ascontiguousarray(
        (rng.standard_normal((INTER, HIDDEN)) * 0.05).astype(np.float16))
        for _ in range(POOL)]
    pool_gs = [(rng.standard_normal((HIDDEN, K * INTER)) * 0.05).astype(np.float16)
               for _ in range(POOL)]
    pool_ds = [(rng.standard_normal((K * INTER, HIDDEN)) * 0.05).astype(np.float16)
               for _ in range(POOL)]
    _cursor = {"a": 0, "d": 0}

    def next_single():
        i = _cursor["a"] = (_cursor["a"] + 1) % POOL
        return pool_g[i], pool_d[i]

    def next_stack():
        i = _cursor["d"] = (_cursor["d"] + 1) % POOL
        return pool_gs[i], pool_ds[i]

    # ---- A: infer singolo
    cm = core.compile_model(build_single(core, device), device)
    req = cm.create_infer_request()
    wg, wd = next_single()
    req.infer({"activation": x, "gate": wg, "up": wg, "down": wd})

    def do_a():
        wg, wd = next_single()
        return req.infer({"activation": x, "gate": wg, "up": wg, "down": wd})

    # ---- A2: request nuovo ogni volta
    def do_a2():
        wg, wd = next_single()
        r = cm.create_infer_request()
        return r.infer({"activation": x, "gate": wg, "up": wg, "down": wd})

    # ---- D: batch k=8
    try:
        cmb = core.compile_model(build_batched(core, device), device)
        reqb = cmb.create_infer_request()
        gs, ds = next_stack()
        reqb.infer({"activation": x, "gate": gs, "up": gs, "down": ds})

        def do_d():
            gs, ds = next_stack()
            return reqb.infer({"activation": x, "gate": gs, "up": gs, "down": ds})

        print("batch k=8: compilato OK")
    except Exception as err:
        print(f"batch k=8: FALLITO -> {str(err).splitlines()[0][:140]}")
        do_d = None

    for _ in range(3):
        do_a()
        do_a2()
        if do_d:
            do_d()

    a_s: list[float] = []
    a2_s: list[float] = []
    d_s: list[float] = []
    for i in range(ROUNDS):
        a_s.append(timeit(do_a, 1)[0])
        a2_s.append(timeit(do_a2, 1)[0])
        if do_d:
            d_s.append(timeit(do_d, 1)[0])
        if (i + 1) % 4 == 0:
            print(f"  round {i + 1:2d}/{ROUNDS}: "
                  f"A={a_s[-1]:6.3f}  A2={a2_s[-1]:6.3f}"
                  + (f"  D={d_s[-1]:7.3f}" if d_s else ""))

    def stat(samples):
        s = sorted(samples)
        return s[len(s) // 2], s[0], s[-1]

    print()
    print("=" * 78)
    print(" RISULTATI (ms per expert equivalente)")
    print("=" * 78)
    med, lo, hi = stat(a_s)
    print(f" A  infer(dict), request riutilizzato     {med:7.3f}  "
          f"[{lo:.3f}..{hi:.3f}]  -> {med * K * 40 / 1000:6.3f} s/token")
    med2, lo2, hi2 = stat(a2_s)
    print(f" A2 infer(dict), request nuovo            {med2:7.3f}  "
          f"[{lo2:.3f}..{hi2:.3f}]  -> {med2 * K * 40 / 1000:6.3f} s/token")
    if d_s:
        medd, lod, hid = stat(d_s)
        per = medd / K
        print(f" D  batch k=8 (1 infer/layer)             {per:7.3f}  "
              f"[{lod / K:.3f}..{hid / K:.3f}]  -> {per * K * 40 / 1000:6.3f} s/token")
        print()
        print(f" guadagno D su A : {med / per:.2f}x")
        print(f" guadagno D su A2: {med2 / per:.2f}x")
        print()
        print(f" A  copia {WEIGHT_BYTES / 1024**2:.2f} MiB in {med:.3f} ms "
              f"= {WEIGHT_BYTES / med / 1e6:.2f} GB/s")
        print(f" D  copia {K * WEIGHT_BYTES / 1024**2:.2f} MiB in {medd:.3f} ms "
              f"= {K * WEIGHT_BYTES / medd / 1e6:.2f} GB/s")

    # ------------------------------------------------------------------
    # Il trasferimento per expert e' forse piu lento della memcpy stessa.
    # Confronto device e baseline BLAS su numpy.
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print(" CONFRONTO DEVICE E BASELINE NUMPY/BLAS (strategia A)")
    print("=" * 78)

    for dev in core.available_devices:
        try:
            cm_d = core.compile_model(build_single(core, device if False else dev), dev)
            r_d = cm_d.create_infer_request()
            wg, wd = next_single()
            r_d.infer({"activation": x, "gate": wg, "up": wg, "down": wd})

            def do_dev():
                wg, wd = next_single()
                return r_d.infer({"activation": x, "gate": wg, "up": wg, "down": wd})

            for _ in range(3):
                do_dev()
            m, l, h = timeit(do_dev, 10)
            print(f" OpenVINO {dev:4s} f16 + pesi per call  {m:7.3f} ms/expert  ->  "
                  f"{m * K * 40 / 1000:6.3f} s/token  "
                  f"({WEIGHT_BYTES / m / 1e6:.2f} GB/s)")
        except Exception as err:
            print(f" OpenVINO {dev:4s}: FALLITO -> {str(err).splitlines()[0][:90]}")

    # numpy: il "copy" dei pesi e' il gather da un pool, poi BLAS f32
    print()
    pool_f32 = [rng.standard_normal((HIDDEN, INTER)).astype(np.float32)
                for _ in range(POOL)]
    pool_f32_d = [rng.standard_normal((INTER, HIDDEN)).astype(np.float32)
                  for _ in range(POOL)]
    xf = x.astype(np.float32)
    cur = {"i": 0}

    def do_numpy():
        i = cur["i"] = (cur["i"] + 1) % POOL
        w = pool_f32[i]
        wd_ = pool_f32_d[i]
        gp = xf @ w
        act = gp * (1.0 / (1.0 + np.exp(-gp)))
        return (act * (xf @ w)) @ wd_

    for _ in range(3):
        do_numpy()
    m, l, h = timeit(do_numpy, 10)
    print(f" numpy f32 BLAS, pesi gia' in RAM          {m:7.3f} ms/expert  ->  "
          f"{m * K * 40 / 1000:6.3f} s/token")

    def do_numpy_f16():
        i = cur["i"] = (cur["i"] + 1) % POOL
        w16_ = pool_g[i]
        wd16_ = pool_d[i]
        wf = w16_.astype(np.float32)
        wdf = wd16_.astype(np.float32)
        gp = xf @ wf
        act = gp * (1.0 / (1.0 + np.exp(-gp)))
        return (act * (xf @ wf)) @ wdf

    for _ in range(3):
        do_numpy_f16()
    m2, l2, h2 = timeit(do_numpy_f16, 10)
    print(f" numpy: cast f16->f32 + BLAS f32          {m2:7.3f} ms/expert  ->  "
          f"{m2 * K * 40 / 1000:6.3f} s/token")

    def do_copy_only():
        i = cur["i"] = (cur["i"] + 1) % POOL
        dst = np.empty((HIDDEN, INTER), dtype=np.float16)
        np.copyto(dst, pool_g[i])
        dst2 = np.empty((INTER, HIDDEN), dtype=np.float16)
        np.copyto(dst2, pool_d[i])
        return dst, dst2

    for _ in range(3):
        do_copy_only()
    m3, l3, h3 = timeit(do_copy_only, 10)
    print(f" sola memcpy di 6 MiB (riferimento)        {m3:7.3f} ms/expert  ->  "
          f"{m3 * K * 40 / 1000:6.3f} s/token  ({WEIGHT_BYTES / m3 / 1e6:.2f} GB/s)")


if __name__ == "__main__":
    main()
