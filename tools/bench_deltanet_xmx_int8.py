"""Benchmarks real DeltaNet projections with OpenVINO GPU FP16 and INT8.

The INT8 graph uses FakeQuantize around activations and per-output-channel
weights so the GPU plugin can lower the MatMuls to its integer hardware path.
This is an isolated experiment: normal model inference is not modified.
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import openvino as ov
import openvino.opset13 as ops

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gguf  # noqa: E402
from run_q4k_f16_openvino import dequant_to_f16  # noqa: E402
from flash_moe.gated_deltanet import load_layer_weights  # noqa: E402
from flash_moe.paths import model_path  # noqa: E402

LAYER = 0
WARMUP = 5
REPEATS = 50


def timed(fn, repeats: int = REPEATS) -> tuple[float, np.ndarray]:
    for _ in range(WARMUP):
        result = fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples)) * 1000.0, np.asarray(result)


def build_projection_model(
    weight_name: str, weight: np.ndarray, mode: str
) -> ov.Model:
    weight = np.ascontiguousarray(weight, dtype=np.float32)
    input_name = f"x_{weight_name}"
    x = ops.parameter([1, int(weight.shape[1])], ov.Type.f32, name=input_name)

    if mode == "f16":
        x16 = ops.convert(x, ov.Type.f16)
        w16 = ops.constant(weight.astype(np.float16), ov.Type.f16)
        result = ops.convert(ops.matmul(x16, w16, False, True), ov.Type.f32)
        return ov.Model([result], [x], f"deltanet_{weight_name}_f16")

    lo = ops.parameter([1, 1], ov.Type.f32, name=f"{input_name}_low")
    hi = ops.parameter([1, 1], ov.Type.f32, name=f"{input_name}_high")
    xq = ops.fake_quantize(
        x, lo, hi, -127.0, 127.0, 255, auto_broadcast="NUMPY"
    )
    input_scale = ops.divide(hi, np.float32(127.0))

    per_channel_absmax = np.max(np.abs(weight), axis=1, keepdims=True)
    per_channel_absmax = np.maximum(per_channel_absmax, 1e-12)
    weight_scale = per_channel_absmax / 127.0
    wq = ops.fake_quantize(
        ops.constant(weight),
        -per_channel_absmax,
        per_channel_absmax,
        -127.0,
        127.0,
        255,
        auto_broadcast="NUMPY",
    )
    product = ops.matmul(xq, wq, False, True)
    scaled = ops.multiply(product, input_scale)
    result = ops.multiply(
        scaled, ops.constant(weight_scale.reshape(-1), ov.Type.f32)
    )
    return ov.Model(
        [result], [x, lo, hi], f"deltanet_{weight_name}_int8"
    )


def describe_runtime_model(compiled: ov.CompiledModel) -> None:
    runtime_model = compiled.get_runtime_model()
    counts = Counter(node.get_type_name() for node in runtime_model.get_ordered_ops())
    print(f"  optimized runtime ops: {dict(counts)}")


def main() -> int:
    path = model_path()
    reader = gguf.GGUFReader(path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    weights_f16 = load_layer_weights(tensors, LAYER, dequant_to_f16)
    weights = {
        name: np.ascontiguousarray(weights_f16[name], dtype=np.float32)
        for name in ("qkv", "gate", "out")
    }

    rng = np.random.default_rng(90210)
    hidden = int(weights["qkv"].shape[1])
    value_dim = int(weights["out"].shape[1])
    inputs = {
        "x_qkv": rng.standard_normal((1, hidden), dtype=np.float32),
        "x_gate": rng.standard_normal((1, hidden), dtype=np.float32),
        "x_out": rng.standard_normal((1, value_dim), dtype=np.float32),
    }
    reference = {
        "qkv": inputs["x_qkv"] @ weights["qkv"].T,
        "gate": inputs["x_gate"] @ weights["gate"].T,
        "out": inputs["x_out"] @ weights["out"].T,
    }

    core = ov.Core()
    if "GPU" not in core.available_devices:
        raise RuntimeError(f"OpenVINO GPU is unavailable: {core.available_devices}")
    print(f"model: {path.name}; layer: {LAYER}; OpenVINO: {ov.__version__}")
    print(f"device: {core.get_property('GPU', 'FULL_DEVICE_NAME')}")
    print(f"optimization capabilities: "
          f"{core.get_property('GPU', 'OPTIMIZATION_CAPABILITIES')}")
    print("This capability list confirms INT8 support, not that a specific "
          "kernel used XMX hardware.")

    print("\nProjection accuracy and latency (median of 50 calls):")
    int8_pass = True
    for weight_name, input_name in (
        ("qkv", "x_qkv"),
        ("gate", "x_gate"),
        ("out", "x_out"),
    ):
        shape = weights[weight_name].shape
        print(f"\n  {weight_name}: {shape[1]} -> {shape[0]}")
        for mode in ("f16", "int8"):
            model = build_projection_model(weight_name, weights[weight_name], mode)
            start = time.perf_counter()
            compiled = core.compile_model(model, "GPU")
            compile_ms = (time.perf_counter() - start) * 1000.0
            print(f"    {mode.upper()} compile: {compile_ms:.1f} ms")
            describe_runtime_model(compiled)

            values = inputs[f"x_{weight_name}"]
            feed = {f"x_{weight_name}": values}
            if mode == "int8":
                bound = max(float(np.max(np.abs(values))), 1e-12)
                feed[f"x_{weight_name}_low"] = np.asarray(
                    [[-bound]], dtype=np.float32
                )
                feed[f"x_{weight_name}_high"] = np.asarray(
                    [[bound]], dtype=np.float32
                )

            def infer_one() -> np.ndarray:
                outputs = compiled(feed)
                return np.asarray(
                    outputs[compiled.output(0)], dtype=np.float32
                )

            latency_ms, actual = timed(infer_one)
            delta = actual.reshape(-1) - reference[weight_name].reshape(-1)
            reference_rms = float(np.sqrt(np.mean(reference[weight_name] ** 2)))
            rel_rms = float(np.sqrt(np.mean(delta ** 2))) / max(reference_rms, 1e-12)
            max_abs = float(np.max(np.abs(delta)))
            print(f"    {mode:4s}: {latency_ms:8.3f} ms; "
                  f"relative RMS error={rel_rms:.5f}; max abs error={max_abs:.5g}")
            if mode == "int8" and (not np.all(np.isfinite(actual)) or rel_rms > 0.03):
                int8_pass = False

    print("\nINT8 projection check: "
          + ("PASS (all finite, relative RMS error <= 3%)"
             if int8_pass else "FAIL (non-finite output or error > 3%)"))
    return 0 if int8_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
