"""OpenVINO FP16 projections for the recurrent DeltaNet layers."""

from __future__ import annotations

import numpy as np

import openvino as ov
import openvino.opset13 as ops


_CORES: dict[str, ov.Core] = {}


def _get_core(device: str) -> ov.Core:
    core = _CORES.get(device)
    if core is None:
        core = ov.Core()
        if device not in core.available_devices:
            raise RuntimeError(
                f"OpenVINO device {device!r} is unavailable; "
                f"available devices: {core.available_devices}"
            )
        _CORES[device] = core
    return core


class OpenVINODeltaNetProjections:
    """Run fixed-weight QKV/gate and output projections on an OpenVINO device."""

    def __init__(self, weights: dict[str, np.ndarray], device: str) -> None:
        self.device = device
        core = _get_core(device)

        x = ops.parameter([1, int(weights["qkv"].shape[1])],
                          ov.Type.f32, name="hidden")
        x16 = ops.convert(x, ov.Type.f16)
        qkv_w = ops.constant(
            np.ascontiguousarray(weights["qkv"], dtype=np.float16),
            ov.Type.f16,
        )
        gate_w = ops.constant(
            np.ascontiguousarray(weights["gate"], dtype=np.float16),
            ov.Type.f16,
        )
        qkv = ops.matmul(x16, qkv_w, False, True)
        gate = ops.matmul(x16, gate_w, False, True)
        input_model = ov.Model(
            [ops.convert(qkv, ov.Type.f32), ops.convert(gate, ov.Type.f32)],
            [x],
            "DeltaNet_input_projections_fp16",
        )
        self.input_model = core.compile_model(input_model, device)
        self.input_request = self.input_model.create_infer_request()

        out_x = ops.parameter([1, int(weights["out"].shape[1])],
                              ov.Type.f32, name="gated")
        out_w = ops.constant(
            np.ascontiguousarray(weights["out"], dtype=np.float16),
            ov.Type.f16,
        )
        out16 = ops.matmul(ops.convert(out_x, ov.Type.f16), out_w, False, True)
        output_model = ov.Model(
            [ops.convert(out16, ov.Type.f32)],
            [out_x],
            "DeltaNet_output_projection_fp16",
        )
        self.output_model = core.compile_model(output_model, device)
        self.output_request = self.output_model.create_infer_request()

    @staticmethod
    def _input(values: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(values, dtype=np.float32).reshape(1, -1)

    def input_projections(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        results = self.input_request.infer(
            {self.input_model.input(0): self._input(x)}
        )
        return (
            np.asarray(results[self.input_model.output(0)], dtype=np.float32).reshape(-1),
            np.asarray(results[self.input_model.output(1)], dtype=np.float32).reshape(-1),
        )

    def output_projection(self, gated: np.ndarray) -> np.ndarray:
        results = self.output_request.infer(
            {self.output_model.input(0): self._input(gated)}
        )
        return np.asarray(
            results[self.output_model.output(0)], dtype=np.float32
        ).reshape(-1)
