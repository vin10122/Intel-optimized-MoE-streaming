"""OpenVINO FP16 projections for full-attention layers."""

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


class OpenVINOAttentionProjections:
    """Run fused Q/K/V and output projections with FP16 weights."""

    def __init__(self, weights: dict[str, np.ndarray], device: str) -> None:
        self.device = device
        core = _get_core(device)
        q, k, v = (weights[name] for name in ("q", "k", "v"))
        hidden_size = int(q.shape[0])
        self.q_size = int(q.shape[1])
        self.k_size = int(k.shape[1])
        self.v_size = int(v.shape[1])

        x = ops.parameter([-1, hidden_size], ov.Type.f32, name="hidden")
        x16 = ops.convert(x, ov.Type.f16)
        qkv_weight = ops.constant(
            np.ascontiguousarray(np.concatenate((q, k, v), axis=1),
                                 dtype=np.float16),
            ov.Type.f16,
        )
        qkv = ops.matmul(x16, qkv_weight, False, False)
        input_model = ov.Model(
            [ops.convert(qkv, ov.Type.f32)],
            [x],
            "full_attention_qkv_projections_fp16",
        )
        self.input_model = core.compile_model(input_model, device)
        self.input_request = self.input_model.create_infer_request()

        out_x = ops.parameter(
            [-1, int(weights["out"].shape[0])],
            ov.Type.f32,
            name="attention_output",
        )
        out_weight = ops.constant(
            np.ascontiguousarray(weights["out"], dtype=np.float16),
            ov.Type.f16,
        )
        out = ops.matmul(
            ops.convert(out_x, ov.Type.f16), out_weight, False, False
        )
        output_model = ov.Model(
            [ops.convert(out, ov.Type.f32)],
            [out_x],
            "full_attention_output_projection_fp16",
        )
        self.output_model = core.compile_model(output_model, device)
        self.output_request = self.output_model.create_infer_request()

    def input_projections(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.ascontiguousarray(x, dtype=np.float32)
        results = self.input_request.infer(
            {self.input_model.input(0): values}
        )
        qkv = np.asarray(results[self.input_model.output(0)], dtype=np.float32)
        q_end = self.q_size
        k_end = q_end + self.k_size
        return qkv[:, :q_end], qkv[:, q_end:k_end], qkv[:, k_end:]

    def output_projection(self, values: np.ndarray) -> np.ndarray:
        activation = np.ascontiguousarray(values, dtype=np.float32)
        results = self.output_request.infer(
            {self.output_model.input(0): activation}
        )
        return np.asarray(
            results[self.output_model.output(0)], dtype=np.float32
        )
