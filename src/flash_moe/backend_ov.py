import numpy as np
import openvino as ov
import openvino.opset13 as ops
from typing import Any
from .key import ExpertKey
from .store import ExpertStore

class OpenVINOExpertBackend:
    """OpenVINO-accelerated SwiGLU expert execution backend (Intel Arc GPU / NPU)."""

    def __init__(self, store: ExpertStore, device: str = "AUTO"):
        self.store = store
        self.core = ov.Core()
        available = self.core.available_devices

        if device == "GPU" and "GPU" in available:
            self.target_device = "GPU"
        elif "GPU" in available:
            self.target_device = "GPU"
        else:
            self.target_device = "CPU"

        self._compiled_models: dict[tuple[int, int], ov.CompiledModel] = {}
        print(f"[OpenVINOBackend] Target Device selected: {self.target_device}")

    def _get_compiled_model(self, hidden_size: int, intermediate_size: int) -> ov.CompiledModel:
        shape_key = (hidden_size, intermediate_size)
        if shape_key in self._compiled_models:
            return self._compiled_models[shape_key]

        x = ops.parameter([1, hidden_size], ov.Type.f32, name="activation")
        gate = ops.parameter([intermediate_size, hidden_size], ov.Type.f32, name="gate")
        up = ops.parameter([intermediate_size, hidden_size], ov.Type.f32, name="up")
        down = ops.parameter([hidden_size, intermediate_size], ov.Type.f32, name="down")

        gate_proj = ops.matmul(x, gate, transpose_a=False, transpose_b=True)
        up_proj = ops.matmul(x, up, transpose_a=False, transpose_b=True)
        swish_gate = ops.swish(gate_proj)
        swiglu_out = ops.multiply(swish_gate, up_proj)
        out = ops.matmul(swiglu_out, down, transpose_a=False, transpose_b=True)

        model = ov.Model([out], [x, gate, up, down], name="SwiGLU_Expert")
        compiled = self.core.compile_model(model, self.target_device)
        self._compiled_models[shape_key] = compiled
        return compiled

    def forward(
        self,
        key: ExpertKey,
        activation: list[float] | np.ndarray,
        hidden_size: int,
        intermediate_size: int,
    ) -> list[float]:
        gate_bytes = self.store.read_tensor(key, "gate")
        up_bytes = self.store.read_tensor(key, "up")
        down_bytes = self.store.read_tensor(key, "down")

        gate_arr = np.frombuffer(gate_bytes, dtype=np.float32).reshape(intermediate_size, hidden_size)
        up_arr = np.frombuffer(up_bytes, dtype=np.float32).reshape(intermediate_size, hidden_size)
        down_arr = np.frombuffer(down_bytes, dtype=np.float32).reshape(hidden_size, intermediate_size)
        
        if isinstance(activation, list):
            x_arr = np.array(activation, dtype=np.float32).reshape(1, hidden_size)
        else:
            x_arr = activation.reshape(1, hidden_size).astype(np.float32)

        compiled = self._get_compiled_model(hidden_size, intermediate_size)
        infer_request = compiled.create_infer_request()

        results = infer_request.infer({
            "activation": x_arr,
            "gate": gate_arr,
            "up": up_arr,
            "down": down_arr,
        })

        output_tensor = list(results.values())[0]
        return output_tensor.flatten().tolist()
