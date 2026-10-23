"""Optional AVX2/FMA/F16C quantized expert kernel for CPU decode only."""
from __future__ import annotations

import ctypes as C
from pathlib import Path

import numpy as np


SUPPORTED_TYPES = {8, 11, 12, 13, 14}  # Q8_0, Q3_K, Q4_K, Q5_K, Q6_K


class NativeExpert:
    def __init__(self, path: Path):
        self.library = C.CDLL(str(path.resolve()))
        self.function = self.library.moe_quant_expert
        self.function.argtypes = [C.c_int, C.c_void_p, C.c_int64] * 3 + [
            C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_int,
        ]
        self.function.restype = C.c_int

    def forward(self, tensors, kinds, activation, hidden, intermediate):
        arrays = [np.frombuffer(tensors[name], np.uint8)
                  for name in ("gate", "up", "down")]
        args = []
        for name, raw in zip(("gate", "up", "down"), arrays):
            args.extend((kinds[name], raw.ctypes.data, raw.nbytes))
        x = np.ascontiguousarray(activation, dtype=np.float32).reshape(-1)
        if x.size != hidden:
            raise ValueError("native expert activation shape mismatch")
        out = np.empty(hidden, np.float32)
        code = self.function(*args, x.ctypes.data, hidden, intermediate,
                             out.ctypes.data, 1)
        if code:
            raise RuntimeError(f"native expert failed with status {code}")
        return out


def load_native_expert(mode, library, device, tensor_types):
    if mode not in ("auto", "native", "openvino"):
        raise ValueError("native_decode must be auto, native, or openvino")
    if mode == "openvino":
        return None
    reason = None
    try:
        # NumPy detects OS support as well as CPU instruction flags. Never
        # enter an AVX2 DLL just to find out whether the CPU supports it.
        from numpy._core._multiarray_umath import __cpu_features__
        features = __cpu_features__
    except ImportError:
        features = {}
    if device != "CPU":
        reason = "native experts require --device CPU"
    elif not all(features.get(flag, False) for flag in ("AVX2", "FMA3", "F16C")):
        reason = "AVX2, FMA and F16C CPU/OS support could not be confirmed"
    elif not tensor_types or any(
        kinds.get(name) not in SUPPORTED_TYPES
        for kinds in tensor_types.values() for name in ("gate", "up", "down")
    ):
        reason = "expert quantization is unsupported"
    path = Path(library) if library is not None else Path(__file__).with_name("moe_quant_dot.dll")
    if reason is None:
        try:
            result = NativeExpert(path)
        except (OSError, AttributeError) as error:
            reason = f"kernel unavailable ({error})"
        else:
            print("[Experts] native quantized CPU decode; OpenVINO prefill")
            return result
    if mode == "native":
        raise RuntimeError(reason)
    print(f"[Experts] OpenVINO decode: {reason}")
    return None
