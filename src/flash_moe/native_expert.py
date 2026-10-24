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
        self.delta = NativeDelta(self.library) if hasattr(self.library, "moe_delta_heads") else None

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
            print("[DeltaNet] " + ("native FP32 decode recurrence" if result.delta is not None else "NumPy recurrence (rebuild native DLL to enable acceleration)"))
            return result
    if mode == "native":
        raise RuntimeError(reason)
    print(f"[Experts] OpenVINO decode: {reason}")
    return None


class NativeDelta:
    """All-head FP32 state update; called only during native decode."""

    def __init__(self, library):
        self.library = library
        self.function = library.moe_delta_heads
        self.function.argtypes = [C.c_void_p] * 7 + [C.c_int] * 3
        self.function.restype = C.c_int

    def forward(self, state, q, k, v, beta, gate):
        if state.dtype != np.float32 or not state.flags.c_contiguous:
            raise ValueError("DeltaNet state must be contiguous FP32")
        nv, dv, dk = state.shape
        arrays = [np.ascontiguousarray(a, dtype=np.float32)
                  for a in (q, k, v, beta, gate)]
        shapes = [(nv, dk), (nv, dk), (nv, dv), (nv,), (nv,)]
        if any(a.shape != shape for a, shape in zip(arrays, shapes)):
            raise ValueError("DeltaNet native input shape mismatch")
        out = np.empty((nv, dv), np.float32)
        pointers = [state.ctypes.data] + [a.ctypes.data for a in arrays] + [out.ctypes.data]
        code = self.function(*pointers, nv, dk, dv)
        if code:
            raise RuntimeError(f"native DeltaNet failed with status {code}")
        return out
