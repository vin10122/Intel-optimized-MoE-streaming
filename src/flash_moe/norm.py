"""RMSNorm: normalizzazione per i transformer layer.

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

Usata in Qwen3: 80 tensor F32 (attn_norm, post_attention_norm) nel GGUF,
uno per layer + output_norm. F32 significa che non servono dequantizzatori.
"""

from __future__ import annotations

import numpy as np


class RMSNorm:
    """Root Mean Square Layer Normalization (Qwen3)."""

    def __init__(self, weight: np.ndarray, eps: float = 1e-6) -> None:
        self.weight = np.ascontiguousarray(weight, dtype=np.float32)
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Normalizza x (shape [..., hidden]) con self.weight (shape [hidden])."""
        x_f32 = np.ascontiguousarray(x, dtype=np.float32)
        rms = np.sqrt(np.mean(x_f32 * x_f32, axis=-1, keepdims=True) + self.eps)
        return (x_f32 / rms) * self.weight

    @classmethod
    def from_bytes(cls, raw: bytes, eps: float = 1e-6) -> "RMSNorm":
        """Costruisce da tensor F32 grezzo del GGUF."""
        weight = np.frombuffer(raw, dtype=np.float32).copy()
        return cls(weight, eps)


def rmsnorm_numpy(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Reference NumPy per RMSNorm (usata nei test)."""
    x_f32 = x.astype(np.float32)
    rms = np.sqrt(np.mean(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
    return (x_f32 / rms) * weight
