from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flash_moe.rope import RoPE, _rope_frequencies, DEFAULT_HEAD_DIM, DEFAULT_ROPE_DIM

print("=== RoPE reference test ===")

def rope_reference(x: np.ndarray, rope_dim: int, freq_base: float, start_pos: int = 0) -> np.ndarray:
    """NumPy reference: GPT-NeoX interleaved rotary."""
    head_dim = x.shape[-1]
    freqs_half = 1.0 / (freq_base ** (np.arange(0, rope_dim, 2, dtype=np.float32) / rope_dim))
    positions = np.arange(start_pos, start_pos + x.shape[-2], dtype=np.float32)
    freqs = np.outer(positions, freqs_half)
    sin = np.repeat(np.sin(freqs), 2, axis=1)
    cos = np.repeat(np.cos(freqs), 2, axis=1)
    sin_full = np.zeros((x.shape[-2], head_dim), dtype=np.float32)
    cos_full = np.zeros((x.shape[-2], head_dim), dtype=np.float32)
    sin_full[:, :rope_dim] = sin
    cos_full[:, :rope_dim] = cos
    sin_full[:, rope_dim:] = 0.0
    cos_full[:, rope_dim:] = 1.0

    x1 = x[..., :rope_dim]
    x2 = x[..., rope_dim:]
    rot = np.stack([-x1[..., 1::2], x1[..., 0::2]], axis=-1)
    rot = rot.reshape(x1.shape[:-1] + (rope_dim,))
    rotated = np.concatenate([x1 * cos_full[..., :rope_dim] + rot * sin_full[..., :rope_dim], x2], axis=-1)
    return rotated

# Test 1: Basic rotation correctness
head_dim = 128
rope_dim = 64
rope = RoPE(head_dim=head_dim, rope_dim=rope_dim, freq_base=1_000_000.0)

x = np.random.randn(3, 16, 10, head_dim).astype(np.float32)
out = rope.apply_rotary(x)
ref = rope_reference(x, rope_dim, 1_000_000.0)
diff = np.abs(out - ref).max()
print(f"  Basic rotation: max diff = {diff:.3e} (shape={out.shape})")
assert diff < 1e-5, f"RoPE diff too large: {diff}"

# Test 2: Identity region (last 64 dims should be unchanged)
out2 = rope.apply_rotary(x)
diff_tail = np.abs(out2[..., 64:] - x[..., 64:]).max()
print(f"  Tail (passive) dims unchanged: max diff = {diff_tail:.3e}")
assert diff_tail < 1e-7, "Passive dims should be unchanged"

# Test 3: seq_offset caching
x1 = np.random.randn(2, 16, head_dim).astype(np.float32)
out_new = rope.apply_rotary(x1, seq_offset=0)
out_cached = rope.apply_rotary(x1, seq_offset=0)
diff_cache = np.abs(out_new - out_cached).max()
print(f"  Cache consistency: max diff = {diff_cache:.3e}")
assert diff_cache < 1e-7, "Cache should produce identical results"

# Test 4: Different positions
x_short = np.random.randn(1, 16, head_dim).astype(np.float32)
cos, sin = rope.get_cache(10)
print(f"  Cache shapes: cos={cos.shape}, sin={sin.shape}")
print(f"  Cache sin[3, 0] = {sin[3, 0]:.6f}, cos[3, 0] = {cos[3, 0]:.6f}")

# Test 5: Verify freqs match expected
freqs = _rope_frequencies(rope_dim, 1_000_000.0)
print(f"  freqs[0:4] = {freqs[:4]}")
print(f"  freqs[-4:] = {freqs[-4:]}")
assert len(freqs) == rope_dim // 2, "freqs should have rope_dim/2 = 32 entries"

print("  ALL RoPE TESTS PASSED")
