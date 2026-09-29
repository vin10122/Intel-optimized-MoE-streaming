import sys
sys.path.insert(0, 'src')
import gguf
import numpy as np
from flash_moe.paths import model_path
from flash_moe.dequant import name_of

p = model_path()
reader = gguf.GGUFReader(p)

# Print only key metadata
key_fields = [
    "general.architecture",
    "general.basename",
    "general.name",
    "qwen35moe.block_count",
    "qwen35moe.embedding_length",
    "qwen35moe.feed_forward_length",
    "qwen35moe.expert_count",
    "qwen35moe.expert_used_count",
    "qwen35moe.expert_feed_forward_length",
    "qwen35moe.expert_shared_feed_forward_length",
    "qwen35moe.attention.head_count",
    "qwen35moe.attention.head_count_kv",
    "qwen35moe.attention.key_length",
    "qwen35moe.attention.value_length",
    "qwen35moe.attention.layer_norm_rms_epsilon",
    "qwen35moe.rope.dimension_count",
    "qwen35moe.rope.dimension_sections",
    "qwen35moe.rope.freq_base",
    "qwen35moe.context_length",
    "qwen35moe.full_attention_interval",
    "qwen35moe.ssm.conv_kernel",
    "qwen35moe.ssm.group_count",
    "qwen35moe.ssm.inner_size",
    "qwen35moe.ssm.state_size",
    "qwen35moe.ssm.time_step_rank",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
]

print("=== KEY METADATA ===")
for key in key_fields:
    if key in reader.fields:
        parts = reader.fields[key].parts
        part = parts[-1]
        if part.dtype == np.float32 or part.dtype == np.float64:
            print(f"  {key} = {float(part[0])}")
        elif part.dtype in (np.int32, np.int64, np.uint32, np.uint64):
            print(f"  {key} = {int(part[0])}")
        elif part.dtype == np.uint8:
            s = bytes(part).decode('utf-8', errors='replace')
            print(f"  {key} = \"{s}\"")

# Check tokenizer file
import os
tokenizer_path = p.parent / "tokenizer"
print(f"\n=== TOKENIZER ===")
print(f"tokenizer dir exists: {tokenizer_path.exists()}")
if tokenizer_path.exists():
    for f in sorted(tokenizer_path.iterdir()):
        print(f"  {f.name}: {f.stat().st_size} bytes")

# Check for separate tokenizer files
gguf_dir = p.parent
print(f"\n=== GGUF DIR CONTENTS ===")
for f in sorted(gguf_dir.iterdir()):
    if f.suffix in ('.gguf', '.json', '.txt', '.py', '.bin', '.model'):
        print(f"  {f.name}: {f.stat().st_size} bytes")

print(f"\n=== ATTENTION ARCH SUMMARY ===")
layer_attn = {}
for t in reader.tensors:
    n = t.name
    if not n.startswith('blk.'):
        continue
    parts = n.split('.')
    if len(parts) < 3:
        continue
    layer = int(parts[1])
    role = parts[2]
    if 'attn' in role:
        layer_attn.setdefault(layer, []).append(n)

attn_types = {}
for layer in sorted(layer_attn.keys()):
    tensors = sorted(layer_attn[layer])
    has_qkv = any('attn_qkv' in n for n in tensors)
    has_q = any(n.endswith('.attn_q.weight') for n in tensors)
    if has_qkv:
        attn_types.setdefault('qkv', []).append(layer)
    elif has_q:
        attn_types.setdefault('qkv_split', []).append(layer)

for atype, layers in attn_types.items():
    print(f"  {atype}: layers {layers[0]}..{layers[-1]} ({len(layers)} layer)")

# Check for qk_norm (Qwen3 has QK normalization)
print("\n=== QK-NORM TENSORS ===")
for t in reader.tensors:
    n = t.name
    if 'attn_q_norm' in n or 'attn_k_norm' in n:
        print(f"  {n}: shape={tuple(int(s) for s in reversed(t.shape))}, type={name_of(t.tensor_type)}")
        if n.endswith('attn_q_norm.weight'):
            break
