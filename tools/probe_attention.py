import sys
sys.path.insert(0, 'src')
import gguf, numpy as np
from flash_moe.paths import model_path
p = model_path()
reader = gguf.GGUFReader(p)
tmap = {t.name: t for t in reader.tensors}

for name in ['blk.0.attn_qkv.weight', 'blk.0.attn_gate.weight', 'blk.3.attn_q.weight', 'blk.3.attn_k.weight', 'blk.3.attn_v.weight', 'blk.3.attn_output.weight']:
    t = tmap[name]
    print(f"{name}:")
    print(f"  raw GGUF shape (in,out): {tuple(int(s) for s in t.shape)}")
    print(f"  reversed (out,in): {tuple(int(s) for s in reversed(t.shape))}")
    print(f"  data dtype: {t.data.dtype}")
    print()

# So for attn_output: raw shape = (2048, 4096) = (in=2048, out=4096)
# This means input=2048, output=4096. But we need to PROJECT to hidden=2048
# Wait... maybe attention output is 4096-dimensional?
# Q = (2048, 8192) → input 2048, output 8192 = 16*512
# But 8192 is not 4096

# Let me think about this differently. If attn_output: in=2048, out=4096
# Then the attention before proj is 2048-dim, output is 4096-dim
# But the next layer expects 2048-dim input...

# UNLESS: attn_qkv produces 8192, but we only use 4096 for attention (Q+K or Q+K)
# Q = first 4096, K = next 4096? But K head_dim would be 4096/16=256... 
# or K = 4096 = 2*2048 = 2 KV heads * 1024? No, kv_head_dim = 256, 2*256=512

# Hmm, let me look at this more carefully. With 8192 = 16*512:
# Q = 8192 = 16*512
# But attn_output: in=2048, out=4096
# If attention output is 4096, that's 8*512 or 16*256

# Actually wait. Let me check if attn_output raw shape is (2048, 4096) or (4096, 2048)
print("Checking attn_output raw shape interpretation:")
t = tmap['blk.3.attn_output.weight']
print(f"  t.shape = {tuple(t.shape)} → (dim0, dim1)")
print(f"  If shape = (in, out): in={t.shape[0]}, out={t.shape[1]}")
print(f"  If shape = (out, in): out={t.shape[0]}, in={t.shape[1]}")
