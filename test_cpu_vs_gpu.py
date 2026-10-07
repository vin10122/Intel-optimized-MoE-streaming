#!/usr/bin/env python
"""Test CPU vs GPU performance for this model"""
import sys
sys.path.insert(0, 'src')

import time
from pathlib import Path
import gguf
from flash_moe import Qwen3MoEModel, StreamingDetokenizer
from flash_moe.paths import model_path
from flash_moe.tokenizer import GGUFTokenizer
from run_q4k_f16_openvino import dequant_to_f16

def test_device(device, cache_size=1):
    """Test model loading and generation with a device."""
    path = model_path()
    print(f"\nTesting {device} with cache={cache_size}...")
    
    t0 = time.perf_counter()
    model = Qwen3MoEModel(
        path, 
        device=device, 
        max_seq_len=128, 
        cache_capacity=cache_size, 
        dequant_fn=dequant_to_f16
    )
    load_time = time.perf_counter() - t0
    print(f"  Model load: {load_time:.1f}s")
    
    tok = GGUFTokenizer(gguf.GGUFReader(path))
    
    # Test with a simple prompt
    prompt = "Say: hello"
    ids = tok.encode(prompt, add_bos=False)
    stream = StreamingDetokenizer(tok)
    
    t0 = time.perf_counter()
    token_count = 0
    for produced, token in model.stream(
        ids, 
        max_new_tokens=3, 
        temperature=0, 
        seed=42
    ):
        text = stream.push(token)
        token_count += 1
    gen_time = time.perf_counter() - t0
    print(f"  Generated {token_count} tokens in: {gen_time:.2f}s ({token_count/gen_time:.2f} tok/s)")
    print(f"  Total stats: {model.store.stats}")
    
    model.close()
    return load_time, gen_time

def main():
    print("CPU vs GPU Performance Test")
    print("="*60)
    
    # Test CPU with small cache
    cpu_load, cpu_gen = test_device("CPU", cache_size=1)
    
    # Test GPU with small cache
    gpu_load, gpu_gen = test_device("GPU", cache_size=1)
    
    print(f"\n{'='*60}")
    print("Comparison:")
    print(f"{'='*60}")
    print(f"CPU: load={cpu_load:.1f}s, gen={cpu_gen:.2f}s")
    print(f"GPU: load={gpu_load:.1f}s, gen={gpu_gen:.2f}s")
    print(f"\nGPU/CPU load ratio: {gpu_load/cpu_load:.1f}x")
    print(f"GPU/CPU gen ratio:  {gpu_gen/cpu_gen:.1f}x")
    
    if cpu_gen < gpu_gen:
        print(f"\n✅ CPU is {gpu_gen/cpu_gen:.1f}x FASTER for generation")
    else:
        print(f"\n⚡ GPU is {cpu_gen/gpu_gen:.1f}x FASTER for generation")

if __name__ == '__main__':
    main()
