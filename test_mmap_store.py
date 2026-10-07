#!/usr/bin/env python
"""Test MemoryMappedExpertStore vs AsyncExpertStore"""
import sys
sys.path.insert(0, 'src')

import time
from pathlib import Path
import gguf
from flash_moe import Qwen3MoEModel, StreamingDetokenizer
from flash_moe.paths import model_path
from flash_moe.tokenizer import GGUFTokenizer
from flash_moe.store import AsyncExpertStore
from flash_moe.mmap_store import MemoryMappedAsyncExpertStore
from run_q4k_f16_openvino import dequant_to_f16, build_manifest_from_gguf

def test_store(store_class, store_name):
    """Test a store implementation."""
    path = model_path()
    manifest = build_manifest_from_gguf(path)
    
    print(f"\n{'='*60}")
    print(f"Testing {store_name}")
    print(f"{'='*60}")
    
    # Create store
    t0 = time.perf_counter()
    store = store_class(path.parent, capacity=1, manifest=manifest)
    init_time = time.perf_counter() - t0
    print(f"Store init: {init_time:.2f}s")
    
    # Read a few experts
    from flash_moe.key import ExpertKey
    keys = [
        ExpertKey(0, 0),
        ExpertKey(0, 1),
        ExpertKey(1, 0),
        ExpertKey(1, 1),
    ]
    
    t0 = time.perf_counter()
    for key in keys:
        store.read_all(key)
    read_time = time.perf_counter() - t0
    print(f"Read 4 experts: {read_time:.2f}s")
    print(f"Stats: {store.stats}")
    
    store.close()
    return init_time, read_time

def main():
    print("Memory-Mapped Store Performance Test")
    print("="*60)
    
    # Test AsyncExpertStore (original)
    init1, read1 = test_store(AsyncExpertStore, "AsyncExpertStore (original)")
    
    # Test MemoryMappedAsyncExpertStore (optimized)
    init2, read2 = test_store(MemoryMappedAsyncExpertStore, "MemoryMappedAsyncExpertStore (optimized)")
    
    print(f"\n{'='*60}")
    print("Comparison:")
    print(f"{'='*60}")
    print(f"AsyncExpertStore:      init={init1:.2f}s, read={read1:.2f}s")
    print(f"MemoryMappedStore:     init={init2:.2f}s, read={read2:.2f}s")
    print(f"Speedup (init): {init1/init2:.1f}x")
    print(f"Speedup (read):  {read1/read2:.1f}x")

if __name__ == '__main__':
    main()
