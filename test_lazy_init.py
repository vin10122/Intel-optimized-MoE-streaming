#!/usr/bin/env python
"""Test tempo di inizializzazione con lazy loading"""
import sys
sys.path.insert(0, 'src')

import time
from pathlib import Path
from flash_moe import Qwen3MoEModel
from flash_moe.paths import model_path
from run_q4k_f16_openvino import dequant_to_f16
import gguf
from flash_moe.tokenizer import GGUFTokenizer

def main():
    path = model_path()

    print("="*60)
    print("Test Lazy Loading Initialization")
    print("="*60)

    # Phase 1: Model initialization
    print("\n1. Inizializzazione modello (con lazy loading)...")
    t0 = time.perf_counter()
    model = Qwen3MoEModel(
        path,
        device='CPU',
        max_seq_len=128,
        cache_capacity=1,
        dequant_fn=dequant_to_f16,
        use_mmap=True
    )
    t1 = time.perf_counter()
    init_time = t1 - t0
    print(f"   Tempo: {init_time:.1f}s")
    print(f"   Blocchi costruiti all'inizializzazione: {sum(1 for b in model.blocks if b is not None)}")
    print(f"   SSM layer costruiti: {sum(1 for s in model.ssm.values() if s is not None)}")

    # Phase 2: Tokenizer initialization
    print("\n2. Inizializzazione tokenizer...")
    t0 = time.perf_counter()
    tok = GGUFTokenizer(gguf.GGUFReader(path))
    t1 = time.perf_counter()
    tok_time = t1 - t0
    print(f"   Tempo: {tok_time:.2f}s")

    # Phase 3: Encoding
    print("\n3. Encoding prompt...")
    t0 = time.perf_counter()
    ids = tok.encode('Ciao', add_bos=False)
    t1 = time.perf_counter()
    encode_time = t1 - t0
    print(f"   Tempo: {encode_time:.2f}s")
    print(f"   Token: {ids}")

    # Phase 4: Generation (first token triggers lazy loading)
    print("\n4. Generazione primo token (trigger lazy loading)...")
    t0 = time.perf_counter()
    tokens_generated = 0
    for produced, token in model.stream(ids, max_new_tokens=1, temperature=0, seed=42):
        tokens_generated += 1
        print(f"   Token {produced}: {token}")
    t1 = time.perf_counter()
    first_token_time = t1 - t0
    print(f"   Tempo per primo token: {first_token_time:.1f}s")
    print(f"   Blocchi costruiti dopo primo token: {sum(1 for b in model.blocks if b is not None)}")
    print(f"   SSM layer costruiti dopo primo token: {sum(1 for s in model.ssm.values() if s is not None)}")

    # Phase 5: Generation of more tokens
    print("\n5. Generazione successivi 2 token...")
    t0 = time.perf_counter()
    for produced, token in model.stream(ids, max_new_tokens=3, temperature=0, seed=42):
        if produced > 1:
            print(f"   Token {produced}: {token}")
    t1 = time.perf_counter()
    more_tokens_time = t1 - t0
    print(f"   Tempo per 2 token: {more_tokens_time:.1f}s")
    print(f"   Token/s: {2/more_tokens_time:.2f}")

    model.close()

    print("\n" + "="*60)
    print("Riepilogo:")
    print(f"  Inizializzazione: {init_time:.1f}s")
    print(f"  Primo token: {first_token_time:.1f}s")
    print(f"  Token successivi (2): {more_tokens_time:.1f}s")
    print(f"  Token/s (successivi): {2/more_tokens_time:.2f}")
    print("="*60)

if __name__ == '__main__':
    main()
