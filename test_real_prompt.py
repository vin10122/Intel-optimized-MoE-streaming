#!/usr/bin/env python
"""Test con prompt reale"""
import sys
sys.path.insert(0, 'src')

from pathlib import Path
import gguf
from flash_moe import Qwen3MoEModel, StreamingDetokenizer
from flash_moe.paths import model_path
from flash_moe.tokenizer import GGUFTokenizer
from run_q4k_f16_openvino import dequant_to_f16

def main():
    path = model_path()
    print('Loading model...', flush=True)
    model = Qwen3MoEModel(path, device='CPU', max_seq_len=128, cache_capacity=50, dequant_fn=dequant_to_f16)
    tok = GGUFTokenizer(gguf.GGUFReader(path))
    print('Model loaded. Testing with prompt...', flush=True)

    # Test 1: Simple prompt
    print('\n=== Test 1: Simple greeting ===')
    ids = tok.encode('Say: hello world', add_bos=False)
    stream = StreamingDetokenizer(tok)
    result = ''
    for produced, token in model.stream(ids, max_new_tokens=5, temperature=0, seed=42):
        text = stream.push(token)
        result += text
        print(text, end='', flush=True)
    print(stream.flush(), end='', flush=True)
    print()
    
    # Test 2: MoE explanation
    print('\n=== Test 2: MoE explanation ===')
    ids = tok.encode('What is Mixture of Experts?', add_bos=False)
    stream = StreamingDetokenizer(tok)
    result = ''
    for produced, token in model.stream(ids, max_new_tokens=8, temperature=0, seed=42):
        text = stream.push(token)
        result += text
        print(text, end='', flush=True)
    print(stream.flush(), end='', flush=True)
    print()
    
    # Test 3: Italian prompt
    print('\n=== Test 3: Italian prompt ===')
    ids = tok.encode('Cosa è un MoE?', add_bos=False)
    stream = StreamingDetokenizer(tok)
    result = ''
    for produced, token in model.stream(ids, max_new_tokens=6, temperature=0, seed=42):
        text = stream.push(token)
        result += text
        print(text, end='', flush=True)
    print(stream.flush(), end='', flush=True)
    print()
    
    model.close()
    print('\nAll tests completed successfully!')

if __name__ == '__main__':
    main()
