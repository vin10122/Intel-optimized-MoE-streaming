#!/usr/bin/env python
"""Test con prompt reale - generazione di pochi token"""
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
    print('Carico modello...', flush=True)
    
    # Carico con cache ridotta per essere più veloce nell'inizializzazione
    model = Qwen3MoEModel(
        path, 
        device='CPU', 
        max_seq_len=128, 
        cache_capacity=50, 
        dequant_fn=dequant_to_f16
    )
    tok = GGUFTokenizer(gguf.GGUFReader(path))
    print('Modello caricato. Eseguo test con prompt reale...', flush=True)
    
    # Test con prompt in italiano
    prompts = [
        'Ciao',
        'Cosa è un Mixture of Experts?',
        'Say: hello',
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f'\n--- Test {i}: "{prompt}" ---')
        ids = tok.encode(prompt, add_bos=False)
        stream = StreamingDetokenizer(tok)
        
        # Genero solo 5 token massimi
        token_count = 0
        output_text = ''
        for produced, token in model.stream(
            ids, 
            max_new_tokens=5, 
            temperature=0,  # Greedy
            seed=42
        ):
            text = stream.push(token)
            output_text += text
            token_count += 1
            print(f'Token {token_count}: "{text}"', flush=True)
        
        # Flush finale
        final_text = stream.flush()
        if final_text:
            output_text += final_text
            print(f'Flushed: "{final_text}"', flush=True)
        
        print(f'Output completo: "{output_text}"')
    
    model.close()
    print('\n✅ Test completato con successo!')

if __name__ == '__main__':
    main()
