"""Verifica del tokenizer ricostruito dai metadati GGUF.

Un encoder sbagliato non produce errori: produce id plausibili e testo
leggibile, solo numeri diversi da quelli che il modello aspetta. Percio' il
test controlla la fedeltà, non solo che non crashi.
"""
import sys
sys.path.insert(0, 'src')

import gguf
from flash_moe.paths import model_path
from flash_moe.tokenizer import GGUFTokenizer

r = gguf.GGUFReader(model_path())
att = {t.name: t for t in r.tensors}["token_embd.weight"]
rows = int(att.shape[1])

tok = GGUFTokenizer(r)
print(f"modello={tok.model} pre={tok.pre} vocab={tok.vocab_size} "
      f"merges={len(tok._missing)} orfani")

ok = bad = 0
for label, res, detail in tok.verify(rows):
    tag = "PASS" if res else "FAIL"
    if res:
        ok += 1
    else:
        bad += 1
    print(f"  {tag}  {label}  ({detail})")

print("\nsample:")
for text in ["The capital of France is Paris.", "x = 42 + y"]:
    enc = tok._tok.encode(text)
    print(f"  {text!r}")
    for piece, tid in zip(enc.tokens, enc.ids):
        print(f"    {tid:6d}  {piece!r}")

# coerenza embedding/token: ogni id deve avere una riga leggibile
print("\ncoerenza con token_embd:")
probe = tok.encode("The capital of France is Paris.")
print(f"  id {probe[0]} -> riga {probe[0]} esiste: {probe[0] < rows}")

print(f"\n{'=' * 62}\n PASSATI: {ok}   FALLITI: {bad}\n{'=' * 62}")
sys.exit(1 if bad else 0)