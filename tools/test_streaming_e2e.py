"""Il percorso streaming, provato sul modello vero.

Cosa viene verificato, e perche' due cose diverse
-------------------------------------------------
Il progetto si chiama streaming perche' i pesi degli expert non stanno in RAM:
si leggono dal GGUF quando il routing li sceglie. Finora questo e' stato
verificato a pezzi, ognuno col suo test, ma mai il percorso intero con la
cache al valore che un client usa davvero. Due rischi concreti:

1. la cache piccola. Se l'eviction della LRU sbaglia, un expert riletto dal
   disco arriva diverso e il testo si corrompe a meta' frase, senza errore.
   Il test genera gli stessi token con cache 64 e con cache 1 e pretende che
   siano identici, e che il run con cache 1 abbia davvero riletto di piu'.

2. la detokenizzazione. Il BPE e' byte-level, quindi un carattere UTF-8 puo'
   essere spaccato fra due token. Su questo modello l'id 248 sono i byte
   \\x9a e l'id 222 sono \\x80, che col token precedente formano l'emoji
   U+1F680: presi uno alla volta non sono caratteri validi e diventano
   spazzatura. E' il difetto che fa da solo: senza errori, con accenti
   apparentemente a posto, perche' 'e' e 'a' stanno in un token solo.

Perche' non si confronta il testo con llama.cpp
-----------------------------------------------
Il campionamento usa PRNG diversi, quindi a parita' di seed i due runtime
danno token diversi anche se il campionamento e' identico. Qui si usa il
greedy (temperature 0), che e' deterministico, e la parita' numerica e'
gia' coperta da tools/test_golden_parity.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

# La console di Windows e' cp1252 e non sa scrivere i caratteri che questi
# test usano nelle etichette: senza questo il test muore su un accento invece
# di dire cosa ha trovato.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

import gguf  # noqa: E402
from run_q4k_f16_openvino import dequant_to_f16  # noqa: E402
from flash_moe import Qwen3MoEModel  # noqa: E402
from flash_moe.paths import model_path  # noqa: E402
from flash_moe.tokenizer import GGUFTokenizer, StreamingDetokenizer  # noqa: E402

NEW_TOKENS = 4
BIG_CACHE = 64
TINY_CACHE = 1

ok = bad = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global ok, bad
    if cond:
        ok += 1
        print(f"  PASS  {label}  ({detail})")
    else:
        bad += 1
        print(f"  FAIL  {label}  ({detail})")


def test_detokenization(tok: GGUFTokenizer) -> None:
    print("1. detokenizzazione incrementale")
    cases = [
        "café",
        "naïve",
        "l'acqua è più fredda",
        "Il sole splende sopra il mare",
        "The capital of France is Paris.",
        "日本語のテキスト",
        "emoji: 🙂🚀",
        "mixed àèìòù e 🚀 e 日本語 fine",
    ]
    for text in cases:
        ids = tok.encode(text)
        block = tok.decode(ids)
        stream = StreamingDetokenizer(tok)
        pieces = [stream.push(i) for i in ids]
        joined = "".join(pieces) + stream.flush()
        check(f"{text[:24]!r} token per token = blocco", joined == block,
              f"{len(pieces)} token, {sum(1 for p in pieces if not p)} vuoti"
              if joined != block else f"{len(ids)} token")

    # Il caso per cui la classe esiste: il razzo e' spaccato su tre token e
    # solo l'ultimo lo completa. Senza questo controllo una regressione
    # silenziosa (perdita della coda di byte) passerebbe: si vedrebbe solo un
    # carattere in meno, e su un output di testo sembrerebbe un refuso.
    ids = tok.encode("emoji: 🙂🚀")
    block = tok.decode(ids)
    naive = "".join(tok.decode([i]) for i in ids)
    stream = StreamingDetokenizer(tok)
    pieces = [stream.push(i) for i in ids]
    joined = "".join(pieces) + stream.flush()
    check("il percorso naïve mette spazzatura dove lo stream no",
          "�" in naive and "�" not in joined,
          f"naïve {naive!r} contro stream {joined!r}")
    check("lo stream ricostruisce lo stesso testo del blocco",
          joined == block, f"{joined!r}")
    empty_at = [i for i, p in enumerate(pieces) if not p]
    check("il token che apre il carattere multi-byte non emette nulla",
          len(empty_at) >= 1, f"token {empty_at} senza testo")

    # Un token di controllo non e' testo: stamparne il nome dentro l'output
    # e' sbagliato anche se non e' un errore.
    control = next((i for i, t in enumerate(tok.token_types) if t != 1), None)
    if control is not None:
        stream = StreamingDetokenizer(tok)
        check("i token di controllo non producono testo",
              stream.push(control) == "", f"id {control} "
              f"{tok.tokens[control]!r} type={tok.token_types[control]}")

    # Stream troncato a meta' carattere: e' un errore da dire, non spazzatura
    # da stampare, altrimenti in coda all'output sembrerebbe un carattere vero.
    rocket = tok.encode("🚀")
    stream = StreamingDetokenizer(tok)
    for i in rocket[:-1]:
        stream.push(i)
    try:
        stream.flush()
        check("stream troncato segnalato", False, "nessuna eccezione")
    except UnicodeDecodeError:
        check("stream troncato segnalato", True, "UnicodeDecodeError")

    # Un detokenizzatore riusato tra un prompt e il successivo deve ripartire
    # pulito: se no, i byte rimasti dal testo precedente finiscono nel nuovo.
    stream = StreamingDetokenizer(tok)
    first = stream.push_all(tok.encode("🚀")) + stream.flush()
    stream.reset()
    second = stream.push_all(tok.encode("ciao")) + stream.flush()
    check("reset tra due prompt", first == "🚀" and second == "ciao",
          f"{first!r} e {second!r}")


def main() -> int:
    path = model_path()
    reader = gguf.GGUFReader(path)
    tok = GGUFTokenizer(reader)

    test_detokenization(tok)

    print("\n2. generazione con cache piccola e riletture da SSD")
    ids = tok.encode("La capital of France is", add_bos=False)
    model = Qwen3MoEModel(path, device="CPU", max_seq_len=256,
                          cache_capacity=BIG_CACHE, dequant_fn=dequant_to_f16)

    def run(capacity: int) -> tuple[list[int], dict[str, int], int]:
        model.store.capacity = capacity
        model.backend.cache_capacity = capacity
        model.store.clear()
        model.backend.clear_cache()
        before = (model.store.stats.bytes_read, model.store.stats.misses,
                  model.store.stats.prefetch_hits, model.store.stats.hits)
        out = model.generate(ids, max_new_tokens=NEW_TOKENS, temperature=0.0)
        after = (model.store.stats.bytes_read, model.store.stats.misses,
                 model.store.stats.prefetch_hits, model.store.stats.hits)
        delta = {n: a - b for n, a, b in zip(
            ("bytes_read", "misses", "prefetch_hits", "hits"),
            after, before)}
        return out, delta, len(model.store.resident_keys())

    big, big_stats, big_res = run(BIG_CACHE)
    tiny, tiny_stats, tiny_res = run(TINY_CACHE)

    print(f"  cache {BIG_CACHE}: {big[len(ids):]} "
          f"({big_stats['bytes_read'] / 2**20:.0f} MiB da disco, "
          f"{big_stats['misses']} miss, {big_stats['prefetch_hits']} prefetch)")
    print(f"  cache {TINY_CACHE}: {tiny[len(ids):]} "
          f"({tiny_stats['bytes_read'] / 2**20:.0f} MiB da disco, "
          f"{tiny_stats['misses']} miss, {tiny_stats['prefetch_hits']} prefetch)")

    check("gli stessi token con cache 1 e cache 64", big == tiny,
          f"{big[len(ids):]} contro {tiny[len(ids):]}")
    check("la cache piccola rilette davvero di piu'",
          tiny_stats["bytes_read"] > big_stats["bytes_read"],
          f"{tiny_stats['bytes_read'] / 2**20:.0f} MiB contro "
          f"{big_stats['bytes_read'] / 2**20:.0f} MiB")
    check("il prefetch viene usato nel run con cache piccola",
          tiny_stats["prefetch_hits"] > 0, f"{tiny_stats['prefetch_hits']} hit")
    check("residenti entro la capacita' dichiarata",
          tiny_res <= TINY_CACHE and big_res <= BIG_CACHE,
          f"{tiny_res} con cache {TINY_CACHE}, {big_res} con cache {BIG_CACHE}")

    print("\n3. i token generati passano dallo stream")
    stream = StreamingDetokenizer(tok)
    text = ""
    for token in tiny[len(ids):]:
        text += stream.push(token)
    text += stream.flush()
    check("il testo generato non contiene spazzatura",
          "�" not in text, f"{text!r}")

    # Il confronto e' sul decode dei soli token normali, non su quello di
    # tutti. Il modello ha emesso un token di controllo (<|think|>) e le due
    # strade lo trattano diversamente di proposito: lo stream lo salta, il
    # decode sul blocco ne stampa il nome letterale. Quello giusto per un
    # client e' lo stream: '<|think|>' non e' testo da mostrare a chi legge.
    normal = [t for t in tiny[len(ids):] if tok.token_types[t] == 1]
    check("il testo e' quello che dà il decode sul blocco",
          text == tok.decode(normal),
          f"{text!r} contro {tok.decode(normal)!r}")
    if len(normal) != len(tiny) - len(ids):
        control = [t for t in tiny[len(ids):] if tok.token_types[t] != 1]
        check("il decode sul blocco stampa i token di controllo, lo stream no",
              tok.decode(tiny[len(ids):]) == text + "".join(
                  tok.tokens[t] for t in control),
              f"token di controllo {[tok.tokens[t] for t in control]}")

    model.close()
    print(f"\n{'=' * 66}\n PASSATI: {ok}   FALLITI: {bad}\n{'=' * 66}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())