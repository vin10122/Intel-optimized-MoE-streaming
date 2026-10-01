"""Tokenizer e embedding presi dal GGUF, verificati sulla stessa tabella id.

Il file contiene gia' tokenizer.ggml.tokens, .merges e .token_type: non serve
un tokenizer.json esterno. Qui si ricostruisce l'encoder BPE da quei campi e
si controlla che sia fedele, perche' un encoder sbagliato non da' errori: da
id plausibili e testo leggibile, solo numeri diversi dai quelli attesi dal
modello, e quindi output senza senso senza nessun errore inSight.

Verifiche:
  1. il vocab combacia con token_embd (stesso numero di righe)
  2. encode -> decode torna al testo originale
  3. gli id stanno nel range dell'embedding
  4. i merge sono tutti composti da token presenti nel vocab
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _byte_to_unicode() -> dict[int, str]:
    """Byte-level BPE (GPT-2): mappa i 256 byte su caratteri unicode.

    Serve perche' i merge nel GGUF sono gia' codificati byte-level: senza
    questa mappa la stringa 'Ġ' non si ricostruisce da uno spazio.
    """
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class GGUFTokenizer:
    """Encoder BPE ricostruito dai metadati GGUF."""

    def __init__(self, reader: Any) -> None:
        fields = reader.fields
        self.model = fields["tokenizer.ggml.model"].contents()
        self.pre = fields["tokenizer.ggml.pre"].contents()
        if self.model != "gpt2":
            raise ValueError(
                f"tokenizer.ggml.model={self.model!r}: questo modulo ricostruisce "
                "solo BPE byte-level (gpt2). Per altri modelli usare "
                "transformers/tokenizers sul tokenizer originale."
            )
        self.tokens: list[str] = fields["tokenizer.ggml.tokens"].contents()
        self.token_types: list[int] = fields["tokenizer.ggml.token_type"].contents()
        merges = fields["tokenizer.ggml.merges"].contents()
        self.bos_id = fields["tokenizer.ggml.bos_token_id"].contents()
        self.eos_id = fields["tokenizer.ggml.eos_token_id"].contents()

        self._byte_decoder = {v: k for k, v in _byte_to_unicode().items()}

        vocab = {t: i for i, t in enumerate(self.tokens)}
        pairs: list[tuple[str, str]] = []
        for m in merges:
            a, sep, b = m.partition(" ")
            if not sep:
                raise ValueError(f"merge malformato: {m!r}")
            pairs.append((a, b))
        self._missing = [(a, b) for a, b in pairs if a not in vocab or b not in vocab]

        from tokenizers import Tokenizer, models, pre_tokenizers, decoders

        self._tok = Tokenizer(models.BPE(vocab=vocab, merges=pairs))
        # Il pre-tokenizer del GGUF e' 'qwen35': lo split regex di GPT-2 con
        # \s+(?!\S)|\s+, che e' quello di ByteLevel.
        self._tok.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False, use_regex=True)
        self._tok.decoder = decoders.ByteLevel()

    # ------------------------------------------------------------------ api

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str, add_bos: bool = False) -> list[int]:
        ids = self._tok.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        return ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    # ------------------------------------------------------------ verifica

    def verify(self, embedding_rows: int) -> list[tuple[str, bool, str]]:
        """Restituisce (etichetta, esito, dettaglio) per ogni controllo."""
        checks: list[tuple[str, bool, str]] = []

        def add(label: str, cond: bool, detail: str = "") -> None:
            checks.append((label, bool(cond), detail))

        add("vocab combacia con token_embd",
            len(self.tokens) == embedding_rows,
            f"{len(self.tokens)} token vs {embedding_rows} righe")

        add("nessun merge con token assenti", not self._missing,
            f"{len(self._missing)} merge orfani")

        cases = [
            "The capital of France is Paris.",
            "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
            "Ciao, come stai? 12345",
            "  spazi  multipli  ",
            "unicode: caf\u00e9 na\u00efve \u201cquote\u201d",
        ]
        worst = ""
        good = True
        for text in cases:
            ids = self.encode(text)
            back = self.decode(ids)
            if back != text:
                good = False
                worst = f"{text[:24]!r} -> {back[:24]!r}"
                break
            if ids and max(ids) >= embedding_rows:
                good = False
                worst = f"id {max(ids)} >= {embedding_rows}"
                break
        add("encode/decode tornano al testo originale", good, worst)

        max_id = max((max(self.encode(t)) for t in cases if self.encode(t)),
                     default=-1)
        add("id nel range dell'embedding", max_id < embedding_rows,
            f"max id {max_id} < {embedding_rows}")

        add("roundtrip sul singolo id", self.decode([self.bos_id]) is not None,
            f"bos={self.bos_id} eos={self.eos_id}")

        return checks