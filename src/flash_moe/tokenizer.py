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

import codecs
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

    NORMAL = 1

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

        from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, decoders

        self._tok = Tokenizer(models.BPE(vocab=vocab, merges=pairs))
        self._tok.add_special_tokens([
            AddedToken(token, special=True)
            for token, token_type in zip(self.tokens, self.token_types)
            if token_type != self.NORMAL
        ])
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


class StreamingDetokenizer:
    """Detokenizzazione incrementale: un token alla volta, come fa un client.

    ``GGUFTokenizer.decode`` lavora su un blocco di id e va bene per un
    renderer che aspetta la risposta intera. Un client streaming deve emettere
    il testo appena il token esiste, e non puo' riusare ``decode([id])``: il
    BPE e' byte-level, quindi un carattere UTF-8 puo' essere spaccato fra due
    token e il pezzo finale da solo non e' un carattere valido.

    Il caso reale su questo modello: l'id 248 sono i byte ``\\x9a`` e l'id 222
    sono ``\\x80``, che insieme all'id precedente formano l'emoji U+1F680.
    Decodificati uno alla volta diventano due caratteri di sostituzione, e il
    client stampa ``emoji: <sostituzione><sostituzione>`` al posto del razzo.
    Non e' un problema di accenti: sono passati, perche' 'e' e 'a' stanno in
    un token solo. E' un problema dei caratteri da 3 e 4 byte.

    Qui i byte di ogni token passano da un decodificatore incrementale di
    UTF-8, che trattiene in coda la sequenza incompleta e la completa quando
    arriva il resto. Quello che esce token per token, concatenato, e' uguale
    a ``decode(ids)`` sul blocco intero.

    Una differenza rispetto a ``decode``, voluta. ``decode`` stampa il nome
    letterale dei token di controllo, cosi' un output del modello contiene
    ``<|think|>``; qui i token non normali non producono nulla, perche' un
    client non deve mostrare all'utente il nome di un token. Per confrontare
    le due strade, il blocco va decodificato sui soli token normali.
    """

    #: token_type 1 in GGUF. Gli altri sono CONTROL (3), USER_DEFINED (4) e
    #: UNUSED (5): non sono testo, e decodificarli stamperebbe il loro
    #: nome-letterale, tipo '<|vision_start|>', dentro l'output.
    NORMAL = GGUFTokenizer.NORMAL

    def __init__(self, tokenizer: GGUFTokenizer) -> None:
        self._tokens = tokenizer.tokens
        self._types = tokenizer.token_types
        self._byte_decoder = tokenizer._byte_decoder
        self._decoder = codecs.getincrementaldecoder("utf-8")()

    def push(self, token_id: int) -> str:
        """Il testo che questo token rende disponibili subito.

        Puo' essere vuoto: non e' un errore, e' il caso normale quando il
        token porta solo l'inizio di un carattere multi-byte. Il testo
        arrivesera' con il token successivo, e non e' andato perso.
        """
        if self._types[token_id] != self.NORMAL:
            return ""
        piece = self._tokens[token_id]
        try:
            raw = bytes(self._byte_decoder[ch] for ch in piece)
        except KeyError as error:
            raise KeyError(
                f"il token {token_id} ({piece!r}) contiene il carattere "
                f"{error.args[0]!r}, che non e' nella mappa byte-level: "
                "questo pezzo non e' ricostruibile"
            ) from error
        return self._decoder.decode(raw, final=False)

    def push_all(self, ids: list[int]) -> str:
        return "".join(self.push(token_id) for token_id in ids)

    def flush(self) -> str:
        """Chiude lo stream: svuota la coda di byte rimasti.

        Se la coda non e' vuota il testo era troncato a meta' carattere.
        Si solleva un errore invece di stampare un carattere di sostituzione:
        in uno stream finito male la spazzatura sarebbe indistinguibile da
        un carattere vero.
        """
        return self._decoder.decode(b"", final=True)

    def reset(self) -> None:
        """Ricomincia da uno stream pulito, per il prompt successivo."""
        self._decoder.reset()