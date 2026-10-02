"""Rotary Positional Embedding, layout NeoX (half-split).

Perche' questo file esiste
-------------------------
Il RoPE ruota coppie di componenti di ogni testa di attenzione. Esistono due
layout, e confonderli produce un errore che non si vede nei numeri: entrambi
restituiscono valori nella stessa scala, quindi un test di magnitudine passa
in entrambi i casi.

- **interleaved (GPT-J)**: coppie adiacenti, (0,1), (2,3), (4,5), ...
- **half-split (NeoX)**: coppie a distanza fissa, (j, j + n_dims/2)

Qwen3.5-MoE usa il **NeoX**. Verificato su llama.cpp: in
``ggml_compute_forward_rope_flt`` il caso ``GGML_ROPE_TYPE_NEOX`` chiama
``rotate_pairs(n_dims, n_dims/2, ...)``, che per ogni indice pari ``i0``
prende ``ic = i0/2`` e ruota ``src[ic]`` con ``src[ic + n_dims/2]``. Su
CUDA, ``ggml_cuda_rope`` fa esattamente lo stesso:
``x0 = x[ix + n_offs/2]``, ``x1 = x[ix + n_offs/2 + n_dims/2]``.

Per il nostro modello ``head_dim = 256`` e ``rope_dim = 64``: ruotano le
prime 64 dimensioni di ogni testa, con 32 coppie, e le restanti 192 passano
attraverso. E' il RoPE parziale di Qwen3.

Le frequenze sono ``theta_j = pos * freq_base ** (-2j / rope_dim)``, con
``freq_base = 1e7`` per questo modello.

Sul ``rope.dimension_sections``
-------------------------------
Il GGUF porta ``rope.dimension_sections = [11, 11, 10, 0]``, la somma e' 32
coppie = 64 dimensioni. In llama.cpp questo attiva il modo IMROPE, dove la
cache ruota scegliendo per ogni coppia quale componente di posizione usare
(``sector % 3`` seleziona fra t, h, w ed e). Per un modello solo testo pero'
le tre componenti valgono tutte ``pos`` e la quarta non viene mai selezionata
(``llama_graph_input_pos::set_input`` scrive ``pos_t = pos_h = pos_w = pos``
e ``pos_e = 0``, e con ``[11,11,10,0]`` i settori 0..31 sono tutti coperti da
t, h, w). Quindi su testo le sezioni sono un no-op e il risultato e' un RoPE
NeoX parziale normale.

Non e' pero' vero in generale: su input multimodale le sezioni cambiano
davvero il risultato, perche' separano la posizione temporale da quella
spaziale. Percio' il valore viene conservato e validato, e questa classe
rifiuta di accettarlo come se fosse irrilevante: se le sezioni non sono
tutte uguali e non c'e' un modo per gestirle, e' un errore esplicito invece
di un silenzioso no-op sbagliato.
"""

from __future__ import annotations

import numpy as np

DEFAULT_HEAD_DIM: int = 256
DEFAULT_NUM_HEADS: int = 16
DEFAULT_NUM_KV_HEADS: int = 2
DEFAULT_ROPE_DIM: int = 64
DEFAULT_FREQ_BASE: float = 10000000.0

# Nomi storici, mantenuti per chi li importa: il default di head_dim era
# 128 (Qwen3 dense) mentre questo modello usa 256.
DEFAULT_HEAD_COUNT = DEFAULT_NUM_HEADS
DEFAULT_KV_HEAD_COUNT = DEFAULT_NUM_KV_HEADS


class RoPE:
    """Rotary Positional Embedding parziale, layout NeoX.

    ``head_dim`` e' la dimensione completa della testa, ``rope_dim`` quante
    dimensioni ricevono la rotazione. Su questo modello 256 e 64.
    """

    def __init__(
        self,
        head_dim: int = DEFAULT_HEAD_DIM,
        rope_dim: int = DEFAULT_ROPE_DIM,
        freq_base: float = DEFAULT_FREQ_BASE,
        max_seq_len: int = 4096,
        dimension_sections: tuple[int, int, int, int] | None = None,
    ) -> None:
        if rope_dim > head_dim:
            raise ValueError(
                f"rope_dim ({rope_dim}) non puo' superare head_dim ({head_dim})")
        if rope_dim % 2 != 0:
            raise ValueError(f"rope_dim ({rope_dim}) deve essere pari")
        if max_seq_len < 1:
            raise ValueError(f"max_seq_len ({max_seq_len}) deve essere >= 1")

        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.freq_base = freq_base
        self.max_seq_len = max_seq_len

        # Le sezioni sono in coppie: la loro somma deve essere rope_dim/2.
        # Se la somma non torna, la rotazione coprirebbe un numero di coppie
        # diverso da quello che il modello si aspetta, e il risultato sarebbe
        # silenziosamente sbagliato.
        if dimension_sections is not None:
            sections = tuple(int(s) for s in dimension_sections)
            if len(sections) != 4:
                raise ValueError(
                    "dimension_sections deve avere 4 elementi, "
                    f"ne ha {len(sections)}")
            total = sum(sections)
            if total * 2 != rope_dim:
                raise ValueError(
                    f"dimension_sections sommano a {total} coppie = "
                    f"{total * 2} dimensioni, ma rope_dim e' {rope_dim}. "
                    "Le sezioni sono contate in coppie: la somma deve essere "
                    "rope_dim/2.")
            if not (sections[0] > 0 and sections[1] > 0):
                # llama.cpp: use_mrope richiede sections[0] > 0 && sections[1] > 0
                raise ValueError(
                    f"dimension_sections {sections}: il modo mrope richiede "
                    "almeno una sezione t e una sezione h non vuote.")
            # Su testo t, h, w valgono tutti 'pos' e la sezione e non viene
            # mai selezionata, quindi le sezioni sono un no-op. Lo si registra
            # per non perdere l'informazione quando il modello sara' usato
            # su input multimodale.
            self.dimension_sections = sections
            self.sections_are_noop_on_text = True
        else:
            self.dimension_sections = None
            self.sections_are_noop_on_text = None

        # theta_j = pos * freq_base ** (-2j / rope_dim), per j in
        # [0, rope_dim/2). Cioe' 32 frequenze per rope_dim = 64.
        j = np.arange(rope_dim // 2, dtype=np.float32)
        inv_freq = (1.0 / (freq_base ** (2.0 * j / rope_dim))).astype(np.float32)
        self._inv_freq = inv_freq
        self._cache_len = 0
        self._cos = np.empty((0, rope_dim // 2), dtype=np.float32)
        self._sin = np.empty((0, rope_dim // 2), dtype=np.float32)

    # ------------------------------------------------------------------ cache

    def _ensure_cache(self, upto: int) -> None:
        """Estende la cache di posizioni fino a ``upto`` esclusa.

        La tabella cresce a scatti: si estende almeno fino alla lunghezza
        richiesta, e per un decode token-per-token si accresce di un blocco
        alla prima chiamata e poi resta valida, senza ricalcolare nulla.
        """
        if upto <= self._cache_len:
            return
        new_len = max(upto, max(16, self._cache_len * 2))
        pos = np.arange(new_len, dtype=np.float32)
        theta = pos[:, None] * self._inv_freq[None, :]     # (pos, rope_dim/2)
        self._cos = np.cos(theta).astype(np.float32)
        self._sin = np.sin(theta).astype(np.float32)
        self._cache_len = new_len

    def _tables(self, n_pos: int, seq_offset: int):
        """Tabelle per ``n_pos`` posizioni che iniziano a ``seq_offset``."""
        self._ensure_cache(seq_offset + n_pos)
        return self._cos[seq_offset:seq_offset + n_pos], \
            self._sin[seq_offset:seq_offset + n_pos]

    # ------------------------------------------------------------------- api

    def apply(self, x: np.ndarray, seq_offset: int = 0) -> np.ndarray:
        """Applica la rotazione a ``x`` di forma ``(..., n_pos, head_dim)``.

        Anche un vettore 1D di ``head_dim`` componenti e' accettato e
        trattato come un solo token; l'uscita ha la stessa forma
        dell'ingresso.

        Le prime ``rope_dim`` componenti ruotano a coppie (j, j+rope_dim/2);
        le restanti passano invariate.

        La posizione e' determinata dall'asse ``n_pos``, non dalla riga: il
        chiamante passa ``(n_heads, n_tokens, head_dim)`` e tutte le teste di
        un token devono ricevere la stessa posizione. Indicizzando cos/sin per
        riga appiattita, la testa ``h`` del token ``t`` prenderebbe la
        posizione ``t * n_heads + h``, e su un modello con 16 teste l'angolo
        della prima testa del secondo token sarebbe 16 volte troppo grande.

        Le tabelle cos/sin vengono inserite negli assi senza toccare ``x``:
        ``x.reshape(n_pos, n_rot, head_dim)`` su un array contiguo
        ``(n_rot, n_pos, head_dim)`` non raggruppa le righe, le mescola, e il
        risultato sembra funzionare solo perche' anche ``reshape`` indietro
        rimette tutto com'era: l'errore si vede solo confrontando i valori.
        """
        arr = np.asarray(x)
        head_dim = arr.shape[-1]
        if head_dim != self.head_dim:
            raise ValueError(
                f"attesa head_dim={self.head_dim}, trovata {head_dim}")

        was_1d = arr.ndim == 1
        if was_1d:
            arr = arr.reshape(1, self.head_dim)

        n_pos = arr.shape[-2]
        cos, sin = self._tables(n_pos, seq_offset)     # (n_pos, half)
        # gli assi delle teste stanno prima di quello delle posizioni: si
        # inserisce un 1 per ciascuno cosi' che il broadcasting funzioni.
        lead = arr.ndim - 2
        cos = cos.reshape((1,) * lead + cos.shape)
        sin = sin.reshape((1,) * lead + sin.shape)

        half = self.rope_dim // 2
        x_rot = arr[..., :self.rope_dim]
        x_rest = arr[..., self.rope_dim:]

        # Coppie NeoX: (j, j + half) per j in [0, half).
        x0 = x_rot[..., :half]
        x1 = x_rot[..., half:]
        rot = np.concatenate([x0 * cos - x1 * sin, x0 * sin + x1 * cos],
                             axis=-1)
        if x_rest.shape[-1] == 0:
            out = rot
        else:
            out = np.concatenate([rot, x_rest], axis=-1)
        return out.reshape(x.shape) if was_1d else out

    # nome storico mantenuto per chi lo chiama
    apply_rotary = apply

    def get_cache(self, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
        """Tabelle (cos, sin) di forma ``(n_pos, rope_dim // 2)``."""
        return self._tables(n_pos, 0)

    def describe(self) -> str:
        sec = self.dimension_sections
        return (f"RoPE NeoX head_dim={self.head_dim} rope_dim={self.rope_dim} "
                f"freq_base={self.freq_base:g} "
                f"sections={sec if sec is not None else 'assenti'}")
