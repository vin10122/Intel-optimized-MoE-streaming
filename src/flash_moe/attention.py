"""Il layer di attenzione dei 10 blocchi full-attention di Qwen3.5-MoE.

Non un transformer generico: questo file implementa esattamente la
variante che il GGUF contiene, con le convenzioni di peso che ne derivano.

Configurazione letta dal GGUF di questo modello
----------------------------------------------
    hidden_size       = 2048
    head_count        = 16
    head_count_kv     = 2          (GQA con gruppo 8)
    key_length        = 256        (head_dim)
    value_length      = 256
    rope.dimension_count = 64      (parziale sulle prime 64 dimensioni)
    rope.freq_base    = 1e7
    layer_norm_rms_epsilon = 1e-6

Tensori di un blocco di attenzione (da blk.3), con i pesi in convenzione ggml
(ne[0] = asse di riduzione, ne[1] = asse di uscita):

    attn_q.weight      (2048, 8192)   = 16 head * (query 256 | gate 256)
    attn_k.weight      (2048,  512)   = 2 kv-head * 256
    attn_v.weight      (2048,  512)
    attn_q_norm.weight (256,)
    attn_k_norm.weight (256,)
    attn_output.weight (4096, 2048)   = (n_head*head_dim, hidden)

Le quattro cose che sbagliano easily
------------------------------------
1. **Il gate e' sigmoid, non silu.** In llama.cpp ``qwen35moe.cpp:347-350``
   e' ``ggml_sigmoid(gate)`` moltiplicato all'output dell'attenzione. silu e
   sigmoid differiscono anche nel segno: silu(negativo) e' negativo, sigmoid
   no.

2. **Q e gate sono interleaved per testa, non separati a meta'.** Le colonne
   di ``attn_q.weight`` sono ``[query(256) | gate(256)]`` ripetuto per 16
   teste, quindi il reshape e' ``(n_tokens, n_heads, 2, head_dim)`` e non
   ``(n_tokens, 2, n_heads*head_dim)``. Dividere a meta' sull'asse sbagliato
   e' un errore di forma su questo modello, ma su un modello con
   ``head_dim == hidden`` passerebbe silenziosamente.

3. **GQA: 2 kv-head verso 16 query-head, e il mapping e' per gruppi.** La
   testa di query ``h`` prende il kv-head ``h // 8``, cioe' ``np.repeat``.
   Nota il contrasto con il delta net, dove il mapping e' ``h % n_k_heads`` e
   serve ``np.tile``: sono due cose diverse e sbagliarle insieme produce
   due errori che si compensano in parte, il peggiore dei casi.

4. **I pesi si applicano senza trasporre.** Nel GGUF ``attn_output.weight`` e
   ``(4096, 2048)``, quindi ``out = attn @ w_out`` e non ``attn @ w_out.T``.

Il percorso di un token
-----------------------
    q_full = x @ attn_q          -> 16 teste da 256 query + 256 gate
    k      = x @ attn_k          -> 2 teste da 256
    v      = x @ attn_v
    q, k   = QK-norm su head_dim (RMSNorm, eps 1e-6), PRIMA del RoPE
    q, k   = RoPE NeoX parziale sulle prime 64 dimensioni
    k, v   = replicate ai gruppi (2 -> 16)
    out    = softmax(q k^T / 16) v, con maschera causale
    out    = out * sigmoid(gate)
    out    = out @ attn_output

Non ci sono attention sink ne bias: in llama.cpp la chiamata a build_attn
passa ``sinks = nullptr`` e ``kq_b = nullptr``, e il GGUF non ha i tensori
corrispondenti.
"""

from __future__ import annotations

import time

import numpy as np

from .norm import RMSNorm
from .rope import RoPE

DEFAULT_HIDDEN_SIZE: int = 2048
DEFAULT_NUM_HEADS: int = 16
DEFAULT_NUM_KV_HEADS: int = 2
DEFAULT_HEAD_DIM: int = 256
DEFAULT_ROPE_DIM: int = 64
DEFAULT_FREQ_BASE: float = 10000000.0
DEFAULT_RMS_EPS: float = 1e-6
DEFAULT_MAX_SEQ_LEN: int = 4096


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax numericamente stabile: si sottrae il massimo prima di ``exp``."""
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    rope: RoPE | None = None,
    seq_offset: int = 0,
    causal_mask: bool = True,
) -> np.ndarray:
    """Adapter sulla firma storica ``(batch, n_head, seq, head_dim)``.

    Va su ``grouped_attention`` e ne applica il GQA e la maschera causale,
    cosi' i chiamanti che ancora usano questa firma non duplicano la logica.
    Restituisce ``(batch, n_head, seq, head_dim)``.
    """
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    batch, n_head, n_q, head_dim = q.shape
    n_kv = k.shape[1]

    if rope is not None:
        # apply() vuole (n_pos, head_dim) per testa, quindi si passa
        # (batch*n_head, n_q, head_dim) e si ricompone.
        q = rope.apply(
            np.transpose(q, (0, 2, 1, 3)).reshape(-1, n_q, head_dim),
            seq_offset=seq_offset).reshape(batch, n_q, n_head, head_dim)
        q = np.transpose(q, (0, 2, 1, 3))
        n_kv_tok = k.shape[2]
        k = rope.apply(
            np.transpose(k, (0, 2, 1, 3)).reshape(-1, n_kv_tok, head_dim),
            seq_offset=seq_offset).reshape(batch, n_kv_tok, n_kv, head_dim)
        k = np.transpose(k, (0, 2, 1, 3))

    outs = []
    for b in range(batch):
        if causal_mask:
            o = grouped_attention(
                np.transpose(q[b], (1, 0, 2)),              # (n_head, T, d)
                k[b], v[b], n_tokens=n_q,
                seq_offset=seq_offset)
        else:
            # senza maschera: si sposta seq_offset a -n_q per neutralizzarla
            o = grouped_attention(
                np.transpose(q[b], (1, 0, 2)),
                k[b], v[b], n_tokens=n_q,
                seq_offset=-(n_q - 1))
        outs.append(np.transpose(o.reshape(n_q, n_head, head_dim), (1, 0, 2)))
    return np.stack(outs, axis=0)


class KVCache:
    """Cache di chiavi e valori per il decode incrementale.

    Tiene ``(seq, num_kv_heads, head_dim)`` e cresce di un token per volta.
    Il formato f16 e' una scelta: 262144 posizioni con 2 kv-head da 256
    sarebbero 256 MiB in f16 e 512 in f32 per layer, e con 10 layer di
    attenzione la differenza non e' trascurabile. I prodotti che ci
   servono sono su pochi token alla volta, quindi la promozione a f32
    avviene dentro l'attenzione e non nella cache.
    """

    def __init__(self, num_kv_heads: int, head_dim: int,
                 max_seq_len: int = 2048, dtype=np.float16) -> None:
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.reset()

    def reset(self) -> None:
        self.keys = np.zeros((0, self.num_kv_heads, self.head_dim),
                             dtype=self.dtype)
        self.values = np.zeros((0, self.num_kv_heads, self.head_dim),
                               dtype=self.dtype)
        self.seq_len = 0

    @property
    def length(self) -> int:
        return self.seq_len

    def append(self, k: np.ndarray, v: np.ndarray) -> None:
        """Aggiunge token in coda.

        ``k`` e ``v`` sono di forma ``(n_tokens, num_kv_heads, head_dim)``.
        """
        k = np.asarray(k)
        v = np.asarray(v)
        if k.ndim == 2:
            k = k.reshape(1, self.num_kv_heads, self.head_dim)
            v = v.reshape(1, self.num_kv_heads, self.head_dim)
        if k.shape != v.shape:
            raise ValueError(
                f"k {k.shape} e v {v.shape} devono avere la stessa forma")
        if k.shape[1] != self.num_kv_heads or k.shape[2] != self.head_dim:
            raise ValueError(
                f"atteso (n_tokens, {self.num_kv_heads}, {self.head_dim}), "
                f"trovato {k.shape}")
        n = k.shape[0]
        if self.seq_len + n > self.max_seq_len:
            raise ValueError(
                f"cache piena: {self.seq_len + n} > {self.max_seq_len}. "
                "Su questo modello context_length e' 262144, quindi un "
                "max_seq_len piu' piccolo va scelto di proposito.")

        self.keys = np.concatenate([self.keys, k.astype(self.dtype, copy=False)],
                                   axis=0)
        self.values = np.concatenate(
            [self.values, v.astype(self.dtype, copy=False)], axis=0)
        self.seq_len += n

    def get(self) -> tuple[np.ndarray, np.ndarray]:
        return self.keys, self.values


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistica a due rami: non overflowa ne' su +800 ne' su -800."""
    out = np.empty_like(x, dtype=np.result_type(x, np.float32))
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def grouped_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    n_tokens: int,
    seq_offset: int,
    timings: dict[str, float] | None = None,
) -> np.ndarray:
    """Attenzione causale con GQA, batch 1.

    ``q`` di forma ``(n_tokens, n_head, head_dim)``; ``k`` e ``v`` di forma
    ``(n_kv_tokens, n_kv_head, head_dim)``. Restituisce
    ``(n_tokens, n_head * head_dim)``.

    Il GQA e' ``np.repeat`` sulle teste: la testa di query ``h`` prende il
    kv-head ``h // (n_head // n_kv)``. E' il mapping standard, diverso dal
    delta net che usa ``np.tile`` su ``h % n_k_heads``.
    """
    n_head = q.shape[1]
    head_dim = q.shape[2]
    n_kv = k.shape[1]

    if n_head % n_kv != 0:
        raise ValueError(f"n_head ({n_head}) deve essere multiplo di n_kv ({n_kv})")
    group = n_head // n_kv

    started = time.perf_counter() if timings is not None else 0.0
    # La cache e' f16: score e softmax in f32, perche' in f16 einsum e' anche
    # molto piu' lento e la somma su head_dim=256 perderebbe i contributi.
    qf = np.asarray(q, dtype=np.float32)
    kf = np.asarray(k, dtype=np.float32)
    vf = np.asarray(v, dtype=np.float32)

    if group > 1:
        kf = np.repeat(kf, group, axis=1)
        vf = np.repeat(vf, group, axis=1)
    if timings is not None:
        timings["attention_cast_and_gqa"] = (
            timings.get("attention_cast_and_gqa", 0.0)
            + time.perf_counter() - started
        )

    started = time.perf_counter() if timings is not None else 0.0
    scores = np.einsum("qhd,khd->qhk", qf, kf) * (1.0 / np.sqrt(head_dim))

    # Maschera causale: il token alla posizione assoluta seq_offset + q puo'
    # vedere i token fino a quella posizione inclusa. Con cache e decode il
    # token corrente e' l'ultimo, quindi di norma non maschera niente; la
    # maschera serve nel prefill, dove piu' token avanzano insieme.
    q_pos = seq_offset + np.arange(n_tokens)
    k_pos = np.arange(kf.shape[0])
    # scores e' (n_tokens, n_head, n_kv_tokens): la maschera va inserita
    # sull'asse delle teste, altrimenti non si trasmette.
    mask = k_pos[None, :] > q_pos[:, None]
    if mask.any():
        scores = np.where(mask[:, None, :], -np.inf, scores)
    if timings is not None:
        timings["attention_scores_and_mask"] = (
            timings.get("attention_scores_and_mask", 0.0)
            + time.perf_counter() - started
        )

    started = time.perf_counter() if timings is not None else 0.0
    scores -= scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    p = e / e.sum(axis=-1, keepdims=True)
    if timings is not None:
        timings["attention_softmax"] = (
            timings.get("attention_softmax", 0.0)
            + time.perf_counter() - started
        )

    started = time.perf_counter() if timings is not None else 0.0
    out = np.einsum("qhk,khd->qhd", p, vf)
    if timings is not None:
        timings["attention_values"] = (
            timings.get("attention_values", 0.0) + time.perf_counter() - started
        )
    return out.reshape(n_tokens, n_head * head_dim)


class AttentionLayer:
    """Il layer di attenzione di Qwen3.5-MoE.

    I pesi si passano gia' in f32 e senza trasporre: sono nella convenzione
    ggml del GGUF. Se si preferisce passarli cosi' come escono dal GGUF,
    sono convertiti una volta sola alla costruzione del blocco, non a ogni
    token.
    """

    def __init__(
        self,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_heads: int = DEFAULT_NUM_HEADS,
        num_kv_heads: int = DEFAULT_NUM_KV_HEADS,
        head_dim: int = DEFAULT_HEAD_DIM,
        rope_dim: int = DEFAULT_ROPE_DIM,
        freq_base: float = DEFAULT_FREQ_BASE,
        eps: float = DEFAULT_RMS_EPS,
        dimension_sections: tuple[int, int, int, int] | None = None,
        kv_head_dim: int | None = None,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    ) -> None:
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) deve essere multiplo di "
                f"num_kv_heads ({num_kv_heads})")
        if kv_head_dim is None:
            kv_head_dim = head_dim
        if rope_dim > kv_head_dim:
            raise ValueError(
                f"rope_dim ({rope_dim}) non puo' superare kv_head_dim "
                f"({kv_head_dim}): il RoPE ruota anche k")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_head_dim = kv_head_dim
        self.eps = eps
        self.group = num_heads // num_kv_heads
        # Q e K possono avere head_dim diverse (Qwen3 dense: 128 e 128,
        # questo modello: 256 e 256). Quando differiscono servono due RoPE,
        # perche' ciascuno e' costruito sulla lunghezza della propria testa e
        # applicarlo sulla lunghezza sbagliata e' un errore di forma che su
        # forme quadrate passerebbe in silenzio. `rope` resta quello di Q.
        self.rope = RoPE(head_dim=head_dim, rope_dim=rope_dim,
                         freq_base=freq_base,
                         max_seq_len=max_seq_len,
                         dimension_sections=dimension_sections)
        self.rope_k = self.rope if kv_head_dim == head_dim else RoPE(
            head_dim=kv_head_dim, rope_dim=rope_dim,
            freq_base=freq_base, max_seq_len=max_seq_len,
            dimension_sections=dimension_sections)
        # scale 1/sqrt(head_dim): su questo modello 1/16. Se il GGUF avesse
        # attention.scale, andrebbe letto da li' invece che derivato.
        self.scale = 1.0 / float(np.sqrt(head_dim))
        self._cache: KVCache | None = None
        self.profile_timings: dict[str, float] | None = None
        self.projection_backend = None

    # ---------------------------------------------------------------- cache

    def init_cache(self, max_seq_len: int) -> None:
        self._cache = KVCache(self.num_kv_heads, self.kv_head_dim,
                              max_seq_len=max_seq_len)

    @property
    def cache(self) -> KVCache | None:
        return self._cache

    def reset_cache(self) -> None:
        if self._cache is not None:
            self._cache.reset()

    # -------------------------------------------------------------- pesi

    def prepare_weights(
        self,
        w_q: np.ndarray,
        w_k: np.ndarray,
        w_v: np.ndarray,
        w_q_norm: np.ndarray,
        w_k_norm: np.ndarray,
        w_out: np.ndarray,
        keep_f16: bool = False,
    ) -> dict[str, np.ndarray]:
        """Converte i pesi una volta sola e ne verifica le forme.

        La verifica e' qui e non al primo forward perche' una forma sbagliata
        che arriva al calcolo fallisce con un messaggio che sembra un bug di
        broadcasting, mentre qui dice cosa non torna.

        Con ``keep_f16`` i pesi restano f16 e non vengono copiati in f32. Serve
        al percorso che tiene l'intero modello in RAM: i pesi di attenzione di
        questo modello sono 1028M parametri, quindi conservarli in f32
        costerebbe 1,9 GiB in piu' senza guadagno, perche' il prodotto
        matrice-matrice e la riduzione della QK-norm sono già in f32.
        """
        want = {
            "q": (self.hidden_size, self.num_heads * 2 * self.head_dim),
            "k": (self.hidden_size, self.num_kv_heads * self.kv_head_dim),
            "v": (self.hidden_size, self.num_kv_heads * self.kv_head_dim),
            "q_norm": (self.head_dim,),
            "k_norm": (self.kv_head_dim,),
            "out": (self.num_heads * self.head_dim, self.hidden_size),
        }
        got = {"q": w_q, "k": w_k, "v": w_v,
               "q_norm": w_q_norm, "k_norm": w_k_norm, "out": w_out}
        for name, arr in got.items():
            if tuple(arr.shape) != want[name]:
                raise ValueError(
                    f"peso '{name}' di forma {tuple(arr.shape)}, attesa "
                    f"{want[name]}. Convenzione ggml: ne[0] e' l'asse di "
                    f"riduzione, ne[1] quello di uscita.")
        if keep_f16:
            return {n: np.ascontiguousarray(a, dtype=np.float16)
                    for n, a in got.items()}
        return {n: np.ascontiguousarray(a, dtype=np.float32)
                for n, a in got.items()}

    # ------------------------------------------------------------ forward

    def forward(
        self,
        x: np.ndarray,
        weights: dict[str, np.ndarray],
        cache: KVCache | None = None,
    ) -> np.ndarray:
        """Avanza l'attenzione di ``n_tokens`` e restituisce ``(n_tokens, hidden)``.

        ``x`` e' di forma ``(n_tokens, hidden_size)``. Con la cache attiva il
        ``seq_offset`` e' la posizione assoluta del primo token di ``x``, e
        la cache deve gia' contenere i token precedenti.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self.hidden_size:
            raise ValueError(
                f"x di forma {x.shape}, attesa (n_tokens, {self.hidden_size})")
        n_tokens = x.shape[0]
        seq_offset = cache.seq_len if cache is not None else 0
        timings = self.profile_timings

        # --- proiezioni, senza trasporre: i pesi sono in convenzione ggml
        started = time.perf_counter() if timings is not None else 0.0
        if self.projection_backend is None:
            q_full = x @ weights["q"]        # (T, n_head * 2 * head_dim)
            k_full = x @ weights["k"]        # (T, n_kv * head_dim)
            v_full = x @ weights["v"]
        else:
            q_full, k_full, v_full = self.projection_backend.input_projections(x)
        if timings is not None:
            timings["projections"] = (
                timings.get("projections", 0.0) + time.perf_counter() - started
            )

        # --- Q e gate sono interleaved per testa: [query | gate] per head.
        started = time.perf_counter() if timings is not None else 0.0
        # Non e' una divisione a meta' sull'asse delle feature.
        qg = q_full.reshape(n_tokens, self.num_heads, 2, self.head_dim)
        q = qg[:, :, 0, :]                  # (T, n_head, head_dim)
        gate = qg[:, :, 1, :]                # (T, n_head, head_dim)

        k = k_full.reshape(n_tokens, self.num_kv_heads, self.kv_head_dim)
        v = v_full.reshape(n_tokens, self.num_kv_heads, self.kv_head_dim)

        # --- QK-norm su head_dim, prima del RoPE (qwen35moe.cpp:305,311)
        q = self._qk_norm(q, weights["q_norm"])
        k = self._qk_norm(k, weights["k_norm"])
        if timings is not None:
            timings["qk_norm_and_reshape"] = (
                timings.get("qk_norm_and_reshape", 0.0)
                + time.perf_counter() - started
            )

        # --- RoPE su q e k. V non ruotata.
        started = time.perf_counter() if timings is not None else 0.0
        # rope.apply prende (..., n_pos, head_dim): la posizione e' l'ultimo
        # asse, quindi q e k (n_tokens, n_head, head_dim) vanno passati come
        # (n_head, n_tokens, head_dim) e poi tornati a token per primo.
        q = np.ascontiguousarray(
            self.rope.apply(np.transpose(q, (1, 0, 2)), seq_offset=seq_offset))
        q = np.ascontiguousarray(np.transpose(q, (1, 0, 2)))
        k = np.ascontiguousarray(
            self.rope_k.apply(np.transpose(k, (1, 0, 2)), seq_offset=seq_offset))
        k = np.ascontiguousarray(np.transpose(k, (1, 0, 2)))
        if timings is not None:
            timings["rope_and_layout"] = (
                timings.get("rope_and_layout", 0.0)
                + time.perf_counter() - started
            )

        # --- K e V per l'attenzione: (n_kv_tokens, n_kv_head, head_dim).
        # k e v sono gia' in quest'ordine: trasporli qui produrrebbe
        # (n_kv_head, n_kv_tokens, head_dim) e far leggere a grouped_attention
        # il numero di token come numero di teste.
        started = time.perf_counter() if timings is not None else 0.0
        if cache is not None:
            cache.append(np.ascontiguousarray(k),
                         np.ascontiguousarray(v))
            kk = cache.keys
            vv = cache.values
        else:
            kk = np.ascontiguousarray(k)
            vv = np.ascontiguousarray(v)
        if timings is not None:
            timings["kv_cache"] = (
                timings.get("kv_cache", 0.0) + time.perf_counter() - started
            )

        started = time.perf_counter() if timings is not None else 0.0
        out = grouped_attention(q, kk, vv, n_tokens, seq_offset, timings)
        if timings is not None:
            timings["grouped_attention"] = (
                timings.get("grouped_attention", 0.0)
                + time.perf_counter() - started
            )

        # --- gate sigmoid, non silu (qwen35moe.cpp:347-350)
        started = time.perf_counter() if timings is not None else 0.0
        out = out * _sigmoid(gate.reshape(n_tokens, -1))

        # --- proiezione di uscita: attn_output e' (n_head*head_dim, hidden)
        if self.projection_backend is None:
            result = out @ weights["out"]
        else:
            result = self.projection_backend.output_projection(out)
        if timings is not None:
            timings["gate_and_output_projection"] = (
                timings.get("gate_and_output_projection", 0.0)
                + time.perf_counter() - started
            )
        return result

    def _qk_norm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """RMSNorm su head_dim, per ogni token e ogni testa.

        Si scrive qui invece di riusare ``RMSNorm`` perche' quello normalizza
        sull'ultimo asse e restituirebbe f16: qui la riduzione e' in f32 e il
        risultato resta f32, che e' quello che serve per il RoPE e per gli
        score.
        """
        w = weight.astype(np.float32)
        rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * w


class Qwen35MoeAttentionBlock:
    """Il blocco completo: norm, attenzione, norm, e lascia al MoE.

    Non esegue la parte MoE: quella vive in ``moe_layer.run_moe_layer`` e non
    deve essere duplicata. Qui finisce tutto quello che e' attenzione.
    """

    def __init__(self, attn: AttentionLayer, w_attn_norm: np.ndarray,
                 w_post_norm: np.ndarray, eps: float = DEFAULT_RMS_EPS) -> None:
        self.attn = attn
        self.attn_norm = RMSNorm(w_attn_norm, eps)
        self.post_norm = RMSNorm(w_post_norm, eps)
        # in f32 perche' la residuo resta in f32 lungo i 40 layer
        self._w_attn_norm = np.ascontiguousarray(w_attn_norm, dtype=np.float32)
        self._w_post_norm = np.ascontiguousarray(w_post_norm, dtype=np.float32)

    def forward(self, x: np.ndarray, weights: dict[str, np.ndarray],
                cache: KVCache | None = None) -> np.ndarray:
        """``x + attn(attn_norm(x))`` e poi la norm che precede il MoE.

        Restituisce la coppia (residuo post-attenzione, sua normalizzazione),
        cosi' il chiamante passa al MoE la normalizzazione e aggiunge la
        residuo: e' l'ordine di llama.cpp, dove il residuo del FFN e' somato
        al valore che esce dal post-norm e non a quello che ci entra.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        normed = self._rms(x, self._w_attn_norm)
        attn_out = self.attn.forward(normed, weights, cache=cache)
        resid = x + attn_out
        return resid, self._rms(resid, self._w_post_norm)

    @staticmethod
    def _rms(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
        return (x / rms) * weight
