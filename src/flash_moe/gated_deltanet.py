"""Gated DeltaNet: i layer ricorrenti (SSM) di Qwen3.5-MoE.

Perche' esiste
-------------
Qwen3.5-MoE non e' un transformer omogeneo: 30 layer su 40 sono *gated delta
net* (linear attention con ricorrenza), gli altri 10 sono attenzione normale.
I layer ricorrenti non hanno una KV cache che cresce con la sequenza: hanno
uno **stato** fisso di grandezza costante che si aggiorna a ogni token. E'
per questo che un modello da 35B sta in 17 GB e ha un contesto dichiarato di
262144 token.

Semantica
---------
Ricostruita da llama.cpp ``src/models/qwen35.cpp`` (``build_layer_attn_linear``,
``build_qkvz``, ``build_delta_net``, ``build_norm_gated``) e verificata contro
l'aritmetica in NVlabs/GatedDeltaNet e l'articolo 2412.06464. Per un token:

1. proiezioni: ``qkv = x @ attn_qkv`` (QKV), ``z = x @ attn_gate``
2. ``beta = sigmoid(x @ ssm_beta)``, ``alpha = x @ ssm_alpha``
3. ``gate = softplus(alpha + ssm_dt) * ssm_a`` — ssm_a contiene gia' -exp(A_log)
4. conv1d depthwise su QKV (kernel 4), poi SiLU
5. q e k sono normalizzati L2, q anche diviso per sqrt(head_k_dim)
6. ricorrenza delta rule per ogni value-head, sullo stato S
7. RMSNorm gated: ``rmsnorm(out, ssm_norm) * silu(z)``
8. proiezione di uscita: ``@ ssm_out``

Attenzione ai versi, che non sono uniformi: ``attn_qkv`` e ``attn_gate``
sono ``(n_embd, n_out)`` e si applicano senza trasporre, mentre
``ssm_out`` e' ``(value_dim, n_embd)``, cioe' con l'asse di uscita per
secondo, e si applica anch'esso senza trasporre. Con hidden 2048 e
value_dim 4096 i due versi sono scambiabili solo apparentemente: un errore
qui si manifesta come errore di forma su questo modello, ma su un modello
con hidden e value_dim uguali passerebbe silenziosamente. Per questo
``tools/test_gated_deltanet.py`` confronta la ricorrenza su forme non
quadrate.

La ricorrenza, per value-head h, stato S di forma (head_v_dim, head_k_dim):

    S1    = S * exp(gate)                decadimento
    kv    = S1 @ k                       cosa lo stato ricorda per questa chiave
    delta = beta * (v - kv)              errore: nuovo meno ricordato
    S     = S1 + outer(delta, k)         scrittura rango-1 della correzione
    out   = S @ q                        lettura con la query

Nella notazione di 2412.06464 (eq. 10) e':
S_t = S_{t-1} ( alpha_t (I - beta_t k k^T) ) + beta_t v k^T, che e' lo stesso.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .dequant import dequantize_to, name_of


def _silu(x: np.ndarray) -> np.ndarray:
    return x * _sigmoid(x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistica numericamente stabile, che conserva il dtype di ingresso.

    La stabilita' viene dalla formula a due rami (che non overflowa ne'
    su +800 ne' su -800) e non da un dtype piu' largo: forzare f64 qui
    promuoverebbe a 64 bit anche tutto cio' che segue, e la ricorrenza
    finirebbe per accumulare in f64 su 2 MiB di stato per layer, con il
    doppio della banda di memoria per un guadagno di precisione che il
    vincolo f32 per-token non chiede.
    """
    out = np.empty_like(x, dtype=np.result_type(x, np.float32))
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _softplus(x: np.ndarray) -> np.ndarray:
    """log(1 + exp(x)), stabile su tutto l'intero reale.

    Serve la forma esplicita perche' ``log1p(exp(x))`` va in overflow a
    x > 709 (inf, e poi log1p(inf) = inf: il gate diverrebbe inf e il
    decadimento exp(gate) esploderebbe invece di spegnersi). Per x molto
    grande softplus ~ x, per x molto negativo ~ exp(x).
    """
    out = np.empty_like(x, dtype=np.result_type(x, np.float32))
    big = x > 30.0
    small = x < -30.0
    mid = ~(big | small)
    out[big] = x[big]                       # log(1+exp(x)) ~ x
    out[small] = np.exp(x[small])           # ~ exp(x)
    out[mid] = np.log1p(np.exp(x[mid]))
    return out


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    norm = np.sqrt(np.sum(x * x, axis=axis, keepdims=True) + eps)
    return x / norm


class GatedDeltaNetLayer:
    """Un layer ricorrente, con stato e coda di convoluzione persistenti.

    Stato: ``(n_v_heads, head_v_dim, head_k_dim)`` in f32, 2 MiB per layer
    (32 * 128 * 128 * 4 B). Coda conv: ``(conv_kernel - 1, conv_dim)``, 96 KiB.
    Entrambi sono piccoli e fissi: non cresce con la sequenza.
    """

    def __init__(self, layer_idx: int, weights: dict[str, np.ndarray],
                 hidden_size: int, n_v_heads: int, n_k_heads: int,
                 head_k_dim: int, head_v_dim: int, conv_kernel: int,
                 eps: float = 1e-6, keep_f32: bool = False) -> None:
        self.layer_idx = layer_idx
        self.w = weights
        self.hidden_size = hidden_size
        self.n_v_heads = n_v_heads
        self.n_k_heads = n_k_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.conv_kernel = conv_kernel

        key_dim = head_k_dim * n_k_heads
        value_dim = head_v_dim * n_v_heads
        self.conv_dim = key_dim * 2 + value_dim

        # I pesi si tengono in f16 (vincolo di memoria sul modello) ma si
        # calcola in f32: la conversione f16 -> f32 di tutti i pesi di un
        # layer costa 52 ms, contro 1.6 ms dei matmul veri. Rifarla a ogni
        # token rendeva due terzi del forward puro spreco.
        #
        # keep_f32=True tiene la copia per sempre: e' la scelta veloce (52 ms
        # per layer risparmiati a ogni token) ma costa 128.6 MiB per layer,
        # quindi 3.75 GiB se i 30 layer sono vivi tutti insieme. Con 6 GiB
        # liberi e 0.95 GiB per embedding e 0.95 per la LM head non ci sta:
        # la copia si puo' tenere per un singolo layer (per misurare il
        # guadagno) o per un gruppo piccolo, non per tutti e 30.
        #
        # Con keep_f32=False i pesi restano f16 e vengono convertiti a ogni
        # passo. E' il percorso lento ma e' l'unico che entra in memoria, e il
        # suo costo e' il debito di velocita' da pagare: finche' non e'
        # risolto, il tempo del forward e' dominato dalla conversione e non
        # dal calcolo.
        self.keep_f32 = keep_f32
        # I pesi grandi sono tre: qkv (2048x8192), gate (2048x4096) e out
        # (4096x2048). Gli altri valgono 0.2M elementi, quindi la loro copia
        # f32 costa meno di un MiB e si tiene per sempre in ogni caso.
        self._big = ("qkv", "gate", "out")
        self.small = {k: np.ascontiguousarray(v, dtype=np.float32)
                      for k, v in weights.items() if k not in self._big}
        self.big32 = ({k: np.ascontiguousarray(weights[k], dtype=np.float32)
                       for k in self._big} if keep_f32 else None)

        # ssm_conv1d e' (conv_kernel, conv_dim) nel GGUF: l'asse veloce e' il
        # canale, quindi per il prodotto scalare per canale serve il
        # trasposto, di forma (conv_dim, conv_kernel). Il trasposto e' fatto
        # una volta sola qui, non a ogni token.
        self.conv_kernel_w = np.ascontiguousarray(
            self.small["conv1d"].reshape(conv_kernel, self.conv_dim).T)

        # Stato in f32: la ricorrenza accumula su se stessa per tutta la
        # sequenza, e in f16 perderebbe il decadimento exp(gate) che si
        # moltiplica a ogni token.
        self.state = np.zeros((n_v_heads, head_v_dim, head_k_dim), dtype=np.float32)
        # Coda di convoluzione: (conv_dim, conv_kernel - 1), cioe' una colonna
        # per canale. Serve la forma con i canali sulle righe perche' la
        # finestra scorrevole si assembla come [coda; token] e il risultato
        # e' un prodotto scalare per canale.
        self.conv_state = np.zeros((self.conv_dim, conv_kernel - 1),
                                   dtype=np.float32)

        # Buffer riusati per non allocare a ogni passo: 64 KiB di testa, che
        # è la stessa ordine di grandezza di una singola colonna di stato
        # letta 128 volte. Allocherli qui significa che il passo non tocca
        # l'allocatore, che per 32 teste su 30 layer sarebbe 960 allocazioni
        # di 64 KiB per token.
        self._kv_buf = np.zeros(head_v_dim, dtype=np.float32)
        # Lo stesso per la replica di q e k: con n_v_heads > n_k_heads la
        # copia e' di 32x128 float32 = 16 KiB per variabile, e senza buffer
        # l'allocatore verrebbe chiamato 60 volte per token.
        self._q_rep = np.zeros((n_v_heads, head_k_dim), dtype=np.float32)
        self._k_rep = np.zeros((n_v_heads, head_k_dim), dtype=np.float32)

    def promote_to_f32(self) -> None:
        """ Tiene la copia f32 dei pesi grandi per sempre.

        Si chiama dopo la costruzione, quando si e' deciso che la memoria
        c'e': costruire il layer gia' con keep_f32=True farebbe la copia
        mentre gli altri layer sono ancora in costruzione, cioe' nel momento
        in cui il picco di memoria e' peggiore. Costa 128.6 MiB per layer.
        """
        if self.big32 is not None:
            return
        self.big32 = {k: np.ascontiguousarray(self.w[k], dtype=np.float32)
                      for k in self._big}
        self.keep_f32 = True

    def reset(self) -> None:
        self.state.fill(0.0)
        self.conv_state.fill(0.0)

    def f32(self, name: str) -> np.ndarray:
        """Il peso ``name`` in f32, convertito ora se non e' tenuto in copia.

        La copia permanente esiste solo con ``keep_f32``; altrimenti si
        converte qui e si lascia che l'allocatore la liberi a fine passo. Il
        costo e' quello che rende lento il percorso senza copia, ed e'
        voluto esplicitamente invece che nascosto dietro un default veloce
        che non entra in memoria.
        """
        if self.big32 is not None and name in self.big32:
            return self.big32[name]
        return np.asarray(self.w[name], dtype=np.float32)

    # ------------------------------------------------------------------ passi

    def _projections(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray]:
        """Proiezioni di ingresso.

        Nel GGUF i pesi sono memorizzati come ``(n_out, n_in)``, cioe' con
        l'asse di uscita per primo: e' la convenzione di ggml, dove
        ``wqkv`` e' ``{n_embd, key_dim*2 + value_dim}`` e la moltiplicazione
        e' ``input @ wqkv``. Qui si usa la stessa convenzione, quindi i pesi
        si applicano senza trasporre.

        Si usa ``self.w32``, la copia f32 fatta una volta alla costruzione,
        non ``self.w``: riconvertire i pesi a ogni token costerebbe piu' di
        tutti i matmul messi insieme. Quando la copia non e' tenuta
        (``keep_f32=False``) la conversione avviene qui, e questa e' la voce
        che domina il tempo del forward.
        """
        qkv = x @ self.f32("qkv")
        z = x @ self.f32("gate")
        beta = _sigmoid(x @ self.small["beta"])
        alpha = x @ self.small["alpha"]
        return qkv, z, beta, alpha

    def _conv(self, qkv: np.ndarray) -> np.ndarray:
        """Conv1d depthwise causale, poi SiLU.

        Il vettore di ingresso e' [coda precedente; token corrente] di
        lunghezza ``conv_kernel``, e la coda si aggiorna con le ultime
        ``conv_kernel - 1`` posizioni. Canale per canale: e' depthwise, quindi
        non mescola i canali.
        """
        # ggml_ssm_conv accumola su finestre scorrevole; con un solo token per
        # chiamata la finestra e' [coda; token] e il risultato e' il prodotto
        # scalare di ogni canale con i suoi 4 pesi.
        seq = np.concatenate([self.conv_state, qkv.reshape(self.conv_dim, 1)], axis=1)
        out = np.einsum("ck,ck->c", seq, self.conv_kernel_w)
        # la coda per il prossimo token sono le ultime conv_kernel-1 colonne
        self.conv_state = np.ascontiguousarray(seq[:, 1:])
        return _silu(out)

    def _delta_step(self, S: np.ndarray, q: np.ndarray, k: np.ndarray,
                    v: np.ndarray, beta: float, gate: float) -> np.ndarray:
        """Un passo della ricorrenza delta rule su un value-head.

        ``q`` e ``k`` sono gia' L2-normalizzati, ``q`` anche scalato per
        1/sqrt(head_k_dim). ``S`` e' lo stato della singola testa, di forma
        (head_v_dim, head_k_dim), e viene modificato in place: e' la vista sul
        blocco corrispondente di ``self.state``, quindi l'aggiornamento si
        propaga allo stato persistente.

        La sequenza e' quella di 2412.06464 eq. 10, riscritta con prodotti
        matriciali invece che con un ciclo sulle head_k_dim colonne:

            S1    = exp(gate) * S          decadimento
            kv    = S1 @ k                 cosa lo stato ricorda
            delta = beta * (v - kv)        errore: nuovo meno ricordato
            S     = S1 + outer(delta, k)   scrittura rango-1
            out   = S @ q                  lettura con la query

        Un ciclo Python sulle colonne costava 396 us per testa contro 14 us
        qui, e con 32 teste su 30 layer faceva 380 ms per token. La forma
        matriciale e' anche piu' vicina all'articolo, quindi elimina una
        classe di errori di ordine di lettura: nel ciclo fused ``out``
        leggeva ogni colonna subito dopo averla aggiornata, mescolando lo
        stato finale con quello solo decaduto. Qui si legge solo S finale.
        """
        decay = np.exp(gate)

        S *= decay
        np.matmul(S, k, out=self._kv_buf[:self.head_v_dim])
        delta = beta * (v - self._kv_buf[:self.head_v_dim])
        # outer(delta, k) e' S += delta[:, None] * k[None, :]: si scrive in
        # place per non allocare 128x128 float32 a ogni passo.
        S += delta[:, None] * k
        return S @ q

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Avanza il layer di un token e restituisce (hidden_size,) f32."""
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size != self.hidden_size:
            raise ValueError(
                f"attivazione con {x.size} elementi, attesi {self.hidden_size}"
            )

        qkv, z, beta, alpha = self._projections(x)

        # gate: ssm_a contiene gia' -exp(A_log), quindi basta il softplus
        # moltiplicato, non un exp in piu'.
        gate = _softplus(alpha + self.small["dt_bias"]) * self.small["a"]

        conv_out = self._conv(qkv)
        key_dim = self.head_k_dim * self.n_k_heads
        value_dim = self.head_v_dim * self.n_v_heads
        q = conv_out[:key_dim]
        k = conv_out[key_dim:2 * key_dim]
        v = conv_out[2 * key_dim:]

        # q e k sono divisi per testa; v è gia' alla forma giusta.
        q = q.reshape(self.n_k_heads, self.head_k_dim)
        k = k.reshape(self.n_k_heads, self.head_k_dim)
        v = v.reshape(self.n_v_heads, self.head_v_dim)

        q = _l2_normalize(q) / np.sqrt(self.head_k_dim)
        k = _l2_normalize(k)

        # n_v_heads (32) e' multiplo di n_k_heads (16): q e k vanno replicati
        # per allinearli alle 32 teste di valore.
        #
        # Il pattern NON e' np.repeat. In llama.cpp la replica e' un
        # ggml_repeat_4d, la cui CPU implementation (ggml-cpu/ops.cpp,
        # ggml_compute_forward_repeat_f32) scrive la testa di destinazione
        # all'indice ``i1*ne01 + k1``, con k1 che corre su ne01 = n_k_heads:
        # quindi la testa di valore h prende la chiave h % n_k_heads, cioe'
        #
        #     testa 0,1,2,3 -> chiave 0,1,0,1
        #
        # np.repeat darebbe h // n_k_heads, cioe' 0,0,1,1: e' il pattern di
        # Qwen3-Next, non quello di Qwen3.5. La differenza non e' un
        # dettaglio, porta a un errore del 53% sull'output del layer, e i
        # test numerici non la colgono perche' entrambe le versioni
        # producono numeri plausibili. Vedi
        # tools/test_gated_deltanet_replication.py.
        reps = self.n_v_heads // self.n_k_heads
        if reps > 1:
            # Il tiling si scrive come una copia di (reps, n_k_heads) da
            # (1, n_k_heads): e' esattamente h % n_k_heads per qualunque
            # rapporto. L'ordine dei due assi e' essenziale: con
            # (n_k_heads, reps) la copia darebbe ogni chiave ripetuta di
            # seguito, che e' proprio np.repeat, cioe' l'errore che questo
            # test esiste per prevenire. copyto evita l'allocazione.
            self._q_rep.reshape(reps, self.n_k_heads,
                                self.head_k_dim)[:] = q[None, :, :]
            self._k_rep.reshape(reps, self.n_k_heads,
                                self.head_k_dim)[:] = k[None, :, :]
            q = self._q_rep
            k = self._k_rep

        # _delta_step modifica lo stato in place, quindi va chiamata una volta
        # sola per testa: chiamarla due volte applicherebbe il delta rule due
        # volte sullo stesso token. self.state[h] e' una vista, quindi la
        # scrittura aggiorna lo stato persistente.
        head_out = np.empty((self.n_v_heads, self.head_v_dim), dtype=np.float32)
        for h in range(self.n_v_heads):
            head_out[h] = self._delta_step(self.state[h], q[h], k[h], v[h],
                                           float(beta[h]), float(gate[h]))

        # gated RMSNorm: rmsnorm(out) * silu(z), z alla forma dei value-head.
        z = z.reshape(self.n_v_heads, self.head_v_dim)
        norm_w = self.small["norm"]
        eps = 1e-6
        rms = head_out * (1.0 / np.sqrt(
            np.mean(head_out * head_out, axis=-1, keepdims=True) + eps))
        gated = rms * norm_w * _silu(z)

        flat = gated.reshape(-1)
        # ssm_out e' l'unico peso SSM memorizzato come (value_dim, n_embd),
        # cioe' con l'asse di uscita per secondo, al contrario di attn_qkv e
        # attn_gate che sono (n_embd, ...). Qui il verso e' proprio quello
        # della memorizzazione: flat (value_dim) @ out -> n_embd.
        return flat @ self.f32("out")


def is_recurrent_layer(tensor_map: dict, layer: int) -> bool:
    """True se il layer e' un gated delta net e non attenzione.

    Si distingue dalla presenza di ``ssm_a``: i layer di attenzione hanno
    ``attn_q``/``attn_output`` e i layer ricorrenti i tensor ``ssm_*``. Non e'
    sufficiente la presenza di ``attn_qkv``, che esiste in entrambi i tipi di
    layer con nomi diversi (``attn_qkv`` sul ricorrente, ``attn_q`` sull'altro).
    """
    return f"blk.{layer}.ssm_a" in tensor_map


def load_layer_weights(tensor_map: dict, layer: int,
                       dequant_fn: Any = None) -> dict[str, np.ndarray]:
    """Carica i tensor di un layer ricorrente, dequantizzati in f16.

    Nomi e forme sono quelli che llama.cpp si aspetta per qwen35 (cfr.
    ``load_tensors``): wqkv da ``attn_qkv``, wqkv_gate da ``attn_gate``,
    ``ssm_out`` gia' con la forma (value_dim, n_embd).

    I pesi sono piccoli rispetto agli expert (32 + 16 MiB per layer in f16
    contro ~1,5 GiB di expert per 40 layer), quindi si caricano interi e si
    tengono in memoria: servono a ogni token, non c'e' nulla da guadagnare
    tenendoli su disco.
    """
    def load(name: str, key: str) -> None:
        tensor = tensor_map[f"blk.{layer}.{name}"]
        shape = tuple(int(v) for v in tensor.shape)
        gtype = int(tensor.tensor_type)
        if name_of(gtype) in ("F32", "F16"):
            dtype = np.float32 if name_of(gtype) == "F32" else np.float16
            arr = np.ascontiguousarray(tensor.data, dtype=dtype)
            weights[key] = np.ascontiguousarray(arr, dtype=np.float16).reshape(*shape)
            return
        raw = np.frombuffer(np.ascontiguousarray(tensor.data, dtype=np.uint8)
                            .tobytes(), dtype=np.uint8)
        n = int(np.prod(shape))
        if dequant_fn is not None and len(shape) > 1:
            out = dequant_fn(raw.tobytes(), gtype, n)
        else:
            out = dequantize_to(raw, gtype, "f16")
        arr = np.asarray(out, dtype=np.float16).reshape(-1)
        if arr.size != n:
            raise ValueError(
                f"blk.{layer}.{name}: {arr.size} elementi, attesi {n}"
            )
        weights[key] = np.ascontiguousarray(arr, dtype=np.float16).reshape(*shape)

    weights: dict[str, np.ndarray] = {}
    load("attn_qkv.weight", "qkv")
    load("attn_gate.weight", "gate")
    load("ssm_conv1d.weight", "conv1d")
    load("ssm_dt.bias", "dt_bias")
    load("ssm_a", "a")
    load("ssm_beta.weight", "beta")
    load("ssm_alpha.weight", "alpha")
    load("ssm_norm.weight", "norm")
    load("ssm_out.weight", "out")
    return weights
