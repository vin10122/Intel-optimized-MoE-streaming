from __future__ import annotations

import time

import numpy as np

from typing import Any
from pathlib import Path

from .key import ExpertKey
from .store import AsyncExpertStore
from .mmap_store import MemoryMappedAsyncExpertStore
from .dequant import dequantize_to, name_of
from .norm import RMSNorm
from .rope import RoPE, DEFAULT_ROPE_DIM, DEFAULT_FREQ_BASE
from .attention import AttentionLayer, KVCache, scaled_dot_product_attention, softmax
from .backend_ov import OpenVINOExpertBackend
from .router import ExpertRouter, routers_from_gguf
from .shared_expert import SharedExpert
from .embedding import EmbeddingLookup, LMHead, sample
from .transformer import TransformerBlock
from .gated_deltanet import GatedDeltaNetLayer, is_recurrent_layer, load_layer_weights

import gguf


class Qwen3MoEModel:
    """Full Qwen3 MoE model for autoregressive generation.

    Integrates:
    - Token embedding (dequantized on-load, GPU-f16 compatible)
    - 40 transformer blocks (2 attention variants × quantization types)
    - MoE routing + OpenVINO SwiGLU backend with async prefetch
    - KV cache for GQA
    - LM head for token logits
    """

    def __init__(
        self,
        gguf_path: Path,
        device: str = "GPU",
        max_seq_len: int = 2048,
        cache_capacity: int = 320,
        dequant_fn: Any = None,
        keep_ssm_f32: bool = False,
        use_mmap: bool = False,
        ssm_device: str | None = None,
        attention_device: str | None = None,
        weight_cache_capacity: int | None = None,
    ) -> None:
        self.gguv_path = gguf_path
        self.device = device
        self.max_seq_len = max_seq_len
        self.cache_capacity = cache_capacity
        self.weight_cache_capacity = (
            cache_capacity if weight_cache_capacity is None
            else weight_cache_capacity
        )
        self._dequant_fn = dequant_fn
        self.ssm_device = ssm_device
        self.attention_device = attention_device
        self.last_stream_timings: dict[str, int | float] = {
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "prefill_seconds": 0.0,
            "generation_seconds": 0.0,
        }

        reader = gguf.GGUFReader(gguf_path)
        self.tmap = {t.name: t for t in reader.tensors}
        self.reader = reader

        arch = str(reader.fields["general.architecture"].parts[-1].tobytes().decode())
        self.arch = arch

        self.block_count = int(reader.fields[f"{arch}.block_count"].parts[-1][0])
        self.hidden_size = int(reader.fields[f"{arch}.embedding_length"].parts[-1][0])
        self.intermediate_size = int(reader.fields[f"{arch}.expert_feed_forward_length"].parts[-1][0])
        self.num_experts = int(reader.fields[f"{arch}.expert_count"].parts[-1][0])
        self.num_heads = int(reader.fields[f"{arch}.attention.head_count"].parts[-1][0])
        self.num_kv_heads = int(reader.fields[f"{arch}.attention.head_count_kv"].parts[-1][0])
        self.rope_dim = int(reader.fields[f"{arch}.rope.dimension_count"].parts[-1][0]) if f"{arch}.rope.dimension_count" in reader.fields else 64
        self.freq_base = float(reader.fields[f"{arch}.rope.freq_base"].parts[-1][0]) if f"{arch}.rope.freq_base" in reader.fields else 1000000.0
        self.head_dim = int(reader.fields[f"{arch}.attention.key_length"].parts[-1][0]) if f"{arch}.attention.key_length" in reader.fields else self.hidden_size // self.num_heads
        self.kv_head_dim = int(reader.fields[f"{arch}.attention.key_length"].parts[-1][0]) if f"{arch}.attention.key_length" in reader.fields else 256
        vlen = int(reader.fields[f"{arch}.attention.value_length"].parts[-1][0]) if f"{arch}.attention.value_length" in reader.fields else self.kv_head_dim
        if vlen != self.kv_head_dim:
            raise ValueError(
                f"attention.value_length ({vlen}) diverso da key_length "
                f"({self.kv_head_dim}): la proiezione di uscita assume teste "
                "dello stesso formato per Q e per K/V")
        self.eps = float(reader.fields[f"{arch}.attention.layer_norm_rms_epsilon"].parts[-1][0])

        self.bos_token_id = int(reader.fields["tokenizer.ggml.bos_token_id"].parts[-1][0])
        self.eos_token_id = int(reader.fields["tokenizer.ggml.eos_token_id"].parts[-1][0])

        # I layer ricorrenti hanno i propri iperparametri. Nei metadati ci sono
        # ssm.group_count (= le key-head), ssm.time_step_rank (= le value-head)
        # e ssm.inner_size (= value_dim): le dimensioni delle singole teste non
        # ci sono, e hidden/num_heads non le dà. Quelle si ricavano dalle forme
        # dei tensori del blocco, che sono l'unica fonte autorevole:
        #   attn_qkv.weight = (hidden, conv_dim)   con conv_dim = key_dim*2 + value_dim
        #   attn_gate.weight = (hidden, value_dim)
        #   ssm_norm.weight = head_k_dim (il norm e' sulla dimensione delle chiavi)
        # Ogni ricostruzione viene controllata contro la forma che il GGUF
        # dichiara, cosi' un modello con proporzioni diverse fallisce qui e non a
        # metà del forward.
        self.ssm_group = int(reader.fields[f"{arch}.ssm.group_count"].parts[-1][0])
        self.ssm_tsr = int(reader.fields[f"{arch}.ssm.time_step_rank"].parts[-1][0])
        self.ssm_inner = int(reader.fields[f"{arch}.ssm.inner_size"].parts[-1][0])
        self.ssm_conv_kernel = int(reader.fields[f"{arch}.ssm.conv_kernel"].parts[-1][0])
        self.ssm_n_k = self.ssm_group
        self.ssm_n_v = self.ssm_tsr
        self.ssm_d_v = self.ssm_inner // self.ssm_n_v
        # head_k_dim viene dal norm: se manca, si ricava da key_dim e n_k_heads,
        # che pero' richiede la forma di attn_qkv, quindi si rimanda a _build_ssm.
        self.ssm_d_k = int(self.tmap["blk.0.ssm_norm.weight"].n_elements)
        if self.ssm_n_v % self.ssm_n_k != 0:
            raise ValueError(
                f"n_v_heads ({self.ssm_n_v}) deve essere multiplo di "
                f"n_k_heads ({self.ssm_n_k}): le chiavi si replicano sulle "
                "teste di valore")
        if self.ssm_inner != self.ssm_d_v * self.ssm_n_v:
            raise ValueError(
                f"ssm.inner_size ({self.ssm_inner}) incoerente con "
                f"n_v_heads ({self.ssm_n_v}) e head_v_dim ({self.ssm_d_v})")

        self.tensor_types = self._build_tensor_types()

        # Build embedding
        embd_t = self.tmap["token_embd.weight"]
        self.embedding = EmbeddingLookup.from_gguf_tensor(embd_t)

        # Build LM head
        out_t = self.tmap["output.weight"]
        self.lm_head = LMHead.from_gguf_tensor(out_t)

        # Build output norm
        out_norm_t = self.tmap["output_norm.weight"]
        out_norm_w = np.asarray(out_norm_t.data, dtype=np.float16)
        self.output_norm = RMSNorm(out_norm_w, self.eps)

        # Build store + manifest
        from run_q4k_f16_openvino import build_manifest_from_gguf
        self.manifest = build_manifest_from_gguf(gguf_path)
        
        # Use memory-mapped store for better I/O performance
        StoreClass = MemoryMappedAsyncExpertStore if use_mmap else AsyncExpertStore
        self.store = StoreClass(
            gguf_path.parent, capacity=cache_capacity, manifest=self.manifest
        )

        # Build backend
        self.backend = OpenVINOExpertBackend(
            self.store,
            device=device,
            weight_dtype="f16",
            cache_capacity=self.weight_cache_capacity,
            tensor_types=self.tensor_types,
            dequant_fn=dequant_fn,
        )

        # Un router per layer: ffn_gate_inp.weight e' (2048, 256) e cambia
        # in ogni strato. Un router unico condiviso farebbe selezionare lo
        # stesso set di expert in tutti i 40 layer.
        self.top_k = int(reader.fields[f"{arch}.expert_used_count"].parts[-1][0])
        self.routers = routers_from_gguf(
            self.tmap, self.block_count, self.hidden_size, self.num_experts,
            self.top_k, arch=arch,
        )
        # Compatibilita': i call site che leggono model.router continuano a
        # funzionare e ottengono il router del layer 0.
        self.router = self.routers[0]

        # Shared expert: MLP denso che OGNI token attraversa, oltre ai top-k
        # selezionati dal router. Senza, l'output di ogni layer e' sbagliato.
        self.shared_expert = None
        if "blk.0.ffn_gate_shexp.weight" in self.tmap:
            self.shared_expert = SharedExpert(
                gguf_path, num_layers=self.block_count,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                dequant_fn=dequant_fn,
            )

        # Build transformer blocks
        self.blocks: list[TransformerBlock] = []
        # I 30 layer ricorrenti non stanno in self.blocks: hanno le norme e il
        # ramo MoE dello stesso blocco, ma al posto dell'attenzione c'e' il
        # delta net, che vive qui. Attn_qkv e attn_gate di un layer SSM sono le
        # proiezioni del delta net, quindi non vanno tenuti anche nel blocco:
        # tenere entrambe le copie costava 1,4 GiB.
        self.ssm: dict[int, GatedDeltaNetLayer] = {}
        for layer in range(self.block_count):
            if is_recurrent_layer(self.tmap, layer):
                self.ssm[layer] = self._build_ssm(layer)
            self.blocks.append(self._build_block(layer))
        # Il blocco di un layer ricorrente riceve il richiamo al delta net:
        # e' l'unico modo che ha il blocco di sapere cosa mettere al posto
        # dell'attenzione, senza duplicare la ricorrenza.
        for layer, lyr in self.ssm.items():
            self.blocks[layer].ssm_runner = (
                lambda x, _lyr=lyr: self._run_ssm(_lyr, x))
        self.ssm_keep_f32 = keep_ssm_f32
        # keep_ssm_f32 riguarda solo il percorso NumPy/CPU. Con ssm_device
        # impostato, le tre proiezioni grandi vengono eseguite in f16 su
        # OpenVINO; la ricorrenza resta in f32 su CPU.
        if keep_ssm_f32:
            for lyr in self.ssm.values():
                lyr.promote_to_f32()

    def _build_ssm(self, layer: int) -> GatedDeltaNetLayer:
        """Costruisce il Gated DeltaNet di un layer ricorrente.

        Le forme dei tensori vengono controllate contro le dimensioni ricavate
        dai metadati. key_dim non e' nei metadati: esce da
        conv_dim = key_dim * 2 + value_dim, dove conv_dim e' la seconda
        dimensione di attn_qkv.weight. Su questo modello conv_dim 8192 e
        value_dim 4096 danno key_dim 2048, e 2048 con 16 key-head da 128
        combacia con ssm_norm (128).
        """
        qkv_shape = tuple(int(v) for v in self.tmap[f"blk.{layer}.attn_qkv.weight"].shape)
        gate_shape = tuple(int(v) for v in self.tmap[f"blk.{layer}.attn_gate.weight"].shape)
        out_shape = tuple(int(v) for v in self.tmap[f"blk.{layer}.ssm_out.weight"].shape)
        norm_n = int(self.tmap[f"blk.{layer}.ssm_norm.weight"].n_elements)

        # Attenzione alle convenzioni: _get_tensor e load_layer_weights
        # restituiscono (out, in), cioe' l'asse 0 e' l'uscita. value_dim e' la
        # dimensione dei value-head ed e' la seconda dimensione di attn_gate.
        value_dim = gate_shape[1]
        if value_dim != self.ssm_inner:
            raise ValueError(
                f"blk.{layer}.attn_gate.weight ha {value_dim} colonne, attese "
                f"{self.ssm_inner} = ssm.inner_size")
        if out_shape[0] != value_dim or out_shape[1] != self.hidden_size:
            raise ValueError(
                f"blk.{layer}.ssm_out.weight di forma {out_shape}, attesa "
                f"{(value_dim, self.hidden_size)}: e' l'unico peso SSM con "
                "l'asse di uscita per secondo")
        conv_dim = qkv_shape[1]
        if (conv_dim - value_dim) % 2 != 0:
            raise ValueError(
                f"blk.{layer}.attn_qkv.weight ha {conv_dim} colonne: "
                f"conv_dim - value_dim = {conv_dim - value_dim} non e' pari, "
                "ma conv_dim = key_dim * 2 + value_dim lo richiede")
        key_dim = (conv_dim - value_dim) // 2
        head_k_dim = norm_n
        if key_dim % self.ssm_n_k != 0:
            raise ValueError(
                f"key_dim ({key_dim}) non divisibile per n_k_heads "
                f"({self.ssm_n_k})")
        if key_dim // self.ssm_n_k != head_k_dim:
            raise ValueError(
                f"key_dim ({key_dim}) con {self.ssm_n_k} key-head da "
                f"{head_k_dim} fa {key_dim // self.ssm_n_k}, non {head_k_dim}: "
                "head_k_dim non combacia")

        weights = load_layer_weights(self.tmap, layer, self._dequant_fn)
        return GatedDeltaNetLayer(
            layer_idx=layer,
            hidden_size=self.hidden_size,
            n_v_heads=self.ssm_n_v,
            n_k_heads=self.ssm_n_k,
            head_k_dim=head_k_dim,
            head_v_dim=self.ssm_d_v,
            conv_kernel=self.ssm_conv_kernel,
            weights=weights,
            eps=self.eps,
            projection_device=self.ssm_device,
        )

    def _build_tensor_types(self) -> dict[int, dict[str, int]]:
        """Reads ggml types per-layer for expert tensors (gate/up/down)."""
        out: dict[int, dict[str, int]] = {}
        for layer in range(self.block_count):
            layer_types: dict[str, int] = {}
            for role in ("gate", "up", "down"):
                name = f"blk.{layer}.ffn_{role}_exps.weight"
                if name in self.tmap:
                    layer_types[role] = int(self.tmap[name].tensor_type)
            if layer_types:
                out[layer] = layer_types
        return out

    def _get_tensor(self, name: str) -> np.ndarray:
        """Reads a tensor from GGUF as float16, dequantizing if needed.

        Tutto quello che resta in RAM e' f16. Restituire f32 raddoppiava i
        1028M parametri di attenzione (da 1,91 a 3,83 GiB) e i 116M dei
        gate SSM senza guadagnare nulla: OpenVINO e i pesi degli expert
        lavorano gia' in f16.
        """
        t = self.tmap[name]
        ttype = int(t.tensor_type)
        shape = tuple(int(s) for s in reversed(t.shape))

        if ttype == 1:  # F16
            return t.data.astype(np.float16).reshape(shape)
        elif ttype == 0:  # F32
            return np.asarray(t.data, dtype=np.float16).reshape(shape)
        else:
            raw = t.data.tobytes() if hasattr(t.data, "tobytes") else t.data
            return dequantize_to(
                np.frombuffer(raw, dtype=np.uint8), ttype, "f16"
            ).reshape(shape)

    def _build_block(self, layer: int) -> TransformerBlock:
        """Build a transformer block for the given layer index."""
        attn_norm_t = self.tmap[f"blk.{layer}.attn_norm.weight"]
        attn_norm_w = np.asarray(attn_norm_t.data, dtype=np.float16)

        post_attn_norm_t = self.tmap.get(f"blk.{layer}.post_attention_norm.weight")
        if post_attn_norm_t is not None:
            post_attn_norm_w = np.asarray(post_attn_norm_t.data, dtype=np.float16)
        else:
            post_attn_norm_w = np.ones(self.hidden_size, dtype=np.float16)

        is_ssm = is_recurrent_layer(self.tmap, layer)

        if is_ssm:
            # Nessuna proiezione di attenzione: su questo layer i tensori
            # attn_qkv e attn_gate sono le proiezioni del delta net e sono
            # gia' caricate dentro GatedDeltaNetLayer. Qui restano solo le
            # norme e il ramo MoE, che sono identici nei due tipi di layer.
            return TransformerBlock(
                layer_idx=layer,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                kv_head_dim=self.kv_head_dim,
                rope_dim=self.rope_dim,
                freq_base=self.freq_base,
                max_seq_len=self.max_seq_len,
                attn_norm_weight=attn_norm_w,
                post_attn_norm_weight=post_attn_norm_w,
                backend=self.backend,
                router=self.routers[layer],
                intermediate_size=self.intermediate_size,
                num_experts=self.num_experts,
                shared_expert=self.shared_expert,
                is_ssm=True,
            )
        else:
            q_w = self._get_tensor(f"blk.{layer}.attn_q.weight")
            k_w = self._get_tensor(f"blk.{layer}.attn_k.weight")
            v_w = self._get_tensor(f"blk.{layer}.attn_v.weight")
            out_w = self._get_tensor(f"blk.{layer}.attn_output.weight")

            qk_norm_t = self.tmap.get(f"blk.{layer}.attn_q_norm.weight")
            k_norm_t = self.tmap.get(f"blk.{layer}.attn_k_norm.weight")
            qk_norm_w = np.asarray(qk_norm_t.data, dtype=np.float16) if qk_norm_t else None
            k_norm_w = np.asarray(k_norm_t.data, dtype=np.float16) if k_norm_t else None

            # head_dim viene dai metadati (attention.key_length = 256), non da
            # hidden/num_heads: su questa architettura Q e' il gate e la query
            # fuse in una sola proiezione, quindi hidden/num_heads non
            # corrisponde a nessuna delle due. I pesi sono in convenzione
            # numpy (out, in): l'asse 0 e' l'uscita della proiezione.
            head_dim = self.head_dim
            if qk_norm_t is not None and int(qk_norm_t.n_elements) != head_dim:
                raise ValueError(
                    f"blk.{layer}.attn_q_norm.weight e' lungo "
                    f"{int(qk_norm_t.n_elements)}, attesi {head_dim}")
            if q_w is not None:
                want_q = (self.num_heads * 2 * head_dim, self.hidden_size)
                if tuple(int(s) for s in q_w.shape) != want_q:
                    raise ValueError(
                        f"blk.{layer}.attn_q.weight di forma "
                        f"{tuple(int(s) for s in q_w.shape)}, attesa "
                        f"{want_q} = ({self.num_heads} head * 2 (query + "
                        f"gate) * {head_dim}, {self.hidden_size})")
            if k_w is not None:
                want_k = (self.num_kv_heads * self.kv_head_dim, self.hidden_size)
                if tuple(int(s) for s in k_w.shape) != want_k:
                    raise ValueError(
                        f"blk.{layer}.attn_k.weight di forma "
                        f"{tuple(int(s) for s in k_w.shape)}, attesa {want_k}")
            if v_w is not None:
                want_v = (self.num_kv_heads * self.kv_head_dim, self.hidden_size)
                if tuple(int(s) for s in v_w.shape) != want_v:
                    raise ValueError(
                        f"blk.{layer}.attn_v.weight di forma "
                        f"{tuple(int(s) for s in v_w.shape)}, attesa {want_v}")
            if out_w is not None:
                want_o = (self.hidden_size, self.num_heads * head_dim)
                if tuple(int(s) for s in out_w.shape) != want_o:
                    raise ValueError(
                        f"blk.{layer}.attn_output.weight di forma "
                        f"{tuple(int(s) for s in out_w.shape)}, attesa {want_o}")
            if k_norm_w is not None and int(k_norm_w.shape[0]) != self.kv_head_dim:
                raise ValueError(
                    f"blk.{layer}.attn_k_norm.weight e' lungo "
                    f"{int(k_norm_w.shape[0])}, attesi {self.kv_head_dim}")

            return TransformerBlock(
                layer_idx=layer,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=head_dim,
                kv_head_dim=self.kv_head_dim,
                rope_dim=self.rope_dim,
                freq_base=self.freq_base,
                max_seq_len=self.max_seq_len,
                attn_norm_weight=attn_norm_w,
                post_attn_norm_weight=post_attn_norm_w,
                q_weight=q_w,
                k_weight=k_w,
                v_weight=v_w,
                attn_output_weight=out_w,
                q_k_norm=qk_norm_w,
                k_k_norm=k_norm_w,
                backend=self.backend,
                router=self.routers[layer],
                intermediate_size=self.intermediate_size,
                num_experts=self.num_experts,
                shared_expert=self.shared_expert,
                attention_device=self.attention_device,
            )

    def _run_ssm(self, layer: GatedDeltaNetLayer, x: np.ndarray) -> np.ndarray:
        """Applica il delta net a tutta la sequenza, token per token.

        ``GatedDeltaNetLayer.forward`` avanza di un token e restituisce un
        vettore: la ricorrenza e' sequenziale per definizione, quindi un
        prefillo di N token non e' un problema di parallelismo ma di corretto
        ordine. Qui i token si attraversano in ordine e si scrive il risultato
        nella posizione giusta.

        Il batch e' 1: lo stato del delta net e' uno solo per layer, quindi
        due sequenze in parallelo si sovrascriverebbero a vicenda lo stato.
        Accettarlo qui vale piu' che accettarlo in silenzio.
        """
        b, s, h = x.shape
        if b != 1:
            raise ValueError(
                f"il delta net ha stato singolo: batch {b} non supportato, "
                "perche' le due sequenze condividerebbero lo stesso stato")
        out = np.empty((b, s, h), dtype=np.float32)
        for si in range(s):
            out[0, si, :] = layer.forward(x[0, si, :])
        return out

    def init_caches(self) -> None:
        for block in self.blocks:
            if not block.is_ssm:
                block.init_kv_cache(self.max_seq_len)
        # Lo stato dei layer ricorrenti va azzerato insieme alle cache KV: se
        # resta com'era dalla generazione precedente, il primo token del nuovo
        # prompt parte da uno stato che non gli appartiene e l'output e'
        # sbagliato senza nessun errore.
        for layer in self.ssm.values():
            layer.reset()

    def forward_layer(
        self,
        layer_idx: int,
        x: np.ndarray,
        seq_offset: int = 0,
    ) -> np.ndarray:
        """Forward pass through a single transformer block.

        Args:
            layer_idx: Layer index (0 to block_count-1)
            x: (batch, seq, hidden) input
            seq_offset: Position offset in KV cache

        Returns:
            (batch, seq, hidden) output
        """
        return self.blocks[layer_idx].forward(x, seq_offset)

    def stream(
        self,
        input_ids: list[int],
        max_new_tokens: int = 1,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 0.95,
        seed: int = 42,
    ):
        """Genera un token alla volta, e lo consegna prima di continuare.

        ``generate`` e' questo metodo con la risposta raccolta in una lista:
        il ciclo e' scritto una volta sola qui, quindi i due percorsi non
        possono divergere. Serve il metodo separato perche' un client
        streaming deve mostrare il token appena esiste, e non puo' aspettare
        che la generazione finisca: a 0,21 tok/s un prompt di 20 token arriva
        dopo un minuto e mezzo.

        Yields:
            (indice da 1, token id) a ogni token prodotto. L'indice parte da 1
            perche' al cliente interessa la posizione nel testo generato, non
            nel prompt.
        """
        if not input_ids:
            raise ValueError("input_ids vuoto: niente da cui generare")
        prefill_started = time.perf_counter()
        self.init_caches()
        rng = np.random.default_rng(seed)

        # Prefillo: tutto il prompt, in un colpo.
        x = self.embedding.forward(np.asarray(input_ids, dtype=np.int64))
        x = x.reshape(1, len(input_ids), self.hidden_size)
        x = self._forward_all(x)
        prefill_seconds = time.perf_counter() - prefill_started
        generation_seconds = 0.0
        pending_decode_seconds = 0.0

        for produced in range(1, max_new_tokens + 1):
            step_started = time.perf_counter()
            logits = self.output_norm.forward(x)
            logits = self.lm_head.forward(logits)
            # (1, seq, vocab) -> solo l'ultima posizione: e' li' che si
            # campiona. logit[-1] avrebbe restituito tutta la sequenza.
            last_logits = np.asarray(logits[0, -1], dtype=np.float32).reshape(-1)

            next_token = sample(
                last_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                rng=rng,
            )
            generation_seconds += (
                pending_decode_seconds + time.perf_counter() - step_started
            )
            pending_decode_seconds = 0.0
            self.last_stream_timings = {
                "prompt_tokens": len(input_ids),
                "generated_tokens": produced,
                "prefill_seconds": prefill_seconds,
                "generation_seconds": generation_seconds,
            }
            yield produced, next_token
            if next_token == self.eos_token_id or produced == max_new_tokens:
                break

            # Decode: un token alla volta, con lo stato dei layer ricorrenti
            # e la cache KV gia' avanti di un passo.
            step_started = time.perf_counter()
            x = self.embedding.forward(np.asarray([next_token], dtype=np.int64))
            x = x.reshape(1, 1, self.hidden_size)
            x = self._forward_all(x)
            pending_decode_seconds = time.perf_counter() - step_started

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 1,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 0.95,
        seed: int = 42,
    ) -> list[int]:
        """Generazione autoregressiva.

        Il prefillo attraversa tutti i token del prompt in un colpo, poi ogni
        passo ne aggiunge uno. L'offset di posizione non viene passato a mano:
        ``AttentionLayer.forward`` lo ricava dalla lunghezza della cache, che
        dopo un prefillo vale il numero di token gia' visti. Passarlo
        esplicitamente era un doppio libro di conto che poteva anche
        sbagliare: nel codice precedente il prefillo partiva da
        ``len(prompt) - 1``, cioe' le posizioni erano gia' sbagliate prima del
        primo token.

        Returns:
            I token generati, prompt compreso.
        """
        return list(input_ids) + [token for _, token in self.stream(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )]

    def _forward_all(self, x: np.ndarray) -> np.ndarray:
        """Attraversa tutti i 40 layer nell'ordine, ricorrenti e attenzione."""
        for layer in range(self.block_count):
            x = self.blocks[layer].forward(x)
        return x

    def close(self) -> None:
        self.store.close()
