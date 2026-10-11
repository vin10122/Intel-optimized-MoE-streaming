"""Profile warmed token decode, separating non-overlapping phase timings.

Example:
    python tools/profile_decode.py --device GPU --ssm-device GPU \
        --attention-device GPU --cache 320
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gguf  # noqa: E402
from run_q4k_f16_openvino import dequant_to_f16  # noqa: E402
import flash_moe.gated_deltanet as gdn  # noqa: E402
import flash_moe.backend_ov as bov  # noqa: E402
import flash_moe.attention as attn_mod  # noqa: E402
import flash_moe.embedding as embedding_mod  # noqa: E402
import flash_moe.router as router_mod  # noqa: E402
from flash_moe.shared_expert import SharedExpert  # noqa: E402
from flash_moe import Qwen3MoEModel  # noqa: E402
from flash_moe.paths import model_path  # noqa: E402
from flash_moe.tokenizer import GGUFTokenizer  # noqa: E402

class Acc:
    __slots__ = ("t", "n")

    def __init__(self):
        self.t = 0.0
        self.n = 0

    def add(self, dt: float) -> None:
        self.t += dt
        self.n += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="GPU", choices=("CPU", "GPU", "NPU"))
    parser.add_argument("--ssm-device", choices=("CPU", "GPU", "NPU"))
    parser.add_argument("--attention-device", choices=("CPU", "GPU", "NPU"))
    parser.add_argument("--cache", type=int, default=64)
    parser.add_argument("--weight-cache", type=int,
                        help="converted expert weights kept in RAM; defaults "
                             "to --cache")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measure", type=int, default=5)
    parser.add_argument("--prompt", default="Il sole splende sopra")
    parser.add_argument("--keep-ssm-f32", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cache < 1 or args.warmup < 0 or args.measure < 1:
        raise ValueError("cache/measure devono essere positivi e warmup >= 0")
    if args.weight_cache is not None and args.weight_cache < 1:
        raise ValueError("weight-cache deve essere positivo")

    acc = {k: Acc() for k in (
        "moe_weights", "moe_infer", "expert_forward", "shared",
        "weight_decode", "ssm_proj", "ssm_total", "attn", "router",
        "embedding", "output")}
    attn_phases: dict[str, float] = {}
    weight_cache = {"hits": 0, "misses": 0}

    orig_ssm_fwd = gdn.GatedDeltaNetLayer.forward
    orig_proj = gdn.GatedDeltaNetLayer._projections

    def ssm_fwd(self, x):
        t0 = time.perf_counter()
        out = orig_ssm_fwd(self, x)
        acc["ssm_total"].add(time.perf_counter() - t0)
        return out

    def proj(self, x):
        t0 = time.perf_counter()
        out = orig_proj(self, x)
        acc["ssm_proj"].add(time.perf_counter() - t0)
        return out

    gdn.GatedDeltaNetLayer.forward = ssm_fwd
    gdn.GatedDeltaNetLayer._projections = proj

    orig_backend_fwd = bov.OpenVINOExpertBackend.forward
    orig_weights = bov.OpenVINOExpertBackend._weights
    orig_infer = bov.OpenVINOExpertBackend._infer
    orig_decode = bov.OpenVINOExpertBackend._decode

    def backend_fwd(self, key, activation, hidden_size,
                        intermediate_size):
        t0 = time.perf_counter()
        out = orig_backend_fwd(self, key, activation,
                                    hidden_size, intermediate_size)
        acc["expert_forward"].add(time.perf_counter() - t0)
        return out

    def weights(self, key, hidden_size, intermediate_size):
        if key in self._weight_cache:
            weight_cache["hits"] += 1
        else:
            weight_cache["misses"] += 1
        t0 = time.perf_counter()
        out = orig_weights(self, key, hidden_size, intermediate_size)
        acc["moe_weights"].add(time.perf_counter() - t0)
        return out

    def decode(self, raw, ggml_type, shape):
        t0 = time.perf_counter()
        out = orig_decode(self, raw, ggml_type, shape)
        acc["weight_decode"].add(time.perf_counter() - t0)
        return out

    def infer(self, request, feed):
        t0 = time.perf_counter()
        out = orig_infer(self, request, feed)
        acc["moe_infer"].add(time.perf_counter() - t0)
        return out

    bov.OpenVINOExpertBackend.forward = backend_fwd
    bov.OpenVINOExpertBackend._weights = weights
    bov.OpenVINOExpertBackend._infer = infer
    bov.OpenVINOExpertBackend._decode = decode

    orig_shared = SharedExpert.forward

    def shared_fwd(self, layer, activation):
        t0 = time.perf_counter()
        out = orig_shared(self, layer, activation)
        acc["shared"].add(time.perf_counter() - t0)
        return out

    SharedExpert.forward = shared_fwd

    orig_attn_fwd = attn_mod.AttentionLayer.forward

    def attn_fwd(self, x, weights, cache=None):
        self.profile_timings = attn_phases
        t0 = time.perf_counter()
        out = orig_attn_fwd(self, x, weights, cache)
        acc["attn"].add(time.perf_counter() - t0)
        return out

    orig_route = router_mod.ExpertRouter.route

    def route(self, activation):
        t0 = time.perf_counter()
        out = orig_route(self, activation)
        acc["router"].add(time.perf_counter() - t0)
        return out

    # Patch only the model's shared embedding class; the input and decode
    # tokens both pass through this method.
    from flash_moe.embedding import EmbeddingLookup
    orig_embedding_fwd = EmbeddingLookup.forward

    def embedding_fwd(self, input_ids):
        t0 = time.perf_counter()
        out = orig_embedding_fwd(self, input_ids)
        acc["embedding"].add(time.perf_counter() - t0)
        return out

    orig_lm_fwd = embedding_mod.LMHead.forward

    def lm_fwd(self, hidden):
        t0 = time.perf_counter()
        out = orig_lm_fwd(self, hidden)
        acc["output"].add(time.perf_counter() - t0)
        return out

    attn_mod.AttentionLayer.forward = attn_fwd
    router_mod.ExpertRouter.route = route
    EmbeddingLookup.forward = embedding_fwd
    embedding_mod.LMHead.forward = lm_fwd

    # --- run -------------------------------------------------------------
    path = model_path()
    tok = GGUFTokenizer(gguf.GGUFReader(path))
    ids = tok.encode(args.prompt, add_bos=False)

    model = Qwen3MoEModel(path, device=args.device, max_seq_len=256,
                          cache_capacity=args.cache,
                          weight_cache_capacity=args.weight_cache,
                          dequant_fn=dequant_to_f16,
                          keep_ssm_f32=args.keep_ssm_f32,
                          ssm_device=args.ssm_device,
                          attention_device=args.attention_device)
    model.init_caches()

    x = model.embedding.forward(np.asarray(ids, dtype=np.int64)).reshape(
        1, len(ids), model.hidden_size)
    for layer in range(model.block_count):
        x = model.blocks[layer].forward(x)

    w = np.asarray(model.tmap["output_norm.weight"].data,
                   dtype=np.float32).reshape(-1)

    def next_token(last):
        f = last / np.sqrt(np.mean(last * last) + 1e-6) * w
        logits = np.asarray(model.lm_head.forward(
            f.astype(np.float32).reshape(1, -1)), dtype=np.float64).reshape(-1)
        return int(np.argmax(logits))

    cur = next_token(x[0, -1, :])
    for _ in range(args.warmup):
        step = model.embedding.forward(np.asarray([cur], dtype=np.int64)).reshape(
            1, 1, model.hidden_size)
        for layer in range(model.block_count):
            step = model.blocks[layer].forward(step)
        cur = next_token(step[0, -1, :])

    for a in acc.values():
        a.t = 0.0
        a.n = 0
    attn_phases.clear()
    weight_cache["hits"] = 0
    weight_cache["misses"] = 0
    store_stats_start = (
        model.store.stats.hits,
        model.store.stats.misses,
        model.store.stats.prefetch_hits,
        model.store.stats.bytes_read,
    )

    t_all = time.perf_counter()
    for _ in range(args.measure):
        step = model.embedding.forward(np.asarray([cur], dtype=np.int64)).reshape(
            1, 1, model.hidden_size)
        for layer in range(model.block_count):
            step = model.blocks[layer].forward(step)
        cur = next_token(step[0, -1, :])
    total = time.perf_counter() - t_all

    per_token = total / args.measure
    print(f"\n=== device={args.device} ssm_device={args.ssm_device or 'NumPy/CPU'} "
          f"attention_device={args.attention_device or 'NumPy/CPU'} "
          f"cache={args.cache} weight_cache="
          f"{args.weight_cache or args.cache} "
          f"keep_ssm_f32={args.keep_ssm_f32} ===")
    print(f"decode: {per_token * 1000:.0f} ms/token "
          f"({1 / per_token:.2f} tok/s)  [{args.measure} passi]")
    print(f"\n{'fase esclusiva':<22}{'ms/token':>10}{'%':>7}")
    ssm_ms = acc["ssm_total"].t / args.measure * 1000
    ssm_proj_ms = acc["ssm_proj"].t / args.measure * 1000
    attn_ms = acc["attn"].t / args.measure * 1000
    shared_ms = acc["shared"].t / args.measure * 1000
    route_ms = acc["router"].t / args.measure * 1000
    embedding_ms = acc["embedding"].t / args.measure * 1000
    output_ms = acc["output"].t / args.measure * 1000
    weights_ms = acc["moe_weights"].t / args.measure * 1000
    decode_ms = acc["weight_decode"].t / args.measure * 1000
    infer_ms = acc["moe_infer"].t / args.measure * 1000
    expert_ms = acc["expert_forward"].t / args.measure * 1000
    expert_read_ms = max(0.0, weights_ms - decode_ms)
    expert_misc_ms = max(0.0, expert_ms - weights_ms - infer_ms)
    exclusive = [
        ("expert read/weight cache", expert_read_ms),
        ("expert dequant/cast", decode_ms),
        ("expert OpenVINO infer", infer_ms),
        ("expert wrapper/other", expert_misc_ms),
        ("shared expert", shared_ms),
        ("DeltaNet total", ssm_ms),
        ("full attention", attn_ms),
        ("router", route_ms),
        ("embedding", embedding_ms),
        ("LM head", output_ms),
    ]
    attributed_ms = sum(ms for _, ms in exclusive)
    exclusive.append(("other/residual", max(0.0, per_token * 1000 - attributed_ms)))
    for label, ms in exclusive:
        print(f"{label:<22}{ms:>10.1f}{ms / (per_token * 1000) * 100:>6.1f}%")
    print(f"\nDeltaNet projection: {ssm_proj_ms:.1f} ms/token, "
          f"rest: {max(0.0, ssm_ms - ssm_proj_ms):.1f} ms/token "
          "(sottoinsieme DeltaNet, non sommato sopra)")
    print("Full-attention phases (ms/token; somme sulle 10 layer):")
    for key, label in (
        ("projections", "  Q/K/V projections"),
        ("qk_norm_and_reshape", "  QK norm/reshape"),
        ("rope_and_layout", "  RoPE/layout"),
        ("kv_cache", "  KV cache append"),
        ("grouped_attention", "  grouped attention inclusive"),
        ("gate_and_output_projection", "  gate/output projection"),
    ):
        print(f"{label:<31}{attn_phases.get(key, 0.0) / args.measure * 1000:>9.1f}")
    print("Grouped-attention breakdown (nested; do not add to inclusive):")
    for key, label in (
        ("attention_cast_and_gqa", "  cast/GQA replication"),
        ("attention_scores_and_mask", "  scores/mask"),
        ("attention_softmax", "  softmax"),
        ("attention_values", "  value aggregation"),
    ):
        print(f"{label:<31}{attn_phases.get(key, 0.0) / args.measure * 1000:>9.1f}")
    print(f"expert forward inclusivo: {expert_ms:.1f} ms/token "
          "(sottoinsieme read/dequant + infer, non sommato sopra)")
    s = model.store.stats
    print(f"expert weight-cache measured: {weight_cache['hits']} hits, "
          f"{weight_cache['misses']} misses "
          f"({weight_cache['misses'] / args.measure:.1f} misses/token)")
    print(f"store measured: "
          f"hit={s.hits - store_stats_start[0]} "
          f"miss={s.misses - store_stats_start[1]} "
          f"prefetch_hit={s.prefetch_hits - store_stats_start[2]} "
          f"bytes={max(0, s.bytes_read - store_stats_start[3]) / 2**20:.0f} MiB")
    print(f"cache expert: {model.backend.cache_size()}")
    model.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
