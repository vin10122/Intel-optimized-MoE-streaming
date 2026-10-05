"""Scompone il tempo di un passo di decode per fase, con parametri.

Uso: python tools/profile_decode.py [cache_capacity] [keep_ssm_f32]

Fasi (nessun doppio conteggio: i blocchi sono attributi a fase,
non sommati ai wrapper):
* moe_weights: read_tensor + dequant nel backend (cache miss)
* moe_infer: infer OpenVINO degli expert
* shared: shared expert
* ssm_proj: proiezioni DeltaNet (conversione f16->f32 se non keep_f32)
* ssm_rest: conv, delta rule, norm, proiezione di uscita
* attn: layer di attenzione piena
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\valer\Desktop\moe-client-prot\Nuova cartella")
sys.path.insert(0, str(ROOT / "src"))

import gguf  # noqa: E402
from run_q4k_f16_openvino import dequant_to_f16  # noqa: E402
import flash_moe.gated_deltanet as gdn  # noqa: E402
import flash_moe.backend_ov as bov  # noqa: E402
import flash_moe.attention as attn_mod  # noqa: E402
from flash_moe.shared_expert import SharedExpert  # noqa: E402
from flash_moe import Qwen3MoEModel  # noqa: E402
from flash_moe.paths import model_path  # noqa: E402
from flash_moe.tokenizer import GGUFTokenizer  # noqa: E402

PROMPT = "Il sole splende sopra"
N_WARM = 2
N_MEAS = 5


class Acc:
    __slots__ = ("t", "n")

    def __init__(self):
        self.t = 0.0
        self.n = 0

    def add(self, dt: float) -> None:
        self.t += dt
        self.n += 1


def main() -> None:
    cache_capacity = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    keep_f32 = (len(sys.argv) > 2 and sys.argv[2] == "f32")

    acc = {k: Acc() for k in (
        "moe_weights", "moe_infer", "expert_forward", "shared",
        "ssm_proj", "ssm_rest", "ssm_total", "attn")}

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

    def backend_fwd(self, key, activation, hidden_size,
                        intermediate_size):
        t0 = time.perf_counter()
        out = orig_backend_fwd(self, key, activation,
                                    hidden_size, intermediate_size)
        acc["expert_forward"].add(time.perf_counter() - t0)
        return out

    def weights(self, key, hidden_size, intermediate_size):
        t0 = time.perf_counter()
        out = orig_weights(self, key, hidden_size, intermediate_size)
        acc["moe_weights"].add(time.perf_counter() - t0)
        return out

    def infer(self, request, feed):
        t0 = time.perf_counter()
        out = orig_infer(self, request, feed)
        acc["moe_infer"].add(time.perf_counter() - t0)
        return out

    bov.OpenVINOExpertBackend.forward = backend_fwd
    bov.OpenVINOExpertBackend._weights = weights
    bov.OpenVINOExpertBackend._infer = infer

    orig_shared = SharedExpert.forward

    def shared_fwd(self, layer, activation):
        t0 = time.perf_counter()
        out = orig_shared(self, layer, activation)
        acc["shared"].add(time.perf_counter() - t0)
        return out

    SharedExpert.forward = shared_fwd

    orig_attn_fwd = attn_mod.AttentionLayer.forward

    def attn_fwd(self, x, weights, cache=None):
        t0 = time.perf_counter()
        out = orig_attn_fwd(self, x, weights, cache)
        acc["attn"].add(time.perf_counter() - t0)
        return out

    attn_mod.AttentionLayer.forward = attn_fwd

    # --- run -------------------------------------------------------------
    path = model_path()
    tok = GGUFTokenizer(gguf.GGUFReader(path))
    ids = tok.encode(PROMPT, add_bos=False)

    model = Qwen3MoEModel(path, device="CPU", max_seq_len=256,
                          cache_capacity=cache_capacity,
                          dequant_fn=dequant_to_f16,
                          keep_ssm_f32=keep_f32)
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
    for _ in range(N_WARM):
        step = model.embedding.forward(np.asarray([cur], dtype=np.int64)).reshape(
            1, 1, model.hidden_size)
        for layer in range(model.block_count):
            step = model.blocks[layer].forward(step)
        cur = next_token(step[0, -1, :])

    for a in acc.values():
        a.t = 0.0
        a.n = 0

    t_all = time.perf_counter()
    for _ in range(N_MEAS):
        step = model.embedding.forward(np.asarray([cur], dtype=np.int64)).reshape(
            1, 1, model.hidden_size)
        for layer in range(model.block_count):
            step = model.blocks[layer].forward(step)
        cur = next_token(step[0, -1, :])
    total = time.perf_counter() - t_all

    per_token = total / N_MEAS
    print(f"\n=== cache={cache_capacity} keep_ssm_f32={keep_f32} ===")
    print(f"decode: {per_token * 1000:.0f} ms/token "
          f"({1 / per_token:.2f} tok/s)  [{N_MEAS} passi]")
    print(f"\n{'fase':<16}{'ms/token':>10}{'%':>7}{'call/tok':>10}")
    rows = [
        ("expert_fwd", "expert_forward"),
        ("moe_weights", "moe_weights"),
        ("moe_infer", "moe_infer"),
        ("shared", "shared"),
        ("ssm_total", "ssm_total"),
        ("ssm_proj", "ssm_proj"),
        ("ssm_rest", None),
        ("attn", "attn"),
    ]
    for label, key in rows:
        if key is not None:
            a = acc[key]
            if a.n == 0:
                continue
            ms = a.t / N_MEAS * 1000
            print(f"{label:<16}{ms:>10.1f}"
                  f"{ms / (per_token * 1000) * 100:>6.1f}%"
                  f"{a.n // N_MEAS:>10}")
        else:
            ms = (acc["ssm_total"].t - acc["ssm_proj"].t) / N_MEAS * 1000
            print(f"{label:<16}{ms:>10.1f}"
                  f"{ms / (per_token * 1000) * 100:>6.1f}%")
    moe = (acc["expert_forward"].t + acc["shared"].t) / N_MEAS * 1000
    print(f"{'MoE totale':<16}{moe:>10.1f}"
          f"{moe / (per_token * 1000) * 100:>6.1f}%")
    s = model.store.stats
    print(f"store: hit={s.hits} miss={s.misses} prefetch_hit={s.prefetch_hits}"
          f" ({N_MEAS} passi)")
    print(f"cache expert: {model.backend.cache_size()}")
    model.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
