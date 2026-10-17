"""Baseline end-to-end: un token dal prompt all'output, con il tempo per fase.

Finora i numeri che abbiamo sono tutti di componente: OpenVINO, dequant,
shared expert. Nessuno ha mai misurato un token intero, quindi la somma dei
 pezzi non e' una misura e non sappiamo a quanti token al secondo gira il
modello. Questo script attraversa il percorso vero e cronometra ogni fase
con lo stesso token, così il totale e' il totale.

Non modifica la libreria: avvolge i metodi già costruiti con un cronometro e
sommando il tempo che ci mettono. Ogni fase attraversa un solo token, e il
decode di questo modello e' per definizione sequenziale (gli expert del
layer L+1 si conoscono solo dopo l'output del layer L), quindi le fasi non
si possono sovrapporre e la somma e' il tempo totale.
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gguf  # noqa: E402
from run_q4k_f16_openvino import dequant_to_f16  # noqa: E402
from flash_moe import Qwen3MoEModel  # noqa: E402
from flash_moe.paths import model_path  # noqa: E402
from flash_moe.tokenizer import GGUFTokenizer  # noqa: E402

PROMPT = "Il sole splende sopra"
NEW_TOKENS = 3


import os


def rss_mib():
    import subprocess
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"(Get-Process -Id {os.getpid()}).WorkingSet64"],
        capture_output=True, text=True)
    try:
        return int(out.stdout.strip()) / 1024 ** 2
    except ValueError:
        return float("nan")


class Clock:
    """Accumula il tempo di ogni fase, senza fermare il resto."""

    def __init__(self):
        self.totals = {}
        self.counts = {}

    def wrap(self, name, fn):
        def timed(*a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                self.totals[name] = self.totals.get(name, 0.0) + \
                    (time.perf_counter() - t0)
                self.counts[name] = self.counts.get(name, 0) + 1
        return timed

    def reset(self):
        self.totals.clear()
        self.counts.clear()


def main():
    path = model_path()
    tok = GGUFTokenizer(gguf.GGUFReader(path))

    print(f"modello: {path.name}")
    t0 = time.perf_counter()
    # Dispositivo per esperti: CPU per minima latenza dispatch
    # Dispositivo per proiezioni fisse: GPU Arc (con accelerazione XMX)
    model = Qwen3MoEModel(
        path,
        device=(sys.argv[1] if len(sys.argv) > 1 else "CPU"),
        max_seq_len=256,
        cache_capacity=64,
        dequant_fn=dequant_to_f16,
        use_mmap=True,
        ssm_device="GPU",
        attention_device="GPU",
    )
    load_s = time.perf_counter() - t0
    print(f"caricamento: {load_s:.1f} s, RSS {rss_mib():.0f} MiB")
    print(f"{model.block_count} layer, {model.num_experts} expert, top-{model.top_k}")

    clock = Clock()
    # Le fasi costose, avvolte dove vengono chiamate davvero. Sul layer
    # ricorrente si avvolge il metodo che _run_ssm chiama una volta per
    # token: avvolgere anche ssm_runner conterebbe il delta net due volte,
    # una dentro l'altra, e il totale non tornerebbe con la somma.
    for block in model.blocks:
        if not block.is_ssm:
            block._attention_forward = clock.wrap(
                "attenzione (10 layer full)", block._attention_forward)
        block._moe_forward = clock.wrap("MoE routed (8 expert x 40)",
                                        block._moe_forward)
    for lyr in model.ssm.values():
        lyr.forward = clock.wrap("delta net (30 layer)", lyr.forward)
    if model.shared_expert is not None:
        se = model.shared_expert
        se.forward = clock.wrap("shared expert (40)", se.forward)

    ids = tok.encode(PROMPT, add_bos=True)
    print(f"\nprompt {PROMPT!r} -> {len(ids)} token, {PROMPT and ''}"
          f"genero {NEW_TOKENS} token")

    # ---- prefillo
    model.init_caches()
    x = model.embedding.forward(np.asarray(ids, dtype=np.int64))
    x = x.reshape(1, len(ids), model.hidden_size)
    t0 = time.perf_counter()
    x = model._forward_all(x)
    prefill_s = time.perf_counter() - t0
    print(f"\nprefillo: {len(ids)} token in {prefill_s:.2f} s "
          f"({len(ids) / prefill_s:.2f} tok/s di prompt)")

    # ---- decode: un passo alla volta, cronometrato
    steps = []
    print("\ndecode (un token alla volta)")
    for step in range(NEW_TOKENS):
        t0 = time.perf_counter()
        normed = model.output_norm.forward(x)
        logits = model.lm_head.forward(normed)
        head_s = time.perf_counter() - t0

        clock.reset()
        t0 = time.perf_counter()
        last = np.asarray(logits[0, -1], dtype=np.float32).reshape(-1)
        next_id = int(np.argmax(last))
        xn = model.embedding.forward(np.asarray([next_id], dtype=np.int64))
        xn = xn.reshape(1, 1, model.hidden_size)
        x = model._forward_all(xn)
        step_s = time.perf_counter() - t0
        phases = dict(clock.totals)
        steps.append((step_s, head_s, phases))
        print(f"  passo {step + 1}: {step_s * 1000:8.1f} ms  "
              f"lm_head {head_s * 1000:6.1f} ms")

    # ---- la fase piu' lenta di tutte, e la sua scomposizione
    worst = max(steps, key=lambda s: s[0] + s[1])
    total = worst[0] + worst[1]
    print(f"\npasso piu' lento: passo {steps.index(worst) + 1}, "
          f"{worst[0] * 1000:.1f} ms di layer + {worst[1] * 1000:.1f} ms "
          f"di lm_head = {total * 1000:.1f} ms")
    accounted = worst[1]
    for name, seconds in sorted(worst[2].items(), key=lambda kv: -kv[1]):
        accounted += seconds
        print(f"  {name:30s} {seconds * 1000:8.1f} ms  "
              f"({seconds / total * 100:4.1f}%)  chiamate {clock.counts.get(name, 0)}")
    rest = total - accounted
    print(f"  {'norme e residui':30s} {rest * 1000:8.1f} ms  "
          f"({rest / total * 100:4.1f}%)")
    print(f"  {'lm_head':30s} {worst[1] * 1000:8.1f} ms  "
          f"({worst[1] / total * 100:4.1f}%)")

    print(f"\n{'=' * 70}")
    per_token = sorted(s[0] + s[1] for s in steps)[len(steps) // 2]
    print(f" DECODE: {per_token * 1000:.0f} ms/token  ->  {1 / per_token:.2f} tok/s")
    print(f" prefillo: {len(ids) / prefill_s:.2f} tok/s")
    print(f"{'=' * 70}")
    model.close()


if __name__ == "__main__":
    main()