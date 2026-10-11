"""I pesi del delta net sono dequantizzati bene?

Il delta net sbaglia dalla prima proiezione, e la prima cosa che fa e'
un matmul con `attn_qkv`:16,8M di parametri, il tensor piu' grosso del
layer. Se la dequantizzazione di quel tensor e' sbagliata, tutto il resto
e' rumore. Qui ogni tensor del layer 0 viene dequantizzato con la DLL C++
e con la reference di gguf, e i due risultati confrontati.

Non e' una verifica di accuratezza sul modello: e' la verifica che i due
percorsi di dequantizzazione dicano la stessa cosa. Se differiscono, uno
dei due e' sbagliato e non si puo' sapere quale senza la reference.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# La console di Windows e' cp1252 e non sa scrivere i caratteri che questo
# test usa nelle etichette: senza questo muore su un accento invece di dire
# cosa ha trovato.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
sys.path.insert(0, str(ROOT / "src"))

import gguf  # noqa: E402
from gguf.constants import GGML_QUANT_SIZES  # noqa: E402
from run_q4k_f16_openvino import dequant_to_f16  # noqa: E402
from flash_moe.paths import model_path  # noqa: E402

path = model_path()
reader = gguf.GGUFReader(path)
tmap = {t.name: t for t in reader.tensors}

NAMES = [
    "blk.0.attn_qkv.weight",
    "blk.0.attn_gate.weight",
    "blk.0.ssm_out.weight",
    "blk.0.ssm_conv1d.weight",
    "blk.0.ssm_alpha.weight",
    "blk.0.ssm_beta.weight",
    "blk.0.ssm_a.weight",
    "blk.0.ssm_dt_bias.weight",
    "blk.0.ssm_norm.weight",
    "blk.1.attn_qkv.weight",
    "blk.3.attn_q.weight",
    "blk.0.ffn_gate_exps.weight",
]

print(f"{'tensor':30s} {'tipo':>10} {'n':>10} {'max|diff|':>11} {'rel':>10}")
for name in NAMES:
    t = tmap.get(name)
    if t is None:
        print(f"{name:30s} {'ASSENTE':>10}")
        continue
    ttype = int(t.tensor_type)
    label = gguf.GGMLType(ttype).name if hasattr(gguf, "GGMLType") else str(ttype)
    blck = GGML_QUANT_SIZES[ttype][1]
    n = t.n_elements
    raw = np.frombuffer(t.data, dtype=np.uint8)

    try:
        ref = gguf.quants.dequantize(raw, ttype)
    except Exception as err:
        print(f"{name:30s} {label:>10} reference fallita: {str(err)[:40]}")
        continue
    if getattr(ref, "dtype", None) != np.float32:
        ref = ref.astype(np.float32)
    ref = ref.reshape(-1)

    got = np.asarray(dequant_to_f16(t.data, ttype, n), dtype=np.float32)
    if got.size != ref.size:
        print(f"{name:30s} {label:>10} {n:10d}  dimensione diversa: "
              f"{got.size} contro {ref.size}")
        continue
    diff = np.abs(got - ref)
    scale = float(np.abs(ref).max())
    rel = float(diff.max()) / scale if scale else 0.0
    flag = "" if rel < 1e-2 else "   <-- DIVERGE"
    print(f"{name:30s} {label:>10} {n:10d} {float(diff.max()):11.4f} "
          f"{rel * 100:9.2f}%{flag}")
