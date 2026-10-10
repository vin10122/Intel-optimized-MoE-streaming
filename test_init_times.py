#!/usr/bin/env python
"""Misura i tempi di inizializzazione separatamente"""
import sys
sys.path.insert(0, 'src')

import time
from pathlib import Path
import gguf
from flash_moe.paths import model_path
from run_q4k_f16_openvino import build_manifest_from_gguf, dequant_to_f16

def main():
    path = model_path()

    print("="*60)
    print("Misurazione tempi di inizializzazione")
    print("="*60)

    # 1. Caricamento GGUF metadata
    print("\n1. Caricamento metadata GGUF...")
    t0 = time.perf_counter()
    reader = gguf.GGUFReader(path)
    tmap = {t.name: t for t in reader.tensors}
    t1 = time.perf_counter()
    print(f"   Tempo: {t1-t0:.2f}s")

    # 2. Costruzione manifest
    print("\n2. Costruzione manifest...")
    t0 = time.perf_counter()
    manifest = build_manifest_from_gguf(path)
    t1 = time.perf_counter()
    print(f"   Tempo: {t1-t0:.2f}s")
    print(f"   Record: {len(manifest.records)}")

    # 3. Inizializzazione OpenVINO Core
    print("\n3. Inizializzazione OpenVINO Core...")
    t0 = time.perf_counter()
    import openvino as ov
    core = ov.Core()
    t1 = time.perf_counter()
    print(f"   Tempo: {t1-t0:.2f}s")
    print(f"   Dispositivi disponibili: {list(core.available_devices)}")

    # 4. Compilazione modello SwiGLU
    print("\n4. Compilazione modello SwiGLU (2048, 512)...")
    import numpy as np
    import openvino.opset13 as ops

    t0 = time.perf_counter()
    wtype = ov.Type.f16
    x = ops.parameter([1, 2048], ov.Type.f32, name="activation")
    gate = ops.parameter([512, 2048], wtype, name="gate")
    up = ops.parameter([512, 2048], wtype, name="up")
    down = ops.parameter([2048, 512], wtype, name="down")

    xc = ops.convert(x, wtype)
    gate_proj = ops.matmul(xc, gate, transpose_a=False, transpose_b=True)
    up_proj = ops.matmul(xc, up, transpose_a=False, transpose_b=True)
    act = ops.multiply(gate_proj, ops.sigmoid(gate_proj))
    out = ops.matmul(ops.multiply(act, up_proj), down,
                     transpose_a=False, transpose_b=True)

    model = ov.Model([ops.convert(out, ov.Type.f32)], [x, gate, up, down],
                     name="SwiGLU_Expert")
    compiled = core.compile_model(model, "CPU")
    t1 = time.perf_counter()
    print(f"   Tempo: {t1-t0:.2f}s")

    # 5. Creazione infer request
    print("\n5. Creazione infer request...")
    t0 = time.perf_counter()
    request = compiled.create_infer_request()
    t1 = time.perf_counter()
    print(f"   Tempo: {t1-t0:.4f}s")

    # 6. Caricamento modello completo
    print("\n6. Caricamento modello completo (Qwen3MoEModel)...")
    t0 = time.perf_counter()
    from flash_moe import Qwen3MoEModel
    model = Qwen3MoEModel(path, device="CPU", max_seq_len=128,
                          cache_capacity=1, dequant_fn=dequant_to_f16)
    t1 = time.perf_counter()
    print(f"   Tempo: {t1-t0:.1f}s")
    model.close()

    print("\n" + "="*60)
    print("Riepilogo:")
    print("  Il collo di bottiglia è l'inizializzazione OpenVINO + compilazione")
    print("="*60)

if __name__ == '__main__':
    main()
