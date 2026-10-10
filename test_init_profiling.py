#!/usr/bin/env python
"""Profiling dettagliato dell'inizializzazione"""
import sys
sys.path.insert(0, 'src')

import time
from pathlib import Path
import gguf
from run_q4k_f16_openvino import build_manifest_from_gguf, expert_tensor_types
from flash_moe.paths import model_path

def main():
    path = model_path()

    print("="*60)
    print("Detailed Initialization Profiling")
    print("="*60)

    # Step 1: GGUFReader
    print("\n1. Creating GGUFReader...")
    t0 = time.perf_counter()
    reader = gguf.GGUFReader(path)
    t1 = time.perf_counter()
    print(f"   Time: {t1-t0:.2f}s")

    # Step 2: Build tensor map
    print("\n2. Building tensor map...")
    t0 = time.perf_counter()
    tmap = {t.name: t for t in reader.tensors}
    t1 = time.perf_counter()
    print(f"   Time: {t1-t0:.2f}s")
    print(f"   Tensors: {len(tmap)}")

    # Step 3: Build manifest
    print("\n3. Building manifest...")
    t0 = time.perf_counter()
    manifest = build_manifest_from_gguf(path)
    t1 = time.perf_counter()
    print(f"   Time: {t1-t0:.2f}s")
    print(f"   Records: {len(manifest.records)}")

    # Step 4: Expert tensor types
    print("\n4. Getting expert tensor types...")
    t0 = time.perf_counter()
    tensor_types = expert_tensor_types(path)
    t1 = time.perf_counter()
    print(f"   Time: {t1-t0:.2f}s")
    print(f"   Layers: {len(tensor_types)}")

    # Step 5: Import OpenVINO
    print("\n5. Importing OpenVINO...")
    t0 = time.perf_counter()
    import openvino as ov
    from flash_moe.mmap_store import MemoryMappedAsyncExpertStore
    from flash_moe.backend_ov import OpenVINOExpertBackend
    t1 = time.perf_counter()
    print(f"   Time: {t1-t0:.2f}s")

    # Step 6: Create store
    print("\n6. Creating MemoryMappedAsyncExpertStore...")
    t0 = time.perf_counter()
    store = MemoryMappedAsyncExpertStore(
        path.parent, capacity=1, manifest=manifest
    )
    t1 = time.perf_counter()
    print(f"   Time: {t1-t0:.2f}s")

    # Step 7: Create backend
    print("\n7. Creating OpenVINOExpertBackend...")
    t0 = time.perf_counter()
    backend = OpenVINOExpertBackend(
        store,
        device="CPU",
        weight_dtype="f16",
        cache_capacity=1,
        tensor_types=tensor_types,
        dequant_fn=dequant_to_f16,
    )
    t1 = time.perf_counter()
    print(f"   Time: {t1-t0:.2f}s")

    # Step 8: routers_from_gguf
    print("\n8. Building routers...")
    t0 = time.perf_counter()
    from flash_moe.router import routers_from_gguf
    routers = routers_from_gguf(
        tmap, 40, 2048, 256, 8
    )
    t1 = time.perf_counter()
    print(f"   Time: {t1-t0:.2f}s")
    print(f"   Routers: {len(routers)}")

    store.close()

    print("\n" + "="*60)
    print("Total estimated initialization time:")
    print("  (sum of all steps above)")
    print("="*60)

if __name__ == '__main__':
    from run_q4k_f16_openvino import dequant_to_f16
    main()
