"""Test full MoE chain: 40 layer end-to-end vs NumPy f32 reference.

Valida l'intero pipeline su TUTTI i 40 layer con i loro tipi ggml diversi:
  layer 0-9:   Q4_K gate/up, Q8_0 down      (10 layer)
  layer 10-38: Q3_K gate/up/down            (29 layer)
  layer 39:    Q4_K gate/up, Q6_K down      (1 layer)

Per ogni layer:
  1. Router seleziona 8 expert (stesso seed = determinismo)
  2. System:  engine.process_token -> store -> C++ dequant -> OpenVINO -> output
  3. Reference: gguf.quants.dequantize(f32) + NumPy SwiGLU + weighted sum
  4. Confronta output entro tolleranza f16

In più: 1 run completo della catena (40 layer concatenati) per timing.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import gguf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flash_moe import (  # noqa: E402
    ExpertKey,
    AsyncExpertStore,
    synthetic_router,
    OpenVINOExpertBackend,
    MoEEngine,
)
from flash_moe.dequant import dequantize_to, name_of  # noqa: E402
from flash_moe.paths import model_path  # noqa: E402
from run_q4k_f16_openvino import (  # noqa: E402
    build_manifest_from_gguf,
    expert_tensor_types,
    dequant_to_f16,
)

HIDDEN = 2048
INTER = 512
N_LAYERS = 40
TOL = 5e-3

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name if ok else (name, detail))
    tag = "  PASS  " if ok else "  FAIL  "
    print(f"{tag}{name}" + (f"  ({detail})" if detail else ""), flush=True)


def numpy_moe_layer(x: np.ndarray, layer: int, expert_ids: list[int],
                    weights: list[float], tmap: dict, n_experts: int,
                    gguf_path: Path) -> np.ndarray:
    """MoE layer in NumPy f32: dequant Python + SwiGLU + weighted sum."""
    combined = np.zeros(HIDDEN, dtype=np.float32)
    for eid, w in zip(expert_ids, weights):
        tensors: dict[str, np.ndarray] = {}
        for role, shp in (("gate", (HIDDEN, INTER)),
                          ("up", (HIDDEN, INTER)),
                          ("down", (INTER, HIDDEN))):
            t = tmap[f"blk.{layer}.ffn_{role}_exps.weight"]
            per = t.n_bytes // n_experts
            with gguf_path.open("rb") as fh:
                fh.seek(t.data_offset + eid * per)
                raw = fh.read(per)
            arr = dequantize_to(raw, int(t.tensor_type), "f32").reshape(shp)
            tensors[role] = arr
        g = x @ tensors["gate"]
        u = x @ tensors["up"]
        h = (g / (1.0 + np.exp(-g))) * u
        out = (h @ tensors["down"]).flatten()
        combined += w * out
    return combined


def main() -> int:
    gguf_path = model_path()
    print("=" * 78)
    print(" FULL MoE CHAIN: 40 layer vs NumPy f32")
    print("=" * 78)
    print(f"modello : {gguf_path.name}")

    reader = gguf.GGUFReader(gguf_path)
    tmap = {t.name: t for t in reader.tensors}
    arch = str(reader.fields["general.architecture"].parts[-1].tobytes().decode())
    n_experts = int(reader.fields[f"{arch}.expert_count"].parts[-1][0])

    layer_types_map = expert_tensor_types(gguf_path)
    manifest = build_manifest_from_gguf(gguf_path)

    # Show type distribution
    type_summary: dict[str, list[int]] = {}
    for layer, types in layer_types_map.items():
        for role, tid in types.items():
            tname = name_of(tid)
            type_summary.setdefault(tname, []).append(layer)
    print(f"tipi    : {len(layer_types_map)} layer")
    for tname, layers in sorted(type_summary.items()):
        print(f"  {tname:6s}: layer {layers[0]}..{layers[-1]} ({len(layers)} tensor)")

    # Create pipeline
    store = AsyncExpertStore(gguf_path.parent, capacity=700, manifest=manifest)
    backend = OpenVINOExpertBackend(
        store, device="GPU", weight_dtype="f16",
        tensor_types=layer_types_map,
        dequant_fn=dequant_to_f16,
    )
    rng = np.random.default_rng(42)
    # Router finto ma deterministico, uno per layer. Il reference qui sotto
    # usa lo stesso router, quindi il confronto OpenVINO/NumPy resta valido:
    # quello che si misura e' l'esecuzione degli expert, non la scelta.
    routers = {
        layer: synthetic_router(HIDDEN, n_experts, top_k=8, seed=42, layer=layer)
        for layer in range(N_LAYERS)
    }
    engine = MoEEngine(store, backend, routers, HIDDEN, INTER)
    activation = rng.standard_normal(HIDDEN).astype(np.float32) * 0.1

    print(f"\nValidazione per-layer (8 expert/layer, top_k=8)...")
    print(f"{'layer':>6s}  {'tipo':<24s}  {'rel_err':>10s}  {'corr':>10s}")
    print("-" * 60)

    max_rel = 0.0
    for layer in range(N_LAYERS):
        layer_types = layer_types_map.get(layer, {})
        type_str = ", ".join(f"{r}={name_of(t)}" for r, t in layer_types.items())

        # System: MoEEngine (C++ dequant + OpenVINO)
        engine_out = np.array(
            engine.process_token(layer, activation.tolist()),
            dtype=np.float64,
        )

        # Reference: NumPy (Python dequant + NumPy SwiGLU)
        eids, weights = routers[layer].route(activation)
        ref_out = numpy_moe_layer(
            activation, layer, eids, weights, tmap, n_experts, gguf_path
        ).astype(np.float64)

        scale = float(np.abs(ref_out).max())
        rel = float(np.abs(engine_out - ref_out).max() / max(scale, 1e-9))
        corr = float(np.corrcoef(engine_out, ref_out)[0, 1])
        max_rel = max(max_rel, rel)

        status = "OK" if rel < TOL else "FAIL"
        print(f"{layer:6d}  {type_str:<24s}  {rel:10.3e}  {corr:10.8f}  {status}")

        if rel >= TOL:
            check(f"layer {layer} OpenVINO == NumPy", False,
                  f"rel={rel:.3e}, corr={corr:.6f}")

    check("tutti i 40 layer entro tolleranza", max_rel < TOL,
          f"max_rel={max_rel:.3e}")

    # Full chain timing (1 token through all 40 layers)
    print("\n--- Full chain timing (1 token, 40 layer) ---")
    store.close()
    store = AsyncExpertStore(gguf_path.parent, capacity=700, manifest=manifest)
    backend = OpenVINOExpertBackend(
        store, device="GPU", weight_dtype="f16",
        tensor_types=layer_types_map,
        dequant_fn=dequant_to_f16,
    )
    routers = {
        layer: synthetic_router(HIDDEN, n_experts, top_k=8, seed=42, layer=layer)
        for layer in range(N_LAYERS)
    }
    engine = MoEEngine(store, backend, routers, HIDDEN, INTER)
    activation = rng.standard_normal(HIDDEN).astype(np.float32) * 0.1

    t0 = time.perf_counter()
    for layer in range(N_LAYERS):
        engine.process_token(layer, activation.tolist())
    dt = time.perf_counter() - t0
    tpr = dt / N_LAYERS * 1000
    print(f"  tempo totale   : {dt:.3f} s")
    print(f"  ms/layer       : {tpr:.1f}")
    print(f"  token/s (est.) : {1.0/dt:.2f}")
    print(f"  cache misses   : {store.stats.misses}")
    print(f"  cache hits     : {store.stats.hits}")
    print(f"  prefetch hits  : {store.stats.prefetch_hits}")
    print(f"  expert da SSD  : {store.stats.disk_loads}")
    print(f"  bytes da SSD   : {store.stats.bytes_read / 1024**2:.1f} MB")

    check(f"pipeline completa {dt < 30}", dt < 30, f"{dt:.2f}s")
    store.close()

    print("\n" + "=" * 78)
    print(f" PASSATI: {len(PASSED)}   FALLITI: {len(FAILED)}")
    for name, detail in FAILED:
        print(f"   - {name}: {detail}")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
