"""Test end-to-end del pipeline completo: manifest -> store -> backend -> engine.

test_expert_e2e.py valida il backend isolatamente (senza store). Questo test
esercita invece l'intero pipeline:

  1. build_manifest_from_gguf  -> slice con offset corretti per-layer
  2. AsyncExpertStore.read_tensor -> byte letti dal GGUF tramite manifest
  3. OpenVINOExpertBackend._decode -> dequant C++ (fast_dequant_k.dll)
  4. OpenVINOExpertBackend.forward -> SwiGLU su OpenVINO GPU (f16 nativo)
  5. MoEEngine.process_token -> routing + weighted sum degli expert

Su layer con tipi ggml diversi per catchesia:
  layer 0  : gate=Q4_K  up=Q4_K  down=Q8_0
  layer 10 : gate=Q3_K  up=Q3_K  down=Q3_K
  layer 39 : gate=Q4_K  up=Q4_K  down=Q6_K

Confronta ogni output contro la reference indipendente gguf.quants in f32.

Uso:
    python tools/test_pipeline_e2e.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import gguf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flash_moe import (  # noqa: E402
    ExpertKey,
    AsyncExpertStore,
    ExpertRouter,
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
TOL = 5e-3
DEQUANT_TOL = 1e-3

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name if ok else (name, detail))
    tag = "  PASS  " if ok else "  FAIL  "
    print(f"{tag}{name}" + (f"  ({detail})" if detail else ""), flush=True)


TEST_LAYERS = [
    (0, 0,   "Q4_K/Q4_K/Q8_0"),
    (0, 150,  "Q4_K/Q4_K/Q8_0"),
    (10, 0,   "Q3_K/Q3_K/Q3_K"),
    (10, 200,  "Q3_K/Q3_K/Q3_K"),
    (39, 0,   "Q4_K/Q4_K/Q6_K"),
]


def swiglu_numpy(x: np.ndarray, gate: np.ndarray, up: np.ndarray,
                 down: np.ndarray) -> np.ndarray:
    g = x @ gate
    u = x @ up
    h = (g / (1.0 + np.exp(-g))) * u
    return (h @ down).flatten()


def read_expert_tensor_raw(gguf_path: Path, tensor, expert_id: int,
                           n_experts: int) -> tuple[bytes, int]:
    """Legge la slice raw dell'experte direttamente dal GGUF (bypassa lo store)."""
    per = tensor.n_bytes // n_experts
    with gguf_path.open("rb") as fh:
        fh.seek(tensor.data_offset + expert_id * per)
        return fh.read(per), int(tensor.tensor_type)


def main() -> int:
    gguf_path = model_path()
    print("=" * 78)
    print(" TEST PIPELINE E2E: STORE -> BACKEND -> ENGINE vs NUMPY f32")
    print("=" * 78)
    print(f"modello : {gguf_path.name}")

    reader = gguf.GGUFReader(gguf_path)
    tmap = {t.name: t for t in reader.tensors}
    arch = str(reader.fields["general.architecture"].parts[-1].tobytes().decode())
    n_experts = int(reader.fields[f"{arch}.expert_count"].parts[-1][0])

    layer_types_map = expert_tensor_types(gguf_path)
    n_q3k = sum(1 for d in layer_types_map.values()
                if name_of(d.get("gate", 0)) == "Q3_K")
    n_q4k = sum(1 for d in layer_types_map.values()
                if name_of(d.get("gate", 0)) == "Q4_K")
    print(f"tipi    : {len(layer_types_map)} layer ({n_q4k} Q4_K, {n_q3k} Q3_K)")

    manifest = build_manifest_from_gguf(gguf_path)
    print(f"manifest: {len(manifest.records)} expert records")

    store = AsyncExpertStore(gguf_path.parent, capacity=64, manifest=manifest)
    backend = OpenVINOExpertBackend(
        store, device="GPU", weight_dtype="f16",
        tensor_types=layer_types_map,
        dequant_fn=dequant_to_f16,
    )
    print(f"backend : device=GPU, dequant={'C++' if True else 'python'}")

    rng = np.random.default_rng(42)
    x = rng.standard_normal(HIDDEN).astype(np.float32)

    for layer, expert, label in TEST_LAYERS:
        key = ExpertKey(layer, expert)
        layer_types = layer_types_map.get(layer, {})
        print(f"\n--- layer {layer} expert {expert} ({label}) ---")

        names = {r: f"blk.{layer}.ffn_{r}_exps.weight" for r in ("gate", "up", "down")}
        shapes = {"gate": (HIDDEN, INTER), "up": (HIDDEN, INTER), "down": (INTER, HIDDEN)}

        # A. Store data size validation
        for r in ("gate", "up", "down"):
            raw = store.read_tensor(key, r)
            expected = tmap[names[r]].n_bytes // n_experts
            check(f"store size {r}", len(raw) == expected,
                  f"read {len(raw)}, expected {expected}")

        # B. Backend dequant (C++ DLL)
        w16 = {}
        for r in ("gate", "up", "down"):
            raw = store.read_tensor(key, r)
            w16[r] = backend._decode(raw, layer_types.get(r), shapes[r])
            nan_count = int(np.isnan(w16[r].astype(np.float32)).sum())
            check(f"backend dequant {r} no NaN", nan_count == 0,
                  f"nan={nan_count}, absmax={float(np.abs(w16[r]).max()):.4f}")

        # C. Dequant correctness: C++ vs Python reference
        w32_ref = {}
        for r in ("gate", "up", "down"):
            raw, gtype = read_expert_tensor_raw(gguf_path, tmap[names[r]], expert, n_experts)
            shape = tuple(int(v) for v in tmap[names[r]].shape)[:2]
            w32_ref[r] = dequantize_to(raw, gtype, "f32").reshape(shape)

        for r in ("gate", "up", "down"):
            ref = w32_ref[r]
            absmax = float(np.abs(ref).max())
            rel = float(np.abs(w16[r].astype(np.float32) - ref).max() / max(absmax, 1e-9))
            check(f"dequant {r} C++ == gguf.ref", rel < DEQUANT_TOL, f"rel={rel:.3e}")

        # D. Forward pass: OpenVINO vs NumPy
        out = backend.forward(key, x, HIDDEN, INTER)
        ref = swiglu_numpy(x, w32_ref["gate"], w32_ref["up"], w32_ref["down"])
        rel = float(np.abs(out.astype(np.float64) - ref).max() / max(float(np.abs(ref).max()), 1e-9))
        corr = float(np.corrcoef(out.astype(np.float64), ref)[0, 1])
        check(f"forward OpenVINO == NumPy ({label})", rel < TOL,
              f"rel={rel:.3e}, corr={corr:.6f}")

    # E. MoEEngine orchestration
    print("\n--- MoEEngine.process_token (layer 0, seed 42) ---")
    router = ExpertRouter(HIDDEN, 256, top_k=8, seed=42)
    engine = MoEEngine(store, backend, router, HIDDEN, INTER)
    out_engine = engine.process_token(0, x.tolist())
    check("process_token ritorna lista di lunghezza corretta",
          isinstance(out_engine, list) and len(out_engine) == HIDDEN,
          f"type={type(out_engine).__name__}, len={len(out_engine) if isinstance(out_engine, list) else 'N/A'}")

    # F. MoEEngine vs somma manuale di expert
    print("\n--- MoEEngine vs manuale (layer 0) ---")
    eids, weights = router.route(x)
    manual = np.zeros(HIDDEN, dtype=np.float32)
    for eid, w in zip(eids, weights):
        ek = ExpertKey(0, eid)
        manual += w * np.asarray(backend.forward(ek, x, HIDDEN, INTER), dtype=np.float32)
    out_engine2 = np.array(engine.process_token(0, x.tolist()), dtype=np.float32)
    diff = float(np.abs(manual - out_engine2).max())
    check("process_token == somma manuale", diff < 1e-5, f"max_diff={diff:.3e}")

    store.close()

    print("\n" + "=" * 78)
    print(f" PASSATI: {len(PASSED)}   FALLITI: {len(FAILED)}")
    for name, detail in FAILED:
        print(f"   - {name}: {detail}")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
