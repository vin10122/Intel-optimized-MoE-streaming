"""Test AsyncExpertStore prefetch + MoEEngine next_expert_key.

La cache asincrona e' un componente chiave del design: prefetch carica gli
expert in background mentre il backend elabora il layer corrente, cosi'
che siano disponibili al prossimo accesso.

Testa:
  A. prefetch -> read_tensor produce dati identici a lettura sincrona
  B. prefetch -> cache hit (stats.hits), non miss
  C. double-prefetch e' deduplicato (pending_loads ha 1 entry)
  D. MoEEngine.process_token(next_expert_key=...) triggera prefetch
  E. backend.forward su expert prefetched == NumPy reference
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
TOL = 5e-3

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name if ok else (name, detail))
    tag = "  PASS  " if ok else "  FAIL  "
    print(f"{tag}{name}" + (f"  ({detail})" if detail else ""), flush=True)


def swiglu_numpy(x: np.ndarray, gate: np.ndarray, up: np.ndarray,
                 down: np.ndarray) -> np.ndarray:
    g = x @ gate
    u = x @ up
    h = (g / (1.0 + np.exp(-g))) * u
    return (h @ down).flatten()


def read_expert_sync(gguf_path: Path, tmap: dict, names: dict,
                     expert_id: int, n_experts: int) -> dict[str, bytes]:
    """Legge tutti i tensor dell'experte in modo sincrono (bypassa prefetch)."""
    out: dict[str, bytes] = {}
    for r in ("gate", "up", "down"):
        t = tmap[names[r]]
        per = t.n_bytes // n_experts
        with gguf_path.open("rb") as fh:
            fh.seek(t.data_offset + expert_id * per)
            out[r] = fh.read(per)
    return out


def make_store_backend(gguf_path: Path, layer_types_map: dict,
                       manifest, capacity: int = 64):
    store = AsyncExpertStore(gguf_path.parent, capacity=capacity, manifest=manifest)
    backend = OpenVINOExpertBackend(
        store, device="GPU", weight_dtype="f16",
        tensor_types=layer_types_map,
        dequant_fn=dequant_to_f16,
    )
    return store, backend


def main() -> int:
    gguf_path = model_path()
    print("=" * 78)
    print(" TEST ASYNC PREFETCH: store.prefetch -> _consume -> cache hit")
    print("=" * 78)
    print(f"modello : {gguf_path.name}")

    reader = gguf.GGUFReader(gguf_path)
    tmap = {t.name: t for t in reader.tensors}
    arch = str(reader.fields["general.architecture"].parts[-1].tobytes().decode())
    n_experts = int(reader.fields[f"{arch}.expert_count"].parts[-1][0])

    layer_types_map = expert_tensor_types(gguf_path)
    manifest = build_manifest_from_gguf(gguf_path)
    rng = np.random.default_rng(42)
    x = rng.standard_normal(HIDDEN).astype(np.float32)

    names = {r: f"blk.0.ffn_{r}_exps.weight" for r in ("gate", "up", "down")}
    layer_types = layer_types_map[0]
    shapes = {"gate": (HIDDEN, INTER), "up": (HIDDEN, INTER), "down": (INTER, HIDDEN)}

    # ---------------------------------------------------------------- A
    print("\n A. prefetch produce dati identici a lettura sincrona")
    store_a, _ = make_store_backend(gguf_path, layer_types_map, manifest)
    sync_bytes = read_expert_sync(gguf_path, tmap, names, 50, n_experts)
    store_a.prefetch(ExpertKey(0, 50))
    prefetched_bytes = {}
    for r in ("gate", "up", "down"):
        prefetched_bytes[r] = store_a.read_tensor(ExpertKey(0, 50), r)
    all_match = all(p == s for p, s in zip(prefetched_bytes.values(), sync_bytes.values()))
    check("dati prefetched == dati sincroni", all_match)
    store_a.close()

    # ---------------------------------------------------------------- B
    print("\n B. prefetch -> cache hit, non miss")
    store_b, _ = make_store_backend(gguf_path, layer_types_map, manifest)
    store_b.prefetch(ExpertKey(0, 30))
    store_b.read_tensor(ExpertKey(0, 30), "gate")
    check("hit count > 0 dopo prefetch", store_b.stats.hits > 0,
          f"hits={store_b.stats.hits}")
    check("miss count == 0 dopo prefetch", store_b.stats.misses == 0,
          f"misses={store_b.stats.misses}")
    # Access an unfetched expert — should be a miss
    store_b.read_tensor(ExpertKey(0, 31), "gate")
    check("miss count > 0 per expert non-prefetched", store_b.stats.misses > 0,
          f"misses={store_b.stats.misses}")
    store_b.close()

    # ---------------------------------------------------------------- C
    print("\n C. double-prefetch deduplicato")
    store_c, _ = make_store_backend(gguf_path, layer_types_map, manifest)
    key = ExpertKey(0, 100)
    store_c.prefetch(key)
    store_c.prefetch(key)
    check("pending_loads ha 1 entry dopo double-prefetch",
          len(store_c.pending_loads) == 1,
          f"pending={len(store_c.pending_loads)}")
    store_c.close()

    # ---------------------------------------------------------------- D
    print("\n D. MoEEngine.process_token(next_expert_key=...) triggera prefetch")
    store_d, backend_d = make_store_backend(gguf_path, layer_types_map, manifest)
    router_d = synthetic_router(HIDDEN, 256, top_k=8, seed=42, layer=0)
    engine_d = MoEEngine(store_d, backend_d, {0: router_d}, HIDDEN, INTER)
    next_key = ExpertKey(0, 70)
    engine_d.process_token(0, x, next_expert_key=next_key)
    check("pending_loads non vuoto dopo process_token con hint",
          len(store_d.pending_loads) > 0,
          f"pending={len(store_d.pending_loads)}")
    # Consume the prefetched expert
    out_d = backend_d.forward(next_key, x, HIDDEN, INTER)
    check("prefetch consumato: pending_loads svuotato",
          len(store_d.pending_loads) == 0,
          f"pending={len(store_d.pending_loads)}")
    check("accesso prefetched produce output valido",
          out_d.shape == (HIDDEN,) and not np.isnan(out_d).any())
    store_d.close()

    # ---------------------------------------------------------------- E
    print("\n E. backend.forward su expert prefetched == NumPy reference")
    store_e, backend_e = make_store_backend(gguf_path, layer_types_map, manifest)
    key_e = ExpertKey(0, 70)
    store_e.prefetch(key_e)
    out_e = backend_e.forward(key_e, x, HIDDEN, INTER)

    # NumPy reference
    sync_bytes = read_expert_sync(gguf_path, tmap, names, 70, n_experts)
    w32 = {}
    for r in ("gate", "up", "down"):
        t = tmap[names[r]]
        shape = tuple(int(v) for v in t.shape)[:2]
        w32[r] = dequantize_to(sync_bytes[r], int(t.tensor_type), "f32").reshape(shape)
    ref = swiglu_numpy(x, w32["gate"], w32["up"], w32["down"])
    rel = float(np.abs(out_e.astype(np.float64) - ref).max() / max(float(np.abs(ref).max()), 1e-9))
    corr = float(np.corrcoef(out_e.astype(np.float64), ref)[0, 1])
    check("forward prefetched == NumPy", rel < TOL, f"rel={rel:.3e}, corr={corr:.6f}")
    store_e.close()

    # ---------------------------------------------------------------- stats
    print("\n--- stats combinate ---")
    store_f, _ = make_store_backend(gguf_path, layer_types_map, manifest)
    # prefetch + access pattern: 2 prefetched, 1 sync
    store_f.prefetch(ExpertKey(0, 10))
    store_f.prefetch(ExpertKey(0, 20))
    store_f.read_tensor(ExpertKey(0, 10), "gate")  # prefetch hit
    store_f.read_tensor(ExpertKey(0, 20), "gate")  # prefetch hit
    store_f.read_tensor(ExpertKey(0, 30), "gate")  # sync miss
    check("1 sync miss", store_f.stats.misses == 1, f"misses={store_f.stats.misses}")
    check("2 prefetch hits", store_f.stats.hits == 2, f"hits={store_f.stats.hits}")
    check("3 read_tensor total", store_f.stats.tensor_reads == 3,
          f"reads={store_f.stats.tensor_reads}")
    store_f.close()

    print("\n" + "=" * 78)
    print(f" PASSATI: {len(PASSED)}   FALLITI: {len(FAILED)}")
    for name, detail in FAILED:
        print(f"   - {name}: {detail}")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
