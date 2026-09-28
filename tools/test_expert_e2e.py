"""Test end-to-end di un expert: OpenVINO GPU contro NumPy f32.

Fino ad ora il gate (tools/test_dequant_k.py) ha validato i kernel e la
precisione, ma non e' mai uscito un numero corretto da OpenVINO. Qui si
calcola lo stesso SwiGLU in due modi indipendenti e si confrontano:

  - OpenVINO su GPU, pesi float16 nativi
  - NumPy in float32, pesi dequantizzati con la reference gguf.quants

Se i due combaciano entro l'errore rappresentabile in f16 il backend e' a
posto. Se divergono, l'entita' della differenza dice subito se e' un
arrotondamento o se il layout dei pesi e' sbagliato.

Lo store NON e' coinvolto di proposito: AsyncExpertStore e
build_manifest_from_gguf non sono mai stati esercitati, e mischiarli
confonderebbe un difetto del backend con un difetto dello store.

Uso:
    python tools/test_expert_e2e.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gguf  # noqa: E402
import openvino as ov  # noqa: E402
import openvino.opset13 as ops  # noqa: E402

from flash_moe.dequant import dequantize_to, name_of  # noqa: E402
from flash_moe.paths import dll_path, model_path  # noqa: E402

HIDDEN = 2048
INTER = 512
LAYER = 0
EXPERT = 0

# L'errore relativo accettabile non e' azzerabile: f16 ha 11 bit di mantissa,
# quindi ~5e-4 per operazione e un paio di 1e-3 accumulati su tre matmul.
TOL = 5e-3

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name if ok else (name, detail))
    tag = "  PASS  " if ok else "  FAIL  "
    print(f"{tag}{name}" + (f"  ({detail})" if detail else ""), flush=True)


def read_expert_bytes(gguf_path: Path, tensor: "gguf.GGUFReaderTensor",
                      n_experts: int) -> bytes:
    """Legge la slice dell'expert REQUESTED, dal file, senza passare dallo store."""
    per = tensor.n_bytes // n_experts
    with gguf_path.open("rb") as fh:
        fh.seek(tensor.data_offset + EXPERT * per)
        return fh.read(per)


def dequant_f16(raw: bytes, type_id: int, n_el: int) -> np.ndarray:
    """float16 nativo: l'unica forma che OpenVINO accetta su porte a 16 bit."""
    return dequantize_to(raw, type_id, "f16").reshape(-1)


def dequant_f32_ref(raw: bytes, type_id: int) -> np.ndarray:
    """Reference indipendente: gguf.quants in float32."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    return gguf.quants.dequantize(arr, type_id).astype(np.float32)


def swiglu_numpy(x: np.ndarray, gate: np.ndarray, up: np.ndarray,
                 down: np.ndarray) -> np.ndarray:
    """SwiGLU in f32.

    Le shape vengono dal GGUF e gia' sono nell'ordine giusto, quindi qui non si
    trasponde niente: gate e up sono (HIDDEN, INTER) e down e' (INTER, HIDDEN).
    """
    g = x @ gate
    u = x @ up
    h = (g / (1.0 + np.exp(-g))) * u
    return h @ down


def build_ov_model(core: ov.Core, device: str):
    """Stesso grafo di src/run_q4k_f16_openvino.py, su f16 nativo."""
    x = ops.parameter([1, HIDDEN], ov.Type.f32, name="activation")
    gate = ops.parameter([HIDDEN, INTER], ov.Type.f16, name="gate")
    up = ops.parameter([HIDDEN, INTER], ov.Type.f16, name="up")
    down = ops.parameter([INTER, HIDDEN], ov.Type.f16, name="down")

    xf = ops.convert(x, destination_type=ov.Type.f16)
    gp = ops.matmul(xf, gate, transpose_a=False, transpose_b=False)
    upp = ops.matmul(xf, up, transpose_a=False, transpose_b=False)
    h = ops.multiply(gp, ops.sigmoid(gp))
    h = ops.multiply(h, upp)
    out = ops.matmul(h, down, transpose_a=False, transpose_b=False)
    out = ops.convert(out, destination_type=ov.Type.f32)

    model = ov.Model([out], [x, gate, up, down], name="SwiGLU_Expert_F16")
    return core.compile_model(model, device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--repeat", type=int, default=20)
    args = ap.parse_args()

    gguf_path = model_path()
    print("=" * 78)
    print(" TEST END-TO-END DI UN EXPERT: OPENVINO vs NUMPY f32")
    print("=" * 78)
    print(f"modello : {gguf_path.name}")
    print(f"device  : {args.device}")
    print(f"DLL     : {dll_path().name} (presente: {dll_path().exists()})")
    print(f"layer   : {LAYER}   expert: {EXPERT}   hidden {HIDDEN}  inter {INTER}")
    print()

    reader = gguf.GGUFReader(gguf_path)
    tmap = {t.name: t for t in reader.tensors}
    arch = str(reader.fields["general.architecture"].parts[-1].tobytes().decode())
    n_experts = int(reader.fields[f"{arch}.expert_count"].parts[-1][0])

    names = {
        "gate": f"blk.{LAYER}.ffn_gate_exps.weight",
        "up": f"blk.{LAYER}.ffn_up_exps.weight",
        "down": f"blk.{LAYER}.ffn_down_exps.weight",
    }
    types = {r: int(tmap[n].tensor_type) for r, n in names.items()}
    print(f"tipi    : " + ", ".join(f"{r}={name_of(t)}" for r, t in types.items()))

    raw = {r: read_expert_bytes(gguf_path, tmap[n], n_experts)
           for r, n in names.items()}

    w16 = {r: dequant_f16(raw[r], types[r], HIDDEN * INTER) for r in raw}

    # Le shape per expert sono quelle del GGUF senza l'asse expert: e' la
    # verifica che intercetta un errore di layout, non una formalita'.
    for role, expect in (("gate", (HIDDEN, INTER)), ("up", (HIDDEN, INTER)),
                         ("down", (INTER, HIDDEN))):
        t = tmap[names[role]]
        shape = tuple(int(v) for v in t.shape)
        check(f"shape {role} attesa {expect}", shape[:2] == expect,
              f"GGUF {shape}, per expert {shape[:2]}")

    w32 = {}
    for r in raw:
        shape = tuple(int(v) for v in tmap[names[r]].shape)[:2]
        w32[r] = dequant_f32_ref(raw[r], types[r]).reshape(shape)

    for r in ("gate", "up", "down"):
        nz = int((w16[r] == 0).sum())
        nan = int(np.isnan(w16[r].astype(np.float32)).sum())
        amax = float(np.abs(w32[r]).max())
        # Gli zeri NON sono un difetto: in Q8_0 qs[j] == 0 e' un valore legittimo
        # e down in questo modello ne ha molti. Si controlla solo che non ci
        # siano NaN e che i pesi non siano degenere.
        check(f"pesi {r}: float16 senza NaN, non degeneri",
              nan == 0 and amax > 1e-4,
              f"nan {nan}, zeri {nz}/{w16[r].size} ({100.0 * nz / w16[r].size:.1f}%), "
              f"absmax {amax:.4f}")

    # attivazione fissa: il confronto deve essere riproducibile
    rng = np.random.default_rng(1234)
    x = rng.standard_normal(HIDDEN).astype(np.float32)

    core = ov.Core()
    available = core.available_devices
    print(f"\ndispositivi disponibili: {available}")
    if args.device not in available:
        print(f"  ERRORE: '{args.device}' non disponibile")
        return 2

    ref = swiglu_numpy(x, w32["gate"], w32["up"], w32["down"])
    scale = float(np.abs(ref).max())
    print(f"\n--- confronto ---")
    print(f"  NumPy f32: max |y| = {scale:.5f}")
    if scale == 0:
        print("  reference degenere")
        return 1

    try:
        compiled = build_ov_model(core, args.device)
    except Exception as err:
        check("compilazione OpenVINO", False, str(err)[:140])
        return 1
    check("compilazione OpenVINO", True, args.device)

    feeds = {
        "activation": x.reshape(1, HIDDEN),
        "gate": w16["gate"].reshape(HIDDEN, INTER),
        "up": w16["up"].reshape(HIDDEN, INTER),
        "down": w16["down"].reshape(INTER, HIDDEN),
    }
    try:
        req = compiled.create_infer_request()
        out = list(req.infer(feeds).values())[0]
        got = out.flatten().astype(np.float64)
    except Exception as err:
        check("inferenza OpenVINO", False, str(err)[:140])
        return 1
    check("inferenza OpenVINO", True, f"output {got.size} elementi")

    gotf = out.astype(np.float32).flatten().astype(np.float64)
    rel = float(np.abs(gotf - ref).max() / scale)
    corr = float(np.corrcoef(gotf, ref)[0, 1])
    rms = float(np.sqrt(np.mean((gotf - ref) ** 2)) / scale)

    print(f"\n  errore relativo max : {rel:.3e}")
    print(f"  errore RMS         : {rms:.3e}")
    print(f"  correlazione       : {corr:.8f}")
    print(f"  soglia             : {TOL:.1e}")
    print(f"\n  prime componenti   NumPy : "
          f"{np.array2string(ref[:4].astype(np.float64), precision=5)}")
    print(f"  prime componenti   OpenVINO: "
          f"{np.array2string(gotf[:4], precision=5)}")

    check("OpenVINO riproduce NumPy entro la precisione f16", rel < TOL,
          f"rel={rel:.3e}, rms={rms:.3e}, corr={corr:.6f}")

    # tempi puramente indicativi: la macchina non e' dedicata
    t0 = time.perf_counter()
    for _ in range(args.repeat):
        req.infer(feeds)
    dt = (time.perf_counter() - t0) / args.repeat * 1000.0
    print(f"\n  latenza OpenVINO: {dt:.3f} ms/expert "
          f"(misura indicativa, macchina non libera)")

    print("\n" + "=" * 78)
    if rel < TOL:
        print(f" ESITO: il backend OpenVINO riproduce NumPy entro f16. Layout OK.")
    else:
        print(f" ESITO: DIVERGE oltre la precisione f16. Non e' arrotondamento.")
    print("=" * 78)

    print(f"\n  PASSATI: {len(PASSED)}   FALLITI: {len(FAILED)}")
    for name, detail in FAILED:
        print(f"   - {name}: {detail}")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
