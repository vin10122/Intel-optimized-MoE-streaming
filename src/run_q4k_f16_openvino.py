import time
import ctypes
import sys
import numpy as np
from pathlib import Path
import gguf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flash_moe import (
    ExpertKey,
    ExpertManifest,
    ExpertRecord,
    ExpertTensorSlice,
    AsyncExpertStore,
    routers_from_gguf,
    OpenVINOExpertBackend,
    MoEEngine,
)
from flash_moe.dequant import dequantize_to, name_of
from flash_moe.paths import dll_path, model_path

# Tipi ggml dei tensor gate/up/down, riempii in main() leggendo il GGUF.
EXPERT_TYPES: dict[int, dict[str, int]] = {}

# Modello: nome in un solo posto (flash_moe.paths), override con FLASH_MOE_GGUF
GGUF_FILE_PATH = model_path()
DLL_PATH = dll_path()

# Dequantizzazione: kernel C++ validato (tools/test_dequant_k.py, gate 162/162),
# con fallback sulla reference gguf.quants.
#
# Il risultato e' float16 nativo. Passare uint16 su porte a 16 bit produce NaN,
# e la vecchia via .astype(np.uint16) sui pesi in 0..0.0855 produceva una
# matrice di soli zero: non un'approssimazione, ma pesi nulli.
_CPP_SYMBOLS = ("dequant_k", "dequant_k_supported", "dequant_k_block_size")
_cpp_dequant = None

if DLL_PATH.exists():
    try:
        _dll = ctypes.CDLL(str(DLL_PATH))
        # Si verifica che gli export esistano DAVVERO prima di dichiarare
        # l'accelerazione attiva: il vecchio codice lasciava la libreria
        # impostata anche se il lookup falliva, e poi crashava a runtime
        # chiamando un simbolo inesistente.
        for _sym in _CPP_SYMBOLS:
            getattr(_dll, _sym)
        _dll.dequant_k.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int,
        ]
        _dll.dequant_k.restype = ctypes.c_int
        _cpp_dequant = _dll
        print(f"[C++] Dequantizzazione accelerata: {DLL_PATH.name}", flush=True)
    except Exception as err:
        print(f"[C++] DLL non utilizzabile ({err}), uso la reference gguf", flush=True)
else:
    print(f"[C++] {DLL_PATH.name} non presente, uso la reference gguf", flush=True)


def dequant_to_f16(raw: bytes, ggml_type: int, n_elements: int,
                   out: np.ndarray | None = None) -> np.ndarray:
    """Dequantizza in float16, C++ se disponibile altrimenti gguf.quants.

    ``out`` permette di dequantizzare in un buffer già allocato: chiedere
    2 MiB nuovi al sistema operativo costa 0,56 ms su 0,96 ms per tensor, e
    il kernel in sé fa 0,35 ms. Se il buffer non è della misura giusta o non
    è scrivibile, si ignora e se ne alloca uno nuovo, così un caller che
    sbaglia l'argomento non ottiene corruzione silenziosa.
    """
    if out is not None and (out.dtype != np.float16 or out.size != n_elements
                            or not out.flags.writeable):
        out = None
    if _cpp_dequant is not None:
        if out is None:
            out = np.empty(n_elements, dtype=np.float16)
        src = np.frombuffer(raw, dtype=np.uint8)
        rc = _cpp_dequant.dequant_k(
            ggml_type, src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            out.ctypes.data, n_elements, 1,
        )
        if rc == 0:
            return out
        print(f"[C++] dequant_k ha restituito rc={rc}, uso la reference", flush=True)
    return dequantize_to(raw, ggml_type, "f16")


def build_manifest_from_gguf(gguf_path: Path) -> ExpertManifest:
    """Crea manifest per accesso diretto a slices GGUF Q4_K."""
    print(f"[GGUF Manifest] Costruzione manifest per {gguf_path.name}...", flush=True)
    reader = gguf.GGUFReader(gguf_path)
    tensor_map = {t.name: t for t in reader.tensors}

    block_count = 40
    num_experts = 256

    records: dict[ExpertKey, ExpertRecord] = {}

    for layer in range(block_count):
        gate_name = f"blk.{layer}.ffn_gate_exps.weight"
        up_name = f"blk.{layer}.ffn_up_exps.weight"
        down_name = f"blk.{layer}.ffn_down_exps.weight"

        if gate_name not in tensor_map:
            continue

        t_gate = tensor_map[gate_name]
        t_up = tensor_map[up_name]
        t_down = tensor_map[down_name]

        gate_expert_bytes = t_gate.n_bytes // num_experts
        up_expert_bytes = t_up.n_bytes // num_experts
        down_expert_bytes = t_down.n_bytes // num_experts

        for eid in range(num_experts):
            key = ExpertKey(layer, eid)

            tensors = {
                "gate": ExpertTensorSlice("gate", offset=t_gate.data_offset + eid * gate_expert_bytes, nbytes=gate_expert_bytes),
                "up": ExpertTensorSlice("up", offset=t_up.data_offset + eid * up_expert_bytes, nbytes=up_expert_bytes),
                "down": ExpertTensorSlice("down", offset=t_down.data_offset + eid * down_expert_bytes, nbytes=down_expert_bytes),
            }

            total_bytes = gate_expert_bytes + up_expert_bytes + down_expert_bytes
            records[key] = ExpertRecord(
                key=key,
                relative_path=gguf_path.name,
                nbytes=total_bytes,
                tensors=tensors
            )

    return ExpertManifest(gguf_path.parent, records)


def expert_tensor_types(gguf_path: Path) -> dict[int, dict[str, int]]:
    """Legge dal GGUF il tipo ggml di gate/up/down per OGNI layer.

    Non si fidare di una costante globale: in questo modello i tipi variano
    per layer (Q4_K layer 0, Q3_K layer 10-30, Q6_K layer 39 down). Usare il
    tipo sbagliato non dà errore: il kernel C++ legge bytes_per_block sbagliati
    e va fuori dal buffer. I tipi vanno letti per-layer dal GGUF.
    """
    reader = gguf.GGUFReader(gguf_path)
    tmap = {t.name: t for t in reader.tensors}
    out: dict[int, dict[str, int]] = {}
    for layer in range(40):
        layer_types: dict[str, int] = {}
        for role in ("gate", "up", "down"):
            name = f"blk.{layer}.ffn_{role}_exps.weight"
            if name in tmap:
                layer_types[role] = int(tmap[name].tensor_type)
        if layer_types:
            out[layer] = layer_types
    return out


def main():
    print("=" * 80, flush=True)
    print(" FLASH-MOE: Xe2 LPG - K-quant -> float16 su OpenVINO GPU", flush=True)
    print("=" * 80, flush=True)

    if not GGUF_FILE_PATH.exists():
        print(f"ERRORE: File GGUF non trovato: {GGUF_FILE_PATH}", flush=True)
        print(f"Imposta FLASH_MOE_GGUF per puntare altrove.", flush=True)
        return

    # Tipi dei tensor expert, letti dal GGUF: gate/up/down non sono tutti
    # lo stesso tipo quantizzato, e la differenza non produce errori.
    global EXPERT_TYPES
    EXPERT_TYPES = expert_tensor_types(GGUF_FILE_PATH)
    type_summary = {name_of(v) for role_dict in EXPERT_TYPES.values() for v in role_dict.values()}
    print(f"[GGUF] tipi per layer: {len(EXPERT_TYPES)} layer, tipi usati: {sorted(type_summary)}",
          flush=True)
    print(f"[GGUF] layer 0: { {k: name_of(v) for k, v in EXPERT_TYPES[0].items()} }", flush=True)

    # 1. Manifest
    manifest = build_manifest_from_gguf(GGUF_FILE_PATH)

    # 2. Store: gli expert in GGUF sono ~2,3 MB ciascuno compressi
    CACHE_CAPACITY = 700
    store = AsyncExpertStore(GGUF_FILE_PATH.parent, capacity=CACHE_CAPACITY, manifest=manifest)

    # 3. Backend K-quant -> float16 (usa il pacchetto flash_moe)
    backend = OpenVINOExpertBackend(
        store, device="GPU", weight_dtype="f16",
        cache_capacity=CACHE_CAPACITY,
        tensor_types=EXPERT_TYPES,
        dequant_fn=dequant_to_f16,
    )

    # Configurazione
    hidden_dim = 2048
    inter_dim = 512
    num_experts = 256
    num_layers = 40

    # Router: uno per layer, dai gate veri ffn_gate_inp.weight. Con un gate
    # finto il benchmark misverrebbe la cache su un pattern di accessi che il
    # modello non produce mai.
    expert_routers = routers_from_gguf(
        {t.name: t for t in gguf.GGUFReader(GGUF_FILE_PATH).tensors},
        num_layers, hidden_dim, num_experts, top_k=8,
    )
    engine = MoEEngine(store, backend, expert_routers, hidden_dim, inter_dim)

    # Benchmark
    num_tokens = 1
    print(f"\n[F16] Inferenza su {num_tokens} token x {num_layers} layer...", flush=True)
    print(f"[Cache] {CACHE_CAPACITY} esperti in RAM", flush=True)
    print(f"[Accelerazione] C++: {'SI' if _cpp_dequant is not None else 'NO (reference gguf)'}",
          flush=True)
    start_time = time.perf_counter()

    tokens = [np.random.randn(hidden_dim).astype(np.float32) for _ in range(num_tokens)]

    for token_idx, activation in enumerate(tokens):
        for layer in range(num_layers):
            engine.process_token(layer, activation)
        print(f" -> Token {token_idx + 1}/{num_tokens} processato!", flush=True)

    total_time = time.perf_counter() - start_time
    tokens_per_sec = num_tokens / total_time
    latency_ms = (total_time / num_tokens) * 1000

    print("\n" + "=" * 80, flush=True)
    print("     BENCHMARK: K-quant -> float16 + CACHE EXPERT", flush=True)
    print("=" * 80, flush=True)
    print(f" File Modello:                    {GGUF_FILE_PATH.name}", flush=True)
    print(f" Quantizzazione:                 Q4_K", flush=True)
    print(f" Output precision:               float16", flush=True)
    print(f" Esperti in cache (max):          {CACHE_CAPACITY}", flush=True)
    print(f" Memoria stimata (compressi):   ~0.7GB (Q4_K in RAM)", flush=True)
    print(f" Memoria pesi dequant (f16):      da misurare", flush=True)
    print(f" Esperti attivi per token:        8 (top-k)", flush=True)
    print(f" Layer elaborati:                 {num_layers}", flush=True)
    print(f" Target GPU:                      Intel Lunar Lake (OpenVINO)", flush=True)
    print(f" Accelerazione C++:               {'C++ dequant_k' if _cpp_dequant is not None else 'reference gguf'}", flush=True)
    print(f" Tempo Totale:                    {total_time:.4f} s", flush=True)
    print(f" Tokens elaborati:                {num_tokens}", flush=True)
    print(f" Velocita:                        {tokens_per_sec:.2f} token/sec", flush=True)
    print(f" Latenza per token:               {latency_ms:.2f} ms", flush=True)
    print(f" Cache Stats - Hits:              {store.stats.hits}", flush=True)
    print(f" Cache Stats - Misses:            {store.stats.misses}", flush=True)
    print(f" Dati letti da SSD:               {store.stats.bytes_read / (1024**2):.2f} MB", flush=True)
    print(f" Esperti in cache:                {backend.cache_size()}", flush=True)
    print("=" * 80, flush=True)

    store.close()


if __name__ == "__main__":
    main()
