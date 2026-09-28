import time
import ctypes
import sys
import numpy as np
from pathlib import Path
import gguf
from gguf import GGMLQuantizationType
import openvino as ov
import openvino.opset13 as ops

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flash_moe import (
    ExpertKey,
    ExpertManifest,
    ExpertRecord,
    ExpertTensorSlice,
    AsyncExpertStore,
    ExpertRouter,
)
from flash_moe.dequant import dequantize_to, name_of
from flash_moe.paths import dll_path, model_path

# Tipi ggml dei tensor gate/up/down, riempii in main() leggendo il GGUF.
EXPERT_TYPES: dict[str, int] = {}

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


def dequant_to_f16(raw: bytes, ggml_type: int, n_elements: int) -> np.ndarray:
    """Dequantizza in float16, C++ se disponibile altrimenti gguf.quants."""
    if _cpp_dequant is not None:
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


def expert_tensor_types(gguf_path: Path) -> dict[str, int]:
    """Legge dal GGUF il tipo ggml di gate/up/down di un expert.

    Non si fidare di una costante scritta a mano: il dequantizer e' scelto in
    base a questo, e sbagliarlo non dà errore, dà pesi sbagliati.
    """
    reader = gguf.GGUFReader(gguf_path)
    tmap = {t.name: t for t in reader.tensors}
    out: dict[str, int] = {}
    for role in ("gate", "up", "down"):
        name = f"blk.0.ffn_{role}_exps.weight"
        if name not in tmap:
            raise KeyError(f"tensor {name} assente dal GGUF: modello diverso?")
        out[role] = int(tmap[name].tensor_type)
    return out


class Q4K_F16_Backend:
    """Backend OpenVINO GPU: pesi K-quant dequantizzati in float16."""

    def __init__(self, store: AsyncExpertStore, device: str = "GPU", cache_capacity: int = 700):
        self.store = store
        self.core = ov.Core()
        
        available = self.core.available_devices
        if device == "GPU" and "GPU" in available:
            self.target_device = "GPU"
        elif "GPU" in available:
            self.target_device = "GPU"
        else:
            self.target_device = "CPU"
        
        self.cache_capacity = cache_capacity
        self._compiled_models: dict[tuple[int, int], ov.CompiledModel] = {}
        self._f16_cache: dict[ExpertKey, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.cache_order: list[ExpertKey] = []

        print(f"[Q4K_F16_Backend] Target Device: {self.target_device}", flush=True)

    def _get_compiled_model(self, hidden_size: int, intermediate_size: int) -> ov.CompiledModel:
        shape_key = (hidden_size, intermediate_size)
        if shape_key in self._compiled_models:
            return self._compiled_models[shape_key]

        # Attivazione in f32, pesi in f16. bf16 su questa piattaforma e' stato
        # verificato non funzionante (tools/check_ov_precisions.py): il metodo
        # che regge e' float16 nativo su porte a 16 bit.
        x = ops.parameter([1, hidden_size], ov.Type.f32, name="activation")
        gate = ops.parameter([hidden_size, intermediate_size], ov.Type.f16, name="gate")
        up = ops.parameter([hidden_size, intermediate_size], ov.Type.f16, name="up")
        down = ops.parameter([intermediate_size, hidden_size], ov.Type.f16, name="down")

        x_f16 = ops.convert(x, destination_type=ov.Type.f16)

        gate_proj = ops.matmul(x_f16, gate, transpose_a=False, transpose_b=False)
        up_proj = ops.matmul(x_f16, up, transpose_a=False, transpose_b=False)

        # SwiGLU
        swish_gate = ops.multiply(gate_proj, ops.sigmoid(gate_proj))
        swiglu_out = ops.multiply(swish_gate, up_proj)

        out = ops.matmul(swiglu_out, down, transpose_a=False, transpose_b=False)

        out_f32 = ops.convert(out, destination_type=ov.Type.f32)

        model = ov.Model([out_f32], [x, gate, up, down], name="SwiGLU_Expert_Q4K_F16")
        compiled = self.core.compile_model(model, self.target_device)
        self._compiled_models[shape_key] = compiled
        return compiled

    def _dequant_to_f16(self, raw_bytes: bytes, ggml_type: int,
                        n_elements: int) -> np.ndarray:
        return dequant_to_f16(raw_bytes, ggml_type, n_elements)

    def _get_f16_weights(self, key: ExpertKey, intermediate_size: int,
                         hidden_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ottieni i pesi dell'expert in float16. Cache LRU."""
        if key in self._f16_cache:
            self.cache_order.remove(key)
            self.cache_order.append(key)
            return self._f16_cache[key]

        print(f"[F16 Cache] MISS: carico esperto {key}", flush=True)

        gate_bytes = self.store.read_tensor(key, "gate")
        up_bytes = self.store.read_tensor(key, "up")
        down_bytes = self.store.read_tensor(key, "down")

        # gate e up sono Q4_K ma down e' Q8_0 nel GGUF: dequantizzarli con il
        # tipo sbagliato produce pesi senza senso, non un errore evidente.
        gate_f16 = dequant_to_f16(gate_bytes, EXPERT_TYPES["gate"],
                                  hidden_size * intermediate_size)
        up_f16 = dequant_to_f16(up_bytes, EXPERT_TYPES["up"],
                                hidden_size * intermediate_size)
        down_f16 = dequant_to_f16(down_bytes, EXPERT_TYPES["down"],
                                  intermediate_size * hidden_size)

        weights = (gate_f16, up_f16, down_f16)
        self._f16_cache[key] = weights
        self.cache_order.append(key)

        if len(self._f16_cache) > self.cache_capacity:
            oldest = self.cache_order.pop(0)
            self._f16_cache.pop(oldest, None)
            print(f"[F16 Cache] EVICT: rimosso {oldest}", flush=True)

        return weights

    def forward(
        self,
        key: ExpertKey,
        activation: list[float] | np.ndarray,
        hidden_size: int,
        intermediate_size: int,
    ) -> list[float]:
        gate_f16, up_f16, down_f16 = self._get_f16_weights(key, intermediate_size, hidden_size)

        if isinstance(activation, list):
            x_arr = np.array(activation, dtype=np.float32).reshape(1, hidden_size)
        else:
            x_arr = activation.reshape(1, hidden_size).astype(np.float32)

        compiled = self._get_compiled_model(hidden_size, intermediate_size)
        infer_request = compiled.create_infer_request()

        # float16 nativo: e' il metodo verificato in tools/check_ov_tensor_feeds.py
        results = infer_request.infer({
            "activation": x_arr,
            "gate": gate_f16.reshape(hidden_size, intermediate_size),
            "up": up_f16.reshape(hidden_size, intermediate_size),
            "down": down_f16.reshape(intermediate_size, hidden_size),
        })

        output_tensor = list(results.values())[0]
        return output_tensor.flatten().tolist()


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
    print(f"[GGUF] tipi expert: "
          f"{ {k: name_of(v) for k, v in EXPERT_TYPES.items()} }", flush=True)

    # 1. Manifest
    manifest = build_manifest_from_gguf(GGUF_FILE_PATH)

    # 2. Store: gli expert in GGUF sono ~2,3 MB ciascuno compressi
    CACHE_CAPACITY = 700
    store = AsyncExpertStore(GGUF_FILE_PATH.parent, capacity=CACHE_CAPACITY, manifest=manifest)

    # 3. Backend K-quant -> float16
    backend = Q4K_F16_Backend(store, device="GPU", cache_capacity=CACHE_CAPACITY)

    # Configurazione
    hidden_dim = 2048
    inter_dim = 512
    num_experts = 256
    num_layers = 40
    router = ExpertRouter(hidden_size=hidden_dim, num_experts=num_experts, top_k=8)

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
            expert_ids, weights = router.route(activation)
            combined_output = np.zeros(hidden_dim, dtype=np.float32)
            for expert_id, weight in zip(expert_ids, weights):
                key = ExpertKey(layer, expert_id)
                expert_out = backend.forward(key, activation, hidden_dim, inter_dim)
                combined_output += weight * np.array(expert_out, dtype=np.float32)
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
    print(f" Esperti in cache:                {len(backend._f16_cache)}", flush=True)
    print("=" * 80, flush=True)

    store.close()


if __name__ == "__main__":
    main()
