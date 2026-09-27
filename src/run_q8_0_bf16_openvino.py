import time
import ctypes
import numpy as np
from pathlib import Path
import gguf
from gguf import GGMLQuantizationType
import openvino as ov
import openvino.opset13 as ops

from flash_moe import (
    ExpertKey,
    ExpertManifest,
    ExpertRecord,
    ExpertTensorSlice,
    AsyncExpertStore,
    ExpertRouter,
)

# Path del modello Q8_0
GGUF_FILE_PATH = Path(r"C:\Users\valer\Desktop\moe-client-prot\Nuova cartella\Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf")
DLL_PATH = Path(r"flash_moe\fast_dequant_q8_0.dll")

# Caricamento DLL C++ per dequantizzazione accelerata Q8_0 -> BF16
_cpp_dequant_q8_0 = None
_cpp_use_avx2 = False

if DLL_PATH.exists():
    try:
        _cpp_dequant_q8_0 = ctypes.CDLL(str(DLL_PATH.resolve()))
        
        # Prova prima la versione AVX2
        try:
            _cpp_dequant_q8_0.dequantize_q8_0_to_bf16_avx2.argtypes = [
                ctypes.c_char_p,      # src_bytes
                ctypes.POINTER(ctypes.c_uint16),  # dst (uint16_t*)
                ctypes.c_int,         # rows
                ctypes.c_int          # cols
            ]
            _cpp_dequant_q8_0.dequantize_q8_0_to_bf16_avx2.restype = None
            _cpp_use_avx2 = True
            print(f"[C++ Acceleration] Caricata DLL Q8_0->BF16 AVX2: {DLL_PATH.name}", flush=True)
        except:
            # Fallback alla versione semplice
            _cpp_dequant_q8_0.dequantize_q8_0_to_bf16.argtypes = [
                ctypes.c_char_p,      # src_bytes
                ctypes.POINTER(ctypes.c_uint16),  # dst (uint16_t*)
                ctypes.c_int,         # rows
                ctypes.c_int          # cols
            ]
            _cpp_dequant_q8_0.dequantize_q8_0_to_bf16.restype = None
            print(f"[C++ Acceleration] Caricata DLL Q8_0->BF16: {DLL_PATH.name}", flush=True)
            
    except Exception as err:
        print(f"[C++ Acceleration] Impossibile caricare DLL Q8_0: {err}", flush=True)


def build_manifest_from_gguf(gguf_path: Path) -> ExpertManifest:
    """Crea manifest per accesso diretto a slices GGUF Q8_0."""
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

        # Q8_0: 2048 x 512 x 256
        # Ogni esperto: 2048 x 512 elementi
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


class Q8_0_BF16_Backend:
    """Backend ottimizzato per Q8_0 -> BF16 con OpenVINO GPU."""

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
        self._bf16_cache: dict[ExpertKey, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.cache_order: list[ExpertKey] = []
        
        print(f"[Q8_0_BF16_Backend] Target Device: {self.target_device}", flush=True)

    def _get_compiled_model(self, hidden_size: int, intermediate_size: int) -> ov.CompiledModel:
        shape_key = (hidden_size, intermediate_size)
        if shape_key in self._compiled_models:
            return self._compiled_models[shape_key]

        # Parametri: attivazione FP32, pesi BF16
        x = ops.parameter([1, hidden_size], ov.Type.f32, name="activation")
        gate = ops.parameter([hidden_size, intermediate_size], ov.Type.bf16, name="gate")
        up = ops.parameter([hidden_size, intermediate_size], ov.Type.bf16, name="up")
        down = ops.parameter([intermediate_size, hidden_size], ov.Type.bf16, name="down")

        # Converti attivazione a BF16 per matmul
        x_bf16 = ops.convert(x, destination_type=ov.Type.bf16)
        
        # MatMul: x_bf16 @ gate (bf16)
        gate_proj = ops.matmul(x_bf16, gate, transpose_a=False, transpose_b=False)
        up_proj = ops.matmul(x_bf16, up, transpose_a=False, transpose_b=False)
        
        # SwiGLU in BF16 - usiamo ops.sigmoid * gate_proj * beta
        # Swish(x) = x * sigmoid(beta * x), con beta=1 per default
        # Per BF16, facciamo: sigmoid(gate_proj) * gate_proj
        swish_gate = ops.multiply(gate_proj, ops.sigmoid(gate_proj))
        swiglu_out = ops.multiply(swish_gate, up_proj)
        
        # Output matmul: swiglu_out @ down
        out = ops.matmul(swiglu_out, down, transpose_a=False, transpose_b=False)
        
        # Converti output a FP32 per compatibilita
        out_f32 = ops.convert(out, destination_type=ov.Type.f32)

        model = ov.Model([out_f32], [x, gate, up, down], name="SwiGLU_Expert_Q8_0_BF16")
        compiled = self.core.compile_model(model, self.target_device)
        self._compiled_models[shape_key] = compiled
        return compiled

    def _dequantize_q8_0_to_bf16_fast(self, raw_bytes: bytes, rows: int, cols: int) -> np.ndarray:
        """Dequantizzazione Q8_0 -> BF16 ultra-rapida."""
        if _cpp_dequant_q8_0 is not None:
            # Usa C++ se disponibile
            dst = np.empty((rows, cols), dtype=np.uint16)
            if _cpp_use_avx2:
                _cpp_dequant_q8_0.dequantize_q8_0_to_bf16_avx2(
                    raw_bytes, 
                    dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                    rows, 
                    cols
                )
                print(f"[DEQUANT] C++ AVX2: {rows}x{cols}", flush=True)
            else:
                _cpp_dequant_q8_0.dequantize_q8_0_to_bf16(
                    raw_bytes, 
                    dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                    rows, 
                    cols
                )
                print(f"[DEQUANT] C++ OpenMP: {rows}x{cols}", flush=True)
            return dst
        else:
            # Fallback Python
            print(f"[DEQUANT] Python fallback: {rows}x{cols}", flush=True)
            raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            dequant = gguf.quants.dequantize(raw_arr, GGMLQuantizationType.Q8_0).astype(np.float32)
            # Converti FP32 a BF16 (semplicemente troncando)
            return dequant.astype(np.uint16)

    def _get_bf16_weights(self, key: ExpertKey, intermediate_size: int, hidden_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ottieni pesi in BF16. Usa cache LRU."""
        if key in self._bf16_cache:
            # Cache hit: sposta in fondo
            self.cache_order.remove(key)
            self.cache_order.append(key)
            return self._bf16_cache[key]
        
        print(f"[BF16 Cache] MISS: carico esperto {key}", flush=True)
        
        # Leggi i tensor compressi Q8_0
        gate_bytes = self.store.read_tensor(key, "gate")
        up_bytes = self.store.read_tensor(key, "up")
        down_bytes = self.store.read_tensor(key, "down")
        
        # Dequantizza Q8_0 -> BF16
        # Note: gate e up hanno shape (hidden_size, intermediate_size)
        #       down ha shape (intermediate_size, hidden_size)
        gate_bf16 = self._dequantize_q8_0_to_bf16_fast(gate_bytes, hidden_size, intermediate_size)
        up_bf16 = self._dequantize_q8_0_to_bf16_fast(up_bytes, hidden_size, intermediate_size)
        down_bf16 = self._dequantize_q8_0_to_bf16_fast(down_bytes, intermediate_size, hidden_size)
        
        weights = (gate_bf16, up_bf16, down_bf16)
        self._bf16_cache[key] = weights
        self.cache_order.append(key)
        
        # Evict se cache pieno
        if len(self._bf16_cache) > self.cache_capacity:
            oldest = self.cache_order.pop(0)
            self._bf16_cache.pop(oldest, None)
            print(f"[BF16 Cache] EVICT: rimosso {oldest}", flush=True)
        
        return weights

    def forward(
        self,
        key: ExpertKey,
        activation: list[float] | np.ndarray,
        hidden_size: int,
        intermediate_size: int,
    ) -> list[float]:
        gate_bf16, up_bf16, down_bf16 = self._get_bf16_weights(key, intermediate_size, hidden_size)
        
        if isinstance(activation, list):
            x_arr = np.array(activation, dtype=np.float32).reshape(1, hidden_size)
        else:
            x_arr = activation.reshape(1, hidden_size).astype(np.float32)

        compiled = self._get_compiled_model(hidden_size, intermediate_size)
        infer_request = compiled.create_infer_request()

        # OpenVINO si aspetta shape specifici, reshape se necessario
        # gate_bf16 e up_bf16 dovrebbero essere (hidden_size, intermediate_size)
        # down_bf16 dovrebbe essere (intermediate_size, hidden_size)
        results = infer_request.infer({
            "activation": x_arr,
            "gate": gate_bf16.reshape(hidden_size, intermediate_size),
            "up": up_bf16.reshape(hidden_size, intermediate_size),
            "down": down_bf16.reshape(intermediate_size, hidden_size),
        })

        output_tensor = list(results.values())[0]
        return output_tensor.flatten().tolist()


def main():
    print("=" * 80, flush=True)
    print(" FLASH-MOE: Lunar Lake - Q8_0 -> BF16 + 700 Esperti Cache", flush=True)
    print("=" * 80, flush=True)

    if not GGUF_FILE_PATH.exists():
        print(f"ERRORE: File GGUF non trovato.", flush=True)
        print(f"Path atteso: {GGUF_FILE_PATH}", flush=True)
        return

    # 1. Manifest
    manifest = build_manifest_from_gguf(GGUF_FILE_PATH)

    # 2. Store con 700 esperti Q8_0 in RAM (~2.1GB compressi)
    CACHE_CAPACITY = 700
    store = AsyncExpertStore(GGUF_FILE_PATH.parent, capacity=CACHE_CAPACITY, manifest=manifest)

    # 3. Backend Q8_0 -> BF16
    backend = Q8_0_BF16_Backend(store, device="GPU", cache_capacity=CACHE_CAPACITY)

    # Configurazione
    hidden_dim = 2048
    inter_dim = 512
    num_experts = 256
    num_layers = 40  # Tutti i layer
    router = ExpertRouter(hidden_size=hidden_dim, num_experts=num_experts, top_k=8)

    # Benchmark
    num_tokens = 1
    print(f"\n[Q8_0->BF16] Inferenza su {num_tokens} token x {num_layers} layer...", flush=True)
    print(f"[Cache] {CACHE_CAPACITY} esperti in RAM", flush=True)
    print(f"[Accelerazione] C++ Q8_0->BF16: {'AVX2' if _cpp_use_avx2 else 'SI' if _cpp_dequant_q8_0 else 'NO (fallback Python)'}", flush=True)
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
    print("     BENCHMARK: Q8_0 -> BF16 + 700 ESPERTI CACHE", flush=True)
    print("=" * 80, flush=True)
    print(f" File Modello:                    {GGUF_FILE_PATH.name}", flush=True)
    print(f" Quantizzazione:                 Q8_0", flush=True)
    print(f" Output precision:               BF16", flush=True)
    print(f" Esperti in cache (max):          {CACHE_CAPACITY}", flush=True)
    print(f" Memoria stimata (compressi):   ~2.1GB (Q8_0 in RAM)", flush=True)
    print(f" Memoria dequant (BF16):         ~4.2GB (700 esperti)", flush=True)
    print(f" Esperti attivi per token:        8 (top-k)", flush=True)
    print(f" Layer elaborati:                 {num_layers}", flush=True)
    print(f" Target GPU:                      Intel Lunar Lake (OpenVINO)", flush=True)
    print(f" Accelerazione C++:               {'OpenMP AVX2' if _cpp_use_avx2 else 'OpenMP' if _cpp_dequant_q8_0 else 'NO (Python fallback)'}", flush=True)
    print(f" Tempo Totale:                    {total_time:.4f} s", flush=True)
    print(f" Tokens elaborati:                {num_tokens}", flush=True)
    print(f" Velocita:                        {tokens_per_sec:.2f} token/sec", flush=True)
    print(f" Latenza per token:               {latency_ms:.2f} ms", flush=True)
    print(f" Cache Stats - Hits:              {store.stats.hits}", flush=True)
    print(f" Cache Stats - Misses:            {store.stats.misses}", flush=True)
    print(f" Dati letti da SSD:               {store.stats.bytes_read / (1024**2):.2f} MB", flush=True)
    print(f" Esperti in cache:                {len(backend._bf16_cache)}", flush=True)
    print("=" * 80, flush=True)

    store.close()


if __name__ == "__main__":
    main()
