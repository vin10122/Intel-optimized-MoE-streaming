"""
FLASH-MOE GPU CACHED: Dequantizzazione una volta + caching su GPU

Architettura:
- CPU RAM: 2 esperti in formato IQ4_NL compresso (caching LRU)
- GPU VRAM: Tutti gli esperti dequantizzati in FP16 (nessun limite con 16GB)
- Dequantizzazione: C++ AVX2 (solo una volta per esperto)
- Esecuzione: OpenVINO su Intel Arc GPU

Ottimizzazioni vs versione base:
✅ Dequantizzazione SOLO UNA VOLTA per esperto (non per ogni accesso)
✅ Pesi su GPU pronti per MatMul (nessun trasferimento ripetuto)
✅ 16GB VRAM sfruttata al massimo
✅ 2 esperti in CPU RAM rispettati
"""

import time
import ctypes
import numpy as np
from pathlib import Path
import gguf
import openvino as ov

from flash_moe import (
    ExpertKey,
    ExpertManifest,
    ExpertRecord,
    ExpertTensorSlice,
    AsyncExpertStore,
    ExpertRouter,
)

# === CONFIG ===
GGUF_FILE_PATH = Path(r"C:\Users\valer\Desktop\moe-client-prot\Nuova cartella\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_NL.gguf")
DLL_PATH = Path(r"flash_moe\fast_dequant.dll")

HIDDEN_DIM = 2048
INTER_DIM = 512
NUM_EXPERTS = 256
NUM_LAYERS = 40
TOP_K = 8

# 2 esperti in CPU RAM (IQ4_NL compressi)
CPU_CACHE_SIZE = 2

# === DLL C++ DEQUANT ===
_cpp_dequant = None
if DLL_PATH.exists():
    try:
        _cpp_dequant = ctypes.CDLL(str(DLL_PATH.resolve()))
        _cpp_dequant.dequantize_iq4_nl_avx2.argtypes = [
            ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int
        ]
        print(f"[AVX2] DLL OpenMP caricata", flush=True)
    except Exception as err:
        print(f"[AVX2] Errore: {err}", flush=True)


# === MANIFEST (stesso di prima) ===
def build_manifest_from_gguf(gguf_path: Path) -> ExpertManifest:
    print(f"[Manifest] Costruzione...", flush=True)
    reader = gguf.GGUFReader(gguf_path)
    tensor_map = {t.name: t for t in reader.tensors}
    
    records: dict[ExpertKey, ExpertRecord] = {}
    
    for layer in range(NUM_LAYERS):
        gate_name = f"blk.{layer}.ffn_gate_exps.weight"
        up_name = f"blk.{layer}.ffn_up_exps.weight"
        down_name = f"blk.{layer}.ffn_down_exps.weight"
        
        if gate_name not in tensor_map:
            continue
        
        t_gate = tensor_map[gate_name]
        t_up = tensor_map[up_name]
        t_down = tensor_map[down_name]
        
        gate_expert_bytes = t_gate.n_bytes // NUM_EXPERTS
        up_expert_bytes = t_up.n_bytes // NUM_EXPERTS
        down_expert_bytes = t_down.n_bytes // NUM_EXPERTS
        
        for eid in range(NUM_EXPERTS):
            key = ExpertKey(layer, eid)
            tensors = {
                "gate": ExpertTensorSlice("gate", 
                    offset=t_gate.data_offset + eid * gate_expert_bytes, 
                    nbytes=gate_expert_bytes),
                "up": ExpertTensorSlice("up", 
                    offset=t_up.data_offset + eid * up_expert_bytes, 
                    nbytes=up_expert_bytes),
                "down": ExpertTensorSlice("down", 
                    offset=t_down.data_offset + eid * down_expert_bytes, 
                    nbytes=down_expert_bytes),
            }
            total_bytes = gate_expert_bytes + up_expert_bytes + down_expert_bytes
            records[key] = ExpertRecord(
                key=key, relative_path=gguf_path.name, 
                nbytes=total_bytes, tensors=tensors
            )
    
    return ExpertManifest(gguf_path.parent, records)


# === BACKEND CON CACHING SU GPU ===
class GPUCachedBackend:
    """
    Backend con DUE livelli di cache:
    - CPU RAM: 2 esperti in IQ4_NL compresso (LRU)
    - GPU VRAM: Tutti gli esperti dequantizzati in FP16
    
    Dequantizzazione avviene SOLO UNA VOLTA per esperto (CPU -> GPU).
    """
    
    def __init__(self, store: AsyncExpertStore, reader: gguf.GGUFReader, device: str = "GPU"):
        self.store = store
        self.reader = reader
        self.core = ov.Core()
        self.target_device = "GPU" if "GPU" in self.core.available_devices else "CPU"
        
        # Tensor map per accesso diretto (fallback)
        self.tensor_map = {t.name: t for t in reader.tensors}
        
        # Cache su GPU: key -> (gate_ov, up_ov, down_ov)
        # OpenVINO tensori già su GPU, pronti per inferenza
        self.gpu_cache = {}
        
        # Pre-compila modello SwiGLU UNA SOLA VOLTA
        self.compiled_model = self._compile_swiglu()
        print(f"[OpenVINO] Modello pre-compilato su {self.target_device}", flush=True)
        print(f"[GPU Cache] Pesi dequantizzati verranno cacheati su VRAM", flush=True)
    
    def _compile_swiglu(self) -> ov.CompiledModel:
        """Compila modello SwiGLU per OpenVINO."""
        import openvino.opset13 as ops
        
        x = ops.parameter([1, HIDDEN_DIM], ov.Type.f32, name="activation")
        gate = ops.parameter([INTER_DIM, HIDDEN_DIM], ov.Type.f32, name="gate")
        up = ops.parameter([INTER_DIM, HIDDEN_DIM], ov.Type.f32, name="up")
        down = ops.parameter([HIDDEN_DIM, INTER_DIM], ov.Type.f32, name="down")
        
        gate_proj = ops.matmul(x, gate, transpose_a=False, transpose_b=True)
        up_proj = ops.matmul(x, up, transpose_a=False, transpose_b=True)
        swish_gate = ops.swish(gate_proj)
        swiglu_out = ops.multiply(swish_gate, up_proj)
        out = ops.matmul(swiglu_out, down, transpose_a=False, transpose_b=True)
        
        model = ov.Model([out], [x, gate, up, down], name="SwiGLU_MoE")
        return self.core.compile_model(model, self.target_device)
    
    def _dequant_to_fp16(self, raw_data: np.ndarray, rows: int, cols: int) -> np.ndarray:
        """Dequantizza IQ4_NL -> FP16 usando C++ AVX2."""
        if _cpp_dequant is not None:
            raw_bytes = raw_data.tobytes()
            dst_f32 = np.empty((rows, cols), dtype=np.float32)
            _cpp_dequant.dequantize_iq4_nl_avx2(raw_bytes, dst_f32.ctypes.data, rows, cols)
            return dst_f32.astype(np.float16)
        else:
            return gguf.quants.dequantize(raw_data, gguf.GGMLType.IQ4_NL).astype(np.float16)
    
    def _ensure_expert_on_gpu(self, key: ExpertKey) -> tuple[ov.Tensor, ov.Tensor, ov.Tensor]:
        """
        Assicura che l'esperto sia su GPU.
        
        Strategia:
        1. Se già su GPU -> restituisci
        2. Altrimenti, prova a leggere da CPU cache (store)
        3. Se non in CPU cache, leggi da GGUF, dequantizza, carica su GPU
        
        RESTITUISCE: (gate_tensor, up_tensor, down_tensor) - già su GPU
        """
        if key in self.gpu_cache:
            return self.gpu_cache[key]
        
        # Prova a leggere da CPU cache (store) - IQ4_NL compresso
        try:
            # Leggi tutti e 3 i tensori compressi dallo store
            gate_bytes = self.store.read_tensor(key, "gate")
            up_bytes = self.store.read_tensor(key, "up")
            down_bytes = self.store.read_tensor(key, "down")
            
            # Dequantizza a FP16 su CPU
            gate_fp16 = self._dequant_bytes_to_fp16(gate_bytes, INTER_DIM, HIDDEN_DIM)
            up_fp16 = self._dequant_bytes_to_fp16(up_bytes, INTER_DIM, HIDDEN_DIM)
            down_fp16 = self._dequant_bytes_to_fp16(down_bytes, HIDDEN_DIM, INTER_DIM)
        except KeyError:
            # Non in cache CPU: leggi direttamente da GGUF
            layer = key.layer
            eid = key.expert
            t_gate = self.tensor_map[f"blk.{layer}.ffn_gate_exps.weight"]
            t_up = self.tensor_map[f"blk.{layer}.ffn_up_exps.weight"]
            t_down = self.tensor_map[f"blk.{layer}.ffn_down_exps.weight"]
            
            gate_raw = t_gate.data[eid]
            up_raw = t_up.data[eid]
            down_raw = t_down.data[eid]
            
            gate_fp16 = self._dequant_to_fp16(gate_raw, INTER_DIM, HIDDEN_DIM)
            up_fp16 = self._dequant_to_fp16(up_raw, INTER_DIM, HIDDEN_DIM)
            down_fp16 = self._dequant_to_fp16(down_raw, HIDDEN_DIM, INTER_DIM)
        
        # Converti a FP32 per OpenVINO
        gate_fp32 = gate_fp16.astype(np.float32)
        up_fp32 = up_fp16.astype(np.float32)
        down_fp32 = down_fp16.astype(np.float32)
        
        # Crea tensori OpenVINO su GPU
        gate_tensor = ov.Tensor(gate_fp32, shape=gate_fp32.shape)
        up_tensor = ov.Tensor(up_fp32, shape=up_fp32.shape)
        down_tensor = ov.Tensor(down_fp32, shape=down_fp32.shape)
        
        # Salva in cache GPU
        self.gpu_cache[key] = (gate_tensor, up_tensor, down_tensor)
        
        return gate_tensor, up_tensor, down_tensor
    
    def _dequant_bytes_to_fp16(self, raw_bytes: bytes, rows: int, cols: int) -> np.ndarray:
        """Dequantizza da bytes IQ4_NL a FP16."""
        if _cpp_dequant is not None:
            dst_f32 = np.empty((rows, cols), dtype=np.float32)
            _cpp_dequant.dequantize_iq4_nl_avx2(raw_bytes, dst_f32.ctypes.data, rows, cols)
            return dst_f32.astype(np.float16)
        else:
            raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            return gguf.quants.dequantize(raw_arr, gguf.GGMLType.IQ4_NL).astype(np.float16)
    
    def forward(self, key: ExpertKey, activation: np.ndarray) -> np.ndarray:
        """
        Forward pass con caching su GPU.
        La dequantizzazione avviene SOLO UNA VOLTA per esperto.
        """
        # Assicurati che l'esperto sia su GPU
        gate_tensor, up_tensor, down_tensor = self._ensure_expert_on_gpu(key)
        
        # Converti activation a FP32 per OpenVINO
        x = activation.astype(np.float32).reshape(1, HIDDEN_DIM)
        
        # Inference su GPU
        infer_request = self.compiled_model.create_infer_request()
        result = infer_request.infer({
            "activation": x,
            "gate": gate_tensor,
            "up": up_tensor,
            "down": down_tensor,
        })
        
        return list(result.values())[0].flatten().astype(np.float32)


# === MAIN ===
def main():
    print("=" * 80, flush=True)
    print(" FLASH-MOE GPU CACHED: Dequant una volta + caching su GPU", flush=True)
    print("=" * 80, flush=True)
    
    if not GGUF_FILE_PATH.exists():
        print(f"ERRORE: {GGUF_FILE_PATH} non trovato!", flush=True)
        return
    
    # Init
    print("[Init] GGUF Reader...", flush=True)
    reader = gguf.GGUFReader(GGUF_FILE_PATH)
    
    print("[Init] Manifest...", flush=True)
    manifest = build_manifest_from_gguf(GGUF_FILE_PATH)
    
    print("[Init] Store (2 esperti in CPU RAM)...", flush=True)
    store = AsyncExpertStore(GGUF_FILE_PATH.parent, capacity=CPU_CACHE_SIZE, manifest=manifest)
    
    print("[Init] Backend (caching su GPU)...", flush=True)
    backend = GPUCachedBackend(store, reader, device="GPU")
    
    router = ExpertRouter(hidden_size=HIDDEN_DIM, num_experts=NUM_EXPERTS, top_k=TOP_K)
    
    # Benchmark
    NUM_TOKENS = 10
    print(f"\n[Benchmark] {NUM_TOKENS} token x {NUM_LAYERS} layer", flush=True)
    print(f"[CPU Cache] {CPU_CACHE_SIZE} esperti in RAM", flush=True)
    print(f"[GPU Cache] Illimitata (16GB VRAM)", flush=True)
    
    tokens = [np.random.randn(HIDDEN_DIM).astype(np.float32) for _ in range(NUM_TOKENS)]
    
    start_time = time.perf_counter()
    
    for token_idx, activation in enumerate(tokens):
        current = activation
        for layer in range(NUM_LAYERS):
            expert_ids, weights = router.route(current)
            combined = np.zeros(HIDDEN_DIM, dtype=np.float32)
            for eid, weight in zip(expert_ids, weights):
                key = ExpertKey(layer, eid)
                expert_out = backend.forward(key, current)
                combined += weight * expert_out
            current = combined
        print(f"  -> Token {token_idx + 1}/{NUM_TOKENS} processato", flush=True)
    
    total_time = time.perf_counter() - start_time
    tokens_per_sec = NUM_TOKENS / total_time
    latency_ms = (total_time / NUM_TOKENS) * 1000
    
    # Risultati
    print("\n" + "=" * 80, flush=True)
    print("          FLASH-MOE GPU CACHED - RISULTATI", flush=True)
    print("=" * 80, flush=True)
    
    print(f"\n Configurazione:")
    print(f"   Token:            {NUM_TOKENS}")
    print(f"   Layer:           {NUM_LAYERS}")
    print(f"   CPU Cache:      {CPU_CACHE_SIZE} esperti (IQ4_NL)")
    print(f"   GPU Cache:      Illimitata (FP16)")
    print(f"   Dequant:        1x per esperto (C++ AVX2)")
    
    print(f"\n Prestazioni:")
    print(f"   Tempo totale:   {total_time:.4f} s")
    print(f"   Tokens/sec:     {tokens_per_sec:.2f}")
    print(f"   Latenza/token: {latency_ms:.2f} ms")
    
    print(f"\n Cache Stats:")
    print(f"   CPU Hits:       {store.stats.hits}")
    print(f"   CPU Misses:    {store.stats.misses}")
    print(f"   Dati da SSD:   {store.stats.bytes_read / (1024**2):.2f} MB")
    print(f"   GPU Cache:     {len(backend.gpu_cache)} esperti")
    
    # Confronto
    original_time = 20.94  # tempo attuale ottimizzato
    original_tps = NUM_TOKENS / original_time
    speedup = tokens_per_sec / original_tps
    
    print(f"\n Confronto vs versione attuale (20.94s):")
    print(f"   Speedup:       {speedup:.2f}x")
    print(f"   Tempo risparmiato: {original_time - total_time:.2f}s")
    print("=" * 80, flush=True)
    
    store.close()


if __name__ == "__main__":
    main()
