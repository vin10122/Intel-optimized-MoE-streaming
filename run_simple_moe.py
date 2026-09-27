"""
FLASH-MOE SIMPLE: Solo cache dequant + lettura diretta da GGUF

Semplificazione estrema:
- Legge direttamente da GGUFReader (mappato in memoria)
- Cache SOLO dequantizzati (FP16) in RAM
- Nessuna cache IQ4_NL (non serve con lettura veloce)
"""

import time
import ctypes
import numpy as np
from pathlib import Path
import gguf
import openvino as ov

from flash_moe import (
    ExpertKey,
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


# === BACKEND SEMPLICE ===
class SimpleBackend:
    def __init__(self, reader: gguf.GGUFReader, device: str = "GPU"):
        self.reader = reader
        self.core = ov.Core()
        self.target_device = "GPU" if "GPU" in self.core.available_devices else "CPU"
        self.tensor_map = {t.name: t for t in reader.tensors}
        self.dequant_cache = {}
        self.stats_dequant = 0
        self.compiled_model = self._compile_swiglu()
        print(f"[OpenVINO] Modello pre-compilato su {self.target_device}", flush=True)
    
    def _compile_swiglu(self) -> ov.CompiledModel:
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
        if _cpp_dequant is not None:
            raw_bytes = raw_data.tobytes()
            dst_f32 = np.empty((rows, cols), dtype=np.float32)
            _cpp_dequant.dequantize_iq4_nl_avx2(raw_bytes, dst_f32.ctypes.data, rows, cols)
            return dst_f32.astype(np.float16)
        else:
            return gguf.quants.dequantize(raw_data, gguf.GGMLType.IQ4_NL).astype(np.float16)
    
    def forward(self, key: ExpertKey, activation: np.ndarray) -> np.ndarray:
        layer, eid = key.layer, key.expert
        
        if key in self.dequant_cache:
            gate_fp16, up_fp16, down_fp16 = self.dequant_cache[key]
        else:
            t_gate = self.tensor_map[f"blk.{layer}.ffn_gate_exps.weight"]
            t_up = self.tensor_map[f"blk.{layer}.ffn_up_exps.weight"]
            t_down = self.tensor_map[f"blk.{layer}.ffn_down_exps.weight"]
            
            gate_raw = t_gate.data[eid]
            up_raw = t_up.data[eid]
            down_raw = t_down.data[eid]
            
            gate_fp16 = self._dequant_to_fp16(gate_raw, INTER_DIM, HIDDEN_DIM)
            up_fp16 = self._dequant_to_fp16(up_raw, INTER_DIM, HIDDEN_DIM)
            down_fp16 = self._dequant_to_fp16(down_raw, HIDDEN_DIM, INTER_DIM)
            
            self.dequant_cache[key] = (gate_fp16, up_fp16, down_fp16)
            self.stats_dequant += 1
        
        gate_fp32 = gate_fp16.astype(np.float32)
        up_fp32 = up_fp16.astype(np.float32)
        down_fp32 = down_fp16.astype(np.float32)
        
        x = activation.astype(np.float32).reshape(1, HIDDEN_DIM)
        
        infer_request = self.compiled_model.create_infer_request()
        result = infer_request.infer({
            "activation": x,
            "gate": gate_fp32,
            "up": up_fp32,
            "down": down_fp32,
        })
        
        return list(result.values())[0].flatten().astype(np.float32)


# === MAIN ===
def main():
    print("=" * 80, flush=True)
    print(" FLASH-MOE SIMPLE: Lettura diretta + cache dequant", flush=True)
    print("=" * 80, flush=True)
    
    if not GGUF_FILE_PATH.exists():
        print(f"ERRORE: {GGUF_FILE_PATH} non trovato!", flush=True)
        return
    
    print("[Init] GGUF Reader...", flush=True)
    reader = gguf.GGUFReader(GGUF_FILE_PATH)
    
    print("[Init] Backend...", flush=True)
    backend = SimpleBackend(reader, device="GPU")
    
    router = ExpertRouter(hidden_size=HIDDEN_DIM, num_experts=NUM_EXPERTS, top_k=TOP_K)
    
    NUM_TOKENS = 10
    print(f"\n[Benchmark] {NUM_TOKENS} token x {NUM_LAYERS} layer", flush=True)
    
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
    
    print("\n" + "=" * 80, flush=True)
    print("          RISULTATI", flush=True)
    print("=" * 80, flush=True)
    print(f"\n Prestazioni:")
    print(f"   Tempo totale:   {total_time:.4f} s")
    print(f"   Tokens/sec:     {tokens_per_sec:.2f}")
    print(f"   Latenza/token: {latency_ms:.2f} ms")
    print(f"\n Cache:")
    print(f"   Dequant:        {backend.stats_dequant} volte")
    print(f"   Cache size:    {len(backend.dequant_cache)} esperti")
    
    baseline_time = 32.62
    baseline_tps = NUM_TOKENS / baseline_time
    speedup_vs_baseline = tokens_per_sec / baseline_tps
    
    improved_time = 20.94
    improved_tps = NUM_TOKENS / improved_time
    speedup_vs_improved = tokens_per_sec / improved_tps
    
    print(f"\n Confronto:")
    print(f"   vs Originale (32.62s):   {speedup_vs_baseline:.2f}x")
    print(f"   vs Attuale (20.94s):    {speedup_vs_improved:.2f}x")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
