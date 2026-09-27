import time
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

# Path del modello INT8 (Q8_0) - 35GB
GGUF_FILE_PATH = Path(r"C:\Users\valer\Desktop\moe-client-prot\Nuova cartella\Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf")


def build_manifest_from_gguf(gguf_path: Path) -> ExpertManifest:
    """Crea manifest per accesso diretto a slices GGUF INT8."""
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


class INT8NativeBackend:
    """Backend che usa INT8 nativamente in OpenVINO - zero conversione."""

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
        self._int8_cache: dict[ExpertKey, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.cache_order: list[ExpertKey] = []
        
        print(f"[OpenVINOBackend] Target Device: {self.target_device}", flush=True)

    def _get_compiled_model(self, hidden_size: int, intermediate_size: int) -> ov.CompiledModel:
        shape_key = (hidden_size, intermediate_size)
        if shape_key in self._compiled_models:
            return self._compiled_models[shape_key]

        # Parametri: pesi BF16, attivazioni FP32 (mixed precision)
        x = ops.parameter([1, hidden_size], ov.Type.f32, name="activation")
        gate = ops.parameter([hidden_size, intermediate_size], ov.Type.bf16, name="gate")
        up = ops.parameter([hidden_size, intermediate_size], ov.Type.bf16, name="up")
        down = ops.parameter([intermediate_size, hidden_size], ov.Type.bf16, name="down")

        # MatMul senza trasporre: x@gate, x@up, swiglu_out@down
        gate_proj = ops.matmul(x, gate, transpose_a=False, transpose_b=False)
        up_proj = ops.matmul(x, up, transpose_a=False, transpose_b=False)
        swish_gate = ops.swish(gate_proj)
        swiglu_out = ops.multiply(swish_gate, up_proj)
        out = ops.matmul(swiglu_out, down, transpose_a=False, transpose_b=False)

        model = ov.Model([out], [x, gate, up, down], name="SwiGLU_Expert_INT8")
        compiled = self.core.compile_model(model, self.target_device)
        self._compiled_models[shape_key] = compiled
        return compiled

    def _get_int8_weights(self, key: ExpertKey, intermediate_size: int, hidden_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ottieni pesi INT8. Usa cache LRU."""
        if key in self._int8_cache:
            self.cache_order.remove(key)
            self.cache_order.append(key)
            return self._int8_cache[key]
        
        print(f"[INT8 Cache] MISS: carico esperto {key}", flush=True)
        
        gate_bytes = self.store.read_tensor(key, "gate")
        up_bytes = self.store.read_tensor(key, "up")
        down_bytes = self.store.read_tensor(key, "down")
        
        # Q8_0 è quantizzazione a blocchi, non INT8 puro. Dequantizza con gguf
        raw_gate = np.frombuffer(gate_bytes, dtype=np.uint8)
        raw_up = np.frombuffer(up_bytes, dtype=np.uint8)
        raw_down = np.frombuffer(down_bytes, dtype=np.uint8)
        
        # Dequantizza Q8_0 -> BF16 (restituisce 1D, dobbiamo reshapeare)
        # Q8_0 memorizza: gate/up come [hidden, inter], down come [inter, hidden]
        gate_arr = gguf.quants.dequantize(raw_gate, GGMLQuantizationType.Q8_0).astype(np.bfloat16).reshape(hidden_size, intermediate_size)
        up_arr = gguf.quants.dequantize(raw_up, GGMLQuantizationType.Q8_0).astype(np.bfloat16).reshape(hidden_size, intermediate_size)
        down_arr = gguf.quants.dequantize(raw_down, GGMLQuantizationType.Q8_0).astype(np.bfloat16).reshape(intermediate_size, hidden_size)
        
        # Non trasponiamo! Mantieni la forma originale:
        # gate: (2048, 512), up: (2048, 512), down: (512, 2048)
        # OpenVINO farà matmul senza trasporre
        
        weights = (gate_arr, up_arr, down_arr)
        self._int8_cache[key] = weights
        self.cache_order.append(key)
        
        if len(self._int8_cache) > self.cache_capacity:
            oldest = self.cache_order.pop(0)
            self._int8_cache.pop(oldest, None)
            print(f"[INT8 Cache] EVICT: rimosso {oldest}", flush=True)
        
        return weights

    def forward(
        self,
        key: ExpertKey,
        activation: list[float] | np.ndarray,
        hidden_size: int,
        intermediate_size: int,
    ) -> list[float]:
        gate_arr, up_arr, down_arr = self._get_int8_weights(key, intermediate_size, hidden_size)
        
        if isinstance(activation, list):
            x_arr = np.array(activation, dtype=np.float32).reshape(1, hidden_size)
        else:
            x_arr = activation.reshape(1, hidden_size).astype(np.float32)

        compiled = self._get_compiled_model(hidden_size, intermediate_size)
        infer_request = compiled.create_infer_request()

        results = infer_request.infer({
            "activation": x_arr,
            "gate": gate_arr,
            "up": up_arr,
            "down": down_arr,
        })

        output_tensor = list(results.values())[0]
        return output_tensor.flatten().tolist()


def main():
    print("=" * 80, flush=True)
    print(" FLASH-MOE: Lunar Lake - INT8 NATIVO + 700 Esperti Cache", flush=True)
    print("=" * 80, flush=True)

    if not GGUF_FILE_PATH.exists():
        print(f"ERRORE: File GGUF non trovato.", flush=True)
        print(f"Path atteso: {GGUF_FILE_PATH}", flush=True)
        return

    # 1. Manifest
    manifest = build_manifest_from_gguf(GGUF_FILE_PATH)

    # 2. Store con 700 esperti INT8 in RAM (~2.1GB)
    CACHE_CAPACITY = 700
    store = AsyncExpertStore(GGUF_FILE_PATH.parent, capacity=CACHE_CAPACITY, manifest=manifest)

    # 3. Backend INT8 nativo
    backend = INT8NativeBackend(store, device="GPU", cache_capacity=CACHE_CAPACITY)

    # Configurazione
    hidden_dim = 2048
    inter_dim = 512
    num_experts = 256
    num_layers = 2  # Test solo 2 layer prima
    router = ExpertRouter(hidden_size=hidden_dim, num_experts=num_experts, top_k=8)

    # Benchmark (test con 1 token prima)
    num_tokens = 1
    print(f"\n[INT8 Native] Inferenza su {num_tokens} token x {num_layers} layer...", flush=True)
    print(f"[Cache] {CACHE_CAPACITY} esperti INT8 in RAM (~2.1GB)", flush=True)
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
    print("     BENCHMARK: INT8 NATIVO + 700 ESPERTI CACHE", flush=True)
    print("=" * 80, flush=True)
    print(f" File Modello:                    {GGUF_FILE_PATH.name}", flush=True)
    print(f" Quantizzazione:                 Q8_0 (INT8)", flush=True)
    print(f" Esperti in cache (max):          {CACHE_CAPACITY}", flush=True)
    print(f" Memoria stimata:                ~2.1GB (INT8 in RAM)", flush=True)
    print(f" Esperti attivi per token:        8 (top-k)", flush=True)
    print(f" Layer elaborati:                 {num_layers}", flush=True)
    print(f" Target GPU:                      Intel Lunar Lake (OpenVINO)", flush=True)
    print(f" Tempo Totale:                    {total_time:.4f} s", flush=True)
    print(f" Tokens elaborati:                {num_tokens}", flush=True)
    print(f" Velocita:                        {tokens_per_sec:.2f} token/sec", flush=True)
    print(f" Latenza per token:               {latency_ms:.2f} ms", flush=True)
    print(f" Cache Stats - Hits:              {store.stats.hits}", flush=True)
    print(f" Cache Stats - Misses:            {store.stats.misses}", flush=True)
    print(f" Dati letti da SSD:               {store.stats.bytes_read / (1024**2):.2f} MB", flush=True)
    print(f" Esperti in cache:                {len(backend._int8_cache)}", flush=True)
    print("=" * 80, flush=True)

    store.close()


if __name__ == "__main__":
    main()
