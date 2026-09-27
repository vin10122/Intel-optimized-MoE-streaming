import time
import ctypes
import numpy as np
from pathlib import Path
import gguf

from flash_moe import (
    ExpertKey,
    ExpertManifest,
    ExpertRecord,
    ExpertTensorSlice,
    AsyncExpertStore,
    OpenVINOExpertBackend,
    ExpertRouter,
)

# Path corretto: il modello è nella cartella del progetto
GGUF_FILE_PATH = Path(r"C:\Users\valer\Desktop\moe-client-prot\Nuova cartella\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_NL.gguf")
DLL_PATH = Path(r"flash_moe\fast_dequant.dll")

# Caricamento DLL C++ per dequantizzazione accelerata
_cpp_dequant = None
if DLL_PATH.exists():
    try:
        _cpp_dequant = ctypes.CDLL(str(DLL_PATH.resolve()))
        _cpp_dequant.dequantize_iq4_nl_avx2.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        print(f"[C++ Acceleration] Caricata DLL OpenMP AVX2: {DLL_PATH.name}", flush=True)
    except Exception as err:
        print(f"[C++ Acceleration] Impossibile caricare DLL: {err}", flush=True)


def build_manifest_from_gguf(gguf_path: Path) -> ExpertManifest:
    """Crea manifest per accesso diretto a slices GGUF."""
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


class GGUFStreamingBackend(OpenVINOExpertBackend):
    """Backend che usa lo store per caching con 2 esperti in RAM."""

    def __init__(self, store: AsyncExpertStore, device: str = "GPU"):
        super().__init__(store, device=device)

    def _dequantize_fast(self, raw_bytes: bytes, rows: int, cols: int) -> np.ndarray:
        """Dequantizzazione ultra-rapida: C++ AVX2/OpenMP se disponibile, altrimenti Python."""
        if _cpp_dequant is not None:
            dst = np.empty((rows, cols), dtype=np.float32)
            _cpp_dequant.dequantize_iq4_nl_avx2(raw_bytes, dst.ctypes.data, rows, cols)
            return dst
        else:
            raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            return gguf.quants.dequantize(raw_arr, gguf.GGMLType.IQ4_NL).astype(np.float32)

    def forward(
        self,
        key: ExpertKey,
        activation: list[float] | np.ndarray,
        hidden_size: int,
        intermediate_size: int,
    ) -> list[float]:
        # Legge i tensori COMPRESSI IQ4_NL dallo store (con caching LRU a 2 esperti)
        gate_bytes = self.store.read_tensor(key, "gate")
        up_bytes = self.store.read_tensor(key, "up")
        down_bytes = self.store.read_tensor(key, "down")

        # Dequantizzazione accelerata
        gate_arr = self._dequantize_fast(gate_bytes, intermediate_size, hidden_size)
        up_arr = self._dequantize_fast(up_bytes, intermediate_size, hidden_size)
        down_arr = self._dequantize_fast(down_bytes, hidden_size, intermediate_size)

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
    print(" FLASH-MOE: Intel Arc GPU - Streaming da GGUF con 2 esperti in RAM", flush=True)
    print("=" * 80, flush=True)

    if not GGUF_FILE_PATH.exists():
        print(f"ERRORE: File GGUF non trovato in {GGUF_FILE_PATH}", flush=True)
        return

    # 1. Costruzione manifest per accesso a slices GGUF
    manifest = build_manifest_from_gguf(GGUF_FILE_PATH)

    # 2. Store con MAX 2 ESPERTI IN RAM (come richiesto)
    store = AsyncExpertStore(GGUF_FILE_PATH.parent, capacity=2, manifest=manifest)

    # 3. Backend che usa lo store per caching
    backend = GGUFStreamingBackend(store, device="GPU")

    # Configurazione modello Qwen3.6 35B MoE
    hidden_dim = 2048
    inter_dim = 512
    num_experts = 256
    num_layers = 40

    # Router MoE (top-8 esperti per token)
    router = ExpertRouter(hidden_size=hidden_dim, num_experts=num_experts, top_k=8)

    # Benchmark su 10 token, tutti i 40 layer
    num_tokens = 10
    print(f"\n[Streaming GPU] Avvio inferenza su {num_tokens} token x {num_layers} layer MoE...", flush=True)
    print(f"[RAM Policy] Massimo 2 esperti in memoria alla volta", flush=True)
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
    print("     BENCHMARK: 2 ESPERTI IN RAM - INTEL ARC GPU + C++ AVX2", flush=True)
    print("=" * 80, flush=True)
    print(f" File Modello:                    {GGUF_FILE_PATH.name}", flush=True)
    print(f" Esperti in RAM (max):            2", flush=True)
    print(f" Esperti attivi per token:        8 (top-k)", flush=True)
    print(f" Layer elaborati:                 {num_layers}", flush=True)
    print(f" Accelerazione C++:               OpenMP AVX2 (se DLL caricata)", flush=True)
    print(f" Target GPU:                      Intel Arc (OpenVINO)", flush=True)
    print(f" Tempo Totale:                    {total_time:.4f} s", flush=True)
    print(f" Tokens elaborati:                {num_tokens}", flush=True)
    print(f" Velocita:                        {tokens_per_sec:.2f} token/sec", flush=True)
    print(f" Latenza per token:               {latency_ms:.2f} ms", flush=True)
    print(f" Cache Stats - Hits:              {store.stats.hits}", flush=True)
    print(f" Cache Stats - Misses:            {store.stats.misses}", flush=True)
    print(f" Dati letti da SSD:               {store.stats.bytes_read / (1024**2):.2f} MB", flush=True)
    print(f" Tensor reads:                    {store.stats.tensor_reads}", flush=True)
    print(f" Esperti residenti in RAM:        {len(store.resident_keys())}", flush=True)
    print("=" * 80, flush=True)

    store.close()


if __name__ == "__main__":
    main()
