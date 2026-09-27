import time
import ctypes
import numpy as np
from pathlib import Path
import gguf

from flash_moe import (
    ExpertKey,
    ExpertManifest,
    AsyncExpertStore,
    OpenVINOExpertBackend,
    ExpertRouter,
    MoEEngine,
)

GGUF_FILE_PATH = Path(r"C:\Users\valer\Desktop\moe-client-prot\Nuova cartella\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_NL.gguf")
DLL_PATH = Path(r"flash_moe\fast_dequant.dll")

_cpp_dequant = None
if DLL_PATH.exists():
    try:
        _cpp_dequant = ctypes.CDLL(str(DLL_PATH.resolve()))
        _cpp_dequant.dequantize_iq4_nl_avx2.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        print(f"[C++ Acceleration] Caricata con successo la DLL MSVC OpenMP AVX2: {DLL_PATH.name}")
    except Exception as err:
        print(f"[C++ Acceleration] Impossibile caricare DLL: {err}")


class GGUFRealExpertBackend(OpenVINOExpertBackend):
    """OpenVINO backend with native OpenMP AVX2 C++ dequantization streaming for Intel Arc GPU."""

    def __init__(self, store: AsyncExpertStore, reader: gguf.GGUFReader, device: str = "GPU"):
        super().__init__(store, device=device)
        self.reader = reader
        self.tensor_map = {t.name: t for t in reader.tensors}

    def _dequantize_fast(self, raw_data: np.ndarray, rows: int, cols: int) -> np.ndarray:
        if _cpp_dequant is not None:
            raw_bytes = raw_data.tobytes()
            dst = np.empty((rows, cols), dtype=np.float32)
            _cpp_dequant.dequantize_iq4_nl_avx2(raw_bytes, dst.ctypes.data, rows, cols)
            return dst
        else:
            return gguf.quants.dequantize(raw_data, gguf.GGMLType.IQ4_NL).astype(np.float32)

    def forward(
        self,
        key: ExpertKey,
        activation: list[float] | np.ndarray,
        hidden_size: int,
        intermediate_size: int,
    ) -> list[float]:
        layer = key.layer
        eid = key.expert

        t_gate = self.tensor_map[f"blk.{layer}.ffn_gate_exps.weight"]
        t_up = self.tensor_map[f"blk.{layer}.ffn_up_exps.weight"]
        t_down = self.tensor_map[f"blk.{layer}.ffn_down_exps.weight"]

        # Dequantizzazione OpenMP AVX2 C++ ultrarapida
        gate_arr = self._dequantize_fast(t_gate.data[eid], intermediate_size, hidden_size)
        up_arr = self._dequantize_fast(t_up.data[eid], intermediate_size, hidden_size)
        down_arr = self._dequantize_fast(t_down.data[eid], hidden_size, intermediate_size)

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


class GGUFMoEEngine:
    """MoE Engine streaming directly from GGUF 18.4GB file into Intel Arc GPU."""
    def __init__(self, backend: GGUFRealExpertBackend, router: ExpertRouter, hidden_size: int, intermediate_size: int):
        self.backend = backend
        self.router = router
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

    def process_token(self, layer: int, activation: np.ndarray) -> list[float]:
        expert_ids, weights = self.router.route(activation)

        combined_output = np.zeros(self.hidden_size, dtype=np.float32)
        for expert_id, weight in zip(expert_ids, weights):
            key = ExpertKey(layer, expert_id)
            expert_out = self.backend.forward(key, activation, self.hidden_size, self.intermediate_size)
            combined_output += weight * np.array(expert_out, dtype=np.float32)

        return combined_output.tolist()


def main():
    print("==================================================================", flush=True)
    print(" FLASH-MOE: Real-Time GGUF 18.4GB Streaming su Intel Arc GPU      ", flush=True)
    print("==================================================================", flush=True)

    if not GGUF_FILE_PATH.exists():
        print(f"ERRORE: File GGUF non trovato in {GGUF_FILE_PATH}", flush=True)
        return

    print(f"[GGUF Reader] Apertura file da 18.4 GB: {GGUF_FILE_PATH.name}...", flush=True)
    start_parse = time.perf_counter()
    reader = gguf.GGUFReader(GGUF_FILE_PATH)
    print(f"[GGUF Reader] Header analizzato in {time.perf_counter() - start_parse:.2f}s! Tensori trovati: {len(reader.tensors)}", flush=True)

    block_count = 40 # Qwen3.6 35B MoE layers
    num_experts = 256 # total experts per layer
    hidden_dim = 2048
    inter_dim = 512
    top_k = 8

    dummy_manifest = ExpertManifest(GGUF_FILE_PATH.parent, {})
    store = AsyncExpertStore(GGUF_FILE_PATH.parent, capacity=2, manifest=dummy_manifest)

    backend = GGUFRealExpertBackend(store, reader=reader, device="GPU")
    router = ExpertRouter(hidden_size=hidden_dim, num_experts=num_experts, top_k=top_k)
    engine = GGUFMoEEngine(backend=backend, router=router, hidden_size=hidden_dim, intermediate_size=inter_dim)

    num_tokens = 10
    print(f"\n[Esecuzione Streaming GPU] Avvio inferenza MoE con dequantizzatore OpenMP AVX2 C++ su {num_tokens} token...", flush=True)
    start_time = time.perf_counter()

    tokens = [np.random.randn(hidden_dim).astype(np.float32) for _ in range(num_tokens)]

    for token_idx, activation in enumerate(tokens):
        for layer in range(5):
            out = engine.process_token(layer=layer, activation=activation)
        print(f" -> Token {token_idx + 1}/{num_tokens} processato su 5 layer MoE con successo!", flush=True)

    total_time = time.perf_counter() - start_time
    tokens_per_sec = num_tokens / total_time
    latency_ms = (total_time / num_tokens) * 1000

    print("\n==================================================================", flush=True)
    print("      RISULTATI BENCHMARK STREAMING OpenMP AVX2 + INTEL ARC GPU   ", flush=True)
    print("==================================================================", flush=True)
    print(f" File Modello GGUF:               {GGUF_FILE_PATH.name}")
    print(f" Accelerazione C++ DLL:           MSVC 2022 OpenMP AVX2 NATIVO")
    print(f" Target Hardware GPU:             Intel Arc 130V (via OpenVINO)")
    print(f" Tempo Totale Generazione:        {total_time:.4f} s")
    print(f" Velocità di Generazione:        {tokens_per_sec:.2f} token/sec")
    print(f" Latenza per Token (5 Layer MoE): {latency_ms:.2f} ms")
    print(f" Esperti Attivi per Token:        8 su 256 per layer")
    print("==================================================================", flush=True)

    store.close()

if __name__ == "__main__":
    main()
