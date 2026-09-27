import time
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
    MoEEngine,
)

GGUF_FILE_PATH = Path(r"C:\Users\valer\Desktop\moe-client-prot\Nuova cartella\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_NL.gguf")

def build_manifest_from_gguf(gguf_path: Path) -> tuple[ExpertManifest, int, int, int, int]:
    """Parses GGUF header and creates an ExpertManifest for all 40 layers and 256 experts."""
    print(f"[GGUF Reader] Apertura file GGUF: {gguf_path.name}...")
    reader = gguf.GGUFReader(gguf_path)
    tensor_map = {t.name: t for t in reader.tensors}

    block_count = 40 # Qwen3.6 35B MoE layers
    num_experts = 256 # total experts per layer
    hidden_dim = 2048
    inter_dim = 512
    top_k = 8

    records: dict[ExpertKey, ExpertRecord] = {}
    print(f"[GGUF Manifest] Mappatura offset di {block_count} layer x {num_experts} esperti = {block_count * num_experts} esperti totali su SSD...")

    for layer in range(block_count):
        gate_name = f"blk.{layer}.ffn_gate_exps.weight"
        up_name = f"blk.{layer}.ffn_up_exps.weight"
        down_name = f"blk.{layer}.ffn_down_exps.weight"

        if gate_name not in tensor_map:
            continue

        t_gate = tensor_map[gate_name]
        t_up = tensor_map[up_name]
        t_down = tensor_map[down_name]

        # Calcolo dimensione per singolo esperto allineata nel buffer GGUF (n_bytes)
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

    manifest = ExpertManifest(gguf_path.parent, records)
    return manifest, block_count, num_experts, hidden_dim, inter_dim


def main():
    print("==================================================================")
    print(" FLASH-MOE: Streaming Diretto da GGUF 18.4GB su Intel Arc GPU     ")
    print("==================================================================")

    if not GGUF_FILE_PATH.exists():
        print(f"ERRORE: File GGUF non trovato in {GGUF_FILE_PATH}")
        return

    # 1. Parsing GGUF Manifest
    manifest, num_layers, num_experts, hidden_dim, inter_dim = build_manifest_from_gguf(GGUF_FILE_PATH)

    # 2. Store asincrono con limite RAM (max 2 esperti residenti in memoria)
    store = AsyncExpertStore(GGUF_FILE_PATH.parent, capacity=2, manifest=manifest)

    # 3. Backend OpenVINO su GPU Intel Arc 130V
    backend = OpenVINOExpertBackend(store, device="GPU")

    # 4. Router Qwen MoE (Top-8 esperti per token)
    router = ExpertRouter(hidden_size=hidden_dim, num_experts=num_experts, top_k=8)

    # 5. Engine MoE
    engine = MoEEngine(
        store=store,
        backend=backend,
        router=router,
        hidden_size=hidden_dim,
        intermediate_size=inter_dim,
    )

    # 6. Benchmark di streaming reale dal file .gguf su SSD
    num_tokens = 10
    print(f"\n[Streaming] Avvio generazione per {num_tokens} token attraverso i {num_layers} layer MoE...")
    start_time = time.perf_counter()

    tokens = [np.random.randn(hidden_dim).astype(np.float32) for _ in range(num_tokens)]

    for token_idx, activation in enumerate(tokens):
        for layer in range(min(num_layers, 5)): # Test sui primi 5 layer MoE reali
            next_key = ExpertKey(layer, (token_idx + 1) % num_experts) if token_idx + 1 < num_tokens else None
            out = engine.process_token(layer=layer, activation=activation, next_expert_key=next_key)

    total_time = time.perf_counter() - start_time
    print("\n==================================================================")
    print("           BENCHMARK STREAMING DAL FILE GGUF REALE (18.4 GB)      ")
    print("==================================================================")
    print(f" File Modello GGUF:              {GGUF_FILE_PATH.name}")
    print(f" Tempo Esecuzione:               {total_time:.4f} secondi")
    print(f" Statistiche Cache Store:        Hit={store.stats.hits}, Miss={store.stats.misses}")
    print(f" Dati Estratti da SSD (GGUF):     {store.stats.bytes_read / (1024 * 1024):.2f} MB")
    print(f" Esperti in RAM:                 {len(store.resident_keys())} (Cap massimo = 2)")
    print("==================================================================")

    store.close()

if __name__ == "__main__":
    main()
