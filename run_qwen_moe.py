import time
import sys
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

# Il pacchetto vive in src/: e' l'unica copia, la radice non ne ha piu'.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from flash_moe import (
    ExpertKey,
    ExpertManifest,
    AsyncExpertStore,
    OpenVINOExpertBackend,
    ExpertRouter,
    synthetic_router,
    MoEEngine,
    pack_expert,
    pack_f16,
    write_manifest,
)

def build_mock_qwen_moe_weights(root: Path, num_layers: int = 2, num_experts: int = 8, hidden_dim: int = 128, inter_dim: int = 256):
    """Builds a synthetic model matching Qwen MoE layout for testing."""
    records = []
    print(f"[QwenMoE Setup] Generazione pesi MoE sintetici su SSD ({num_layers} layer, {num_experts} esperti/layer)...")

    for layer in range(num_layers):
        for eid in range(num_experts):
            key = ExpertKey(layer, eid)
            # f16: e' il formato con cui il backend legge gli expert non
            # quantizzati. Con pack_f32 i byte verrebbero riletti come f16 e
            # il conteggio degli elementi risulterebbe doppio.
            tensors = {
                "gate": pack_f16(np.random.randn(inter_dim * hidden_dim).astype(np.float16).tolist()),
                "up": pack_f16(np.random.randn(inter_dim * hidden_dim).astype(np.float16).tolist()),
                "down": pack_f16(np.random.randn(hidden_dim * inter_dim).astype(np.float16).tolist()),
            }
            records.append(pack_expert(root, key, tensors))

    manifest_path = write_manifest(root, records)
    return ExpertManifest.load(manifest_path)


def main():
    print("==================================================================")
    print(" FLASH-MOE: Qwen MoE A3B Offloading Engine (Windows + OpenVINO)  ")
    print("==================================================================")

    with TemporaryDirectory() as temp_dir:
        model_root = Path(temp_dir)
        num_layers = 4
        num_experts_per_layer = 8
        hidden_dim = 128
        inter_dim = 256
        top_k = 2

        # 1. Caricamento Manifest
        manifest = build_mock_qwen_moe_weights(
            model_root,
            num_layers=num_layers,
            num_experts=num_experts_per_layer,
            hidden_dim=hidden_dim,
            inter_dim=inter_dim,
        )

        # 2. Inizializzazione Store con vincolo di memoria RAM (max 2 esperti residenti)
        store = AsyncExpertStore(model_root, capacity=2, manifest=manifest)

        # 3. Inizializzazione Backend OpenVINO (GPU Intel Arc 130V)
        backend = OpenVINOExpertBackend(store, device="GPU")

        # 4. Router MoE: uno per layer. Su questo benchmark i pesi sono
        # sintetici, quindi il gate e' finto ma deterministico e diverso per
        # layer; sul modello vero i gate vengono da ffn_gate_inp.weight.
        routers = {
            layer: synthetic_router(
                hidden_size=hidden_dim, num_experts=num_experts_per_layer,
                top_k=top_k, seed=7, layer=layer,
            )
            for layer in range(num_layers)
        }

        # 5. Engine MoE con Lookahead Prefetching
        engine = MoEEngine(
            store=store,
            backend=backend,
            routers=routers,
            hidden_size=hidden_dim,
            intermediate_size=inter_dim,
        )

        # 6. Simulazione Generazione 30 Token per Prompt
        num_tokens = 30
        print(f"\n[Simulazione] Inizio generazione di {num_tokens} token sequenziali su Intel Arc GPU...")
        start_time = time.perf_counter()

        # Generiamo attivazioni casuali simulate per ogni token
        np.random.seed(123)
        tokens = [np.random.randn(hidden_dim).astype(np.float32) for _ in range(num_tokens)]

        for i, activation in enumerate(tokens):
            # Per ogni token, iteriamo attraverso i layer del modello
            for layer in range(num_layers):
                # Look-ahead: segnala lo store di pre-caricare l'esperto per il token/layer successivo
                next_key = ExpertKey(layer, (i + 1) % num_experts_per_layer) if i + 1 < num_tokens else None
                out = engine.process_token(layer=layer, activation=activation, next_expert_key=next_key)

        elapsed = time.perf_counter() - start_time
        tokens_per_sec = num_tokens / elapsed
        latency_per_token_ms = (elapsed / num_tokens) * 1000

        print("\n==================================================================")
        print("                  RISULTATI BENCHMARK QWEN MOE                    ")
        print("==================================================================")
        print(f" Tempo Totale Esecuzione:         {elapsed:.4f} s")
        print(f" Throughput Generazione:         {tokens_per_sec:.2f} token/sec")
        print(f" Latenza Media per Token:        {latency_per_token_ms:.2f} ms")
        print(f" Statistiche Cache Store:        Hit={store.stats.hits}, Miss={store.stats.misses}")
        print(f" Dati Letti da SSD:              {store.stats.bytes_read / (1024 * 1024):.2f} MB")
        print(f" Esperti Residenti in RAM:       {len(store.resident_keys())} (Cap massimo = 2)")
        print("==================================================================")

        store.close()

if __name__ == "__main__":
    main()
