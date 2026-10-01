import numpy as np
from typing import Any
from .key import ExpertKey
from .store import AsyncExpertStore
from .router import ExpertRouter
from .moe_layer import run_moe_layer

class MoEEngine:
    """Flash-MoE Inference Engine managing layer execution, expert routing and async look-ahead prefetching."""

    def __init__(
        self,
        store: AsyncExpertStore,
        backend: Any,
        routers: dict[int, ExpertRouter],
        hidden_size: int,
        intermediate_size: int,
        shared_expert: Any = None,
    ):
        self.store = store
        self.backend = backend
        # Un router per layer: i gate ffn_gate_inp.weight cambiano in ogni
        # strato. Un solo router condiviso selezionerebbe lo stesso set di
        # expert in tutti i 40 layer e il routing non sarebbe quello del
        # modello.
        self.routers = routers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # Shared expert: se assente (None) il modello gira senza, ma su
        # Qwen3/Qwen3.5-MoE la sua assenza cambia l'output di ogni token.
        self.shared_expert = shared_expert

    def router_for(self, layer: int) -> ExpertRouter:
        try:
            return self.routers[layer]
        except KeyError as error:
            raise KeyError(
                f"nessun router per il layer {layer}: ffn_gate_inp.weight "
                f"mancante, gli expert da eseguire non sono determinabili"
            ) from error

    def process_token(
        self,
        layer: int,
        activation: list[float] | np.ndarray,
        next_expert_key: ExpertKey | None = None,
    ) -> list[float]:
        # 1. Hint opzionale del chiamante. Non serve perche' il routing qui
        # sotto e' gia' noto: si mantiene solo per chi vuole anticipare un
        # expert di un altro layer a piacere. I runner storici passavano
        # ExpertKey(layer, (i+1) % n), cioe' un indice inventato che non
        # corrispondeva a nessun expert che il layer avrebbe usato.
        if next_expert_key is not None:
            self.store.prefetch(next_expert_key)

        # 2-5. Routing, esecuzione degli expert e contributo dello shared
        # expert: calcolati in flash_moe.moe_layer, l'unica implementazione.
        # TransformerBlock chiama la stessa funzione, cosi' i due percorsi non
        # possono divergere.
        combined_output = run_moe_layer(
            layer=layer,
            activation=activation,
            router=self.router_for(layer),
            backend=self.backend,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            shared_expert=self.shared_expert,
        )

        return combined_output.tolist()
