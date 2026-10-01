import numpy as np
from typing import Any
from .key import ExpertKey
from .store import AsyncExpertStore
from .router import ExpertRouter

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

        # 2. Select top-k experts and compute routing weights
        expert_ids, weights = self.router_for(layer).route(activation)
        keys = [ExpertKey(layer, expert_id) for expert_id in expert_ids]

        # 3. Accoda tutta la decisione di routing in un colpo. Con 4 worker
        # le 8 letture girano in parallelo invece che in fila: 5,7 ms contro
        # 19,2 ms a cache svuotata. Non e' un prefetch a due layer di
        # distanza, perche' quello non puo' esistere: E_{L+1} dipende
        # dall'output di E_L. E' il parallelismo interno al layer.
        self.store.prefetch_many(keys)

        # 4. Execute active experts and compute weighted output sum
        combined_output = np.zeros(self.hidden_size, dtype=np.float32)
        for key, weight in zip(keys, weights):
            # No store.get() here: it would concatenate the whole expert just
            # to throw it away. backend.forward() -> store.read_tensor() takes
            # the prefetch path and only pulls the tensor it needs.
            expert_out = self.backend.forward(key, activation, self.hidden_size, self.intermediate_size)
            combined_output += weight * np.asarray(expert_out, dtype=np.float32)

        # 5. Contributo dello shared expert: NON e' uno degli 8 selezionati dal
        # router, e ogni token lo attraversa. Va sommato dopo gli expert, non
        # pesato: si somma al totale, gia' scalato dal suo gate sigmoid.
        if self.shared_expert is not None:
            shared_out = self.shared_expert.forward(layer, activation)
            if shared_out is not None:
                combined_output += shared_out

        return combined_output.tolist()
