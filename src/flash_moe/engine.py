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
        router: ExpertRouter,
        hidden_size: int,
        intermediate_size: int,
    ):
        self.store = store
        self.backend = backend
        self.router = router
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

    def process_token(
        self,
        layer: int,
        activation: list[float] | np.ndarray,
        next_expert_key: ExpertKey | None = None,
    ) -> list[float]:
        # 1. Trigger look-ahead prefetch for next token/expert
        if next_expert_key is not None:
            self.store.prefetch(next_expert_key)

        # 2. Select top-k experts and compute routing weights
        expert_ids, weights = self.router.route(activation)

        # 3. Execute active experts and compute weighted output sum
        combined_output = np.zeros(self.hidden_size, dtype=np.float32)
        for expert_id, weight in zip(expert_ids, weights):
            key = ExpertKey(layer, expert_id)
            # No store.get() here: it would concatenate the whole expert just
            # to throw it away. backend.forward() -> store.read_tensor() takes
            # the prefetch path and only pulls the tensor it needs.
            expert_out = self.backend.forward(key, activation, self.hidden_size, self.intermediate_size)
            combined_output += weight * np.asarray(expert_out, dtype=np.float32)

        return combined_output.tolist()
