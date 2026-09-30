import numpy as np

class ExpertRouter:
    """Qwen MoE Gating network that selects top-k experts and computes normalized Softmax weights."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2, seed: int = 42):
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        np.random.seed(seed)
        self.gate_weights = np.random.randn(hidden_size, num_experts).astype(np.float16)

    def route(self, activation: list[float] | np.ndarray) -> tuple[list[int], list[float]]:
        # Gate e pesi in f16, indici e softmax in f32: il confronto dei punteggi
        # in f16 puo'pare due expert su 256 e sceglierne uno a caso.
        scores = np.dot(np.asarray(activation, dtype=np.float16), self.gate_weights).astype(np.float32)
        top_k_indices = np.argsort(scores)[-self.top_k:][::-1].tolist()
        top_k_scores = scores[top_k_indices]

        exp_scores = np.exp(top_k_scores - np.max(top_k_scores))
        weights = (exp_scores / np.sum(exp_scores)).tolist()
        return top_k_indices, weights
