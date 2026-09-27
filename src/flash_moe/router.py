import numpy as np

class ExpertRouter:
    """Qwen MoE Gating network that selects top-k experts and computes normalized Softmax weights."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2, seed: int = 42):
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        np.random.seed(seed)
        self.gate_weights = np.random.randn(hidden_size, num_experts).astype(np.float32)

    def route(self, activation: list[float] | np.ndarray) -> tuple[list[int], list[float]]:
        scores = np.dot(activation, self.gate_weights)
        top_k_indices = np.argsort(scores)[-self.top_k:][::-1].tolist()
        top_k_scores = scores[top_k_indices]

        exp_scores = np.exp(top_k_scores - np.max(top_k_scores))
        weights = (exp_scores / np.sum(exp_scores)).tolist()
        return top_k_indices, weights
