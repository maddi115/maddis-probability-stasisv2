import numpy as np
from typing import List, Tuple

class ProbabilityStasis:
    """Compute stability of token/step probabilities."""
    def __init__(self, lambda_instability: float = 1.0, max_keep: int = 3):
        self.lambda_instability = lambda_instability
        self.max_keep = max_keep

    def stasis_score(self, probs: List[float]) -> float:
        p = np.array(probs)
        if len(p) < 2:
            return 0.0
        mean_p = np.mean(p)
        variance = np.var(p)
        prob_range = np.ptp(p)
        instability = variance + prob_range
        return float(mean_p - self.lambda_instability * instability)

    def filter_paths(self, paths: List[Tuple[str, List[float]]]) -> List[Tuple[str, float]]:
        scores = [(n, self.stasis_score(p)) for n, p in paths]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[: self.max_keep]
