"""
Generates a stream of requests-per-tick according to a workload scenario.
"""
import numpy as np


class RequestGenerator:
    def __init__(self, scenario="normal", seed=None):
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)

    def requests_at(self, t):
        """Return the number of new requests arriving at time step t."""
        if self.scenario == "normal":
            base = 4
        elif self.scenario == "high":
            base = 10
        elif self.scenario == "spike":
            base = 4 if t < 40 else 20 if t < 80 else 4
        elif self.scenario == "heterogeneous":
            base = 6
        elif self.scenario == "degradation":
            base = 6
        else:
            base = 5
        # Poisson noise around the base rate
        return int(self.rng.poisson(lam=max(base, 1)))
