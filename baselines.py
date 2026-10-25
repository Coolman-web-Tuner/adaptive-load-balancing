"""
baselines.py
------------
Traditional load-balancing algorithms used as a comparison point
against the RL agent (Step 9 of the plan).
"""


class RoundRobin:
    def __init__(self, num_servers):
        self.num_servers = num_servers
        self.next_idx = 0

    def act(self, state=None, env=None):
        a = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.num_servers
        return a


class LeastConnections:
    """Routes to whichever server currently has the shortest queue."""

    def act(self, state=None, env=None):
        queues = [s.queue for s in env.servers]
        return int(min(range(len(queues)), key=lambda i: queues[i]))


class Random:
    def __init__(self, num_servers, seed=0):
        import numpy as np
        self.rng = np.random.default_rng(seed)
        self.num_servers = num_servers

    def act(self, state=None, env=None):
        return int(self.rng.integers(0, self.num_servers))
