"""
Traditional load balancing baselines to compare against the RL agent.
"""


class RoundRobinBalancer:
    def __init__(self, n_servers):
        self.n_servers = n_servers
        self._next = 0

    def choose(self, env):
        idx = self._next
        self._next = (self._next + 1) % self.n_servers
        return idx

    def reset(self):
        self._next = 0


class LeastConnectionsBalancer:
    def choose(self, env):
        # pick server with fewest active_requests + queue
        loads = [s.active_requests + s.queue for s in env.servers]
        return int(min(range(len(loads)), key=lambda i: loads[i]))

    def reset(self):
        pass
