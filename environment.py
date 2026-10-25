"""
environment.py
----------------
Simulates a small distributed system of backend servers that receive
incoming requests. A "load balancer" (either a fixed heuristic, or an
RL agent) decides which server each new request is routed to.

This implements the State / Action / Reward / Next-State loop described
in the project plan (Steps 2-4, 10-11).
"""

import numpy as np


class Server:
    """A single simulated backend server."""

    def __init__(self, server_id, capacity=1.0, rng=None):
        self.id = server_id
        self.capacity = capacity        # requests/sec it can comfortably serve
        self.rng = rng if rng is not None else np.random.default_rng()
        self.queue = 0                  # pending requests
        self.load = 0.0                 # smoothed utilization, 0..1+ (>1 = overloaded)
        self.memory = 0.2               # baseline memory usage, 0..1
        self.latency = 20.0             # ms
        self.total_served = 0
        self.degraded_until = 0         # for "server degradation" scenario

    def effective_capacity(self, t):
        if t < self.degraded_until:
            return self.capacity * 0.4   # simulate a slow/degraded server
        return self.capacity

    def add_request(self):
        self.queue += 1

    def step(self, t, dt=1.0):
        """Advance this server by one timestep: serve what it can,
        update utilization/memory/latency."""
        cap = self.effective_capacity(t)
        served = min(self.queue, cap * dt)
        self.queue -= served
        self.total_served += served

        instantaneous_load = served / (cap * dt + 1e-6)
        # exponential moving average -> smoother, more realistic signal
        self.load = 0.7 * self.load + 0.3 * min(instantaneous_load + self.queue / (cap * 5 + 1e-6), 2.0)
        self.memory = np.clip(0.6 * self.memory + 0.4 * (0.15 + 0.5 * self.load) +
                               self.rng.normal(0, 0.01), 0.0, 1.0)
        base = 15.0 / max(cap, 0.1)
        self.latency = base + (self.queue / max(cap, 0.1)) * 40.0 + max(self.load - 1.0, 0) * 100.0
        return served

    def snapshot(self):
        return {
            "cpu": float(np.clip(self.load, 0, 1.5)),
            "memory": float(self.memory),
            "queue": float(self.queue),
            "latency": float(self.latency),
        }


class LoadBalancerEnv:
    """
    RL environment wrapping N servers.

    State  = normalized [cpu, memory, queue, latency] for every server
    Action = index of the server to route the next request to
    Reward = negative cost of routing to that server, plus a small
             bonus for keeping load balanced across the cluster
    """

    def __init__(self, num_servers=4, capacities=None, base_request_rate=6,
                 seed=None, degradation_scenario=False, spike_scenario=False,
                 traffic_weights=None, reward_weights=None,
                 spike_multiplier=2.5, spike_start=150, spike_duration=60):
        self.num_servers = num_servers
        self.rng = np.random.default_rng(seed)
        if capacities is None:
            # heterogeneous cluster by default (Scenario 4 in the plan)
            capacities = np.linspace(0.7, 1.3, num_servers)
        if len(capacities) != num_servers or any(float(c) <= 0 for c in capacities):
            raise ValueError("capacities must contain one positive value per server")
        self.rng = np.random.default_rng(seed)
        self.servers = [Server(i, c, rng=self.rng) for i, c in enumerate(capacities)]
        self.base_request_rate = base_request_rate
        if traffic_weights is None:
            traffic_weights = [1.0] * num_servers
        if len(traffic_weights) != num_servers or any(float(w) < 0 for w in traffic_weights):
            raise ValueError("traffic_weights must contain one non-negative value per server")
        if sum(traffic_weights) == 0:
            raise ValueError("at least one traffic weight must be greater than zero")
        self.traffic_weights = np.asarray(traffic_weights, dtype=np.float64)
        self.traffic_weights /= self.traffic_weights.sum()
        self.reward_weights = reward_weights or (1.0, 0.5, 1.5, 0.5)
        if len(self.reward_weights) != 4 or any(float(w) < 0 for w in self.reward_weights):
            raise ValueError("reward_weights must contain four non-negative values")
        self.spike_multiplier = float(spike_multiplier)
        self.spike_start = int(spike_start)
        self.spike_duration = int(spike_duration)
        self.t = 0
        self.degradation_scenario = degradation_scenario
        self.spike_scenario = spike_scenario
        self.max_queue_norm = 20.0
        self.max_latency_norm = 150.0

    # ---------- core RL API ----------
    def reset(self):
        for s in self.servers:
            s.queue = 0
            s.load = 0.0
            s.memory = 0.2
            s.latency = 20.0
            s.degraded_until = 0
        self.t = 0
        return self._get_state()

    def current_request_rate(self):
        rate = self.base_request_rate
        if self.spike_scenario and self.spike_start <= self.t < self.spike_start + self.spike_duration:
            rate = self.base_request_rate * self.spike_multiplier
        return rate

    def maybe_trigger_degradation(self):
        if self.degradation_scenario and self.t == 120:
            # Server 1 (index 1) suddenly becomes slow for a while
            self.servers[min(1, self.num_servers - 1)].degraded_until = self.t + 100

    def step(self, action):
        """Route ONE request to `action`, advance the whole cluster by
        one timestep, and return (next_state, reward, info)."""
        self.maybe_trigger_degradation()
        if not isinstance(action, (int, np.integer)) or not 0 <= action < self.num_servers:
            raise ValueError(
                f"action must be an integer in [0, {self.num_servers}), got {action!r}"
            )

        chosen = self.servers[action]
        chosen.add_request()

        # every server also receives background traffic proportional to the
        # overall request rate, so the environment keeps evolving even for
        # servers the agent didn't just pick (models a live production system)
        rate = self.current_request_rate()
        extra_count = int(self.rng.poisson(max(rate - 1, 0)))
        extra = self.rng.multinomial(extra_count, self.traffic_weights)
        arrivals = extra.astype(np.int64)
        arrivals[action] += 1
        for s, n in zip(self.servers, extra):
            for _ in range(n):
                s.add_request()

        for s in self.servers:
            s.step(self.t)

        reward, info = self._compute_reward(action, arrivals)
        self.t += 1
        next_state = self._get_state()
        return next_state, reward, info

    # ---------- helpers ----------
    def _get_state(self):
        feats = []
        for s in self.servers:
            snap = s.snapshot()
            feats.extend([
                snap["cpu"],
                snap["memory"],
                min(snap["queue"] / self.max_queue_norm, 1.5),
                min(snap["latency"] / self.max_latency_norm, 1.5),
            ])
        return np.array(feats, dtype=np.float32)

    def _compute_reward(self, action, arrivals=None):
        chosen = self.servers[action]
        latencies = np.array([s.latency for s in self.servers])
        loads = np.array([s.load for s in self.servers])

        norm_latency = chosen.latency / self.max_latency_norm
        norm_queue = chosen.queue / self.max_queue_norm
        overload_penalty = max(chosen.load - 1.0, 0) * 2.0
        balance_bonus = -np.std(loads)          # less spread -> less negative -> better

        alpha, beta, gamma, delta = self.reward_weights
        reward = -(alpha * norm_latency + beta * norm_queue + gamma * overload_penalty) + delta * balance_bonus

        info = {
            "avg_latency": float(np.mean(latencies)),
            "p95_latency": float(np.percentile(latencies, 95)),
            "throughput": float(sum(s.total_served for s in self.servers)),
            "avg_cpu": float(np.mean(loads)),
            "chosen_latency": float(chosen.latency),
            "queues": [float(s.queue) for s in self.servers],
            "arrivals": arrivals.tolist() if arrivals is not None else [0] * self.num_servers,
        }
        return float(reward), info

    def state_dim(self):
        return self.num_servers * 4

    def action_dim(self):
        return self.num_servers
