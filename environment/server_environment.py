"""
Server simulation environment.

Each Server has: cpu (%), memory (%), queue (# pending requests),
active_requests, and derives latency/throughput from those.

This is a discrete-time simulation: at every tick we (a) let running
requests make progress / finish, (b) route new incoming requests to a
server, and (c) recompute each server's stats.
"""
import numpy as np


class Server:
    def __init__(self, server_id, capacity=1.0, base_latency=20.0):
        """
        capacity: relative processing power (1.0 = normal, 0.5 = weak, 1.5 = powerful)
        base_latency: latency (ms) for a single request with an empty queue
        """
        self.id = server_id
        self.capacity = capacity
        self.base_latency = base_latency

        self.cpu = 5.0          # %
        self.memory = 20.0      # %
        self.queue = 0          # pending requests
        self.active_requests = 0
        self.completed = 0
        self.total_latency_recorded = 0.0
        self.degraded = False   # used for "server degradation" scenario

    def receive_request(self):
        self.queue += 1
        self.active_requests += 1

    def tick(self, service_rate=3):
        """
        Advance the server by one time step:
        - process up to `service_rate` requests from the queue (scaled by capacity)
        - update cpu/memory load based on how busy the server is
        """
        effective_rate = max(1, int(round(service_rate * self.capacity)))
        if self.degraded:
            effective_rate = max(1, effective_rate // 3)

        processed = min(self.queue, effective_rate)
        self.queue -= processed
        self.active_requests = max(0, self.active_requests - processed)
        self.completed += processed

        # CPU/memory rise with queue+active load, decay when idle
        load_signal = self.active_requests + self.queue
        target_cpu = min(100.0, 8.0 * load_signal / max(self.capacity, 0.1))
        target_mem = min(100.0, 5.0 * load_signal / max(self.capacity, 0.1) + 15.0)
        if self.degraded:
            target_cpu = min(100.0, target_cpu * 1.8)

        self.cpu += (target_cpu - self.cpu) * 0.4
        self.memory += (target_mem - self.memory) * 0.3
        self.cpu = float(np.clip(self.cpu, 0, 100))
        self.memory = float(np.clip(self.memory, 0, 100))

    def current_latency(self):
        """Estimated latency (ms) for a request arriving right now."""
        queue_penalty = self.queue * (15.0 / max(self.capacity, 0.1))
        cpu_penalty = (self.cpu / 100.0) * 40.0
        latency = self.base_latency + queue_penalty + cpu_penalty
        if self.degraded:
            latency *= 2.5
        return latency

    def is_overloaded(self, cpu_threshold=85.0, queue_threshold=25):
        return self.cpu >= cpu_threshold or self.queue >= queue_threshold

    def stats(self):
        return {
            "id": self.id,
            "cpu": round(self.cpu, 2),
            "memory": round(self.memory, 2),
            "queue": self.queue,
            "active_requests": self.active_requests,
            "latency": round(self.current_latency(), 2),
            "completed": self.completed,
        }


class ServerEnvironment:
    """
    Holds N servers, accepts routed requests, advances simulation time,
    and exposes state/metrics used by both baselines and the RL agent.
    """

    def __init__(self, n_servers=4, capacities=None, seed=None):
        self.n_servers = n_servers
        if capacities is None:
            capacities = [1.0] * n_servers
        self.servers = [Server(i, capacity=capacities[i]) for i in range(n_servers)]
        self.time_step = 0
        self.rng = np.random.default_rng(seed)
        self.total_requests_sent = 0

    def reset(self):
        for s in self.servers:
            s.__init__(s.id, capacity=s.capacity, base_latency=s.base_latency)
        self.time_step = 0
        self.total_requests_sent = 0
        return self.get_state()

    def route_request(self, server_index):
        """Send one request to a chosen server; returns latency observed."""
        server = self.servers[server_index]
        latency_before = server.current_latency()
        server.receive_request()
        self.total_requests_sent += 1
        return latency_before

    def tick(self, service_rate=3):
        for s in self.servers:
            s.tick(service_rate=service_rate)
        self.time_step += 1

    def get_state(self):
        """
        Flat normalized state vector for the RL agent:
        [cpu_i/100, mem_i/100, queue_i/50, latency_i/300] for each server i
        """
        state = []
        for s in self.servers:
            state.extend([
                s.cpu / 100.0,
                s.memory / 100.0,
                min(s.queue, 50) / 50.0,
                min(s.current_latency(), 300.0) / 300.0,
            ])
        return np.array(state, dtype=np.float32)

    def state_size(self):
        return self.n_servers * 4

    def action_size(self):
        return self.n_servers

    def compute_reward(self, chosen_index, alpha=1.0, beta=0.6, gamma=2.0):
        """
        Reward = -(alpha * normalized_latency + beta * normalized_queue_imbalance
                    + gamma * overload_penalty)
        Encourages low latency, balanced queues, and avoiding overloaded servers.
        """
        chosen = self.servers[chosen_index]
        latencies = np.array([s.current_latency() for s in self.servers])
        queues = np.array([s.queue for s in self.servers])

        norm_latency = chosen.current_latency() / 300.0
        imbalance = float(np.std(queues)) / 25.0
        overload_penalty = 1.0 if chosen.is_overloaded() else 0.0

        reward = -(alpha * norm_latency + beta * imbalance + gamma * overload_penalty)
        # small bonus for picking the least-loaded reasonable server
        if chosen_index == int(np.argmin(latencies)):
            reward += 0.15
        return float(reward)

    def snapshot(self):
        return {
            "time_step": self.time_step,
            "servers": [s.stats() for s in self.servers],
        }

    def set_degraded(self, server_index, degraded=True):
        self.servers[server_index].degraded = degraded
