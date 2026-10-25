"""
Collects per-tick metrics from the environment for later analysis/plots.
"""
import numpy as np


class MetricsCollector:
    def __init__(self):
        self.records = []  # list of dicts, one per tick
        self.latencies_all = []
        self.rewards = []

    def record(self, env, latency_this_tick=None, reward=None):
        snap = env.snapshot()
        cpus = [s["cpu"] for s in snap["servers"]]
        queues = [s["queue"] for s in snap["servers"]]
        latencies = [s["latency"] for s in snap["servers"]]
        completed = sum(s["completed"] for s in snap["servers"])

        row = {
            "t": snap["time_step"],
            "avg_cpu": float(np.mean(cpus)),
            "avg_queue": float(np.mean(queues)),
            "avg_latency": float(np.mean(latencies)),
            "load_imbalance": float(np.std(queues)),
            "throughput": completed,
        }
        self.records.append(row)
        if latency_this_tick is not None:
            self.latencies_all.append(latency_this_tick)
        if reward is not None:
            self.rewards.append(reward)

    def summary(self):
        if not self.records:
            return {}
        avg_latency = float(np.mean([r["avg_latency"] for r in self.records]))
        p95 = float(np.percentile(self.latencies_all, 95)) if self.latencies_all else 0.0
        p99 = float(np.percentile(self.latencies_all, 99)) if self.latencies_all else 0.0
        avg_cpu = float(np.mean([r["avg_cpu"] for r in self.records]))
        avg_imbalance = float(np.mean([r["load_imbalance"] for r in self.records]))
        total_throughput = self.records[-1]["throughput"]
        return {
            "avg_latency": round(avg_latency, 2),
            "p95_latency": round(p95, 2),
            "p99_latency": round(p99, 2),
            "avg_cpu_utilization": round(avg_cpu, 2),
            "avg_load_imbalance": round(avg_imbalance, 2),
            "total_throughput": total_throughput,
        }
