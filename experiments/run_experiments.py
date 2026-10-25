"""
Orchestrates:
  1. Training the DQN agent on a normal workload.
  2. Running Round Robin, Least Connections, and the trained DQN across
     several scenarios (normal, high load, traffic spike, heterogeneous
     servers, server degradation).
  3. Collecting metrics for comparison + plotting.
"""
import sys, os, json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "environment"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "baselines"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "agent"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "monitoring"))

import numpy as np
from server_environment import ServerEnvironment
from request_generator import RequestGenerator
from baselines import RoundRobinBalancer, LeastConnectionsBalancer
from dqn_agent import DQNAgent
from metrics_collector import MetricsCollector

N_SERVERS = 4
TICKS_PER_EPISODE = 150
SERVICE_RATE = 3


def make_env(scenario, seed=0):
    if scenario == "heterogeneous":
        capacities = [1.5, 1.0, 0.6, 1.3]
    else:
        capacities = [1.0, 1.0, 1.0, 1.0]
    env = ServerEnvironment(n_servers=N_SERVERS, capacities=capacities, seed=seed)
    if scenario == "degradation":
        pass  # degradation triggered mid-episode
    return env


def run_baseline_episode(balancer, scenario, seed=0):
    env = make_env(scenario, seed)
    gen = RequestGenerator(scenario=scenario, seed=seed)
    balancer.reset()
    collector = MetricsCollector()

    for t in range(TICKS_PER_EPISODE):
        if scenario == "degradation" and t == 60:
            env.set_degraded(1, True)

        n_new = gen.requests_at(t)
        tick_latencies = []
        for _ in range(n_new):
            idx = balancer.choose(env)
            lat = env.route_request(idx)
            tick_latencies.append(lat)

        env.tick(service_rate=SERVICE_RATE)
        avg_tick_latency = float(np.mean(tick_latencies)) if tick_latencies else None
        collector.record(env, latency_this_tick=avg_tick_latency)

    return collector


def train_dqn(episodes=60, scenario="normal", seed=0):
    env = make_env(scenario, seed)
    gen = RequestGenerator(scenario=scenario, seed=seed)
    agent = DQNAgent(env.state_size(), env.action_size(), seed=seed)

    episode_rewards = []
    for ep in range(episodes):
        env.reset()
        gen = RequestGenerator(scenario=scenario, seed=seed + ep)
        ep_reward = 0.0

        for t in range(TICKS_PER_EPISODE):
            state = env.get_state()
            n_new = gen.requests_at(t)
            for _ in range(n_new):
                state = env.get_state()
                action = agent.act(state, explore=True)
                env.route_request(action)
                reward = env.compute_reward(action)
                env.tick(service_rate=0)  # partial tick just to refresh latency calc cheaply
                next_state = env.get_state()
                agent.remember(state, action, reward, next_state, False)
                agent.replay()
                ep_reward += reward
            env.tick(service_rate=SERVICE_RATE)

        agent.decay_epsilon()
        episode_rewards.append(ep_reward)
        if (ep + 1) % 10 == 0:
            print(f"  [DQN training] episode {ep+1}/{episodes} "
                  f"reward={ep_reward:.2f} epsilon={agent.epsilon:.3f}")

    return agent, episode_rewards


def run_dqn_episode(agent, scenario, seed=0):
    env = make_env(scenario, seed)
    gen = RequestGenerator(scenario=scenario, seed=seed)
    collector = MetricsCollector()

    for t in range(TICKS_PER_EPISODE):
        if scenario == "degradation" and t == 60:
            env.set_degraded(1, True)

        n_new = gen.requests_at(t)
        tick_latencies = []
        for _ in range(n_new):
            state = env.get_state()
            action = agent.act(state, explore=False)
            lat = env.route_request(action)
            tick_latencies.append(lat)

        env.tick(service_rate=SERVICE_RATE)
        avg_tick_latency = float(np.mean(tick_latencies)) if tick_latencies else None
        collector.record(env, latency_this_tick=avg_tick_latency)

    return collector


def main():
    scenarios = ["normal", "high", "spike", "heterogeneous", "degradation"]
    results = {}

    print("Training DQN agent on normal workload...")
    agent, training_rewards = train_dqn(episodes=60, scenario="normal", seed=1)
    results["_training_rewards"] = training_rewards

    for scenario in scenarios:
        print(f"\nRunning scenario: {scenario}")
        rr = run_baseline_episode(RoundRobinBalancer(N_SERVERS), scenario, seed=42)
        lc = run_baseline_episode(LeastConnectionsBalancer(), scenario, seed=42)
        dqn = run_dqn_episode(agent, scenario, seed=42)

        results[scenario] = {
            "round_robin": {"summary": rr.summary(), "series": rr.records},
            "least_connections": {"summary": lc.summary(), "series": lc.records},
            "dqn": {"summary": dqn.summary(), "series": dqn.records},
        }
        for name, res in results[scenario].items():
            print(f"  {name:18s} -> {res['summary']}")

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "results", "experiment_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")
    return results


if __name__ == "__main__":
    main()
