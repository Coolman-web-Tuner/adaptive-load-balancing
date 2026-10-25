"""
train.py
--------
1. Trains the DQN load-balancing agent (Steps 12-13 of the plan).
2. Evaluates it against Round Robin / Least Connections / Random
   under several scenarios (Step 14): normal load, heavy load,
   traffic spike, heterogeneous servers, server degradation.
3. Saves per-step metrics + reward curve as CSVs so visualize.py can
   plot them.
"""

import os
import numpy as np
import pandas as pd

from environment import LoadBalancerEnv
from dqn_agent import DQNAgent
from baselines import RoundRobin, LeastConnections, Random

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_SERVERS = 4
STEPS_PER_EPISODE = 300
TRAIN_EPISODES = 120


def train_dqn(num_servers=NUM_SERVERS, episodes=TRAIN_EPISODES,
              steps_per_episode=STEPS_PER_EPISODE, seed=0):
    env = LoadBalancerEnv(num_servers=num_servers, base_request_rate=2.8, seed=seed)
    agent = DQNAgent(state_dim=env.state_dim(), action_dim=env.action_dim(), seed=seed)

    reward_history = []
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0.0
        for _ in range(steps_per_episode):
            action = agent.act(state)
            next_state, reward, info = env.step(action)
            done = 0.0
            agent.remember(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            ep_reward += reward
        reward_history.append(ep_reward / steps_per_episode)
        if (ep + 1) % 10 == 0:
            print(f"[train] episode {ep + 1}/{episodes}  "
                  f"avg_reward={reward_history[-1]:.3f}  eps={agent.epsilon():.2f}")

    pd.DataFrame({"episode": range(1, episodes + 1),
                  "avg_reward": reward_history}).to_csv(
        os.path.join(RESULTS_DIR, "dqn_training_curve.csv"), index=False)
    return agent


def run_policy(policy_name, act_fn, env, steps):
    """Runs one policy on one environment instance and logs metrics per step."""
    rows = []
    state = env.reset()
    for t in range(steps):
        action = act_fn(state, env)
        next_state, reward, info = env.step(action)
        rows.append({
            "t": t, "policy": policy_name, "reward": reward,
            "avg_latency": info["avg_latency"], "p95_latency": info["p95_latency"],
            "throughput": info["throughput"], "avg_cpu": info["avg_cpu"],
        })
        state = next_state
    return pd.DataFrame(rows)


SCENARIOS = {
    "normal": dict(base_request_rate=2.8, degradation_scenario=False, spike_scenario=False),
    "heavy_load": dict(base_request_rate=3.9, degradation_scenario=False, spike_scenario=False),
    "traffic_spike": dict(base_request_rate=2.8, degradation_scenario=False, spike_scenario=True),
    "server_degradation": dict(base_request_rate=2.8, degradation_scenario=True, spike_scenario=False),
}


def run_all_scenarios(agent, num_servers=NUM_SERVERS, steps=STEPS_PER_EPISODE, seed=1):
    all_rows = []
    for scenario_name, kwargs in SCENARIOS.items():
        policies = {
            "RoundRobin": RoundRobin(num_servers),
            "LeastConnections": LeastConnections(),
            "Random": Random(num_servers, seed=seed),
            "DQN": None,  # handled specially (greedy, no exploration)
        }
        for policy_name, policy in policies.items():
            env = LoadBalancerEnv(num_servers=num_servers, seed=seed, **kwargs)
            if policy_name == "DQN":
                act_fn = lambda s, e: agent.act(s, greedy=True)
            else:
                act_fn = lambda s, e, p=policy: p.act(s, e)
            df = run_policy(policy_name, act_fn, env, steps)
            df["scenario"] = scenario_name
            all_rows.append(df)
            print(f"[eval] scenario={scenario_name:18s} policy={policy_name:16s} "
                  f"avg_latency={df['avg_latency'].mean():6.2f}  "
                  f"avg_reward={df['reward'].mean():6.3f}")

    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(os.path.join(RESULTS_DIR, "scenario_comparison.csv"), index=False)
    return result


if __name__ == "__main__":
    print("=== Phase 1: training DQN agent ===")
    agent = train_dqn()
    print("\n=== Phase 2: evaluating DQN vs baselines across scenarios ===")
    run_all_scenarios(agent)
    print(f"\nAll results saved under: {RESULTS_DIR}")
