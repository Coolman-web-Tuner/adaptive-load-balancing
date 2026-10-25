"""Interactive Streamlit dashboard for the adaptive load-balancing simulator."""

import html
import time

import numpy as np
import pandas as pd
import streamlit as st

from baselines import LeastConnections, Random, RoundRobin
from dqn_agent import DQNAgent
from environment import LoadBalancerEnv


st.set_page_config(page_title="Adaptive Load Balancing", layout="wide")
st.title("Adaptive Load Balancing Simulator")
st.caption("Tune the workload and cluster, then compare routing policies in a simulated run.")


def train_agent(env, episodes, steps, seed):
    agent = DQNAgent(env.state_dim(), env.action_dim(), seed=seed)
    for episode in range(episodes):
        state = env.reset()
        for _ in range(steps):
            action = agent.act(state)
            next_state, reward, _ = env.step(action)
            agent.remember(state, action, reward, next_state, False)
            agent.train_step()
            state = next_state
    return agent


def simulate(env, policy_name, steps, seed, training_episodes):
    if policy_name == "DQN":
        agent = train_agent(env, training_episodes, min(steps, 150), seed)
        choose = lambda state: agent.act(state, greedy=True)
    elif policy_name == "Round Robin":
        policy = RoundRobin(env.num_servers)
        choose = lambda state: policy.act(state, env)
    elif policy_name == "Least Connections":
        policy = LeastConnections()
        choose = lambda state: policy.act(state, env)
    else:
        policy = Random(env.num_servers, seed=seed)
        choose = lambda state: policy.act(state, env)

    rows = []
    state = env.reset()
    for t in range(steps):
        action = choose(state)
        state, reward, info = env.step(action)
        row = {"timestep": t, "reward": reward, "selected_server": action}
        row.update({key: info[key] for key in ("avg_latency", "p95_latency", "throughput", "avg_cpu")})
        row.update({f"queue_{i + 1}": queue for i, queue in enumerate(info["queues"])})
        row.update({f"arrivals_{i + 1}": count for i, count in enumerate(info["arrivals"])})
        row.update({f"server_{i + 1}_latency": server.latency for i, server in enumerate(env.servers)})
        row.update({f"server_{i + 1}_cpu": server.load for i, server in enumerate(env.servers)})
        rows.append(row)
    return pd.DataFrame(rows)


def render_cluster_frame(row, server_count, frame, total_frames):
    """Render one routing timestep with a traffic lane for every server."""
    selected = int(row["selected_server"])
    queue_columns = [f"queue_{i + 1}" for i in range(server_count)]
    server_lanes = []
    for index, queue_column in enumerate(queue_columns):
        queue = float(row[queue_column])
        load = float(row[f"server_{index + 1}_cpu"])
        latency = float(row[f"server_{index + 1}_latency"])
        is_selected = index == selected
        status = "ROUTED HERE" if is_selected else "READY"
        status_class = "selected" if is_selected else ""
        arrivals = int(row[f"arrivals_{index + 1}"])
        job_count = min(arrivals, 12)
        jobs = "".join(
            '<span class="job-dot">⚡</span>' if item == 0 and is_selected
            else '<span class="job-dot">•</span>'
            for item in range(job_count)
        )
        if arrivals > job_count:
            jobs += f'<span class="job-more">+{arrivals - job_count}</span>'
        server_lanes.append(
            f"""
            <div class="server-lane {status_class}">
              <div class="lane-label">
                <b>Server {index + 1}</b><span>{status}</span>
              </div>
              <div class="lane-track">
                <div class="traffic-source">📥</div>
                <div class="lane-line"><div class="jobs">{jobs}</div></div>
                <div class="server-end">🖥️</div>
              </div>
              <div class="lane-stats">
                Queue <b>{queue:.0f}</b> · Load <b>{load:.2f}</b> · Latency <b>{latency:.1f} ms</b>
              </div>
              <div class="bar"><i style="width:{min(load / 2 * 100, 100):.1f}%"></i></div>
            </div>
            """
        )

    progress = (frame + 1) / max(total_frames, 1) * 100
    return f"""
    <style>
      .animation-shell {{font-family: sans-serif; border: 1px solid #d9e2ec; border-radius: 12px;
        padding: 18px; background: linear-gradient(110deg, #f8fbff, #eef6ff);}}
      .animation-header {{display:flex; justify-content:space-between; align-items:center;
        margin-bottom:14px; color:#183b56;}}
      .animation-header b {{font-size:1.1rem;}}
      .server-lanes {{display:flex; flex-direction:column; gap:10px;}}
      .server-lane {{background:white; border:2px solid #bcccdc; border-radius:10px; padding:10px 12px;
        box-shadow:0 2px 5px #d9e2ec;}}
      .server-lane.selected {{border-color:#f59e0b; box-shadow:0 0 0 3px #fde68a;}}
      .lane-label {{display:flex; justify-content:space-between; color:#243b53; margin-bottom:5px;}}
      .lane-label span {{font-size:.65rem; color:#829ab1;}}
      .selected .lane-label span {{color:#b45309; font-weight:700;}}
      .lane-track {{display:flex; align-items:center; gap:8px;}}
      .traffic-source,.server-end {{font-size:1.2rem; width:28px; text-align:center;}}
      .lane-line {{height:6px; flex:1; background:#9fb3c8; position:relative; border-radius:5px;}}
      .jobs {{position:absolute; top:-13px; left:{progress:.1f}%; transition:left .18s ease;
        display:flex; gap:2px; align-items:center;}}
      .job-dot {{background:#2f80ed; color:white; border:2px solid white; box-shadow:0 1px 4px #829ab1;
        width:23px; height:23px; border-radius:50%; text-align:center; line-height:19px; font-size:.7rem;}}
      .selected .job-dot:first-child {{background:#f59e0b;}}
      .job-more {{font-size:.7rem; color:#52606d; margin-left:3px; white-space:nowrap;}}
      .job {{position:absolute; top:-11px; left:{progress:.1f}%; transition:left .18s ease;
        background:#f59e0b; border:3px solid white; box-shadow:0 1px 5px #829ab1;
        width:28px; height:28px; border-radius:50%; text-align:center; line-height:23px;}}
      .lane-stats {{font-size:.78rem; color:#52606d; margin:6px 0 4px 36px;}}
      .bar {{height:7px; background:#e4e7eb; border-radius:6px; overflow:hidden;}}
      .bar i {{display:block; height:100%; background:#2f80ed; border-radius:6px;}}
      .selected .bar i {{background:#f59e0b;}}
      .progress {{height:6px; margin-top:14px; background:#d9e2ec; border-radius:4px;}}
      .progress i {{display:block; width:{progress:.1f}%; height:100%; background:#2f80ed; border-radius:4px;}}
    </style>
    <div class="animation-shell">
      <div class="animation-header">
        <b>Live job routing across all servers</b>
        <span>timestep {int(row["timestep"]) + 1} / {total_frames} · highlighted job routed to Server {selected + 1}</span>
      </div>
      <div class="server-lanes">{"".join(server_lanes)}</div>
      <div class="progress"><i></i></div>
    </div>
    """


with st.sidebar:
    st.header("Simulation controls")
    server_count = st.slider("Number of servers", 2, 10, 4)
    steps = st.slider("Simulation timesteps", 20, 500, 150, 10)
    base_rate = st.slider("Base traffic rate", 0.0, 20.0, 2.8, 0.1)
    traffic_size = st.slider("Request size multiplier", 0.1, 5.0, 1.0, 0.1)
    policy_name = st.selectbox(
        "Routing policy",
        ["DQN", "Least Connections", "Round Robin", "Random"],
    )
    scenario = st.selectbox("Traffic scenario", ["Normal", "Traffic spike", "Server degradation"])
    seed = st.number_input("Random seed", min_value=0, value=7, step=1)

    st.subheader("Traffic distribution weights")
    traffic_weights = [
        st.number_input(f"Server {i + 1} traffic weight", 0.0, 100.0, 1.0, 0.1, key=f"traffic_{i}")
        for i in range(server_count)
    ]
    st.subheader("Server capacities")
    capacities = [
        st.number_input(f"Server {i + 1} capacity", 0.1, 10.0, 1.0, 0.1, key=f"capacity_{i}")
        for i in range(server_count)
    ]

    spike_multiplier = 1.0
    spike_start = 0
    spike_duration = 0
    if scenario == "Traffic spike":
        spike_multiplier = st.slider("Spike multiplier", 1.0, 10.0, 2.5, 0.1)
        spike_start = st.slider("Spike starts at timestep", 0, max(0, steps - 1), min(50, max(0, steps - 1)))
        spike_duration = st.slider("Spike duration", 1, steps, min(60, steps))

    st.subheader("Reward weights")
    reward_weights = (
        st.slider("Latency weight", 0.0, 5.0, 1.0, 0.1),
        st.slider("Queue weight", 0.0, 5.0, 0.5, 0.1),
        st.slider("Overload weight", 0.0, 5.0, 1.5, 0.1),
        st.slider("Balance weight", 0.0, 5.0, 0.5, 0.1),
    )
    training_episodes = st.slider("DQN training episodes", 1, 100, 10) if policy_name == "DQN" else 0
    run = st.button("Run simulation", type="primary", use_container_width=True)


if run:
    if sum(traffic_weights) <= 0:
        st.error("At least one traffic weight must be greater than zero.")
        st.stop()

    scenario_kwargs = {
        "degradation_scenario": scenario == "Server degradation",
        "spike_scenario": scenario == "Traffic spike",
        "spike_multiplier": spike_multiplier,
        "spike_start": spike_start,
        "spike_duration": spike_duration,
    }
    env = LoadBalancerEnv(
        num_servers=server_count,
        capacities=capacities,
        base_request_rate=base_rate * traffic_size,
        traffic_weights=traffic_weights,
        reward_weights=reward_weights,
        seed=int(seed),
        **scenario_kwargs,
    )
    with st.spinner("Running simulation..."):
        results = simulate(env, policy_name, steps, int(seed), training_episodes)

    st.session_state["results"] = results
    st.session_state["config"] = {
        "policy": policy_name,
        "servers": server_count,
        "scenario": scenario,
    }


if "results" in st.session_state:
    results = st.session_state["results"]
    config = st.session_state["config"]
    st.success(
        f"Completed {len(results)} timesteps using {config['policy']} with "
        f"{config['servers']} servers ({config['scenario']})."
    )
    metrics = st.columns(4)
    metrics[0].metric("Average latency", f"{results['avg_latency'].mean():.1f} ms")
    metrics[1].metric("P95 latency", f"{results['p95_latency'].mean():.1f} ms")
    metrics[2].metric("Requests served", f"{results['throughput'].iloc[-1]:.0f}")
    metrics[3].metric("Average CPU/load", f"{results['avg_cpu'].mean():.2f}")

    st.subheader("Cluster performance")
    st.line_chart(results.set_index("timestep")[["avg_latency", "p95_latency"]])
    st.line_chart(results.set_index("timestep")[["throughput", "avg_cpu"]])

    queue_columns = [column for column in results if column.startswith("queue_")]
    st.subheader("Per-server queues")
    st.line_chart(results.set_index("timestep")[queue_columns])

    arrival_columns = [column for column in results if column.startswith("arrivals_")]
    st.subheader("Incoming jobs per server")
    st.line_chart(results.set_index("timestep")[arrival_columns])

    cpu_columns = [column for column in results if column.startswith("server_") and column.endswith("_cpu")]
    st.subheader("Per-server load")
    st.line_chart(results.set_index("timestep")[cpu_columns])

    st.subheader("Job balancing animation")
    st.caption("Watch each incoming job move through the load balancer to the selected server.")
    animation_speed = st.slider("Animation speed (steps/second)", 1, 20, 6)
    play_animation = st.button("▶ Play routing animation", use_container_width=True)
    animation_frame = st.empty()
    if play_animation:
        for frame, (_, row) in enumerate(results.iterrows()):
            animation_frame.html(
                render_cluster_frame(row, config["servers"], frame, len(results)),
            )
            time.sleep(1 / animation_speed)
        st.success("Animation complete. Run the simulation again or replay it here.")

    st.download_button(
        "Download simulation CSV",
        results.to_csv(index=False).encode("utf-8"),
        "simulation_results.csv",
        "text/csv",
    )
else:
    st.info("Adjust the controls and click Run simulation to generate charts.")
