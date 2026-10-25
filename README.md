# Adaptive Load Balancing with Deep Reinforcement Learning (DQN)

A working implementation of the RL-based load balancer described in the
project plan: a Deep Q-Network learns to route incoming requests across
a cluster of simulated servers, and is benchmarked against Round Robin,
Least Connections, and Random routing under four traffic scenarios.

## What's included

| File | Purpose |
|---|---|
| `environment.py` | Simulated cluster (`Server`, `LoadBalancerEnv`). Defines State / Action / Reward / Next-State. |
| `dqn_agent.py` | The deep learning core: a 2-hidden-layer neural network (`MLP`), trained with backprop + Adam, plus experience replay and a target network — a full DQN. |
| `baselines.py` | `RoundRobin`, `LeastConnections`, `Random` — the traditional algorithms to compare against. |
| `train.py` | Trains the DQN, then evaluates every policy across 4 scenarios; saves CSVs to `results/`. |
| `visualize.py` | Turns the CSVs into the plots in `plots/`. |
| `run_experiment.py` | One command that runs everything end to end. |

## Running it

```bash
pip install -r requirements.txt
python run_experiment.py
```

This trains for 120 episodes (~300 steps each), evaluates all 4 policies
on all 4 scenarios, and writes:
- `results/dqn_training_curve.csv` — reward per training episode
- `results/scenario_comparison.csv` — per-timestep metrics for every (policy, scenario) pair
- `plots/training_reward_curve.png`
- `plots/latency_over_time.png`, `plots/throughput_over_time.png`
- `plots/summary_avg_latency.png`, `summary_throughput.png`, `summary_avg_cpu.png`

On Windows, if `python` is a runtime without `pip`, use the Python launcher:

```powershell
py -3.14 -m pip install -r requirements.txt
py -3.14 run_experiment.py
```

The repository also contains an alternate, lower-level experiment runner:

```powershell
py -3.14 experiments\run_experiments.py
```

The top-level `run_experiment.py` is the recommended entry point because it
generates the CSV and plot outputs documented above.

## Interactive dashboard

To review and control a simulation in the browser, run:

```powershell
py -3.14 -m streamlit run app.py
```

Then open `http://localhost:8501`. The dashboard lets you change the number
of servers, each server's capacity, traffic rate and per-server traffic
weights, traffic-spike settings, routing policy, simulation length, random
seed, and all four reward weights. It displays latency, throughput, CPU/load,
and per-server queue charts and provides a CSV download for each run.

After a run, click **Play routing animation** to watch each simulated job
travel from the incoming traffic stream to the server selected by the policy.
Every server has its own traffic lane, while the destination lane is
highlighted and updates its queue, load, and latency at every timestep.

## How the RL problem is set up

- **State**: `[cpu, memory, queue, latency]` for every server, normalized.
- **Action**: which server to send the next request to.
- **Reward**: `-(latency + queue + overload_penalty) + balance_bonus`, so the
  agent is pushed toward low-latency, non-overloaded, evenly-loaded servers.
- **Scenarios** (`train.py: SCENARIOS`): `normal`, `heavy_load`,
  `traffic_spike` (a 2.5x burst mid-run), `server_degradation` (one server
  suddenly slows to 40% capacity) — this is what tests adaptability.

Sample result from a run in this environment (yours will vary slightly
with the random seed):

| Scenario | RoundRobin | Random | LeastConnections | **DQN** |
|---|---|---|---|---|
| normal | 148 ms | 120 ms | 56 ms | **47 ms** |
| heavy_load | 997 ms | 943 ms | 387 ms | **379 ms** |
| traffic_spike | 939 ms | 910 ms | 766 ms | **735 ms** |
| server_degradation | 436 ms | 426 ms | 150 ms | **130 ms** |

(avg latency per step, lower is better — DQN matches or beats the
strongest baseline, Least Connections, in every scenario, and clearly
beats Round Robin / Random everywhere.)

## About the "deep learning library" in `dqn_agent.py`

The network is a real multi-layer perceptron trained with backprop and
Adam — a genuine DQN, not a lookup table. It's written directly in NumPy
rather than importing PyTorch/TensorFlow because this sandboxed build
environment doesn't have enough free disk space for the CUDA-bundled
wheels those libraries pull from PyPI (a few GB of NVIDIA packages get
installed even for CPU-only use). On your own machine you have two
options:

1. **Run it as-is** — `dqn_agent.py` has no dependency on torch/TF at all,
   so it just works with `pip install -r requirements.txt`.
2. **Swap in PyTorch** — replace the `MLP` class with an `nn.Module` and
   `DQNAgent.train_step` with the standard PyTorch training loop
   (`optimizer.zero_grad(); loss.backward(); optimizer.step()`). The
   state/action/reward interface in `environment.py` doesn't change at
   all, so `train.py` and `visualize.py` keep working unmodified.

## Extending toward the full plan

This covers plan phases 1–14 (server simulation → baselines → DQN →
scenario experiments → plots). Natural next steps from here:
- Add Double DQN / Dueling DQN and compare against the vanilla DQN here.
- Sweep the reward weights (`alpha, beta, gamma, delta` in
  `environment.py: _compute_reward`) as its own experiment.
- Increase `NUM_SERVERS` / `TRAIN_EPISODES` in `train.py` for a more
  thorough result, and write up `results/*.csv` into the final report
  (Step 15 / Phase 15 of the plan).
