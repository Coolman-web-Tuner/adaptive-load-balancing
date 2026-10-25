"""
run_experiment.py
------------------
One-command entry point: trains the DQN, evaluates it against the
baselines across all scenarios, and generates every plot.

    python run_experiment.py
"""

from train import train_dqn, run_all_scenarios
import visualize

if __name__ == "__main__":
    print("=== Phase 1: training DQN agent ===")
    agent = train_dqn()

    print("\n=== Phase 2: evaluating DQN vs baselines across scenarios ===")
    run_all_scenarios(agent)

    print("\n=== Phase 3: generating plots ===")
    visualize.plot_training_curve()
    import pandas as pd, os
    comp = pd.read_csv(os.path.join(visualize.RESULTS_DIR, "scenario_comparison.csv"))
    visualize.plot_scenario_metric(comp, "avg_latency", "Avg latency (ms)", "latency_over_time.png")
    visualize.plot_scenario_metric(comp, "throughput", "Cumulative requests served", "throughput_over_time.png")
    visualize.plot_summary_bars(comp)

    print(f"\nDone. Results: {visualize.RESULTS_DIR}\nPlots: {visualize.PLOTS_DIR}")
