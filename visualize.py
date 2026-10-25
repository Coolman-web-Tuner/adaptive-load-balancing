"""
visualize.py
------------
Turns the CSVs produced by train.py into the plots described in
Step 16-17 of the plan: reward learning curve, latency comparison,
throughput comparison, and per-scenario summaries.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_training_curve():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "dqn_training_curve.csv"))
    plt.figure(figsize=(7, 4))
    plt.plot(df["episode"], df["avg_reward"], color="#2563eb")
    plt.title("DQN Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Average reward per step")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_reward_curve.png"), dpi=150)
    plt.close()


def plot_scenario_metric(df, metric, ylabel, filename):
    scenarios = df["scenario"].unique()
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios):
        sub = df[df["scenario"] == scenario]
        for policy in sub["policy"].unique():
            p = sub[sub["policy"] == policy]
            ax.plot(p["t"], p[metric], label=policy, linewidth=1.5)
        ax.set_title(scenario.replace("_", " ").title())
        ax.set_xlabel("Timestep")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close()


def plot_summary_bars(df):
    summary = df.groupby(["scenario", "policy"])[["avg_latency", "throughput", "avg_cpu"]].mean().reset_index()
    metrics = [("avg_latency", "Avg latency (ms) - lower is better"),
               ("throughput", "Total requests served - higher is better"),
               ("avg_cpu", "Avg CPU utilization - closer to balanced is better")]
    for metric, title in metrics:
        pivot = summary.pivot(index="scenario", columns="policy", values=metric)
        pivot.plot(kind="bar", figsize=(8, 4.5))
        plt.title(title)
        plt.ylabel(metric)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"summary_{metric}.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    plot_training_curve()
    comp = pd.read_csv(os.path.join(RESULTS_DIR, "scenario_comparison.csv"))
    plot_scenario_metric(comp, "avg_latency", "Avg latency (ms)", "latency_over_time.png")
    plot_scenario_metric(comp, "throughput", "Cumulative requests served", "throughput_over_time.png")
    plot_summary_bars(comp)
    print(f"Plots saved under: {PLOTS_DIR}")
