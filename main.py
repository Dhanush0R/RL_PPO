# command line usage:

# python main.py --task ppo                --> runs only PPO
# python main.py --task compare_ppo        --> compares A2C, PPO, and Baseline
# python main.py --task ablation_clip      --> sweeps PPO clipping parameter
# python main.py --task ablation_batch     --> sweeps PPO minibatch size
# python main.py --task ablation_entropy   --> sweeps PPO entropy coefficient
# python main.py --task ablation_anneal    --> sweeps LR and Entropy annealing
# python main.py --task all_ablations      --> runs all 4 ablation studies
# python main.py --task all                --> runs EVERYTHING 

import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import torch
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

try:
    import a2c
    import ppo_naive
    import ppo
except ImportError as e:
    print(f"[warn] Could not import one of the required PG modules: {e}")

def run_pg(module, config, max_steps, seeds, workers):
    def _run(seed):
        torch.set_num_threads(1)
        warnings.filterwarnings("ignore")
        return module.train_agent(config, max_steps, seed, show_progress=False)
    
    return Parallel(n_jobs=workers, backend="loky", verbose=0)(
        delayed(_run)(s) for s in range(seeds)
    )

def plot(results: dict, title: str, filename: str, max_steps: int, outdir: str, window: int):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    palette = sns.color_palette("colorblind", n_colors=len(results))
    fig, ax = plt.subplots(figsize=(9, 5))
    grid = np.linspace(0, max_steps, num=2000)

    for (label, runs), color in zip(results.items(), palette):
        interp_r = []
        for run_data in runs:
            steps = np.array(run_data[0], dtype=float)
            rewards = np.array(run_data[1], dtype=float)
            
            if len(steps) == 0:
                print(f"  [warn] No data recorded for {label} in one of the seeds.")
                continue
                
            u_steps, idx = np.unique(steps, return_index=True)
            interp_r.append(np.interp(grid, u_steps, rewards[idx]))

        if not interp_r: 
            print(f"  [error] Entire run '{label}' failed to generate valid data.")
            continue

        arr = np.array(interp_r)
        sm_mean = pd.Series(arr.mean(0)).rolling(window, min_periods=1).mean().values
        sm_std  = pd.Series(arr.std(0)).rolling(window,  min_periods=1).mean().values

        ax.plot(grid, sm_mean, label=label, linewidth=1.8, color=color)
        ax.fill_between(grid, sm_mean - sm_std, sm_mean + sm_std, alpha=0.15, color=color)

    ax.axhline(500, color="green", linestyle="--", linewidth=1.0, alpha=0.7, label="Max return (500)")
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k" if x >= 1e3 else f"{x:.0f}"
    ))
    ax.set_xlabel("Environment Steps", fontsize=12, labelpad=6)
    ax.set_ylabel("Return (Episode Reward)", fontsize=12, labelpad=6)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlim(0, max_steps)
    ax.set_ylim(bottom=0, top=510)
    
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9, edgecolor="lightgray")
    sns.despine(ax=ax, top=True, right=True)
    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{filename}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved plot → {out_path}")

def summary_table(results: dict, title: str):
    print(f"\n {title} Summary")
    for label, runs in results.items():
        finals = []
        for run_data in runs:
            rewards = run_data[1]
            if len(rewards) > 0:
                finals.append(np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards))
        if finals:
            print(f"  {label:<25}: {np.mean(finals):.1f} ± {np.std(finals):.1f}")
        else:
            print(f"  {label:<25}: FAILED TO RUN")

def add_baseline(results_dict, csv_path):
    try:
        df = pd.read_csv(csv_path)
        b_steps = df["env_step"].tolist()
        b_rewards = df["Episode_Return_smooth"].tolist()
        results_dict["DQN Baseline (CSV)"] = [(b_steps, b_rewards)]
    except FileNotFoundError:
        pass

def run_compare_ppo(args):
    print("\nComparing A2C, PPO, and Baseline...")
    config_a2c = {"gamma": 0.99, "actor_lr": 1e-3, "critic_lr": 2e-3, "h": 128, "entropy_coef": 0.01}
    config_ppo = {"gamma": 0.99, "actor_lr": 3e-4, "critic_lr": 1e-3, "h": 128, "clip_eps": 0.2, "ppo_epochs": 4}
    config_ppo_sota = {"gamma": 0.99, "actor_lr": 3e-4, "critic_lr": 1e-3, "h": 128, "clip_eps": 0.2, "ppo_epochs": 3, "num_envs": 4, "rollout_steps": 128, "minibatch_size": 64, "entropy_coef": 0.01}
    
    res = {}
    res["A2C"] = run_pg(a2c, config_a2c, args.steps, args.seeds, args.workers)
    res["PPO (Naive)"] = run_pg(ppo_naive, config_ppo, args.steps, args.seeds, args.workers)
    res["PPO (SOTA)"] = run_pg(ppo, config_ppo_sota, args.steps, args.seeds, args.workers)
    add_baseline(res, args.baseline_csv)
    
    plot(res, "PPO vs A2C vs DQN Baseline — CartPole-v1", "compare_ppo", args.steps, args.outdir, args.window)
    summary_table(res, "PPO Comparison")

def run_ablation_anneal(args):
    print("\nRunning Ablation: Annealing (Learning Rate & Entropy)...")
    base_cfg = {"gamma": 0.99, "actor_lr": 3e-4, "critic_lr": 1e-3, "h": 128, "clip_eps": 0.2, "ppo_epochs": 3, "num_envs": 4, "rollout_steps": 128, "minibatch_size": 128, "entropy_coef": 0.01}
    res = {}
    
    print("  Testing Static LR & Entropy...")
    cfg_static = base_cfg.copy()
    res["Static (No Decay)"] = run_pg(ppo, cfg_static, args.steps, args.seeds, args.workers)
    
    print("  Testing LR Annealing Only...")
    cfg_lr = base_cfg.copy()
    cfg_lr["anneal_lr"] = True
    res["LR Decay Only"] = run_pg(ppo, cfg_lr, args.steps, args.seeds, args.workers)
    
    print("  Testing LR + Entropy Annealing...")
    cfg_both = base_cfg.copy()
    cfg_both["anneal_lr"] = True
    cfg_both["anneal_entropy"] = True
    res["LR + Entropy Decay"] = run_pg(ppo, cfg_both, args.steps, args.seeds, args.workers)
    
    plot(res, "Ablation: Annealing Strategies", "ablation_anneal", args.steps, args.outdir, args.window)
    summary_table(res, "Annealing Ablation")

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="compare_ppo")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--outdir", type=str, default="plots")
    p.add_argument("--window", type=int, default=150)
    p.add_argument("--baseline_csv", type=str, default="BaselineDataCartPole.csv")
    return p.parse_args()

def main():
    args = get_args()
    print(f"Steps: {args.steps:,} | Seeds: {args.seeds} | Workers: {args.workers}")
    
    if args.task == "compare_ppo": run_compare_ppo(args)
    if args.task == "ablation_anneal": run_ablation_anneal(args)
    
    print("\nExperiments completed!\n")

if __name__ == "__main__":
    main()