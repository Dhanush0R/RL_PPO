# command line usage:

# python main.py --task ppo                --> runs only PPO
# python main.py --task compare_ppo        --> compares A2C, PPO, and Baseline
# python main.py --task ablation_clip      --> sweeps PPO clipping parameter
# python main.py --task ablation_batch     --> sweeps PPO minibatch size
# python main.py --task ablation_entropy   --> sweeps PPO entropy coefficient
# python main.py --task all_ablations      --> runs all 3 ablation studies
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

# reinforce, ac, a2c, ppo imports
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
            
            # Safety check for empty arrays
            if len(steps) == 0:
                print(f"  [warn] No data recorded for {label} in one of the seeds.")
                continue
                
            u_steps, idx = np.unique(steps, return_index=True)
            interp_r.append(np.interp(grid, u_steps, rewards[idx]))

        if not interp_r: 
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
            finals.append(np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards))
        print(f"  {label:<25}: {np.mean(finals):.1f} ± {np.std(finals):.1f}")

def add_baseline(results_dict, csv_path):
    try:
        df = pd.read_csv(csv_path)
        b_steps = df["env_step"].tolist()
        b_rewards = df["Episode_Return_smooth"].tolist()
        results_dict["DQN Baseline (CSV)"] = [(b_steps, b_rewards)]
        print(f"  Loaded baseline from '{csv_path}'")
    except FileNotFoundError:
        print(f"  [warn] Baseline CSV '{csv_path}' not found — skipping baseline.")

def run_ppo_only(args):
    print("\nRunning standalone PPO...")
    config_ppo = {"gamma": 0.99, "actor_lr": 3e-4, "critic_lr": 1e-3, "h": 128, "clip_eps": 0.2, "ppo_epochs": 4}
    res = {}
    res["PPO"] = run_pg(ppo_naive, config_ppo, args.steps, args.seeds, args.workers)
    
    plot(res, "Proximal Policy Optimization (PPO) — CartPole-v1", "ppo_standalone", args.steps, args.outdir, args.window)
    summary_table(res, "PPO Only")

def run_compare_ppo(args):
    print("\nComparing A2C, PPO, and Baseline...")
    config_a2c = {"gamma": 0.99, "actor_lr": 1e-3, "critic_lr": 2e-3, "h": 128, "entropy_coef": 0.01}
    config_ppo = {"gamma": 0.99, "actor_lr": 3e-4, "critic_lr": 1e-3, "h": 128, "clip_eps": 0.2, "ppo_epochs": 4}
    # Adjusted SOTA baseline for more stability: lower entropy, slightly larger minibatches
    config_ppo_sota = {"gamma": 0.99, "actor_lr": 3e-4, "critic_lr": 1e-3, "h": 128, "clip_eps": 0.2, "ppo_epochs": 3, "num_envs": 4, "rollout_steps": 128, "minibatch_size": 128, "entropy_coef": 0.001}
    
    res = {}
    print(f"Running A2C...")
    res["A2C"] = run_pg(a2c, config_a2c, args.steps, args.seeds, args.workers)
    
    print(f"Running PPO (Naive)...")
    res["PPO"] = run_pg(ppo_naive, config_ppo, args.steps, args.seeds, args.workers)
    
    print(f"Running PPO (SOTA)...")
    res["PPO (SOTA)"] = run_pg(ppo, config_ppo_sota, args.steps, args.seeds, args.workers)
    
    add_baseline(res, args.baseline_csv)
    
    plot(res, "PPO vs A2C vs DQN Baseline — CartPole-v1", "compare_ppo", args.steps, args.outdir, args.window)
    summary_table(res, "PPO Comparison")

# --- ABLATION STUDIES ---

def run_ablation_clip(args):
    print("\nRunning Ablation: PPO Clipping Parameter (epsilon)...")
    # Base config: fixed low entropy to isolate clipping effects
    base_cfg = {"gamma": 0.99, "actor_lr": 3e-4, "critic_lr": 1e-3, "h": 128, "ppo_epochs": 3, "num_envs": 4, "rollout_steps": 128, "minibatch_size": 128, "entropy_coef": 0.001}
    res = {}
    
    for clip in [0.1, 0.2, 0.3]:
        cfg = base_cfg.copy()
        cfg["clip_eps"] = clip
        print(f"  Testing clip_eps = {clip}...")
        res[f"PPO (clip={clip})"] = run_pg(ppo, cfg, args.steps, args.seeds, args.workers)
        
    plot(res, "Ablation: PPO Clipping Parameter ($\epsilon$)", "ablation_clip", args.steps, args.outdir, args.window)
    summary_table(res, "Clipping Ablation")

def run_ablation_batch(args):
    print("\nRunning Ablation: PPO Mini-Batch Size...")
    # Base config: fixed clip to isolate batch effects. Note total batch size is 4 * 128 = 512
    base_cfg = {"gamma": 0.99, "actor_lr": 3e-4, "critic_lr": 1e-3, "h": 128, "clip_eps": 0.2, "ppo_epochs": 3, "num_envs": 4, "rollout_steps": 128, "entropy_coef": 0.001}
    res = {}
    
    for mb in [32, 64, 256, 512]:
        cfg = base_cfg.copy()
        cfg["minibatch_size"] = mb
        print(f"  Testing minibatch_size = {mb}...")
        res[f"PPO (mb={mb})"] = run_pg(ppo, cfg, args.steps, args.seeds, args.workers)
        
    plot(res, "Ablation: PPO Mini-Batch Size", "ablation_batch", args.steps, args.outdir, args.window)
    summary_table(res, "Mini-Batch Ablation")

def run_ablation_entropy(args):
    print("\nRunning Ablation: PPO Entropy Regularization Impact...")
    base_cfg = {"gamma": 0.99, "actor_lr": 3e-4, "critic_lr": 1e-3, "h": 128, "clip_eps": 0.2, "ppo_epochs": 3, "num_envs": 4, "rollout_steps": 128, "minibatch_size": 128}
    res = {}
    
    for ent in [0.0, 0.01, 0.05]:
        cfg = base_cfg.copy()
        cfg["entropy_coef"] = ent
        print(f"  Testing entropy_coef = {ent}...")
        res[f"PPO (ent={ent})"] = run_pg(ppo, cfg, args.steps, args.seeds, args.workers)
        
    plot(res, "Ablation: PPO Entropy Coefficient", "ablation_entropy", args.steps, args.outdir, args.window)
    summary_table(res, "Entropy Ablation")


def get_args():
    p = argparse.ArgumentParser(description="PPO Comparison and Ablation Runner")
    p.add_argument("--task", type=str, default="all",
                   choices=["ppo", "compare_ppo", "ablation_clip", "ablation_batch", "ablation_entropy", "all_ablations", "all"])
    p.add_argument("--steps",   type=int,   default=1_000_000)
    p.add_argument("--seeds",   type=int,   default=5)
    p.add_argument("--workers", type=int,   default=5)
    p.add_argument("--outdir",  type=str,   default="plots")
    p.add_argument("--window",  type=int,   default=150)
    p.add_argument("--baseline_csv", type=str, default="BaselineDataCartPole.csv",
                   help="Path to Assignment 2 baseline CSV")
    return p.parse_args()

def main():
    args = get_args()
    print(f"Steps: {args.steps:,} | Seeds: {args.seeds} | Workers: {args.workers}")
    
    # Standard Tasks
    if args.task in ["all", "ppo"]: run_ppo_only(args)
    if args.task in ["all", "compare_ppo"]: run_compare_ppo(args)
    
    # Ablation Tasks
    if args.task in ["all", "all_ablations", "ablation_clip"]: run_ablation_clip(args)
    if args.task in ["all", "all_ablations", "ablation_batch"]: run_ablation_batch(args)
    if args.task in ["all", "all_ablations", "ablation_entropy"]: run_ablation_entropy(args)
    
    print("\nExperiments completed!\n")

if __name__ == "__main__":
    main()