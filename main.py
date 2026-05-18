# To run the algorithm comparison (Task 1): python main.py --task all_algos
# To run the minibatch ablation (Task 2): python main.py --task ablation_batch
# To run the entropy ablation (Task 2): python main.py --task ablation_entropy
# To run the PPO step-by-step optimization (Task 3): python main.py --task ppo_opt
# To run all tasks sequentially: python main.py --task all
# python main.py --task quick ---> Runs a very quick smoke test of the All Algorithms comparison and PPO optimizations with reduced steps and seeds.

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

# importing all agorithms 
try:
    import reinforce
    import ac
    import a2c
    import dqn
    import ppo
except ImportError as e:
    print(f"[warn] Could not import one of the required modules: {e}")

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
    print(f"  saved plot in : {out_path}")

def summary_table(results: dict, title: str):
    print(f"\n {title} Summary")
    for label, runs in results.items():
        finals = []
        for run_data in runs:
            rewards = run_data[1]
            if len(rewards) > 0:
                finals.append(np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards))
        if finals:
            print(f"  {label:<35}: {np.mean(finals):.1f} ± {np.std(finals):.1f}")
        else:
            print(f"  {label:<35}: FAILED TO RUN")

# task 1: plots all algorithms 
def run_all_algos(args):
    print("\ntraining and comparing all algorithms...")
    
    cfg_reinforce = {"gamma": 0.99, "lr": 1e-3, "h": 128}
    cfg_ac = {"gamma": 0.99, "actor_lr": 1e-4, "critic_lr": 1e-3, "h": 128, "entropy_coef": 0.01}
    cfg_a2c = {"gamma": 0.99, "actor_lr": 1e-3, "critic_lr": 2e-3, "h": 128, "entropy_coef": 0.01}
    cfg_dqn = {"tn_switch": True, "er_switch": True, "lr": 1e-4, "h": 256, "gamma": 0.99, "main_net_update_freq": 1, "e_decay": 0.9999, "e_start": 1.0, "e_min": 0.01, "start_learning_at": 5000, "target_u_freq": 5000, "buff_sz": 100000}
    cfg_ppo = {"gamma": 0.99, "actor_lr": 3e-4, "critic_lr": 1e-3, "h": 128, "clip_eps": 0.2, "ppo_epochs": 4, "num_envs": 4, "rollout_steps": 128, "minibatch_size": 64, "entropy_coef": 0.0}

    res = {}
    res["REINFORCE"] = run_pg(reinforce, cfg_reinforce, args.steps, args.seeds, args.workers)
    res["Actor-Critic (AC)"] = run_pg(ac, cfg_ac, args.steps, args.seeds, args.workers)
    res["A2C"] = run_pg(a2c, cfg_a2c, args.steps, args.seeds, args.workers)
    res["DQN (TN + ER)"] = run_pg(dqn, cfg_dqn, args.steps, args.seeds, args.workers)
    res["PPO (Clipped)"] = run_pg(ppo, cfg_ppo, args.steps, args.seeds, args.workers)
    
    plot(res, "Performance Comparison of All Algorithms — CartPole-v1", "all_algorithms", args.steps, args.outdir, args.window)
    summary_table(res, "All Algorithms Comparison")

# task 2: ablation studies on PPO minibatch size and entropy coefficient
def run_ablation_batch(args):
    print("\nrunning ablation: PPO minibatch Size")
    base_cfg = {"gamma": 0.99, "actor_lr": 3e-4, "critic_lr": 1e-3, "h": 128, "clip_eps": 0.2, "ppo_epochs": 4, "num_envs": 4, "rollout_steps": 128, "entropy_coef": 0.01}
    
    batch_sizes = [64, 128, 256, 512]
    res = {}
    for size in batch_sizes:
        cfg = base_cfg.copy()
        cfg["minibatch_size"] = size
        label = f"Minibatch Size: {size}"
        print(f"  testing {label}")
        res[label] = run_pg(ppo, cfg, args.steps, args.seeds, args.workers)
        
    plot(res, "PPO Ablation: Minibatch Size", "ablation_batch", args.steps, args.outdir, args.window)
    summary_table(res, "Minibatch Ablation")

def run_ablation_entropy(args):
    print("\nrunning ablation: PPO entropy coefficient")
    base_cfg = {"gamma": 0.99, "actor_lr": 3e-4, "critic_lr": 1e-3, "h": 128, "clip_eps": 0.2, "ppo_epochs": 4, "num_envs": 4, "rollout_steps": 128, "minibatch_size": 64}
    
    entropy_coeffs = [0.0, 0.01, 0.05, 0.1]
    res = {}
    for coef in entropy_coeffs:
        cfg = base_cfg.copy()
        cfg["entropy_coef"] = coef
        label = f"Entropy Coef: {coef}"
        print(f"  testing {label}")
        res[label] = run_pg(ppo, cfg, args.steps, args.seeds, args.workers)
        
    plot(res, "PPO Ablation: Entropy Coefficient", "ablation_entropy", args.steps, args.outdir, args.window)
    summary_table(res, "Entropy Ablation")

#task 3: PPO optimization breakdown
def run_optimization_comparison(args):
    print("\nrunning PPO optimization breakdown")
    base_cfg = {"gamma": 0.99, "actor_lr": 3e-4, "critic_lr": 1e-3, "h": 128, "clip_eps": 0.2, "ppo_epochs": 4, "num_envs": 4, "rollout_steps": 128}
    
    res = {}
    
    # naive PPO (no minibatch, no norm)
    print("  testing naive PPO")
    cfg_1 = base_cfg.copy()
    # 4 envs * 128 steps = 512 (one full batch) -->
    cfg_1["minibatch_size"] = 512
    cfg_1["norm_adv"] = False
    cfg_1["entropy_coef"] = 0.0
    res["1. Naive PPO (Clip + GAE)"] = run_pg(ppo, cfg_1, args.steps, args.seeds, args.workers)
    
    # naive + minibatch
    print("  testing naive PPO + minibatch")
    cfg_2 = base_cfg.copy()
    cfg_2["minibatch_size"] = 64
    cfg_2["norm_adv"] = False
    cfg_2["entropy_coef"] = 0.0
    res["2. Naive + Minibatch"] = run_pg(ppo, cfg_2, args.steps, args.seeds, args.workers)
    
    # naive + minibatch + normalization
    print("  testing naive PPO + minibatch + normalization")
    cfg_3 = base_cfg.copy()
    cfg_3["minibatch_size"] = 64
    cfg_3["norm_adv"] = True
    cfg_3["entropy_coef"] = 0.0
    res["3. Naive + Minibatch + Norm"] = run_pg(ppo, cfg_3, args.steps, args.seeds, args.workers)
    

    plot(res, "PPO Optimizations Comparison", "ppo_optimizations", args.steps, args.outdir, args.window)
    summary_table(res, "PPO Optimizations Breakdown")

def run_quick(args):
    print("\n\nQuick Test\n\n")
    
    # 1. override arguments for a fast test
    original_outdir = args.outdir
    args.steps = 10_000
    args.seeds = 2
    args.workers = 2
    args.outdir = os.path.join(args.outdir, "quick")
    
    print(f"  Running Quick Test: {args.steps:,} steps, {args.seeds} seeds...")
    
    print("\n(quick) testing all algorithms comparison")
    run_all_algos(args)
    print("\n(quick) testing PPO optimizations")
    run_optimization_comparison(args)
    print(f"\n(quick) test done. plots successfully saved to '{args.outdir}/'")

    args.outdir = original_outdir

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="all", choices=["all_algos", "ablation_batch", "ablation_entropy", "ppo_opt", "all", "quick"])
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--outdir", type=str, default="plots")
    p.add_argument("--window", type=int, default=150)
    
    return p.parse_args()

def main():
    args = get_args()

    if args.task == "quick":
        run_quick(args)
        return

    print(f"Steps: {args.steps:,} | Seeds: {args.seeds} | Workers: {args.workers}")
    
    if args.task in ["all_algos", "all"]:
        run_all_algos(args)
    if args.task in ["ablation_batch", "all"]:
        run_ablation_batch(args)
    if args.task in ["ablation_entropy", "all"]:
        run_ablation_entropy(args)
    if args.task in ["ppo_opt", "all"]:
        run_optimization_comparison(args)
        
    print("\nexperiments completed!\n")

if __name__ == "__main__":
    main()