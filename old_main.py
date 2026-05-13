# command line usage:

# python main.py --task all       --> runs everything (1M steps, 5 seeds)
# python main.py --task compare   --> runs only PG methods comparison
# python main.py --task quick     --> quick test (5k steps)

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
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from joblib import Parallel, delayed
from torch.distributions import Categorical

warnings.filterwarnings("ignore")

# reinforce, ac, a2c imports
try:
    from networks import PolicyNetwork, ValueNetwork
except ImportError as e:
    print(f"[warn] Could not import networks.py: {e}")

try:
    import reinforce
    import ac
    import a2c
except ImportError as e:
    print(f"[warn] Could not import one of the required PG modules: {e}")

# mc
def _compute_mc_returns(rewards, gamma):
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

# td(0) targets
def _compute_td0_targets(rewards, next_states_t, dones, critic, gamma):
    with torch.no_grad():
        next_vals = critic(next_states_t).squeeze(-1)
    dones_t = torch.FloatTensor(dones)
    rewards_t = torch.FloatTensor(rewards)
    return (rewards_t + gamma * next_vals * (1.0 - dones_t)).tolist()

# n-step td targets
def _compute_nstep_targets(rewards, next_states, dones, critic, gamma, n):
    T = len(rewards)
    targets = []
    with torch.no_grad():
        ns_tensor = torch.FloatTensor(np.array(next_states))
        all_next_v = critic(ns_tensor).squeeze(-1).numpy()
    for t in range(T):
        end = t
        G = 0.0
        discount = 1.0
        for k in range(n):
            idx = t + k
            if idx >= T:
                break
            G += discount * rewards[idx]
            discount *= gamma
            end = idx
            if dones[idx]:
                break
        else:
            if not dones[end]:
                G += discount * float(all_next_v[end])
        targets.append(G)
    return targets


def train_agent_variants(config: dict, max_steps: int, seed: int, estimator: str = "mc", n_step: int = 10) -> tuple:
    warnings.filterwarnings("ignore")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("CartPole-v1")
    state, _ = env.reset(seed=seed)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    hidden       = config.get("h", 128)
    actor_lr     = config.get("actor_lr", config.get("lr", 1e-3))
    critic_lr    = config.get("critic_lr", config.get("lr", 2e-3))
    gamma        = config.get("gamma", 0.99)
    entropy_coef = config.get("entropy_coef", 0.01)

    actor  = PolicyNetwork(state_dim, action_dim, hidden)
    critic = ValueNetwork(state_dim, hidden)

    actor_optim  = optim.Adam(actor.parameters(),  lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)
    mse = nn.MSELoss()

    steps_list, rewards_list = [], []
    total_steps = 0

    while total_steps < max_steps:
        states, actions, rewards, next_states, dones = [], [], [], [], []
        epi_reward = 0.0
        done = False

        while not done and total_steps < max_steps:
            action, _ = actor.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(float(done))

            state = next_state
            epi_reward += reward
            total_steps += 1

        steps_list.append(total_steps)
        rewards_list.append(epi_reward)
        state, _ = env.reset()

        states_t      = torch.FloatTensor(np.array(states))
        next_states_t = torch.FloatTensor(np.array(next_states))
        actions_t     = torch.LongTensor(actions)

        if estimator == "mc":
            q_targets = _compute_mc_returns(rewards, gamma)
        elif estimator == "td0":
            q_targets = _compute_td0_targets(rewards, next_states_t, dones, critic, gamma)
        elif estimator == "nstep":
            q_targets = _compute_nstep_targets(rewards, next_states, dones, critic, gamma, n_step)
        else:
            raise ValueError(f"Unknown estimator: {estimator!r}")

        q_targets_t = torch.FloatTensor(q_targets)

        values = critic(states_t).squeeze(-1)
        critic_loss = mse(values, q_targets_t)
        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_optim.step()

        with torch.no_grad():
            values_det = critic(states_t).squeeze(-1)
        advantages = q_targets_t - values_det

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        probs     = actor(states_t)
        dist      = Categorical(probs)
        log_probs = dist.log_prob(actions_t)
        entropy   = dist.entropy().mean()

        actor_loss = -(log_probs * advantages).mean() - entropy_coef * entropy
        actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_optim.step()

    env.close()
    return steps_list, rewards_list

def _run_one_variant(config, max_steps, seed, estimator, n_step):
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    return train_agent_variants(config, max_steps, seed, estimator=estimator, n_step=n_step)

def parallelize_variants(config, max_steps, seeds, estimator, n_step, n_workers):
    return Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
        delayed(_run_one_variant)(config, max_steps, s, estimator, n_step)
        for s in range(seeds)
    )

# entropy ablation
def train_a2c_metrics(config: dict, max_steps: int, seed: int) -> tuple:
    warnings.filterwarnings("ignore")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("CartPole-v1")
    state, _ = env.reset(seed=seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    hidden = config.get("h", 128)
    actor_lr = config.get("actor_lr", 1e-3)
    critic_lr = config.get("critic_lr", 2e-3)
    gamma = config.get("gamma", 0.99)
    entropy_coef = config.get("entropy_coef", 0.01)

    actor = PolicyNetwork(state_dim, action_dim, hidden)
    critic = ValueNetwork(state_dim, hidden)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)
    loss_fn = nn.MSELoss()

    steps_list = []
    rewards_list = []
    critic_loss_list = []
    total_steps = 0
    
    while total_steps < max_steps:
        states, actions, rewards = [], [], []
        epi_reward = 0.0
        done = False

        while not done and total_steps < max_steps:
            action, _ = actor.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            epi_reward += reward
            total_steps += 1

        steps_list.append(total_steps)
        rewards_list.append(epi_reward)
        state, _ = env.reset()

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        returns_t = torch.FloatTensor(returns)
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)

        values = critic(states_t).squeeze(-1)
        advantages = returns_t - values.detach()

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        critic_loss = loss_fn(values, returns_t)
        critic_loss_list.append(critic_loss.item())

        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_optim.step()

        probs = actor(states_t)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()
        
        actor_loss = -(log_probs * advantages).mean() - (entropy_coef * entropy)

        actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_optim.step()

    env.close()
    return steps_list, rewards_list, critic_loss_list

def run_experiment_entropy(config, max_steps, seeds, n_workers=5):
    def run_seed(seed):
        torch.set_num_threads(1)
        warnings.filterwarnings("ignore")
        return train_a2c_metrics(config, max_steps, seed)
    return Parallel(n_jobs=n_workers, backend="loky", verbose=0)(delayed(run_seed)(s) for s in range(seeds))



def get_args():
    p = argparse.ArgumentParser(description="Consolidated Experiments Runner")
    p.add_argument("--task", type=str, default="all",
                   choices=["quick", "compare", "ac_lr", "a2c_vars", "entropy", "all"])
    p.add_argument("--steps",   type=int,   default=1_000_000)
    p.add_argument("--seeds",   type=int,   default=5)
    p.add_argument("--workers", type=int,   default=5)
    p.add_argument("--outdir",  type=str,   default="plots")
    p.add_argument("--window",  type=int,   default=150)
    p.add_argument("--baseline_csv", type=str, default="BaselineDataCartPole.csv",
                   help="Path to Assignment 2 baseline CSV")
    return p.parse_args()

def run_pg(module, config, max_steps, seeds, workers):
    def _run(seed):
        torch.set_num_threads(1)
        warnings.filterwarnings("ignore")
        return module.train_agent(config, max_steps, seed, show_progress=False)
    
    return Parallel(n_jobs=workers, backend="loky", verbose=0)(
        delayed(_run)(s) for s in range(seeds)
    )

# plotting and summary func
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
            u_steps, idx = np.unique(steps, return_index=True)
            interp_r.append(np.interp(grid, u_steps, rewards[idx]))

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

# for plotting learning curves
def run_compare(args):
    print("compare PG methods")
    config_reinforce = {"gamma": 0.99, "lr": 1e-3, "h": 128}
    config_ac = {"gamma": 0.99, "actor_lr": 1e-4, "critic_lr": 1e-3, "h": 128}
    config_a2c= {"gamma": 0.99, "actor_lr": 1e-3, "critic_lr": 2e-3, "h": 128, "entropy_coef": 0.01}
    
    res = {}
    print(f"running REINFORCE")
    res["REINFORCE"] = run_pg(reinforce, config_reinforce, args.steps, args.seeds, args.workers)
    
    print(f"running AC")
    res["Actor-Critic (AC)"] = run_pg(ac, config_ac, args.steps, args.seeds, args.workers)
    
    print(f"running A2C")
    res["A2C"] = run_pg(a2c, config_a2c, args.steps, args.seeds, args.workers)
    
    add_baseline(res, args.baseline_csv)
    
    plot(res, "Policy Gradient Methods vs DQN Baseline — CartPole-v1", "pg_compare", args.steps, args.outdir, args.window)
    summary_table(res, "PG Compare")

# for plotting learning rate ablation curves
def run_ac_lr(args):
    print("\nAC learning rate ablation")
    config_equal = {"gamma": 0.99, "actor_lr": 1e-3, "critic_lr": 1e-3, "h": 128}
    config_split = {"gamma": 0.99, "actor_lr": 1e-4, "critic_lr": 1e-3, "h": 128}
    
    res = {}
    print(f"running AC (equal LR)")
    res["Actor=1e-3, Critic=1e-3"] = run_pg(ac, config_equal, args.steps, args.seeds, args.workers)
    
    print(f"running AC (split LR)")
    res["Actor=1e-4, Critic=1e-3"] = run_pg(ac, config_split, args.steps, args.seeds, args.workers)
    
    plot(res, "Actor-Critic Learning Rate Ablation — CartPole-v1", "ac_lr_ablation", args.steps, args.outdir, args.window)
    summary_table(res, "AC LR Ablation")

# for plotting q-estimator ablation curves
def run_a2c_vars(args):
    print("\nA2C estimator variants")
    config = {"gamma": 0.99, "actor_lr": 1e-3, "critic_lr": 2e-3, "h": 128, "entropy_coef": 0.01}
    
    res = {}
    print(f"Running MC")
    res["A2C (Monte-Carlo)"] = parallelize_variants(config, args.steps, args.seeds, "mc", 10, args.workers)
    
    print(f"running TD(0)")
    res["A2C (TD(0))"] = parallelize_variants(config, args.steps, args.seeds, "td0", 10, args.workers)
    
    print(f"Running 10-step TD")
    res["A2C (10-step TD)"] = parallelize_variants(config, args.steps, args.seeds, "nstep", 10, args.workers)
    
    add_baseline(res, args.baseline_csv)

    plot(res, "A2C Q-Estimator Comparison — CartPole-v1", "a2c_estimators", args.steps, args.outdir, args.window)
    summary_table(res, "A2C Estimators")

# for plotting entropy ablation curves
def run_entropy(args):
    print("\nA2C entropy sweep")
    res = {}
    configs = {
        "No Entropy (0.0)":   {"entropy_coef": 0.0,  "h": 128, "actor_lr": 1e-3, "critic_lr": 2e-3},
        "Standard (0.01)":    {"entropy_coef": 0.01, "h": 128, "actor_lr": 1e-3, "critic_lr": 2e-3},
        "High Entropy (0.1)": {"entropy_coef": 0.1,  "h": 128, "actor_lr": 1e-3, "critic_lr": 2e-3}
    }
    
    for label, cfg in configs.items():
        print(f"  Running {label}...")
        res[label] = run_experiment_entropy(cfg, args.steps, args.seeds, args.workers)
        
    plot(res, "Ablation: Entropy Regularization Coefficient in A2C", "entropy_sweep", args.steps, args.outdir, args.window)
    summary_table(res, "Entropy Sweep")

def main():
    args = get_args()
    
    if args.task == "quick":
        args.steps = 5000
        args.seeds = 2
        args.workers = 2
        args.window = 5
        args.outdir = os.path.join(args.outdir, "quick")
        print("quick test mode: 5k steps, 2 seeds, 2 workers")
        run_compare(args)
        return

    print(f"Steps: {args.steps:,} | Seeds: {args.seeds} | Workers: {args.workers}")
    
    if args.task in ["all", "compare"]:  run_compare(args)
    if args.task in ["all", "ac_lr"]:    run_ac_lr(args)
    if args.task in ["all", "a2c_vars"]: run_a2c_vars(args)
    if args.task in ["all", "entropy"]:  run_entropy(args)
    
    print("\nAll experiments completed!\n")

if __name__ == "__main__":
    main()