import argparse
import os
import random
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from joblib import Parallel, delayed
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# STORAGE
SEED = 0

# we cant use tensorboard while having parallel workers, 
# so we conditionally import it and disable if not available or if parallelism is used later in the code.
try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

# couple of args for easier tests and runs
def inputargs():
    p = argparse.ArgumentParser(
        description="DQN CartPole",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--task", type=str, default="all",
        choices=["quick", "2.1", "2.2", "2.4", "best", "extra", "all"],
        help=(
            "Task to run "
            "'quick' = fast smoke test (approx 2 min)"
            "'extra' = extra sweeps for generating tables"
            "'all' = full pipeline"
        ),
    )

    p.add_argument("--steps",type=int, default=1_000_000, help="Max env steps per run")
    p.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    p.add_argument("--workers", type=int, default=5, help="Parallel workers")

    p.add_argument("--lr", type=float, default=0.0001,  help="Lr")
    p.add_argument("--h", type=int,   default=256, help="Hidden layer size")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--e_decay", type=float, default=0.9999, help="Epsilon decay per step")
    p.add_argument("--e_start", type=float, default=1.0, help="Starting epsilon")
    p.add_argument("--e_min", type=float, default=0.01, help="Minimum epsilon") # when we hit this, we stop decaying and just keep it at the min value to ensure very low exploration late training.
    p.add_argument("--main_net_update_freq", type=int, default=1, help="Steps between updates")
    p.add_argument("--start_learning_at", type=int, default=5000, help="Steps before training starts") # insync with target update
    p.add_argument("--tn_update_freq",type=int,default=5000, help="Target network update freq") # we cranked this up from 1000 to 5000 for better performance and stability
    p.add_argument("--buff_sz", type=int, default=100_000, help="Replay buffer size")

    # we saved the final best plots in "Final-Plots"
    
    p.add_argument("--outdir", type=str, default="plots-test", help="Plot output directory for runs") # overwites prev tests
    p.add_argument("--model_path", type=str, default="best_cartpole_dqn-test.pt", help="Where to save best model")
    p.add_argument("--baseline_csv", type=str, default="BaselineDataCartPole.csv")
    p.add_argument("--window", type=int, default=300, help="Smoothing")
    p.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard")

    return p.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######### Model and Agent #########

# lets write the agent and model. 
# we will implement the algo with settings for er and tn, 
# as well as hyperparams for lr, expn, etc.
class DQN(nn.Module):
    def __init__(self, s, a, h):
        super().__init__()
        self.fc1 = nn.Linear(s, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, a)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, s, a, config):
        self.a = a
        self.lr = config.get("lr", 0.0001)
        self.h = config.get("h", 256) #hidden layer size
        self.gamma = config.get("gamma", 0.99)
        self.e = config.get("e_start", 1.0)
        self.e_min = config.get("e_min", 0.01)
        self.e_decay = config.get("e_decay", 0.9999)
        self.main_net_update_freq = config.get("main_net_update_freq", 1)
        self.start_learning_at = config.get("start_learning_at", 5000)
        self.er_switch = config.get("er_switch", True)
        self.tn_switch = config.get("tn_switch", True)
        self.target_u_freq = config.get("target_u_freq", 5000)

        # we used a smaller batch size (32) when not using ER since we only train on the most recent transitions, 
        # so we can afford to update more frequently without needing as much data per update. 
        # with ER, we want a larger batch size to get more stable updates from the replay buffer.
        # we also tried 32, 128 and 256 for ER and they all sucked.

        self.batch_sz = config.get("batch_sz", 64 if self.er_switch else 32) 
        buff = config.get("buff_sz", 100_000) #we cranked this up from 10k to 100k for better performance.
        self.mem = deque(maxlen=buff) if self.er_switch else deque(maxlen=32)
        self.model = DQN(s, a, self.h).to(device)
        self.tm = DQN(s, a, self.h).to(device)
        self.update_tm()
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr) #going with a classic one
        self.loss_func = nn.MSELoss()
        self.total_steps = 0

    def update_tm(self):
        self.tm.load_state_dict(self.model.state_dict()) #copying main net wgts to tnet

    def push(self, s, a, r, ns_s, done): # push it down the replay buff
        self.mem.append((s, a, r, ns_s, done))

    def act(self, s): #espsilon greedy
        if np.random.rand() < self.e:
            return np.random.randint(self.a)
        s = torch.FloatTensor(s).unsqueeze(0).to(device)
        with torch.no_grad():
            return self.model(s).argmax(dim=1).item()

    def replay(self):
        if self.er_switch:
            if len(self.mem) < max(self.batch_sz, self.start_learning_at):
                return
        else:
            if len(self.mem) < self.batch_sz:
                return

        if self.total_steps % self.main_net_update_freq != 0:
            return

        #sampling
        if self.er_switch:
            batch = random.sample(self.mem, self.batch_sz)
        else:
            batch = list(self.mem)[-self.batch_sz:]

        ss, acts, rs, next_ss, dones = zip(*batch)
        ss = torch.FloatTensor(np.array(ss)).to(device)
        acts = torch.LongTensor(acts).to(device)
        rs = torch.FloatTensor(rs).to(device)
        next_ss = torch.FloatTensor(np.array(next_ss)).to(device)
        dones = torch.FloatTensor(dones).to(device)

        curr_q = self.model(ss).gather(1, acts.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.tn_switch:
                next_q = self.tm(next_ss).max(1)[0]
            else:
                next_q = self.model(next_ss).max(1)[0]

        tar_q = rs + (1 - dones) * self.gamma * next_q
        loss = self.loss_func(curr_q, tar_q)
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()

####### Training #######

def train_agent(config, max_steps, seed, pth=None, n_envs=1, show_progress=True, tb_logdir=None):
    
    warnings.filterwarnings("ignore")

    w = SummaryWriter(log_dir=tb_logdir) if (tb_logdir and TB_AVAILABLE) else None

    # lets run multiple cartpolrs in parallel to speed up data collection
    envs = gym.vector.SyncVectorEnv( [lambda i=i: gym.make("CartPole-v1") for i in range(n_envs)])
    np.random.seed(seed) 
    random.seed(seed)
    torch.manual_seed(seed)
    s = envs.single_observation_space.shape[0]
    a = envs.single_action_space.n
    agent = Agent(s, a, config)

    steps_l = []
    rewards_l = []
    total_steps = 0
    epi_rewards = np.zeros(n_envs)
    epi_count = 0
    ss, _ = envs.reset(seed=[seed + i for i in range(n_envs)])

    with tqdm(total=max_steps, desc=f"seed={seed}", leave=False, disable=not show_progress) as pbar:
        
        while total_steps < max_steps:
            acts = np.array([agent.act(s) for s in ss])
            ns, rs, ter, trunc, _ = envs.step(acts)
            dones = ter | trunc

            for i in range(n_envs):
                agent.push(ss[i], acts[i], rs[i],
                           ns[i], dones[i])
                epi_rewards[i] += rs[i]
                if dones[i]:
                    steps_l.append(total_steps)
                    rewards_l.append(epi_rewards[i])
                    if w:
                        w.add_scalar("Episode/Return", epi_rewards[i], total_steps)
                        w.add_scalar("Episode/Epsilon", agent.e, total_steps)
                    epi_rewards[i] = 0.0
                    epi_count += 1

            agent.total_steps = total_steps

            # after a lot of tests, we added the below chunk...
            # when the agent hits 480 mark of rewards (last 20 epi avg), we can be pretty sure it has basically solved the env, 
            # so we can just freeze exploration (e->min) at that point to let it fully exploit and get those last few points to max reward of 500 
            # without risking any more exploration that could cause it to drop back down 
            # The major reason we added this is it aligns well with the baseline data!
            frz = config.get("freeze", False)
            latest_mean_r = np.mean(rewards_l[-20:]) if len(rewards_l) >= 20 else 0

            if frz and latest_mean_r >= 480:
                agent.e = agent.e_min #this
            else:
                agent.replay()
                if agent.tn_switch and total_steps % agent.target_u_freq == 0:
                    agent.update_tm()
                if agent.e > agent.e_min:
                    agent.e *= agent.e_decay
                else:
                    agent.e = agent.e_min

            ss = ns
            total_steps += n_envs
            pbar.update(n_envs)

    if w:
        w.close()
    envs.close()

    if pth:
        torch.save({"config": config, "s": s, "a": a, "model_state_dict": agent.model.state_dict()}, pth)

    return steps_l, rewards_l

# we spawn 5 parallel workers to run 5 seeds at the same time.
def parallelize(config, max_steps, seeds, pth=None, n_workers=5):
    def run_seed(seed):
        import torch as _t
        global device
        _t.set_num_threads(1)
        device = _t.device("cpu") #to avoid memory isssues
        warnings.filterwarnings("ignore")
        return train_agent(config, max_steps, seed, 
                            pth=(pth if seed == 0 else None), 
                            show_progress=False)
    return Parallel(n_jobs=n_workers, backend="loky", verbose=10)(delayed(run_seed)(s) for s in range(seeds))

def plotting(res, t, fn, max_steps, outdir="plots", window=300):

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    palette = sns.color_palette("colorblind", n_colors=len(res))
    fig, ax = plt.subplots(figsize=(9, 5))
    grid = np.linspace(0, max_steps, num=2000)

    for (label, runs), color in zip(res.items(), palette):
        interp_r = []
        for steps, r in runs:
            steps = np.array(steps)
            r = np.array(r)
            u_steps, idx = np.unique(steps, return_index=True)
            interp_r.append(np.interp(grid, u_steps, r[idx]))

        arr = np.array(interp_r)
        sm_mean = pd.Series(arr.mean(0)).rolling(window, min_periods=1).mean().values
        sm_std = pd.Series(arr.std(0)).rolling(window,  min_periods=1).mean().values
        ax.plot(grid, sm_mean, label=label, linewidth=1.8, color=color)
        ax.fill_between(grid, sm_mean - sm_std, sm_mean + sm_std, alpha=0.15, color=color)

    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.7, color="gray")
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.4, color="gray")
    ax.tick_params(which="minor", length=0)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6
                             else f"{x/1e3:.0f}k" if x >= 1e3 else f"{x:.0f}"))
    ax.set_xlabel("Environment Steps", fontsize=12, labelpad=6)
    ax.set_ylabel("Return (Episode Reward)", fontsize=12, labelpad=6)
    ax.set_title(t, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlim(0, max_steps)
    ax.set_ylim(bottom=0)

    legend = ax.legend(loc="upper left", fontsize=9.5,
        framealpha=0.9, edgecolor="lightgray",
        borderpad=0.6, labelspacing=0.4)
    sns.despine(ax=ax, top=True, right=True)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{fn}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved to {out_path}")

def summ_table(res_dict, title):
    print(f"\n--- {title} ---")
    data = []
    
    for label, runs in res_dict.items():
        final = []
        for steps, r in runs:
            if len(r) >= 20:
                final.append(np.mean(r[-20:]))
            elif len(r) > 0:
                final.append(np.mean(r))
            else:
                final.append(0)
                
        mean = np.mean(final)
        std = np.std(final)
        
        data.append({"Configuration": label, 
            "Final Mean Return": f"{mean:.1f} ± {std:.1f}"})
        
    df = pd.DataFrame(data)
    print(df.to_markdown(index=False))
    return df

def run_task_21(base_config, args):
    print("\n\n------Task 2.1: Naive DQN (No TN, No ER)-----\n\n")
    cfg = {**base_config, "tn_switch":False, "er_switch":False} # we can just use the base configs and switch off tn and er
    results = {"Naive DQN": parallelize(cfg, args.steps, args.seeds, n_workers=args.workers)}
    plotting(results, "Task 2.1: Naive DQN Learning Curve", "naive_dqn", args.steps, args.outdir, args.window)


def run_task_22(base_config, args):
    print("\n\n------Task 2.2: Hyperparameter Sweeps-----\n\n")

    # 1 Learning Rate
    print("  -- LR --")
    lr_cfgs = {
        "Small LR (1e-4)": {**base_config, "lr": 1e-4},
        "Medium LR (1e-3)": {**base_config, "lr": 1e-3},
        "High LR (1e-2)": {**base_config, "lr": 1e-2}
    }
    lr_res = {n: parallelize(c, args.steps, args.seeds, n_workers=args.workers) for n, c in lr_cfgs.items()}
    plotting(lr_res, "Task 2.2: Learning Rate Sweep", "hyperparam_lr", args.steps, args.outdir, args.window)

    # 2 Network Size
    print("  -- Network Size --")
    size_cfgs = {
        "Small Net (64)":{**base_config, "h": 64},
        "Medium Net (256)":{**base_config, "h": 256},
        "Large Net (512)":{**base_config, "h": 512}
    }
    size_res = {n: parallelize(c, args.steps, args.seeds, n_workers=args.workers) for n, c in size_cfgs.items()}
    plotting(size_res, "Task 2.2: Network Size Sweep", "hyperparam_size", args.steps, args.outdir, args.window)

    # 3 Update to Data Ratio
    print("  -- Update to Data ratio --")
    freq_cfgs = {
        "High Ratio (1 step)":{**base_config, "main_net_update_freq": 1},
        "Medium Ratio (4 steps)":{**base_config, "main_net_update_freq": 4},
        "Low Ratio (16 steps)":{**base_config, "main_net_update_freq": 16}
    }
    freq_res = {n: parallelize(c, args.steps, args.seeds, n_workers=args.workers) for n, c in freq_cfgs.items()}
    plotting(freq_res, "Task 2.2: Update to Data Ratio Sweep", "hyperparam_freq", args.steps, args.outdir, args.window)

    # 4 Exploration Factor
    print("  -- Exploration Factor --")
    eps_cfgs = {
        "Fast Decay (0.9999)": {**base_config, "e_decay": 0.9999},
        "Medium Decay (0.99995)":{**base_config, "e_decay": 0.99995},
        "Slow Decay (0.99999)":{**base_config, "e_decay": 0.99999}
    }
    eps_res = {n: parallelize(c, args.steps, args.seeds, n_workers=args.workers) for n, c in eps_cfgs.items()}
    plotting(eps_res, "Task 2.2: Exploration Factor Sweep", "hyperparam_eps", args.steps, args.outdir, args.window)


def run_task_24(base_config, args):
    print("\n\n------Task 2.4: Feature Ablation-----\n\n")
    feat_cfgs = {
        "Naive (No TN, No ER)":{**base_config, "tn_switch": False, "er_switch": False},
        "Only TN": {**base_config, "tn_switch": True, "er_switch": False},
        "Only ER": {**base_config, "tn_switch": False, "er_switch": True},
        "TN & ER": {**base_config, "tn_switch": True, "er_switch": True}
    }
    feat_res = {n: parallelize(c, args.steps, args.seeds, n_workers=args.workers) for n, c in feat_cfgs.items()}
    plotting(feat_res, "Task 2.4: Feature Ablation", "feature_ablation", args.steps, args.outdir, args.window)

def run_extra_sweeps(base_config, args):
    print("\n\n------Extra Sweeps-----\n\n")

    # 1 Replay Buff Size
    print("  --Replay Buffer Size--")
    buff_cfgs = {
        "Small (5,000)": {**base_config, "buff_sz": 5000},
        "Medium (20,000)": {**base_config, "buff_sz": 20000},
        "Large (100,000)": {**base_config, "buff_sz": 100000}
    }
    buff_res = {n: parallelize(c, args.steps, args.seeds, n_workers=args.workers) for n, c in buff_cfgs.items()}
    summ_table(buff_res, "Buffer Size Sweep")

    # 2 Target Network Update Frequency
    print("\n  --Target Network Update Frequency--")
    tn_cfgs = {
        "Fast TN (500)": {**base_config, "target_u_freq": 500},
        "Medium TN (5,000)": {**base_config, "target_u_freq": 5000},
        "Slow TN (20,000)": {**base_config, "target_u_freq": 20000}
    }
    tn_res = {n: parallelize(c, args.steps, args.seeds, n_workers=args.workers) for n, c in tn_cfgs.items()}
    summ_table(tn_res, "Target Network Frequency Sweep")

    # 3 Discount Factor
    print("\n  --Discount Factor Gamma--")
    gamma_cfgs = {
        "Short-sighted (0.85)": {**base_config, "gamma": 0.85},
        "Standard (0.99)": {**base_config, "gamma": 0.99},
        "Far-sighted (0.999)": {**base_config, "gamma": 0.999}
    }
    gamma_res = {n: parallelize(c, args.steps, args.seeds, n_workers=args.workers) for n, c in gamma_cfgs.items()}
    summ_table(gamma_res, "Discount Factor Sweep")

    # 4 Batch Size
    print("\n  --Batch Size--")
    batch_cfgs = {
        "Small Batch (32)": {**base_config, "batch_sz": 32},
        "Medium Batch (128)": {**base_config, "batch_sz": 128},
        "Large Batch (512)": {**base_config, "batch_sz": 512}
    }
    batch_res = {n: parallelize(c, args.steps, args.seeds, n_workers=args.workers) for n, c in batch_cfgs.items()}
    summ_table(batch_res, "Batch Size Sweep")

def run_best_model(base_config, args): # final model average with 5 seeds and comparison with baseline
    print("\n\n------Best Model (TN+ER, single seed + baseline comparison)-----\n\n")
    best_config = {**base_config, "freeze": True}
    best_res = {"Best Model (TN+ER)": parallelize(best_config, args.steps, args.seeds, n_workers=args.workers)}
    print(f"  Model saved to {args.model_path}")
    plotting(best_res, "Best Final Model Training Graph", "best_model_reward_graph", 
            args.steps, args.outdir, window=100)

    # vs baseline
    try:
        df=pd.read_csv(args.baseline_csv)
        b_steps=df["env_step"].tolist()
        b_rew=df["Episode_Return_smooth"].tolist()
        cmp = {"Best Model (TN+ER)": best_res["Best Model (TN+ER)"],
            "Instructor Baseline (Smoothed)": [(b_steps, b_rew)]}
    except FileNotFoundError:
        cmp = best_res

    plotting(cmp, "CartPole: Best Model vs Instructor Baseline", "final_model_vs_baseline", 
            args.steps, args.outdir, window=100)


def run_quick(args):
    # to make it easy for you guys to test the training we added this small bit :)
    print("\n\n------QUICK TEST MODE-----\n\n")

    Q_STEPS=10000
    Q_SEEDS=2
    Q_WORKERS=2
    outdir=os.path.join(args.outdir, "quick")
    quick_base={"tn_switch":True, "er_switch": True, "lr": 0.0001, "h":128, "main_net_update_freq": 4, 
                "e_decay": 0.999, "e_start":1.0, "e_min":0.01, "start_learning_at":200, "gamma": 0.99,
                "target_u_freq":200, "buff_sz": 5000} #keeping it same ratio as main

    #Task 2.1
    print("---[2.1] Naive DQN---")
    naive = {"Naive DQN": parallelize({**quick_base, "tn_switch": False, "er_switch": False},
                                    Q_STEPS, Q_SEEDS, n_workers=Q_WORKERS)}
    plotting(naive, "Quick: Naive DQN", "quick_naive", Q_STEPS, outdir, window=100)

    # Task 2.2
    configs = [("LR","lr",[1e-4, 1e-3, 1e-2],["Small (1e-4)", "Medium (1e-3)", "High (1e-2)"]),
        ("Size", "h",[64, 256, 512], ["Small (64)", "Medium (256)", "Large (512)"]),
        ("Freq", "main_net_update_freq",[1, 4, 16],["High (1)", "Medium (4)", "Low (16)"]),
        ("Eps", "e_decay", [0.999, 0.99995, 0.99999],["Fast", "Medium", "Slow"])
    ]
    print("---[2.2] Hyperparameter Sweeps---")
    for sweep, key, vals, labels in configs:
        print(f"    {sweep} sweep")
        res = {lbl: parallelize({**quick_base, key: v}, Q_STEPS, Q_SEEDS, n_workers=Q_WORKERS) for lbl, v in zip(labels, vals)}
        plotting(res, f"Quick: {sweep} Sweep", f"quick_{sweep.lower()}", Q_STEPS, outdir, window=100)

    # Task 2.4
    print("---[2.4] Feature ablation---")
    feat_res = {}
    features = [("Naive", False, False), 
                ("Only TN", True, False), 
                ("Only ER", False, True), 
                ("TN & ER", True, True)]
    for name, tn, er in features:
        feat_res[name] = parallelize({**quick_base, "tn_switch": tn, "er_switch": er},Q_STEPS, Q_SEEDS, n_workers=Q_WORKERS)
    plotting(feat_res, "Quick: Feature Ablation","quick_ablation", Q_STEPS, outdir, window=100)

    # Best model
    print("---Best Model (TN+ER)---")
    train_agent(quick_base, Q_STEPS, seed=SEED,pth="quick_cartpole_dqn.pt", show_progress=True)

    print(f"\nQuick test done. Plots saved to '{outdir}/'")

## Main entry point
def main():
    args = inputargs()

    print(f"Using device: {device}")
    task = args.task.lower()
    if task =="quick":
        Q_STEPS=10000
        Q_SEEDS=2
        print(f"Task: {args.task}  |  Steps: {Q_STEPS}  |  Seeds: {Q_SEEDS}\n")
    else:
        print(f"Task: {args.task}  |  Steps: {args.steps}  |  Seeds: {args.seeds}\n")

    # take all from args, and we can easily pass this around. 
    # we can also override specific ones for different runs 
    base_config = { "tn_switch": True, "er_switch":True, "lr":args.lr,
        "h":args.h, "gamma":args.gamma,"main_net_update_freq":args.main_net_update_freq, 
        "e_decay":args.e_decay,"e_start":args.e_start, "e_min":args.e_min,
        "start_learning_at": args.start_learning_at, "target_u_freq":args.tn_update_freq,"buff_sz":args.buff_sz,
    }

    if task =="quick":
        run_quick(args)
    elif task== "2.1":
        run_task_21(base_config, args)
    elif task == "2.2":
        run_task_22(base_config, args)
    elif task =="2.4":
        run_task_24(base_config, args)
    elif task =="best":
        run_best_model(base_config, args)
    elif task =="extra":
        run_extra_sweeps(base_config, args)
    elif task =="all":
        run_best_model(base_config, args)
        run_task_21(base_config, args)
        run_task_22(base_config, args)
        run_task_24(base_config, args)
        run_extra_sweeps(base_config, args)
    else:
        raise ValueError(f"Unknown task: {task}")

    print("\nAll done!\n")


if __name__ == "__main__":
    main()

#EOF