# RL Algorithm Comparison — CartPole-v1

A reinforcement learning benchmark suite comparing REINFORCE, Actor-Critic (AC), A2C, DQN, and PPO on the CartPole-v1 environment, with ablation studies and step-by-step PPO optimization analysis.

---

## Project Structure

| File | Description |
|------|-------------|
| `main.py` | Entry point — orchestrates all experiments, plotting, and summary tables |
| `reinforce.py` | REINFORCE (vanilla policy gradient) implementation |
| `ac.py` | Actor-Critic (online, one-step TD) implementation |
| `a2c.py` | Advantage Actor-Critic (A2C) with parallel rollouts |
| `dqn.py` | Deep Q-Network with target network and experience replay |
| `ppo.py` | Proximal Policy Optimization (clipped surrogate) with GAE |
| `networks.py` | Shared neural network architectures used across algorithms |
| `requirements.txt` | Python dependencies |
| `plots/` | Output directory for generated experiment plots |

---

## Setup

### 1. Create and activate a virtual environment

```bash
# Create the venv
python -m venv venv

# Activate — Linux/macOS
source venv/bin/activate

# Activate — Windows (Command Prompt)
venv\Scripts\activate.bat

# Activate — Windows (PowerShell)
venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running Experiments

All experiments are launched through `main.py` using the `--task` argument.

### Quick syntax

```bash
python main.py --task <task_name> [options]
```

---

## Tasks

### `all_algos` — Algorithm Comparison (Task 1)

Trains and compares all five algorithms (REINFORCE, AC, A2C, DQN, PPO) on CartPole-v1.

```bash
python main.py --task all_algos
```

Produces: `plots/all_algorithms.png`

---

### `ablation_batch` — PPO Minibatch Size Ablation (Task 2)

Sweeps PPO minibatch sizes `[64, 128, 256, 512]`, holding all other hyperparameters fixed.

```bash
python main.py --task ablation_batch
```

Produces: `plots/ablation_batch.png`

---

### `ablation_entropy` — PPO Entropy Coefficient Ablation (Task 2)

Sweeps entropy regularization coefficients `[0.0, 0.01, 0.05, 0.1]` in PPO.

```bash
python main.py --task ablation_entropy
```

Produces: `plots/ablation_entropy.png`

---

### `ppo_opt` — PPO Optimization Breakdown (Task 3)

Trains three progressive PPO variants to isolate the contribution of each optimization:

1. **Naive PPO** — clipping + GAE only, full batch, no normalization  
2. **Naive + Minibatch** — adds minibatch updates (size 64)  
3. **Naive + Minibatch + Norm** — adds advantage normalization on top

```bash
python main.py --task ppo_opt
```

Produces: `plots/ppo_optimizations.png`

---

### `all` — Run Everything

Runs all four tasks sequentially.

```bash
python main.py --task all
```

---

### `quick` — Smoke Test

Runs a fast sanity check with reduced steps (10k) and fewer seeds (2) to verify everything works before committing to a full run.

```bash
python main.py --task quick
```

Produces outputs under `plots/quick/`.

---

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `all` | Which experiment to run. Choices: `all_algos`, `ablation_batch`, `ablation_entropy`, `ppo_opt`, `all`, `quick` |
| `--steps` | `1_000_000` | Total environment steps per training run |
| `--seeds` | `5` | Number of random seeds (independent runs) per configuration |
| `--workers` | `5` | Number of parallel workers for seed-level parallelism (via `joblib`) |
| `--outdir` | `plots` | Directory where generated plots are saved |
| `--window` | `150` | Rolling average window size for smoothing learning curves |

### Example: faster run with fewer seeds

```bash
python main.py --task all_algos --steps 500000 --seeds 3 --workers 3
```

### Example: custom output directory

```bash
python main.py --task ppo_opt --outdir results/ppo --steps 1000000
```

---

## Output

Each task saves a `.png` plot under `--outdir` and prints a summary table to stdout showing **mean ± std of final episode returns** (averaged over the last 20 episodes across seeds):

```
 All Algorithms Comparison Summary
  REINFORCE                          : 312.4 ± 88.2
  Actor-Critic (AC)                  : 421.7 ± 54.1
  A2C                                : 467.3 ± 31.8
  DQN (TN + ER)                      : 489.1 ± 12.4
  PPO (Clipped)                      : 498.6 ± 3.2
```

The dashed green line at **500** marks the maximum achievable return in CartPole-v1.
