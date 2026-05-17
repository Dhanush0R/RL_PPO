import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym
import random
import warnings

from networks import PolicyNetwork


def compute_returns(rewards: list, gamma: float) -> torch.Tensor:
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)
    returns = (returns - returns.mean())/(returns.std() + 1e-8)
    return returns


def train_agent(config: dict, max_steps: int, seed: int,
                pth=None, show_progress=True) -> tuple:
    warnings.filterwarnings("ignore")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    gamma = config.get("gamma",0.99)
    lr = config.get("lr", 1e-3)
    hidden = config.get("h",128)
    env_name = config.get("env","CartPole-v1")
    max_ep_len = config.get("max_ep", 500)

    env = gym.make(env_name)
    env.reset(seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim, hidden)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    steps_list, rewards_list = [], []
    total_steps = 0

    while total_steps < max_steps:
        state, _ = env.reset()
        log_probs, rewards = [], []

        for _ in range(max_ep_len):
            action, log_prob = policy.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            total_steps += 1
            state = next_state
            if terminated or truncated:
                break

        returns = compute_returns(rewards, gamma)

        # REINFORCE policy gradient loss
        policy_loss = torch.stack(
            [-lp * G for lp, G in zip(log_probs, returns)]
        ).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        steps_list.append(total_steps)
        rewards_list.append(sum(rewards))

    env.close()

    if pth:
        torch.save({
            "config": config, "state_dim": state_dim,
            "action_dim": action_dim,
            "model_state_dict": policy.state_dict(),
        }, pth)

    return steps_list, rewards_list
