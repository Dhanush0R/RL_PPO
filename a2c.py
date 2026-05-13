import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random
import warnings
from tqdm import tqdm
from torch.distributions import Categorical

from networks import PolicyNetwork, ValueNetwork

def train_agent(config: dict, max_steps: int, seed: int,
                pth=None, show_progress=True) -> tuple:
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
    total_steps = 0
    
    with tqdm(total=max_steps, desc=f"A2C seed={seed}", leave=False, disable=not show_progress) as pbar:
        while total_steps < max_steps:
            states, actions, rewards = [], [], []
            epi_reward = 0.0
            done = False

            # full episode to sample data 
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
                pbar.update(1)

            steps_list.append(total_steps)
            rewards_list.append(epi_reward)
            state, _ = env.reset()

            # monte carlo returns 
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            
            returns_t = torch.FloatTensor(returns)
            states_t = torch.FloatTensor(np.array(states))
            actions_t = torch.LongTensor(actions)

            # V(s)
            values = critic(states_t).squeeze(-1)

            # advantage:G_t - V(s)
            advantages = returns_t - values.detach()

            # advantage normalization
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            critic_loss = loss_fn(values, returns_t)
            
            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_optim.step()

            probs = actor(states_t)
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions_t)
            
            # entropy regularization
            entropy = dist.entropy().mean()
            
            # actor loss using the advantage
            actor_loss = -(log_probs * advantages).mean() - (entropy_coef * entropy)

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_optim.step()

    env.close()

    if pth:
        torch.save({
            "config": config,
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict()
        }, pth)

    return steps_list, rewards_list