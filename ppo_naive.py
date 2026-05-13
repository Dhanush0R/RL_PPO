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
    actor_lr = config.get("actor_lr", 3e-4) 
    critic_lr = config.get("critic_lr", 1e-3) 
    gamma = config.get("gamma", 0.99)
    gae_lambda = config.get("gae_lambda", 0.95) # Added GAE parameter
    entropy_coef = config.get("entropy_coef", 0.01)
    clip_eps = config.get("clip_eps", 0.2)
    ppo_epochs = config.get("ppo_epochs", 4)

    actor = PolicyNetwork(state_dim, action_dim, hidden)
    critic = ValueNetwork(state_dim, hidden)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)
    loss_fn = nn.MSELoss()

    steps_list = []
    rewards_list = []
    total_steps = 0
    
    with tqdm(total=max_steps, desc=f"PPO seed={seed}", leave=False, disable=not show_progress) as pbar:
        while total_steps < max_steps:
            # Added next_states and dones for GAE calculation
            states, actions, rewards, old_log_probs, next_states, dones = [], [], [], [], [], []
            epi_reward = 0.0
            done = False

            # 1. Collect full episode trajectory
            while not done and total_steps < max_steps:
                state_t = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    probs = actor(state_t)
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                next_state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated

                states.append(state)
                actions.append(action.item())
                rewards.append(reward)
                old_log_probs.append(log_prob.item())
                next_states.append(next_state)
                dones.append(float(done)) # Store done as float (0.0 or 1.0)

                state = next_state
                epi_reward += reward
                total_steps += 1
                pbar.update(1)

            steps_list.append(total_steps)
            rewards_list.append(epi_reward)
            state, _ = env.reset()

            # Convert trajectory lists to tensors
            states_t = torch.FloatTensor(np.array(states))
            next_states_t = torch.FloatTensor(np.array(next_states))
            actions_t = torch.LongTensor(actions)
            rewards_t = torch.FloatTensor(rewards)
            dones_t = torch.FloatTensor(dones)
            old_log_probs_t = torch.FloatTensor(old_log_probs)

            # 2. Compute Generalized Advantage Estimation (GAE)
            with torch.no_grad():
                values = critic(states_t).squeeze(-1)
                next_values = critic(next_states_t).squeeze(-1)
            
            advantages = torch.zeros_like(rewards_t)
            last_gae_lam = 0.0
            
            # Traverse backwards to compute GAE
            for t in reversed(range(len(rewards))):
                mask = 1.0 - dones_t[t]
                # delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
                delta = rewards_t[t] + gamma * next_values[t] * mask - values[t]
                
                # A_t = delta_t + gamma * lambda * (1 - done) * A_{t+1}
                advantages[t] = last_gae_lam = delta + gamma * gae_lambda * mask * last_gae_lam

            # TD(lambda) Returns are simply advantages + values
            returns_t = advantages + values

            # Advantage normalization for optimization stability
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 3. PPO Optimization Epochs
            for _ in range(ppo_epochs):
                probs = actor(states_t)
                dist = Categorical(probs)
                curr_log_probs = dist.log_prob(actions_t)
                entropy = dist.entropy().mean()

                curr_values = critic(states_t).squeeze(-1)

                # Ratio: exp(current - old)
                ratios = torch.exp(curr_log_probs - old_log_probs_t)

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

                # Actor & Critic losses
                actor_loss = -torch.min(surr1, surr2).mean() - (entropy_coef * entropy)
                critic_loss = loss_fn(curr_values, returns_t)

                # Update Critic
                critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_optim.step()

                # Update Actor
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