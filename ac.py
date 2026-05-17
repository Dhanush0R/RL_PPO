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
    # lr = config.get("lr", 1e-3)
    gamma = config.get("gamma", 0.99)

    actor_lr = config.get("actor_lr", 1e-4)
    critic_lr = config.get("critic_lr", 1e-3)

    entropy_coef = config.get("entropy_coef", 0.01)

    actor = PolicyNetwork(state_dim, action_dim, hidden)
    critic = ValueNetwork(state_dim, hidden)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)
    
    #mse as loss fn
    loss_fn = nn.MSELoss()

    steps_list = []
    rewards_list = []
    total_steps = 0
    
    with tqdm(total=max_steps, desc=f"AC seed={seed}", leave=False, disable=not show_progress) as pbar:
        while total_steps < max_steps:
            states, actions, rewards, next_states, dones = [], [], [], [], []
            epi_reward = 0.0
            done = False

            #a full episode to sample data
            while not done and total_steps < max_steps:
                action, _ = actor.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                state = next_state
                epi_reward += reward
                total_steps += 1
                pbar.update(1)

            steps_list.append(total_steps)
            rewards_list.append(epi_reward)
            state, _ = env.reset()

            states_t = torch.FloatTensor(np.array(states))
            actions_t = torch.LongTensor(actions)
            rewards_t = torch.FloatTensor(rewards)
            next_states_t = torch.FloatTensor(np.array(next_states))
            dones_t = torch.FloatTensor(dones)

            # Critic Estimate of td targets
            with torch.no_grad():
                next_values = critic(next_states_t).squeeze(-1)
                td_targets = rewards_t + gamma * next_values * (1.0 - dones_t)

            # V(s)
            values = critic(states_t).squeeze(-1)

            # critic loss
            critic_loss = loss_fn(values, td_targets)
            
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            # normalizing targets
            td_targets_norm = (td_targets - td_targets.mean()) / (td_targets.std() + 1e-8)

            probs = actor(states_t)
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions_t)
            
            #entropy regularization 
            entropy = dist.entropy().mean()
            
            actor_loss = -(log_probs * td_targets_norm.detach()).mean() - (entropy_coef * entropy)

            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

    env.close()

    if pth:
        torch.save({
            "config": config,
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict()
        }, pth)

    return steps_list, rewards_list