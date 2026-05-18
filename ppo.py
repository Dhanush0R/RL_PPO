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

def make_env(seed):
    def thunk():
        env = gym.make("CartPole-v1")
        env.action_space.seed(seed)
        return env
    return thunk

def train_agent(config: dict, max_steps: int, seed: int,
                pth=None, show_progress=True) -> tuple:
    warnings.filterwarnings("ignore")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_envs = config.get("num_envs", 4)
    envs = gym.vector.SyncVectorEnv([make_env(seed + i) for i in range(num_envs)])
    
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs)
    next_done = torch.zeros(num_envs)

    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.n

    #hyperparameters 
    hidden = config.get("h", 128)
    actor_lr = config.get("actor_lr", 3e-4) 
    critic_lr = config.get("critic_lr", 1e-3) 
    gamma = config.get("gamma", 0.99)
    gae_lambda = config.get("gae_lambda", 0.95)
    entropy_coef = config.get("entropy_coef", 0.0)
    clip_eps = config.get("clip_eps", 0.2)
    
    ppo_epochs = config.get("ppo_epochs", 4)
    rollout_steps = config.get("rollout_steps", 128) 
    batch_size = num_envs * rollout_steps       
    minibatch_size = config.get("minibatch_size", 64) 

    actor = PolicyNetwork(state_dim, action_dim, hidden)
    critic = ValueNetwork(state_dim, hidden)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr, eps=1e-5)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr, eps=1e-5)
    loss_fn = nn.MSELoss()

    obs = torch.zeros((rollout_steps, num_envs, state_dim))
    actions = torch.zeros((rollout_steps, num_envs))
    logprobs = torch.zeros((rollout_steps, num_envs))
    rewards = torch.zeros((rollout_steps, num_envs))
    dones = torch.zeros((rollout_steps, num_envs))
    values = torch.zeros((rollout_steps, num_envs))

    steps_list = []
    rewards_list = []
    total_steps = 0
    num_updates = max_steps // batch_size
    current_ep_rewards = np.zeros(num_envs)

    with tqdm(total=max_steps, desc=f"PPO seed={seed}", leave=False, disable=not show_progress) as pbar:
        for update in range(num_updates):
            
            # fixed length rollout
            for step in range(rollout_steps):
                total_steps += num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    probs = actor(next_obs)
                    dist = Categorical(probs)
                    action = dist.sample()
                    logprob = dist.log_prob(action)
                    value = critic(next_obs).squeeze(-1)

                values[step] = value
                actions[step] = action
                logprobs[step] = logprob

                next_obs_np, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
                
                rewards[step] = torch.tensor(reward).view(-1)
                next_obs = torch.Tensor(next_obs_np)
                next_done = torch.Tensor(terminated | truncated)

                #reward tracking
                current_ep_rewards += reward
                dones_np = terminated | truncated
                
                for i in range(num_envs):
                    if dones_np[i]:
                        steps_list.append(total_steps)
                        rewards_list.append(float(current_ep_rewards[i]))
                        current_ep_rewards[i] = 0.0
                            
                pbar.update(num_envs)

            # GAE
            with torch.no_grad():
                next_value = critic(next_obs).squeeze(-1)
                advantages = torch.zeros_like(rewards)
                lastgaelam = 0
                # traverseing backwards through the trajectory to compute advantages
                for t in reversed(range(rollout_steps)):
                    if t == rollout_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    
                    delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                
                returns = advantages + values

            b_obs = obs.reshape((-1, state_dim))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)

            b_inds = np.arange(batch_size)
            
            # mini - batches optimization
            for epoch in range(ppo_epochs):
                np.random.shuffle(b_inds) 
                
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    mb_obs = b_obs[mb_inds]
                    mb_actions = b_actions[mb_inds]
                    mb_logprobs = b_logprobs[mb_inds]
                    mb_returns = b_returns[mb_inds]
                    mb_advantages = b_advantages[mb_inds]
                    
                    # advantage normalization for optimization stability
                    if config.get("norm_adv", True):
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    probs = actor(mb_obs)
                    dist = Categorical(probs)
                    newlogprob = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()
                    newvalue = critic(mb_obs).squeeze(-1)

                    # ratio for clipping
                    logratio = newlogprob - mb_logprobs
                    ratio = logratio.exp()

                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
                    
                    # actor and critic losses
                    actor_loss = -torch.min(surr1, surr2).mean() - (entropy_coef * entropy)
                    critic_loss = loss_fn(newvalue, mb_returns)

                    actor_optim.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    actor_optim.step()

                    critic_optim.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                    critic_optim.step()

    envs.close()

    if pth:
        torch.save({
            "config": config,
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict()
        }, pth)

    return steps_list, rewards_list