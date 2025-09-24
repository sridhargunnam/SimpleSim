# -*- coding: utf-8 -*-
#
# PPO (Proximal Policy Optimization) Implementation for Clawbot
#
# 🤖 UPDATED FOR PURE REWARD TRAINING (2025-09-23):
# 
# ✅ FIXED: Removed hardcoded rotation logic from simulation
# ✅ FIXED: Implemented pure angle/distance reward function
# 🎯 NEW GOAL: Train model that learns to turn naturally without environment assistance
#
# TRAINING CHANGES:
# 1. ✅ Pure reward function based only on angle alignment and distance minimization
# 2. ✅ No forced rotation - model must learn differential drive behavior
# 3. ✅ Exponential angle reward strongly incentivizes turning
# 4. ✅ Combined rewards maximize coordinated approach + alignment
#
# This model should work directly on real robot without rotation overrides!
# Previous model (9644.36 reward) relied on simulation's automatic turning.
# See reward_function_backup.py for original complex reward implementation.

# Import and create an environment, such as Pendulum
import gymnasium as gym

# Import some libraries that we will use in this example to do RL
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, List
import signal
import sys
import time
import datetime
import os
import glob

class ValueNetwork(nn.Module):
  """Value network (critic) that estimates V(s) - the value of a state"""
  def __init__(self, obs: int):
    super(ValueNetwork, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(obs, 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, 1)
    )

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.net(obs)

class PPOPolicyNetwork(nn.Module):
  """Stochastic policy network for PPO"""
  def __init__(self, obs: int, act: int):
    super(PPOPolicyNetwork, self).__init__()
    self.act_dim = act
    
    # Shared layers
    self.net = nn.Sequential(
      nn.Linear(obs, 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU()
    )
    
    # Output layers for mean and log_std
    self.mean_layer = nn.Linear(128, act)
    # For PPO, we often use a learnable parameter for log_std instead of outputting it
    self.log_std = nn.Parameter(torch.zeros(act))

  def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass that returns action distribution parameters
    Args:
        obs: Observation tensor
    Returns:
        Tuple of (mean, std)
    """
    x = self.net(obs)
    mean = self.mean_layer(x)
    std = torch.exp(self.log_std.expand_as(mean))
    return mean, std

  def get_action_and_log_prob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample action and return its log probability"""
    mean, std = self.forward(obs)
    
    # PPO Fix: Add numerical stability checks
    mean = torch.clamp(mean, -10, 10)  # Prevent extreme values
    std = torch.clamp(std, 1e-6, 10)   # Ensure positive std, prevent extreme values
    
    normal = Normal(mean, std)
    action = normal.sample()
    log_prob = normal.log_prob(action).sum(dim=-1)
    
    # Apply tanh to bound the action and correct log prob
    action = torch.tanh(action)
    log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
    
    return action, log_prob

  def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the log probability and entropy of given actions"""
    mean, std = self.forward(obs)
    
    # PPO Fix: Add numerical stability checks
    mean = torch.clamp(mean, -10, 10)  # Prevent extreme values
    std = torch.clamp(std, 1e-6, 10)   # Ensure positive std, prevent extreme values
    
    normal = Normal(mean, std)
    
    # Inverse tanh to get the pre-tanh action (with numerical stability)
    actions_clamped = torch.clamp(actions, -0.999, 0.999)  # Prevent log(0)
    actions_pretanh = 0.5 * torch.log((1 + actions_clamped) / (1 - actions_clamped + 1e-8))
    
    log_prob = normal.log_prob(actions_pretanh).sum(dim=-1)
    log_prob = log_prob - torch.log(1 - actions_clamped.pow(2) + 1e-8).sum(dim=-1)
    
    entropy = normal.entropy().sum(dim=-1)
    
    return log_prob, entropy

class PPO:
  def __init__(self, obs_dim: int, act_dim: int):
    """Proximal Policy Optimization - An on-policy algorithm that takes safe, small policy steps.
    
    Key features:
    1. On-policy learning (no replay buffer)
    2. Clipped surrogate objective for stable updates
    3. Value function for advantage estimation  
    4. Multiple epochs per data collection
    
    Args:
        obs_dim (int): dim of observation space
        act_dim (int): dim of action space
    """
    self.obs_dim = obs_dim
    self.act_dim = act_dim
    self.step = 0

    # PPO networks
    self.policy = PPOPolicyNetwork(obs_dim, act_dim)
    self.value = ValueNetwork(obs_dim)

    # Organic learning PPO hyperparameters for natural behavior discovery
    self.lr = 3e-4  # Higher learning rate for faster exploration
    self.gamma = 0.99  # Standard discount for balanced short/long-term learning
    self.lam = 0.95  # Higher GAE for better advantage estimation
    self.clip_range = 0.2  # Standard clipping for stable but flexible updates
    self.value_coeff = 0.5  # Standard value loss weight
    self.entropy_coeff = 0.05  # Higher entropy for extensive exploration
    self.max_grad_norm = 0.5  # Standard gradient clipping

    # Optimizers
    self.optimizer = torch.optim.Adam(
        list(self.policy.parameters()) + list(self.value.parameters()), 
        lr=self.lr
    )

  def get_action_and_value(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get action, log probability, and value for given observation"""
    with torch.no_grad():
      action, log_prob = self.policy.get_action_and_log_prob(obs)
      value = self.value(obs).squeeze(-1)
      return action, log_prob, value

  def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool], last_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE)
    Args:
        rewards: List of rewards for each step
        values: List of value estimates for each step  
        dones: List of done flags for each step
        last_value: Value estimate for the state after the last step (for bootstrapping truncated episodes)
    """
    advantages = []
    gae = 0
    
    for i in reversed(range(len(rewards))):
      if i == len(rewards) - 1:
        # PPO Fix: Use last_value for truncated episodes, 0 for terminal episodes
        next_value = 0.0 if dones[i] else last_value
      else:
        next_value = values[i + 1]
      
      delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
      gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
      advantages.insert(0, gae)
    
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + torch.tensor(values, dtype=torch.float32)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns

  def update(self, observations: torch.Tensor, actions: torch.Tensor, log_probs_old: torch.Tensor,
            advantages: torch.Tensor, returns: torch.Tensor, epochs: int = 10) -> Tuple[float, float]:
    """Update policy and value networks using PPO"""
    
    total_policy_loss = 0
    total_value_loss = 0
    
    for epoch in range(epochs):
      # Evaluate current policy
      log_probs_new, entropy = self.policy.evaluate_actions(observations, actions)
      values_new = self.value(observations).squeeze(-1)
      
      # PPO clipped surrogate objective
      ratio = torch.exp(log_probs_new - log_probs_old)
      clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
      
      policy_loss1 = -advantages * ratio
      policy_loss2 = -advantages * clipped_ratio
      policy_loss = torch.max(policy_loss1, policy_loss2).mean()
      
      # Value loss (MSE)
      value_loss = F.mse_loss(values_new, returns)
      
      # Entropy bonus (encourages exploration)
      entropy_loss = -entropy.mean()
      
      # Total loss
      loss = policy_loss + self.value_coeff * value_loss + self.entropy_coeff * entropy_loss
      
      # Optimize
      self.optimizer.zero_grad()
      loss.backward()
      
      # PPO Fix: Enhanced gradient clipping for stability
      grad_norm = nn.utils.clip_grad_norm_(
          list(self.policy.parameters()) + list(self.value.parameters()),
          self.max_grad_norm
      )
      
      # Skip update if gradients are too large (numerical instability)
      if torch.isfinite(grad_norm):
          self.optimizer.step()
      else:
          print(f"⚠️  Skipping update due to infinite gradients at epoch {epoch}")
      
      total_policy_loss += policy_loss.item()
      total_value_loss += value_loss.item()
    
    return total_policy_loss / epochs, total_value_loss / epochs

  def sample(self, obs: torch.Tensor) -> np.ndarray:
    """Sample an action for given observation"""
    self.step += 1
    with torch.no_grad():
      action, _, _ = self.get_action_and_value(obs)
      return action.cpu().numpy()

import gymnasium as gym
from mujoco_env import ClawbotCan
from model_manager import cleanup_all_old_models


def save_model(agent, best_reward, episode, base_filename='ppo_clawbot_model', 
               save_type='completed'):
  """Save model parameters to timestamped file
  
  Args:
    agent: The PPO agent
    best_reward: Best reward achieved
    episode: Current episode number
    base_filename: Base name for the model file
    save_type: Type of save ('completed', 'interrupted', 'checkpoint')
  """
  # Create timestamp for unique filename
  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  filename = f"{base_filename}_{save_type}_{timestamp}.pth"
  
  # Save model data
  model_data = {
    'policy_state_dict': agent.policy.state_dict(),
    'value_state_dict': agent.value.state_dict(),
    'optimizer_state_dict': agent.optimizer.state_dict(),
    'best_total_reward': best_reward,
    'last_episode': episode,
    'save_type': save_type,
    'algorithm': 'PPO',
    'timestamp': timestamp,
    'save_time': datetime.datetime.now().isoformat()
  }
  
  torch.save(model_data, filename)
  print(f"💾 PPO Model saved to {filename} at episode {episode} with best reward: {best_reward:.2f}")
  
  return filename

def signal_handler(signum, frame, agent, best_reward, episode):
  """Handle Ctrl+C interruption by saving the model"""
  print(f"\n🛑 Training interrupted! Saving PPO model at episode {episode}...")
  filename = save_model(agent, best_reward, episode, save_type='interrupted')
  print(f"✅ Model safely saved. You can resume training or evaluate using: {filename}")
  sys.exit(0)

if __name__ == "__main__":
  # Clean up old models before starting new training
  print("🧹 Cleaning up old PPO model files before starting training...")
  cleanup_all_old_models(keep_count=3, model_pattern='ppo_clawbot_model_*.pth')
  
  # Environment configuration parameters
  CAN_X_RANGE = (-0.75, 0.75)      # X position range for can
  CAN_Y_RANGE = (0.4, 0.75)        # Y position range for can (forward hemisphere, away from claw)
  MIN_DISTANCE = 0.4               # Minimum distance between robot and can (always away from claw)
  
  print(f"🤖 PPO Training Configuration:")
  print(f"   Can X range: {CAN_X_RANGE[0]} to {CAN_X_RANGE[1]}")
  print(f"   Can Y range: {CAN_Y_RANGE[0]} to {CAN_Y_RANGE[1]}")
  print(f"   Min distance: {MIN_DISTANCE}")
  
  # Create environment with configurable parameters
  env = ClawbotCan(can_x_range=CAN_X_RANGE, can_y_range=CAN_Y_RANGE, min_distance=MIN_DISTANCE)
  # Create agent (PPO doesn't use a replay buffer)
  agent = PPO(obs_dim=3, act_dim=4)

  # Organic learning PPO training parameters for natural behavior discovery
  rollout_length = 2048  # Longer rollouts for more diverse experience
  update_epochs = 10     # More epochs for thorough learning
  batch_size = 64        # Standard mini-batch size
  
  # Training loop variables
  best_total_reward = -float('inf')
  last_policy_loss = 0.0
  last_value_loss = 0.0
  total_episodes = 0
  current_episode = 0
  render_new_best = False
  checkpoint_interval = 50  # Save checkpoint every 50 rollouts (different from episode-based)
  last_checkpoint_rollout = 0
  rollout_count = 0

  # Set up signal handler for Ctrl+C
  def interrupt_handler(signum, frame):
    signal_handler(signum, frame, agent, best_total_reward, current_episode)
  
  signal.signal(signal.SIGINT, interrupt_handler)

  print(f"🚀 Starting PPO training with rollout length: {rollout_length}")

  # PPO Training Loop (rollout-based, extended for organic learning)
  for rollout in range(2000):  # Extended training for natural behavior discovery
    rollout_count = rollout
    
    # Collect rollout data
    observations = []
    actions = []
    log_probs = []
    values = []
    rewards = []
    dones = []
    
    obs, info = env.reset()
    episode_reward = 0
    episode_length = 0
    
    for step in range(rollout_length):
      obs_tensor = torch.tensor(obs, dtype=torch.float32).view(1, -1)
      action, log_prob, value = agent.get_action_and_value(obs_tensor)
      
      # Store the data
      observations.append(obs.copy())
      actions.append(action.squeeze(0).cpu().numpy())
      log_probs.append(log_prob.cpu().item())
      values.append(value.cpu().item())
      
      # Step environment
      new_obs, reward, done, info = env.step(action.squeeze(0).cpu().numpy())
      
      rewards.append(reward)
      dones.append(done)
      episode_reward += reward
      episode_length += 1
      
      # Disable rendering during training for performance
      # Original logic: if render_new_best and episode_reward > 0:
      if False:  # Disabled for training efficiency
        env.render()
        time.sleep(0.05)
      
      obs = new_obs
      
      # Reset environment if done (longer episodes for organic learning)
      if done or episode_length >= 1000:
        # Track episode statistics
        if episode_reward > best_total_reward and episode_reward > 0:
          render_new_best = True
          print(f'🎉 NEW BEST REWARD: {episode_reward:.2f} (episode {total_episodes})')
        else:
          render_new_best = False
        
        best_total_reward = max(best_total_reward, episode_reward)
        # Enhanced logging with curriculum info
        curriculum_info = env.get_curriculum_info()
        print(f'Episode {total_episodes}: Reward: {episode_reward:.2f}, Steps: {episode_length} | {curriculum_info}')
        
        total_episodes += 1
        current_episode = total_episodes
        
        # Reset for next episode
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
    
    # PPO Fix: Compute value of the last observed state for bootstrapping
    with torch.no_grad():
      last_obs_tensor = torch.tensor(obs, dtype=torch.float32).view(1, -1)
      last_value = agent.value(last_obs_tensor).squeeze(-1).item()
    
    # Compute advantages and returns using GAE with proper bootstrapping
    advantages, returns = agent.compute_gae(rewards, values, dones, last_value)
    
    # Convert lists to tensors
    observations = torch.tensor(np.array(observations), dtype=torch.float32)
    actions = torch.tensor(np.array(actions), dtype=torch.float32)
    log_probs_old = torch.tensor(log_probs, dtype=torch.float32)
    
    # Update networks
    policy_loss, value_loss = agent.update(
        observations, actions, log_probs_old, advantages, returns, epochs=update_epochs
    )
    
    last_policy_loss = policy_loss
    last_value_loss = value_loss
    
    # Print rollout statistics
    avg_reward = np.mean(rewards)
    print(f'Rollout {rollout}: Avg Reward: {avg_reward:.3f}, Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}')
    
    # Save checkpoint periodically
    if rollout - last_checkpoint_rollout >= checkpoint_interval:
      print(f"💾 Saving PPO checkpoint at rollout {rollout}...")
      save_model(agent, best_total_reward, total_episodes, save_type='checkpoint')
      last_checkpoint_rollout = rollout
      
      # Clean up old models every few checkpoints
      if rollout % (checkpoint_interval * 3) == 0:
        cleanup_all_old_models(keep_count=2, model_pattern='ppo_clawbot_model_*.pth', verbose=False)

  # Save model parameters after training completion
  print(f"\n🎉 PPO Training completed successfully!")
  filename = save_model(agent, best_total_reward, total_episodes, save_type='completed')
  print(f"✅ Final PPO model saved as: {filename}")
  print(f"📊 Total episodes completed: {total_episodes}")
  
  # Final cleanup to keep only the most important models
  print("🧹 Final cleanup of old PPO model files...")
  cleanup_all_old_models(keep_count=2, model_pattern='ppo_clawbot_model_*.pth')
  
  # Close environment
  env.close()
