# -*- coding: utf-8 -*-
# 
# TD3 (Twin Delayed Deep Deterministic Policy Gradient) Implementation for Clawbot
# 
# ⚠️  KNOWN ISSUE (2025-09-22): This TD3 implementation has a behavioral problem where
# the trained model moves BACKWARD while trying to align its angle with the target,
# instead of moving toward the target object. This results in poor performance.
# 
# The issue likely stems from:
# - Reward function not properly encouraging forward movement
# - Action space interpretation problems
# - Training instability in the twin critic networks
# 
# TODO: Fix the backward movement issue by:
# 1. Reviewing the reward function in mujoco_env.py
# 2. Checking action mapping (positive/negative action interpretation)
# 3. Potentially retraining with modified reward weights
# 
# For comparison, PPO achieves 9644.36 reward and approaches targets correctly.
# See KNOWN_ISSUES.md for detailed analysis.

# Import and create an environment, such as Pendulum
import gymnasium as gym

# Import some libraries that we will use in this example to do RL
import torch
from torch import nn
import random
from collections import deque
import numpy as np
from typing import Tuple
import signal
import sys
import time
import datetime
import os
import glob

class QNetwork(nn.Module):
  def __init__(self, obs: int, act: int):
    super(QNetwork, self).__init__()
    # Create a basic neural network and optimizer which takes in an observation and action
    # then outputs a value representing the Q-value of doing that action on this state
    self.net = nn.Sequential(
      nn.Linear(obs + act, 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, 1)
    )

  def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    Given an observation and an action, return the Q-value of that action on that state.
    """
    # Combine observations and actions to send into the neural network
    obs_act = torch.cat([obs, action], dim=1)
    # Grab Q-values for all (s, a) pairs
    q_values = self.net(obs_act)
    return q_values
  
  def copyfrom(self, other, polyak=0.0):
    """
    Copy the weights from another QNetwork with Polyak averaging.
    Args:
        other (QNetwork): The other QNetwork to copy weights from.
        polyak (float, optional): The Polyak averaging factor. Defaults to 0.0.
    """
    for param, other_param in zip(self.net.parameters(), other.net.parameters()):
      param.data.copy_(polyak * param.data + (1 - polyak) * other_param.data)
    self.net.eval()  # Set to evaluation mode

class PolicyNetwork(nn.Module):
  def __init__(self, obs: int, act: int):
    super(PolicyNetwork, self).__init__()

    self.net = nn.Sequential(
      nn.Linear(obs, 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, act)
    )

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    mean_action = self.net(obs)
    return mean_action

  def copyfrom(self, other, polyak=0.0):
    """
    Copy the weights from another PolicyNetwork with Polyak averaging.
    Args:
        other (PolicyNetwork): The other PolicyNetwork to copy weights from.
        polyak (float, optional): The Polyak averaging factor. Defaults to 0.0.
    """
    for param, other_param in zip(self.net.parameters(), other.net.parameters()):
      param.data.copy_(polyak * param.data + (1 - polyak) * other_param.data)
    self.net.eval()  # Set to evaluation mode

class TD3(nn.Module):
  def __init__(self, obs_dim: int, act_dim: int):
    """Twin Delayed DDPG - An improved version of DDPG.
    
    Key improvements over DDPG:
    1. Uses two Q-networks to reduce overestimation bias
    2. Adds noise to target actions for smoother Q-functions  
    3. Delays policy updates to let Q-networks stabilize
    
    Args:
        obs_dim (int): dim of observation space
        act_dim (int): dim of action space
    """
    super(TD3, self).__init__()
    self.obs_dim = obs_dim
    self.act_dim = act_dim
    self.step = 0

    # Create two Q-networks and their targets (key TD3 improvement)
    self.Q1 = QNetwork(obs_dim, act_dim)
    self.Q1_target = QNetwork(obs_dim, act_dim)
    self.Q1_target.copyfrom(self.Q1)

    self.Q2 = QNetwork(obs_dim, act_dim)
    self.Q2_target = QNetwork(obs_dim, act_dim)
    self.Q2_target.copyfrom(self.Q2)

    # Create the policy network and target policy network
    self.policy = PolicyNetwork(obs_dim, act_dim)
    self.policy_target = PolicyNetwork(obs_dim, act_dim)
    self.policy_target.copyfrom(self.policy)

    # Create optimizers for both Q-networks and policy network
    self.q1_optim = torch.optim.Adam(self.Q1.parameters(), lr=0.0001)
    self.q2_optim = torch.optim.Adam(self.Q2.parameters(), lr=0.0001)
    self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=0.0001)

    # TD3 hyperparameters
    self.policy_noise = 0.2        # Noise added to target policy
    self.noise_clip = 0.5          # Clip noise to this range
    self.policy_freq = 2           # How often to update policy (delayed updates)

  def fit(self, dataset: deque, num_samples: int=10, skip_every: int=5) -> Tuple[float, float]:
    """Sample from the dataset and do a training step of TD3.

    Args:
        dataset (deque): A deque of (obs, act, new_obs, rew, terminal) tuples.
        num_samples (int, optional): Number of samples. Defaults to 10.
        skip_every (int, optional): Do a training step only once every {skip_every} steps. Defaults to 5.

    Returns:
        Tuple[float, float]: Q-loss (average of Q1 and Q2), policy-loss (0.0 if no policy update)
    """
    if (self.step % skip_every) != 0: return 0.0, 0.0
    if len(dataset) < num_samples: return 0.0, 0.0

    # Sample a bunch of datapoints (s, a, s', r, term)
    with torch.no_grad():
      minibatch  = random.sample(dataset, k=num_samples)
      state      = torch.tensor(np.array([item[0] for item in minibatch]), dtype=torch.float32)
      action     = torch.tensor(np.array([item[1] for item in minibatch]), dtype=torch.float32)
      next_state = torch.tensor(np.array([item[2] for item in minibatch]), dtype=torch.float32)
      reward     = torch.tensor(np.array([item[3] for item in minibatch]), dtype=torch.float32)
      term       = torch.tensor(np.array([item[4] for item in minibatch]), dtype=torch.float32)

      # TD3 Improvement #2: Target Policy Smoothing
      # Add clipped noise to target action to smooth Q-function
      noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
      next_action = (torch.tanh(self.policy_target(next_state)) + noise).clamp(-1.0, 1.0)
      
      # TD3 Improvement #1: Clipped Double Q-Learning
      # Use minimum of two target Q-networks to reduce overestimation bias
      next_q1_value = self.Q1_target(next_state, next_action).flatten()
      next_q2_value = self.Q2_target(next_state, next_action).flatten()
      next_q_value = torch.min(next_q1_value, next_q2_value)
      target_q_value = reward + 0.99 * next_q_value * (1 - term)

    # Update both Q-networks with stability improvements
    self.q1_optim.zero_grad()
    pred_q1_value = self.Q1(state, action).flatten()
    # TD3 Fix: Clip Q-values to prevent explosion (adjusted for new reward scale)
    pred_q1_value = torch.clamp(pred_q1_value, -2000, 2000)
    target_q_value_clipped = torch.clamp(target_q_value, -2000, 2000)
    q1_error = nn.MSELoss()(pred_q1_value, target_q_value_clipped)
    q1_error.backward()
    # TD3 Fix: Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(self.Q1.parameters(), max_norm=10.0)
    self.q1_optim.step()

    self.q2_optim.zero_grad()
    pred_q2_value = self.Q2(state, action).flatten()
    pred_q2_value = torch.clamp(pred_q2_value, -2000, 2000)
    q2_error = nn.MSELoss()(pred_q2_value, target_q_value_clipped)
    q2_error.backward()
    torch.nn.utils.clip_grad_norm_(self.Q2.parameters(), max_norm=10.0)
    self.q2_optim.step()

    # Copy weights to target Q-networks
    with torch.no_grad():
      self.Q1_target.copyfrom(self.Q1, polyak=0.99)
      self.Q2_target.copyfrom(self.Q2, polyak=0.99)

    # TD3 Improvement #3: Delayed Policy Updates
    # Update policy less frequently to let Q-networks stabilize
    policy_loss = 0.0
    if self.step % self.policy_freq == 0:
      # Update policy by maximizing Q1 (we could use either Q1 or Q2)
      self.policy_optim.zero_grad()
      policy_actions = torch.tanh(self.policy(state))
      q_values = self.Q1(state, policy_actions)
      # TD3 Fix: Clip Q-values in policy loss to prevent explosion (adjusted for new reward scale)
      q_values = torch.clamp(q_values, -2000, 2000)
      policy_loss = -torch.mean(q_values)
      policy_loss.backward()
      # TD3 Fix: Gradient clipping for policy network
      torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
      self.policy_optim.step()
      
      # Update target policy network
      with torch.no_grad():
        self.policy_target.copyfrom(self.policy, polyak=0.99)
    
    avg_q_error = (q1_error.detach().cpu().item() + q2_error.detach().cpu().item()) / 2
    policy_loss_value = policy_loss.detach().cpu().item() if isinstance(policy_loss, torch.Tensor) else 0.0
    
    return avg_q_error, policy_loss_value

  def sample(self, obs: torch.Tensor, noise=0.1) -> np.ndarray:
    """Sample an action from the policy network, adding some noise for exploration.
    Args:
        obs (torch.Tensor): The current observation.
        noise (float, optional): The standard deviation of the Gaussian noise to add. Defaults to 0.1.
    Returns:
        np.ndarray: The action to take.
    """
    self.step += 1
    with torch.no_grad():
      # TD3 Fix: Get deterministic action first, then add noise in action space
      mean_action = torch.tanh(self.policy(obs))
      # Add noise and clamp to ensure action stays in [-1, 1] bounds
      action = (mean_action + noise * torch.randn(1, self.act_dim)).clamp(-1.0, 1.0)
      return action.cpu().numpy()

import gymnasium as gym
from mujoco_env import ClawbotCan
from model_manager import cleanup_all_old_models


def save_model(agent, best_reward, episode, base_filename='td3_clawbot_model', 
               save_type='completed'):
  """Save model parameters to timestamped file
  
  Args:
    agent: The TD3 agent
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
    'q1_network_state_dict': agent.Q1.state_dict(),
    'q2_network_state_dict': agent.Q2.state_dict(),
    'policy_optimizer_state_dict': agent.policy_optim.state_dict(),
    'q1_optimizer_state_dict': agent.q1_optim.state_dict(),
    'q2_optimizer_state_dict': agent.q2_optim.state_dict(),
    'best_total_reward': best_reward,
    'last_episode': episode,
    'save_type': save_type,
    'algorithm': 'TD3',
    'timestamp': timestamp,
    'save_time': datetime.datetime.now().isoformat()
  }
  
  torch.save(model_data, filename)
  print(f"💾 TD3 Model saved to {filename} at episode {episode} with best reward: {best_reward:.2f}")
  
  return filename

def signal_handler(signum, frame, agent, best_reward, episode):
  """Handle Ctrl+C interruption by saving the model"""
  print(f"\n🛑 Training interrupted! Saving TD3 model at episode {episode}...")
  filename = save_model(agent, best_reward, episode, save_type='interrupted')
  print(f"✅ Model safely saved. You can resume training or evaluate using: {filename}")
  sys.exit(0)

if __name__ == "__main__":
  # Clean up old models before starting new training
  print("🧹 Cleaning up old TD3 model files before starting training...")
  cleanup_all_old_models(keep_count=3, model_pattern='td3_clawbot_model_*.pth')
  
  # Environment configuration parameters
  CAN_X_RANGE = (-0.75, 0.75)      # X position range for can
  CAN_Y_RANGE = (0.4, 0.75)        # Y position range for can (forward hemisphere, away from claw)
  MIN_DISTANCE = 0.4               # Minimum distance between robot and can (always away from claw)
  
  print(f"🤖 TD3 Training Configuration:")
  print(f"   Can X range: {CAN_X_RANGE[0]} to {CAN_X_RANGE[1]}")
  print(f"   Can Y range: {CAN_Y_RANGE[0]} to {CAN_Y_RANGE[1]}")
  print(f"   Min distance: {MIN_DISTANCE}")
  
  # Create environment with configurable parameters
  env = ClawbotCan(can_x_range=CAN_X_RANGE, can_y_range=CAN_Y_RANGE, min_distance=MIN_DISTANCE)
  # Create agent and dataset storage
  agent = TD3(obs_dim=3, act_dim=4)
  dataset = deque(maxlen=100000) # (obs, act, new_obs, rew, terminal)

  # Training loop variables
  best_total_reward = -float('inf')
  last_q_error = 0.0
  last_policy_error = 0.0
  total_episodes = 10000
  render_threshold = total_episodes - 5  # 90% of training
  current_episode = 0
  render_new_best = False  # Flag to render when hitting new best reward
  checkpoint_interval = 1000  # Save checkpoint every 1000 episodes
  last_checkpoint_episode = 0

  # Set up signal handler for Ctrl+C
  def interrupt_handler(signum, frame):
    signal_handler(signum, frame, agent, best_total_reward, current_episode)
  
  signal.signal(signal.SIGINT, interrupt_handler)

  # Train for episodes
  for episode in range(total_episodes):
    current_episode = episode
    obs, info = env.reset()
    total_reward = 0

    # For each epoch, balanced episode length for 8-hour training efficiency
    for step in range(512):
      # Sample an action
      action = agent.sample(torch.tensor(obs, dtype=torch.float32).view(1, -1)).flatten()

      # print(action)
      # Try the action out
      new_obs, rew, term, info = env.step(action)
      
      # Disable rendering during training for performance (can be re-enabled for final episodes)
      should_render = False  # Disabled for training efficiency
      # Original logic: should_render = (episode >= render_threshold or (total_reward > 0 or render_new_best))
      if should_render:
        env.render()
        time.sleep(0.05)  # Slow down rendering for better visibility

      # Store the result in the dataset and redefine the current observation
      term = term or step == 511 # terminal if we reached the time limit (512 steps)
      dataset.append([obs.flatten(), action, new_obs.flatten(), rew, int(term)])
      obs = new_obs
      total_reward += rew

      # Train our agent
      q_err, policy_err = agent.fit(dataset)
      if q_err != 0.0 or policy_err != 0.0: # only update if we did a training step
        last_q_error = q_err
        last_policy_error = policy_err
      if term:
        # Check if we hit a new best reward and should render next episode
        if total_reward > best_total_reward and total_reward > 0:
          render_new_best = True
          print(f'🎉 NEW BEST REWARD: {total_reward:.2f} (rendering next episode)')
        else:
          render_new_best = False
          
        best_total_reward = max(best_total_reward, total_reward)
        print(f'Total Reward: {total_reward:.2f} Episode: {episode} Steps: {step} Q Loss: {last_q_error:.4f} Policy Loss: {last_policy_error:.4f}')
        
        # Save checkpoint periodically
        if episode - last_checkpoint_episode >= checkpoint_interval:
          print(f"💾 Saving TD3 checkpoint at episode {episode}...")
          save_model(agent, best_total_reward, episode, save_type='checkpoint')
          last_checkpoint_episode = episode
          
          # Clean up old models every few checkpoints to keep storage manageable
          if episode % (checkpoint_interval * 3) == 0:  # Every 3000 episodes
            cleanup_all_old_models(keep_count=2, model_pattern='td3_clawbot_model_*.pth', verbose=False)
        
        break

  # Save model parameters after training completion
  print(f"\n🎉 TD3 Training completed successfully!")
  filename = save_model(agent, best_total_reward, total_episodes, save_type='completed')
  print(f"✅ Final TD3 model saved as: {filename}")
  
  # Final cleanup to keep only the most important models
  print("🧹 Final cleanup of old TD3 model files...")
  cleanup_all_old_models(keep_count=2, model_pattern='td3_clawbot_model_*.pth')
  
  # Close environment
  env.close()
