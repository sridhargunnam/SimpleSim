# -*- coding: utf-8 -*-

# Import and create an environment, such as Pendulum
import gymnasium as gym

# Import some libraries that we will use in this example to do RL
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
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
import math

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

class SACPolicyNetwork(nn.Module):
  def __init__(self, obs: int, act: int):
    super(SACPolicyNetwork, self).__init__()
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
    self.log_std_layer = nn.Linear(128, act)
    
    # Constrain log_std to reasonable range
    self.log_std_min = -20
    self.log_std_max = 2

  def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass that returns sampled action, log probability, and mean action
    Args:
        obs: Observation tensor
    Returns:
        Tuple of (sampled_action, log_prob, mean_action)
    """
    x = self.net(obs)
    
    mean = self.mean_layer(x)
    log_std = self.log_std_layer(x)
    log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
    std = log_std.exp()
    
    # Create normal distribution and sample using reparameterization trick
    normal = Normal(mean, std)
    x_t = normal.rsample()  # Reparameterization trick (rsample instead of sample)
    
    # Apply tanh to bound the action
    action = torch.tanh(x_t)
    
    # Calculate log probability with correction for tanh
    # log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
    # More stable version:
    log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
    log_prob = log_prob.sum(dim=-1, keepdim=True)
    
    # Mean action for deterministic evaluation
    mean_action = torch.tanh(mean)
    
    return action, log_prob, mean_action

  def sample_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
    """Sample deterministically (for evaluation)"""
    with torch.no_grad():
      _, _, mean_action = self.forward(obs)
      return mean_action

class SAC(nn.Module):
  def __init__(self, obs_dim: int, act_dim: int):
    """Soft Actor-Critic - An off-policy algorithm that maximizes both reward and entropy.
    
    Key features:
    1. Uses two Q-networks to reduce overestimation bias (like TD3)
    2. Maximizes entropy for better exploration 
    3. Automatic temperature (alpha) tuning
    4. More sample efficient than DDPG/TD3
    
    Args:
        obs_dim (int): dim of observation space
        act_dim (int): dim of action space
    """
    super(SAC, self).__init__()
    self.obs_dim = obs_dim
    self.act_dim = act_dim
    self.step = 0

    # Create two Q-networks and their targets (for double Q-learning)
    self.Q1 = QNetwork(obs_dim, act_dim)
    self.Q1_target = QNetwork(obs_dim, act_dim)
    self.Q1_target.copyfrom(self.Q1)

    self.Q2 = QNetwork(obs_dim, act_dim)
    self.Q2_target = QNetwork(obs_dim, act_dim)
    self.Q2_target.copyfrom(self.Q2)

    # SAC uses a stochastic policy (unlike DDPG/TD3)
    self.policy = SACPolicyNetwork(obs_dim, act_dim)

    # Automatic temperature tuning
    self.target_entropy = -act_dim  # Heuristic: -|A|
    self.log_alpha = torch.zeros(1, requires_grad=True)
    
    # Create optimizers
    self.q1_optim = torch.optim.Adam(self.Q1.parameters(), lr=0.0003)
    self.q2_optim = torch.optim.Adam(self.Q2.parameters(), lr=0.0003)
    self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=0.0003)
    self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=0.0003)

  @property
  def alpha(self):
    """Get the current alpha (temperature) value"""
    return self.log_alpha.exp()

  def fit(self, dataset: deque, num_samples: int=10, skip_every: int=5) -> Tuple[float, float]:
    """Sample from the dataset and do a training step of SAC.

    Args:
        dataset (deque): A deque of (obs, act, new_obs, rew, terminal) tuples.
        num_samples (int, optional): Number of samples. Defaults to 10.
        skip_every (int, optional): Do a training step only once every {skip_every} steps. Defaults to 5.

    Returns:
        Tuple[float, float]: Average Q-loss, policy-loss
    """
    if (self.step % skip_every) != 0: return 0.0, 0.0
    if len(dataset) < num_samples: return 0.0, 0.0

    # Sample a batch of experiences
    with torch.no_grad():
      minibatch  = random.sample(dataset, k=num_samples)
      state      = torch.tensor(np.array([item[0] for item in minibatch]), dtype=torch.float32)
      action     = torch.tensor(np.array([item[1] for item in minibatch]), dtype=torch.float32)
      next_state = torch.tensor(np.array([item[2] for item in minibatch]), dtype=torch.float32)
      reward     = torch.tensor(np.array([item[3] for item in minibatch]), dtype=torch.float32)
      term       = torch.tensor(np.array([item[4] for item in minibatch]), dtype=torch.float32)

      # Calculate target Q-value with entropy term
      next_action, next_log_prob, _ = self.policy(next_state)
      next_q1_target = self.Q1_target(next_state, next_action).flatten()
      next_q2_target = self.Q2_target(next_state, next_action).flatten()
      min_q_target = torch.min(next_q1_target, next_q2_target)
      
      # SAC adds entropy bonus to the target
      target_q_value = reward + 0.99 * (1 - term) * (min_q_target - self.alpha * next_log_prob.flatten())

    # Update Q-networks
    self.q1_optim.zero_grad()
    pred_q1_value = self.Q1(state, action).flatten()
    q1_loss = nn.MSELoss()(pred_q1_value, target_q_value)
    q1_loss.backward()
    self.q1_optim.step()

    self.q2_optim.zero_grad()
    pred_q2_value = self.Q2(state, action).flatten()
    q2_loss = nn.MSELoss()(pred_q2_value, target_q_value)
    q2_loss.backward()
    self.q2_optim.step()

    # Update policy network
    self.policy_optim.zero_grad()
    action_pi, log_prob_pi, _ = self.policy(state)
    q1_pi = self.Q1(state, action_pi).flatten()
    q2_pi = self.Q2(state, action_pi).flatten()
    min_q = torch.min(q1_pi, q2_pi)
    
    # SAC policy objective: maximize Q-value and entropy
    policy_loss = (self.alpha.detach() * log_prob_pi.flatten() - min_q).mean()
    policy_loss.backward()
    self.policy_optim.step()

    # Update alpha (temperature parameter)
    self.alpha_optim.zero_grad()
    alpha_loss = -(self.log_alpha * (log_prob_pi.detach().flatten() + self.target_entropy)).mean()
    alpha_loss.backward()
    self.alpha_optim.step()

    # Update target networks
    with torch.no_grad():
      self.Q1_target.copyfrom(self.Q1, polyak=0.995)  # SAC uses slower target updates
      self.Q2_target.copyfrom(self.Q2, polyak=0.995)
    
    avg_q_loss = (q1_loss.detach().cpu().item() + q2_loss.detach().cpu().item()) / 2
    policy_loss_value = policy_loss.detach().cpu().item()
    
    return avg_q_loss, policy_loss_value

  def sample(self, obs: torch.Tensor, deterministic=False) -> np.ndarray:
    """Sample an action from the policy network.
    
    Args:
        obs (torch.Tensor): The current observation.
        deterministic (bool): If True, use mean action (for evaluation)
        
    Returns:
        np.ndarray: The action to take.
    """
    self.step += 1
    with torch.no_grad():
      if deterministic:
        action = self.policy.sample_deterministic(obs)
      else:
        action, _, _ = self.policy(obs)
      return action.cpu().numpy()

import gymnasium as gym
from mujoco_env import ClawbotCan
from model_manager import cleanup_all_old_models


def save_model(agent, best_reward, episode, base_filename='sac_clawbot_model', 
               save_type='completed'):
  """Save model parameters to timestamped file
  
  Args:
    agent: The SAC agent
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
    'alpha_optimizer_state_dict': agent.alpha_optim.state_dict(),
    'log_alpha': agent.log_alpha,
    'target_entropy': agent.target_entropy,
    'best_total_reward': best_reward,
    'last_episode': episode,
    'save_type': save_type,
    'algorithm': 'SAC',
    'timestamp': timestamp,
    'save_time': datetime.datetime.now().isoformat()
  }
  
  torch.save(model_data, filename)
  print(f"💾 SAC Model saved to {filename} at episode {episode} with best reward: {best_reward:.2f}")
  
  return filename

def signal_handler(signum, frame, agent, best_reward, episode):
  """Handle Ctrl+C interruption by saving the model"""
  print(f"\n🛑 Training interrupted! Saving SAC model at episode {episode}...")
  filename = save_model(agent, best_reward, episode, save_type='interrupted')
  print(f"✅ Model safely saved. You can resume training or evaluate using: {filename}")
  sys.exit(0)

if __name__ == "__main__":
  # Clean up old models before starting new training
  print("🧹 Cleaning up old SAC model files before starting training...")
  cleanup_all_old_models(keep_count=3, model_pattern='sac_clawbot_model_*.pth')
  
  # Environment configuration parameters
  CAN_X_RANGE = (-0.75, 0.75)      # X position range for can
  CAN_Y_RANGE = (0.4, 0.75)        # Y position range for can (forward hemisphere, away from claw)
  MIN_DISTANCE = 0.4               # Minimum distance between robot and can (always away from claw)
  
  print(f"🤖 SAC Training Configuration:")
  print(f"   Can X range: {CAN_X_RANGE[0]} to {CAN_X_RANGE[1]}")
  print(f"   Can Y range: {CAN_Y_RANGE[0]} to {CAN_Y_RANGE[1]}")
  print(f"   Min distance: {MIN_DISTANCE}")
  
  # Create environment with configurable parameters
  env = ClawbotCan(can_x_range=CAN_X_RANGE, can_y_range=CAN_Y_RANGE, min_distance=MIN_DISTANCE)
  # Create agent and dataset storage
  agent = SAC(obs_dim=3, act_dim=4)
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
      # Sample an action (SAC naturally explores via its stochastic policy)
      action = agent.sample(torch.tensor(obs, dtype=torch.float32).view(1, -1), deterministic=False).flatten()

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
        alpha_value = agent.alpha.item()
        print(f'Total Reward: {total_reward:.2f} Episode: {episode} Steps: {step} Q Loss: {last_q_error:.4f} Policy Loss: {last_policy_error:.4f} Alpha: {alpha_value:.4f}')
        
        # Save checkpoint periodically
        if episode - last_checkpoint_episode >= checkpoint_interval:
          print(f"💾 Saving SAC checkpoint at episode {episode}...")
          save_model(agent, best_total_reward, episode, save_type='checkpoint')
          last_checkpoint_episode = episode
          
          # Clean up old models every few checkpoints to keep storage manageable
          if episode % (checkpoint_interval * 3) == 0:  # Every 3000 episodes
            cleanup_all_old_models(keep_count=2, model_pattern='sac_clawbot_model_*.pth', verbose=False)
        
        break

  # Save model parameters after training completion
  print(f"\n🎉 SAC Training completed successfully!")
  filename = save_model(agent, best_total_reward, total_episodes, save_type='completed')
  print(f"✅ Final SAC model saved as: {filename}")
  
  # Final cleanup to keep only the most important models
  print("🧹 Final cleanup of old SAC model files...")
  cleanup_all_old_models(keep_count=2, model_pattern='sac_clawbot_model_*.pth')
  
  # Close environment
  env.close()
