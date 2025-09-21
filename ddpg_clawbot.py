# -*- coding: utf-8 -*-

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

class DDPG(nn.Module):
  def __init__(self, obs_dim: int, act_dim: int):
    """See https://spinningup.openai.com/en/latest/algorithms/ddpg.html for reference.

    Args:
        obs_dim (int): dim of observation space
        act_dim (int): dim of action space
    """
    super(DDPG, self).__init__()
    self.obs_dim = obs_dim
    self.act_dim = act_dim
    self.step = 0

    # Create the Q-network and target Q-network
    self.Q = QNetwork(obs_dim, act_dim)
    self.Q_target = QNetwork(obs_dim, act_dim)
    self.Q_target.copyfrom(self.Q)  # Copy initial weights

    # Create the policy network and target policy network
    self.policy = PolicyNetwork(obs_dim, act_dim)
    self.policy_target = PolicyNetwork(obs_dim, act_dim)
    self.policy_target.copyfrom(self.policy)  # Copy initial weights

    # Create optimizers for the Q-network and policy network, but not the target networks
    self.q_optim = torch.optim.Adam(self.Q.parameters(), lr=0.0001)
    self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=0.0001)

  def fit(self, dataset: deque, num_samples: int=10, skip_every: int=5) -> Tuple[float, float]:
    """Sample from the dataset and do a training step of DDPG.

    Args:
        dataset (deque): A deque of (obs, act, new_obs, rew, terminal) tuples.
        num_samples (int, optional): Number of samples. Defaults to 10.
        skip_every (int, optional): Do a training step only once every {skip_every} steps. Defaults to 5.

    Returns:
        Tuple[float, float, float]: Q-loss, policy-loss, (0.0 if no update was done)
    """
    if (self.step % skip_every) != 0: return 0.0, 0.0
    if len(dataset) < num_samples: return 0.0, 0.0

    # Sample a bunch of datapoints (s, a, s', r, term)
    # Note that the neural network is typed float32 by default, so we have to convert the dtype
    with torch.no_grad():
      minibatch  = random.sample(dataset, k=num_samples)
      state      = torch.tensor(np.array([item[0] for item in minibatch]), dtype=torch.float32)
      action     = torch.tensor(np.array([item[1] for item in minibatch]), dtype=torch.float32)
      next_state = torch.tensor(np.array([item[2] for item in minibatch]), dtype=torch.float32)
      reward     = torch.tensor(np.array([item[3] for item in minibatch]), dtype=torch.float32)
      term       = torch.tensor(np.array([item[4] for item in minibatch]), dtype=torch.float32)

      # Calculate what the Q-value "should" be using the Bellman Equation and use that as our target value
      # Bellman Equation: Q(s, a) = ∑{ {R(s,a,s') + γ max(Q(s', a') for a' in A)*(1-T)} * {P(s'|s,a) for s' in S} }
      next_action = torch.tanh(self.policy_target(next_state))
      next_q_value = self.Q_target(next_state, next_action).flatten()
      target_q_value = reward + 0.99 * next_q_value * (1 - term) # set discount factor to 0.99

    # backprop the Q-network using the action we took
    self.q_optim.zero_grad()
    pred_q_value = self.Q(state, action).flatten()
    q_error = nn.MSELoss()(pred_q_value, target_q_value)
    q_error.backward()
    self.q_optim.step()

    # copy the weights to our target network for Q
    with torch.no_grad():
      self.Q_target.copyfrom(self.Q, polyak=0.99)

    # backprop the policy network using the action from the policy network
    # note that we do not use the action we took in the environment here, since we want to improve the policy
    # we want to maximize the Q-value for the action the policy network would take
    self.policy_optim.zero_grad()
    policy_loss = -torch.mean(self.Q(state, torch.tanh(self.policy(state)))) # maximize Q value instead of minimizing loss
    policy_loss.backward()
    self.policy_optim.step()
    with torch.no_grad():
      self.policy_target.copyfrom(self.policy, polyak=0.99)
    
    return q_error.detach().cpu().item(), policy_loss.detach().cpu().item()

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
      action = torch.tanh(self.policy(obs) + noise * torch.randn(1, self.act_dim))
      return action.cpu().numpy()

import gymnasium as gym
from mujoco_env import ClawbotCan
from model_manager import cleanup_all_old_models


def save_model(agent, best_reward, episode, base_filename='ddpg_clawbot_model', 
               save_type='completed'):
  """Save model parameters to timestamped file
  
  Args:
    agent: The DDPG agent
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
    'q_network_state_dict': agent.Q.state_dict(),
    'policy_optimizer_state_dict': agent.policy_optim.state_dict(),
    'q_optimizer_state_dict': agent.q_optim.state_dict(),
    'best_total_reward': best_reward,
    'last_episode': episode,
    'save_type': save_type,
    'timestamp': timestamp,
    'save_time': datetime.datetime.now().isoformat()
  }
  
  torch.save(model_data, filename)
  print(f"💾 Model saved to {filename} at episode {episode} with best reward: {best_reward:.2f}")
  
  return filename

def signal_handler(signum, frame, agent, best_reward, episode):
  """Handle Ctrl+C interruption by saving the model"""
  print(f"\n🛑 Training interrupted! Saving model at episode {episode}...")
  filename = save_model(agent, best_reward, episode, save_type='interrupted')
  print(f"✅ Model safely saved. You can resume training or evaluate using: {filename}")
  sys.exit(0)

if __name__ == "__main__":
  # Clean up old models before starting new training
  print("🧹 Cleaning up old model files before starting training...")
  cleanup_all_old_models(keep_count=3)
  
  # Environment configuration parameters
  CAN_X_RANGE = (-0.75, 0.75)      # X position range for can
  CAN_Y_RANGE = (0.4, 0.75)        # Y position range for can (forward hemisphere, away from claw)
  MIN_DISTANCE = 0.4               # Minimum distance between robot and can (always away from claw)
  
  print(f"🤖 Environment Configuration:")
  print(f"   Can X range: {CAN_X_RANGE[0]} to {CAN_X_RANGE[1]}")
  print(f"   Can Y range: {CAN_Y_RANGE[0]} to {CAN_Y_RANGE[1]}")
  print(f"   Min distance: {MIN_DISTANCE}")
  
  # Create environment with configurable parameters
  env = ClawbotCan(can_x_range=CAN_X_RANGE, can_y_range=CAN_Y_RANGE, min_distance=MIN_DISTANCE)
  # Create agent and dataset storage
  agent = DDPG(obs_dim=3, act_dim=4)
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

    # For each epoch, try 200 steps before ending the episode and resetting pendulum position
    for step in range(200):
      # Sample an action
      action = agent.sample(torch.tensor(obs, dtype=torch.float32).view(1, -1)).flatten()

      # print(action)
      # Try the action out
      new_obs, rew, term, info = env.step(action)
      
      # Render environment if conditions are met
      should_render = (episode >= render_threshold or (total_reward > 0 or render_new_best))
      if should_render:
        env.render()
        time.sleep(0.05)  # Slow down rendering for better visibility

      # Store the result in the dataset and redefine the current observation
      term = term or step == 199 # terminal if we reached the time limit
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
          print(f"💾 Saving checkpoint at episode {episode}...")
          save_model(agent, best_total_reward, episode, save_type='checkpoint')
          last_checkpoint_episode = episode
          
          # Clean up old models every few checkpoints to keep storage manageable
          if episode % (checkpoint_interval * 3) == 0:  # Every 3000 episodes
            cleanup_all_old_models(keep_count=2, verbose=False)
        
        break

  # Save model parameters after training completion
  print(f"\n🎉 Training completed successfully!")
  filename = save_model(agent, best_total_reward, total_episodes, save_type='completed')
  print(f"✅ Final model saved as: {filename}")
  
  # Final cleanup to keep only the most important models
  print("🧹 Final cleanup of old model files...")
  cleanup_all_old_models(keep_count=2)
  
  # Close environment
  env.close()