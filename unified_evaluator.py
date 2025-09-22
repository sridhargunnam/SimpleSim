# -*- coding: utf-8 -*-
"""
Unified Model Evaluator for All RL Algorithms
Evaluates saved models from DDPG, TD3, SAC, and PPO algorithms
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import time
import math
from mujoco_env import ClawbotCan
import mujoco
import glob
import os

# Import network classes for all algorithms

# DDPG/TD3 Networks
class QNetwork(nn.Module):
    def __init__(self, obs: int, act: int):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs + act, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs_act = torch.cat([obs, action], dim=1)
        return self.net(obs_act)

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
        return self.net(obs)

# SAC Networks
class SACPolicyNetwork(nn.Module):
    def __init__(self, obs: int, act: int):
        super(SACPolicyNetwork, self).__init__()
        self.act_dim = act
        
        self.net = nn.Sequential(
            nn.Linear(obs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        self.mean_layer = nn.Linear(128, act)
        self.log_std_layer = nn.Linear(128, act)
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, obs: torch.Tensor):
        x = self.net(obs)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        mean_action = torch.tanh(mean)
        return action, None, mean_action

    def sample_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, _, mean_action = self.forward(obs)
            return mean_action

# PPO Networks
class ValueNetwork(nn.Module):
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
    def __init__(self, obs: int, act: int):
        super(PPOPolicyNetwork, self).__init__()
        self.act_dim = act
        
        self.net = nn.Sequential(
            nn.Linear(obs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        self.mean_layer = nn.Linear(128, act)
        self.log_std = nn.Parameter(torch.zeros(act))

    def forward(self, obs: torch.Tensor):
        x = self.net(obs)
        mean = self.mean_layer(x)
        std = torch.exp(self.log_std.expand_as(mean))
        return mean, std

    def get_action_and_log_prob(self, obs: torch.Tensor):
        mean, std = self.forward(obs)
        normal = Normal(mean, std)
        action = normal.sample()
        action = torch.tanh(action)
        return action, None

    def sample_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mean, _ = self.forward(obs)
            return torch.tanh(mean)

class UnifiedModelEvaluator:
    def __init__(self, model_path: str, obs_dim: int = 3, act_dim: int = 4,
                 can_x_range: tuple = (-0.75, 0.75),
                 can_y_range: tuple = (0.4, 0.75),
                 min_distance: float = 0.4,
                 angle_range_deg: tuple = (-90, 90),
                 distance_range: tuple = (0.4, 0.75)):
        """Initialize the unified model evaluator
        
        Args:
            model_path (str): Path to the saved model
            obs_dim (int): Observation space dimension
            act_dim (int): Action space dimension
            can_x_range: X position range for training environment
            can_y_range: Y position range for training environment  
            min_distance: Minimum distance for training environment
            angle_range_deg: Angle range in degrees for evaluation tests
            distance_range: Distance range for evaluation tests
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.env = ClawbotCan(can_x_range=can_x_range, can_y_range=can_y_range, min_distance=min_distance)
        
        # Store evaluation parameters
        self.angle_range_rad = (np.radians(angle_range_deg[0]), np.radians(angle_range_deg[1]))
        self.distance_range = distance_range
        
        # Load model and detect algorithm type
        self.algorithm_type, self.model_data = self.detect_algorithm_and_load(model_path)
        
        # Initialize networks based on algorithm type
        self.policy = None
        self.q_network = None
        self.value_network = None
        
        self.load_networks()
        
        print(f"✅ Loaded {self.algorithm_type} model from {model_path}")
        
    def detect_algorithm_and_load(self, model_path: str):
        """Detect algorithm type from saved model and load it"""
        try:
            # Try loading with weights_only=True first (secure)
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            except Exception:
                # Fall back to weights_only=False for compatibility
                print("⚠️  Falling back to weights_only=False for model compatibility...")
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Detect algorithm type
            algorithm = checkpoint.get('algorithm', None)
            
            if algorithm:
                return algorithm, checkpoint
            else:
                # Try to infer from filename or available keys
                if 'td3' in model_path.lower():
                    return 'TD3', checkpoint
                elif 'sac' in model_path.lower():
                    return 'SAC', checkpoint
                elif 'ppo' in model_path.lower():
                    return 'PPO', checkpoint
                else:
                    # Default to DDPG for backward compatibility
                    return 'DDPG', checkpoint
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def load_networks(self):
        """Load networks based on algorithm type"""
        if self.algorithm_type == 'DDPG':
            self.policy = PolicyNetwork(self.obs_dim, self.act_dim)
            self.policy.load_state_dict(self.model_data['policy_state_dict'])
            
            if 'q_network_state_dict' in self.model_data:
                self.q_network = QNetwork(self.obs_dim, self.act_dim)
                self.q_network.load_state_dict(self.model_data['q_network_state_dict'])
                
        elif self.algorithm_type == 'TD3':
            self.policy = PolicyNetwork(self.obs_dim, self.act_dim)
            self.policy.load_state_dict(self.model_data['policy_state_dict'])
            
            # TD3 has two Q-networks, load Q1 for evaluation
            if 'q1_network_state_dict' in self.model_data:
                self.q_network = QNetwork(self.obs_dim, self.act_dim)
                self.q_network.load_state_dict(self.model_data['q1_network_state_dict'])
                
        elif self.algorithm_type == 'SAC':
            self.policy = SACPolicyNetwork(self.obs_dim, self.act_dim)
            self.policy.load_state_dict(self.model_data['policy_state_dict'])
            
            # SAC also has two Q-networks, load Q1 for evaluation
            if 'q1_network_state_dict' in self.model_data:
                self.q_network = QNetwork(self.obs_dim, self.act_dim)
                self.q_network.load_state_dict(self.model_data['q1_network_state_dict'])
                
        elif self.algorithm_type == 'PPO':
            self.policy = PPOPolicyNetwork(self.obs_dim, self.act_dim)
            self.policy.load_state_dict(self.model_data['policy_state_dict'])
            
            if 'value_state_dict' in self.model_data:
                self.value_network = ValueNetwork(self.obs_dim)
                self.value_network.load_state_dict(self.model_data['value_state_dict'])
        
        # Set all networks to evaluation mode
        if self.policy:
            self.policy.eval()
        if self.q_network:
            self.q_network.eval()
        if self.value_network:
            self.value_network.eval()
    
    def get_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Get action from the policy network based on algorithm type
        
        Args:
            obs: Current observation
            deterministic: If True, use deterministic action (for evaluation)
            
        Returns:
            Action array
        """
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).view(1, -1)
            
            if self.algorithm_type == 'DDPG':
                action = torch.tanh(self.policy(obs_tensor))
                if not deterministic:
                    # Add small amount of noise for exploration
                    action += 0.05 * torch.randn_like(action)
                    action = torch.tanh(action)
            elif self.algorithm_type == 'TD3':
                action = torch.tanh(self.policy(obs_tensor))
                if not deterministic:
                    # TD3 Fix: Add noise in action space and clamp (consistent with training)
                    action = (action + 0.05 * torch.randn_like(action)).clamp(-1.0, 1.0)
                    
            elif self.algorithm_type == 'SAC':
                if deterministic:
                    action = self.policy.sample_deterministic(obs_tensor)
                else:
                    action, _, _ = self.policy(obs_tensor)
                    
            elif self.algorithm_type == 'PPO':
                if deterministic:
                    action = self.policy.sample_deterministic(obs_tensor)
                else:
                    action, _ = self.policy.get_action_and_log_prob(obs_tensor)
            
            return action.cpu().numpy().flatten()
    
    def set_can_position(self, x: float, y: float):
        """Manually set the can position for directed testing"""
        can1_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "can1")
        can1_jntadr = self.env.model.body_jntadr[can1_id]
        can1_qposadr = self.env.model.jnt_qposadr[can1_jntadr]
        
        self.env.data.qpos[can1_qposadr+0] = x
        self.env.data.qpos[can1_qposadr+1] = y
        mujoco.mj_forward(self.env.model, self.env.data)
    
    def run_test_scenario(self, scenario_name: str, can_x: float, can_y: float, 
                         max_steps: int = 100, render: bool = True, 
                         render_delay: float = 0.1) -> dict:
        """Run a specific test scenario"""
        print(f"\n🧪 Running Test ({self.algorithm_type}): {scenario_name}")
        print(f"   Can position: ({can_x:.2f}, {can_y:.2f})")
        
        # Reset environment and set can position
        obs, info = self.env.reset()
        self.set_can_position(can_x, can_y)
        obs, info = self.env._calc_state()
        
        # Track metrics
        total_reward = 0
        distance_history = []
        angle_history = []
        grabbed = False
        
        # Initial state information
        distance, dtheta, objectGrabbed = obs
        print(f"   Initial distance: {distance:.3f}")
        print(f"   Initial angle difference: {dtheta:.3f} rad ({math.degrees(dtheta):.1f}°)")
        
        if render:
            print("   🎬 Rendering enabled - watch the robot!")
        
        for step in range(max_steps):
            # Get action from policy
            action = self.get_action(obs, deterministic=True)
            
            # Step environment
            new_obs, reward, terminal, info = self.env.step(action)
            
            # Track metrics
            total_reward += reward
            distance, dtheta, objectGrabbed = new_obs
            distance_history.append(distance)
            angle_history.append(abs(dtheta))
            
            if objectGrabbed:
                grabbed = True
                print(f"   🎉 Object grabbed at step {step}!")
            
            # Render if requested
            if render:
                self.env.render()
                time.sleep(render_delay)
            
            obs = new_obs
            
            if terminal:
                print(f"   ✅ Episode terminated at step {step}")
                break
        
        # Calculate results
        final_distance = distance_history[-1] if distance_history else float('inf')
        avg_angle_error = np.mean(angle_history) if angle_history else float('inf')
        min_distance = min(distance_history) if distance_history else float('inf')
        
        results = {
            'algorithm': self.algorithm_type,
            'scenario': scenario_name,
            'can_position': (can_x, can_y),
            'total_reward': total_reward,
            'final_distance': final_distance,
            'min_distance_achieved': min_distance,
            'avg_angle_error_rad': avg_angle_error,
            'avg_angle_error_deg': math.degrees(avg_angle_error),
            'object_grabbed': grabbed,
            'steps_taken': step + 1,
            'success': grabbed or final_distance < 0.1
        }
        
        # Print results
        print(f"   📊 Results:")
        print(f"      Total reward: {total_reward:.2f}")
        print(f"      Final distance: {final_distance:.3f}")
        print(f"      Min distance: {min_distance:.3f}")
        print(f"      Avg angle error: {math.degrees(avg_angle_error):.1f}°")
        print(f"      Object grabbed: {'✅ Yes' if grabbed else '❌ No'}")
        print(f"      Success: {'✅ Yes' if results['success'] else '❌ No'}")
        
        return results
    
    def generate_test_scenarios(self, num_tests: int = 12, seed: int = 42):
        """Generate randomized test scenarios"""
        np.random.seed(seed)
        test_scenarios = []
        
        for i in range(num_tests):
            # Generate random angle and distance within specified ranges
            angle = np.random.uniform(self.angle_range_rad[0], self.angle_range_rad[1])
            distance = np.random.uniform(self.distance_range[0], self.distance_range[1])
            
            # Convert polar to cartesian coordinates
            can_x = distance * np.sin(angle)
            can_y = distance * np.cos(angle)
            
            # Ensure Y is positive if angle range is in forward hemisphere
            if self.angle_range_rad[0] >= -np.pi/2 and self.angle_range_rad[1] <= np.pi/2:
                can_y = abs(can_y)
            
            # Create scenario name
            angle_deg = np.degrees(angle)
            dist_category = "Close" if distance < 0.3 else "Medium" if distance < 0.5 else "Far"
            angle_category = "Front" if abs(angle_deg) < 30 else "Angled" if abs(angle_deg) < 60 else "Side"
            direction = "Right" if angle_deg > 0 else "Left"
            
            scenario_name = f"{dist_category} - {angle_category} {direction} ({angle_deg:.0f}°)"
            test_scenarios.append((scenario_name, can_x, can_y))
        
        return test_scenarios

    def run_comprehensive_evaluation(self, render: bool = True, render_delay: float = 0.08):
        """Run comprehensive evaluation with multiple test scenarios"""
        print(f"🚀 Starting Comprehensive {self.algorithm_type} Model Evaluation")
        print("=" * 60)
        
        # Print model info
        best_reward = self.model_data.get('best_total_reward', 'Unknown')
        last_episode = self.model_data.get('last_episode', 'Unknown')
        print(f"📊 Model Info: Best Training Reward: {best_reward}, Episode: {last_episode}")
        
        # Generate test scenarios
        test_scenarios = self.generate_test_scenarios(num_tests=12)
        results = []
        
        for scenario_name, can_x, can_y in test_scenarios:
            result = self.run_test_scenario(
                scenario_name=scenario_name,
                can_x=can_x, 
                can_y=can_y,
                max_steps=150,
                render=render,
                render_delay=render_delay
            )
            results.append(result)
            
            # Brief pause between scenarios
            if render:
                time.sleep(0.5)
        
        # Generate summary report
        self.generate_summary_report(results)
        return results
    
    def run_quick_evaluation(self, render: bool = False):
        """Run quick evaluation for comparison purposes"""
        test_scenarios = self.generate_test_scenarios(num_tests=6)
        
        results = []
        for scenario_name, can_x, can_y in test_scenarios:
            result = self.run_test_scenario(
                scenario_name=scenario_name,
                can_x=can_x, 
                can_y=can_y,
                max_steps=100,
                render=render,
                render_delay=0.02
            )
            results.append(result)
        
        # Calculate summary metrics
        avg_reward = np.mean([r['total_reward'] for r in results])
        avg_final_distance = np.mean([r['final_distance'] for r in results])
        avg_angle_error = np.mean([r['avg_angle_error_deg'] for r in results])
        success_rate = sum(1 for r in results if r['success']) / len(results)
        grab_rate = sum(1 for r in results if r['object_grabbed']) / len(results)
        
        return {
            'algorithm': self.algorithm_type,
            'avg_reward': avg_reward,
            'avg_final_distance': avg_final_distance,
            'avg_angle_error': avg_angle_error,
            'success_rate': success_rate,
            'grab_rate': grab_rate,
            'num_tests': len(results)
        }
    
    def generate_summary_report(self, results: list):
        """Generate a summary report of all test results"""
        print("\n" + "=" * 60)
        print(f"📈 {self.algorithm_type} EVALUATION SUMMARY REPORT")
        print("=" * 60)
        
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r['success'])
        grabbed_tests = sum(1 for r in results if r['object_grabbed'])
        
        avg_reward = np.mean([r['total_reward'] for r in results])
        avg_final_distance = np.mean([r['final_distance'] for r in results])
        avg_min_distance = np.mean([r['min_distance_achieved'] for r in results])
        avg_angle_error = np.mean([r['avg_angle_error_deg'] for r in results])
        
        print(f"🎯 Overall Performance:")
        print(f"   Algorithm: {self.algorithm_type}")
        print(f"   Tests passed: {successful_tests}/{total_tests} ({100*successful_tests/total_tests:.1f}%)")
        print(f"   Objects grabbed: {grabbed_tests}/{total_tests} ({100*grabbed_tests/total_tests:.1f}%)")
        print(f"   Average reward: {avg_reward:.2f}")
        print(f"   Average final distance: {avg_final_distance:.3f}")
        print(f"   Average minimum distance: {avg_min_distance:.3f}")
        print(f"   Average angle error: {avg_angle_error:.1f}°")

def find_latest_model():
    """Find the most recent model file of any algorithm"""
    patterns = [
        'ddpg_clawbot_model_*.pth',
        'td3_clawbot_model_*.pth',
        'sac_clawbot_model_*.pth',
        'ppo_clawbot_model_*.pth'
    ]
    
    all_models = []
    for pattern in patterns:
        all_models.extend(glob.glob(pattern))
    
    if not all_models:
        return None
    
    # Sort by modification time, newest first
    all_models.sort(key=os.path.getmtime, reverse=True)
    return all_models[0]

def list_all_algorithm_models():
    """List all available model files from all algorithms"""
    patterns = {
        'DDPG': 'ddpg_clawbot_model_*.pth',
        'TD3': 'td3_clawbot_model_*.pth', 
        'SAC': 'sac_clawbot_model_*.pth',
        'PPO': 'ppo_clawbot_model_*.pth'
    }
    
    all_models = []
    for algorithm, pattern in patterns.items():
        models = glob.glob(pattern)
        for model in models:
            all_models.append((algorithm, model))
    
    if not all_models:
        print("❌ No saved models found for any algorithm.")
        return []
    
    # Sort by modification time, newest first
    all_models.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
    
    print("📋 Available models (all algorithms):")
    for i, (algorithm, model) in enumerate(all_models):
        mtime = os.path.getmtime(model)
        mod_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
        size_mb = os.path.getsize(model) / (1024 * 1024)
        print(f"   {i+1}. [{algorithm}] {model} ({mod_time}, {size_mb:.1f}MB)")
    
    return all_models

def main():
    """Main evaluation function"""
    import sys
    
    # Check if a specific model was provided as command line argument
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        if not os.path.exists(model_path):
            print(f"❌ Specified model file not found: {model_path}")
            sys.exit(1)
        print(f"🎯 Using specified model: {model_path}")
    else:
        # List available models from all algorithms
        available_models = list_all_algorithm_models()
        
        if not available_models:
            print("❌ No saved models found. Train a model first using one of: ddpg_clawbot.py, td3_clawbot.py, sac_clawbot.py, ppo_clawbot.py")
            sys.exit(1)
        
        # Use the most recent model by default
        algorithm, model_path = available_models[0]
        print(f"\n🤖 Using most recent {algorithm} model: {model_path}")
        print("💡 Tip: Specify a model: python unified_evaluator.py <model_file>")
    
    # Create evaluator
    evaluator = UnifiedModelEvaluator(model_path)
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(
        render=True,
        render_delay=0.01
    )
    
    # Close environment
    evaluator.env.close()
    print("\n✅ Evaluation completed!")

if __name__ == "__main__":
    main()
