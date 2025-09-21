# -*- coding: utf-8 -*-
"""
DDPG Clawbot Model Evaluator
Evaluates saved models with directed test scenarios to assess performance
"""

import torch
from torch import nn
import numpy as np
import time
import math
from mujoco_env import ClawbotCan
import mujoco

# Import the network classes from the training script
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
        q_values = self.net(obs_act)
        return q_values

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

class ModelEvaluator:
    def __init__(self, model_path: str, obs_dim: int = 3, act_dim: int = 4,
                 can_x_range: tuple = (-0.75, 0.75),
                 can_y_range: tuple = (0.4, 0.75),
                 min_distance: float = 0.4,
                 angle_range_deg: tuple = (-90, 90),
                 distance_range: tuple = (0.4, 0.75)):
        """Initialize the model evaluator
        
        Args:
            model_path (str): Path to the saved model
            obs_dim (int): Observation space dimension (distance, dtheta, objectGrabbed)
            act_dim (int): Action space dimension (4 actuators)
            can_x_range: (min_x, max_x) range for training environment
            can_y_range: (min_y, max_y) range for training environment
            min_distance: Minimum distance for training environment
            angle_range_deg: (min_angle, max_angle) in degrees for evaluation tests
            distance_range: (min_dist, max_dist) for evaluation tests
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.env = ClawbotCan(can_x_range=can_x_range, can_y_range=can_y_range, min_distance=min_distance)
        
        # Store evaluation parameters
        self.angle_range_rad = (np.radians(angle_range_deg[0]), np.radians(angle_range_deg[1]))
        self.distance_range = distance_range
        
        # Initialize networks
        self.policy = PolicyNetwork(obs_dim, act_dim)
        self.q_network = QNetwork(obs_dim, act_dim)
        
        # Load the saved model
        self.load_model(model_path)
        
        # Set networks to evaluation mode
        self.policy.eval()
        self.q_network.eval()
        
    def load_model(self, model_path: str):
        """Load the saved model parameters"""
        try:
            # Try loading with weights_only=True first (secure)
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            except Exception:
                # Fall back to weights_only=False for compatibility with older models
                print("⚠️  Falling back to weights_only=False for model compatibility...")
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            
            print(f"✅ Model loaded successfully from {model_path}")
            print(f"   Best reward during training: {checkpoint.get('best_total_reward', 'Unknown')}")
            print(f"   Last episode: {checkpoint.get('last_episode', 'Unknown')}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def get_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Get action from the policy network
        
        Args:
            obs: Current observation
            deterministic: If True, no noise is added to the action
            
        Returns:
            Action array
        """
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).view(1, -1)
            action = torch.tanh(self.policy(obs_tensor))
            if not deterministic:
                # Add small amount of noise for exploration
                action += 0.05 * torch.randn_like(action)
                action = torch.tanh(action)  # Ensure bounds
            return action.cpu().numpy().flatten()
    
    def set_can_position(self, x: float, y: float):
        """Manually set the can position for directed testing
        
        Args:
            x: X coordinate of the can
            y: Y coordinate of the can
        """
        can1_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "can1")
        can1_jntadr = self.env.model.body_jntadr[can1_id]
        can1_qposadr = self.env.model.jnt_qposadr[can1_jntadr]
        
        self.env.data.qpos[can1_qposadr+0] = x
        self.env.data.qpos[can1_qposadr+1] = y
        mujoco.mj_forward(self.env.model, self.env.data)
    
    def run_test_scenario(self, scenario_name: str, can_x: float, can_y: float, 
                         max_steps: int = 100, render: bool = True, 
                         render_delay: float = 0.1) -> dict:
        """Run a specific test scenario
        
        Args:
            scenario_name: Name of the test scenario
            can_x, can_y: Can position for this test
            max_steps: Maximum steps to run
            render: Whether to render the environment
            render_delay: Delay between rendered frames
            
        Returns:
            Dictionary with test results
        """
        print(f"\n🧪 Running Test: {scenario_name}")
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
        """Generate randomized test scenarios with angle and distance constraints"""
        np.random.seed(seed)  # For reproducible tests
        test_scenarios = []
        
        # Generate tests with configurable angle and distance constraints
        for i in range(num_tests):
            # Generate random angle within specified range
            angle = np.random.uniform(self.angle_range_rad[0], self.angle_range_rad[1])
            
            # Generate random distance within specified range
            distance = np.random.uniform(self.distance_range[0], self.distance_range[1])
            
            # Convert polar to cartesian coordinates
            # Robot is at (0,0) facing North, so:
            can_x = distance * np.sin(angle)  # X = distance * sin(angle)
            can_y = distance * np.cos(angle)  # Y = distance * cos(angle)
            
            # Apply Y constraint based on angle range
            if self.angle_range_rad[0] >= -np.pi/2 and self.angle_range_rad[1] <= np.pi/2:
                # If angle range is within forward hemisphere, ensure Y is positive
                can_y = abs(can_y)
            
            # Create scenario name based on characteristics
            angle_deg = np.degrees(angle)
            if distance < 0.3:
                dist_category = "Close"
            elif distance < 0.5:
                dist_category = "Medium"
            else:
                dist_category = "Far"
            
            if abs(angle_deg) < 30:
                angle_category = "Front"
            elif abs(angle_deg) < 60:
                angle_category = "Angled"
            else:
                angle_category = "Side"
            
            direction = "Right" if angle_deg > 0 else "Left"
            scenario_name = f"{dist_category} - {angle_category} {direction} ({angle_deg:.0f}°)"
            
            test_scenarios.append((scenario_name, can_x, can_y))
        
        return test_scenarios

    def run_comprehensive_evaluation(self, render: bool = True, render_delay: float = 0.08):
        """Run a comprehensive evaluation with multiple test scenarios"""
        print("🚀 Starting Comprehensive Model Evaluation")
        print("=" * 60)
        
        # Generate randomized test scenarios
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
        """Run a quick evaluation for comparison purposes"""
        # Generate fewer test scenarios for quick comparison
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
        print("📈 EVALUATION SUMMARY REPORT")
        print("=" * 60)
        
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r['success'])
        grabbed_tests = sum(1 for r in results if r['object_grabbed'])
        
        avg_reward = np.mean([r['total_reward'] for r in results])
        avg_final_distance = np.mean([r['final_distance'] for r in results])
        avg_min_distance = np.mean([r['min_distance_achieved'] for r in results])
        avg_angle_error = np.mean([r['avg_angle_error_deg'] for r in results])
        
        print(f"🎯 Overall Performance:")
        print(f"   Tests passed: {successful_tests}/{total_tests} ({100*successful_tests/total_tests:.1f}%)")
        print(f"   Objects grabbed: {grabbed_tests}/{total_tests} ({100*grabbed_tests/total_tests:.1f}%)")
        print(f"   Average reward: {avg_reward:.2f}")
        print(f"   Average final distance: {avg_final_distance:.3f}")
        print(f"   Average minimum distance: {avg_min_distance:.3f}")
        print(f"   Average angle error: {avg_angle_error:.1f}°")
        
        # Performance by scenario type
        print(f"\n📊 Performance by Scenario Type:")
        scenario_types = ['Close', 'Medium', 'Far', 'Edge']
        
        for scenario_type in scenario_types:
            type_results = [r for r in results if r['scenario'].startswith(scenario_type)]
            if type_results:
                type_success = sum(1 for r in type_results if r['success'])
                type_grabbed = sum(1 for r in type_results if r['object_grabbed'])
                type_avg_reward = np.mean([r['total_reward'] for r in type_results])
                
                print(f"   {scenario_type:>6}: {type_success}/{len(type_results)} passed, "
                      f"{type_grabbed}/{len(type_results)} grabbed, "
                      f"avg reward: {type_avg_reward:.1f}")
        
        # Best and worst performances
        best_result = max(results, key=lambda x: x['total_reward'])
        worst_result = min(results, key=lambda x: x['total_reward'])
        
        print(f"\n🏆 Best Performance:")
        print(f"   Scenario: {best_result['scenario']}")
        print(f"   Reward: {best_result['total_reward']:.2f}")
        print(f"   Success: {'✅ Yes' if best_result['success'] else '❌ No'}")
        
        print(f"\n⚠️  Worst Performance:")
        print(f"   Scenario: {worst_result['scenario']}")
        print(f"   Reward: {worst_result['total_reward']:.2f}")
        print(f"   Success: {'✅ Yes' if worst_result['success'] else '❌ No'}")

def find_latest_model():
    """Find the most recent model file"""
    import glob
    import os
    
    # Look for timestamped model files
    patterns = [
        'ddpg_clawbot_model_completed_*.pth',
        'ddpg_clawbot_model_interrupted_*.pth', 
        'ddpg_clawbot_model_checkpoint_*.pth',
        # Fallback to old naming convention
        'ddpg_clawbot_model_completed.pth',
        'ddpg_clawbot_model_interrupted.pth',
        'ddpg_clawbot_model.pth'
    ]
    
    all_models = []
    for pattern in patterns:
        all_models.extend(glob.glob(pattern))
    
    if not all_models:
        return None
    
    # Sort by modification time, newest first
    all_models.sort(key=os.path.getmtime, reverse=True)
    return all_models[0]

def list_available_models():
    """List all available model files"""
    import glob
    import os
    import datetime
    
    patterns = [
        'ddpg_clawbot_model_*.pth',
        'ddpg_clawbot_model.pth'
    ]
    
    all_models = []
    for pattern in patterns:
        all_models.extend(glob.glob(pattern))
    
    if not all_models:
        print("❌ No saved models found.")
        return []
    
    # Sort by modification time, newest first
    all_models.sort(key=os.path.getmtime, reverse=True)
    
    print("📋 Available models:")
    for i, model in enumerate(all_models):
        mtime = os.path.getmtime(model)
        mod_time = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        size_mb = os.path.getsize(model) / (1024 * 1024)
        print(f"   {i+1}. {model} ({mod_time}, {size_mb:.1f}MB)")
    
    return all_models

def compare_all_models():
    """Compare all available models to show training progression"""
    import glob
    import os
    import datetime
    
    print("🔍 MULTI-MODEL COMPARISON MODE")
    print("=" * 80)
    
    # Find all model files
    patterns = [
        'ddpg_clawbot_model_*.pth',
        'ddpg_clawbot_model.pth'
    ]
    
    all_models = []
    for pattern in patterns:
        all_models.extend(glob.glob(pattern))
    
    if len(all_models) < 2:
        print("❌ Need at least 2 models for comparison")
        return
    
    # Sort by modification time (oldest first for progression analysis)
    all_models.sort(key=os.path.getmtime)
    
    print(f"📋 Found {len(all_models)} models to compare:")
    model_results = []
    
    for i, model_path in enumerate(all_models):
        print(f"\n{'='*60}")
        print(f"🤖 Evaluating Model {i+1}/{len(all_models)}: {model_path}")
        
        # Get model metadata
        try:
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            except:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            training_reward = checkpoint.get('best_total_reward', 'Unknown')
            last_episode = checkpoint.get('last_episode', 'Unknown')
            save_type = checkpoint.get('save_type', 'legacy')
            
            print(f"   📊 Training Info: Episode {last_episode}, Best Reward: {training_reward}")
            print(f"   🏷️  Save Type: {save_type}")
        except:
            training_reward = 'Unknown'
            last_episode = 'Unknown'
            save_type = 'Unknown'
        
        # Quick evaluation
        try:
            evaluator = ModelEvaluator(model_path)
            eval_results = evaluator.run_quick_evaluation(render=False)
            evaluator.env.close()
            
            model_results.append({
                'model_path': model_path,
                'training_reward': training_reward,
                'last_episode': last_episode,
                'save_type': save_type,
                'eval_results': eval_results,
                'mod_time': os.path.getmtime(model_path)
            })
            
            print(f"   ✅ Evaluation completed: Avg Reward {eval_results['avg_reward']:.1f}")
            
        except Exception as e:
            print(f"   ❌ Evaluation failed: {e}")
    
    # Generate comparison report
    generate_progression_report(model_results)

def generate_progression_report(model_results):
    """Generate a detailed progression report comparing all models"""
    import os
    
    print(f"\n{'='*80}")
    print("📈 TRAINING PROGRESSION ANALYSIS")
    print(f"{'='*80}")
    
    if len(model_results) < 2:
        print("❌ Need at least 2 successful evaluations for comparison")
        return
    
    # Sort by episode number for progression analysis
    valid_results = [r for r in model_results if isinstance(r['last_episode'], (int, float))]
    valid_results.sort(key=lambda x: x['last_episode'])
    
    print(f"📊 Model Performance Progression:")
    print(f"{'Episode':<8} {'Type':<12} {'Training':<10} {'Eval Avg':<10} {'Distance':<10} {'Angle°':<8} {'Success%':<9} {'Model'}")
    print("-" * 80)
    
    for result in valid_results:
        episode = result['last_episode']
        save_type = result['save_type'][:11]
        training_reward = f"{result['training_reward']:.1f}" if isinstance(result['training_reward'], (int, float)) else "Unknown"
        eval_reward = f"{result['eval_results']['avg_reward']:.1f}"
        distance = f"{result['eval_results']['avg_final_distance']:.3f}"
        angle = f"{result['eval_results']['avg_angle_error']:.1f}"
        success = f"{result['eval_results']['success_rate']*100:.0f}%"
        model_name = os.path.basename(result['model_path'])[:20]
        
        print(f"{episode:<8} {save_type:<12} {training_reward:<10} {eval_reward:<10} {distance:<10} {angle:<8} {success:<9} {model_name}")
    
    # Calculate improvement metrics
    if len(valid_results) >= 2:
        first_model = valid_results[0]['eval_results']
        last_model = valid_results[-1]['eval_results']
        
        reward_improvement = last_model['avg_reward'] - first_model['avg_reward']
        distance_improvement = first_model['avg_final_distance'] - last_model['avg_final_distance']
        angle_improvement = first_model['avg_angle_error'] - last_model['avg_angle_error']
        success_improvement = last_model['success_rate'] - first_model['success_rate']
        
        print(f"\n🎯 Overall Training Progression:")
        print(f"   Reward Change: {reward_improvement:+.1f} ({'✅ Better' if reward_improvement > 0 else '❌ Worse'})")
        print(f"   Distance Improvement: {distance_improvement:+.3f} ({'✅ Closer' if distance_improvement > 0 else '❌ Further'})")
        print(f"   Angle Accuracy: {angle_improvement:+.1f}° ({'✅ Better' if angle_improvement > 0 else '❌ Worse'})")
        print(f"   Success Rate Change: {success_improvement*100:+.1f}% ({'✅ Better' if success_improvement > 0 else '❌ Worse'})")
        
        # Best performing model
        best_model = max(valid_results, key=lambda x: x['eval_results']['avg_reward'])
        print(f"\n🏆 Best Performing Model:")
        print(f"   Model: {os.path.basename(best_model['model_path'])}")
        print(f"   Episode: {best_model['last_episode']}")
        print(f"   Training Reward: {best_model['training_reward']}")
        print(f"   Evaluation Reward: {best_model['eval_results']['avg_reward']:.1f}")
        print(f"   Success Rate: {best_model['eval_results']['success_rate']*100:.1f}%")

def main():
    """Main evaluation function"""
    import sys
    import os
    
    # Check for comparison mode
    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        compare_all_models()
        return
    
    # Check if a specific model was provided as command line argument
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        if not os.path.exists(model_path):
            print(f"❌ Specified model file not found: {model_path}")
            sys.exit(1)
        print(f"🎯 Using specified model: {model_path}")
    else:
        # List available models
        available_models = list_available_models()
        
        if not available_models:
            print("❌ No saved models found. Train a model first using ddpg_clawbot.py")
            sys.exit(1)
        
        # Use the most recent model by default
        model_path = available_models[0]
        print(f"\n🤖 Using most recent model: {model_path}")
        print("💡 Tips:")
        print("   - Specify a model: python model_evaluator.py <model_file>")
        print("   - Compare all models: python model_evaluator.py --compare")
    
    # Create evaluator
    evaluator = ModelEvaluator(model_path)
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(
        render=True,
        render_delay=0.05  # Slow rendering for better visibility
    )
    
    # Close environment
    evaluator.env.close()
    print("\n✅ Evaluation completed!")

if __name__ == "__main__":
    main()
