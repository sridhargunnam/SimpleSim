#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick Evaluation Script for All RL Algorithms
A lightweight version of the evaluator for quick testing and comparison
"""

from unified_evaluator import UnifiedModelEvaluator
import numpy as np
import time
import glob
import os

def quick_test(model_path=None, num_episodes=5, render=True, verbose=True, 
               angle_range=(-90, 90), distance_range=(0.4, 0.75),
               can_x_range=(-0.75, 0.75), can_y_range=(0.4, 0.75)):
    """Quick test of any trained model"""
    
    if not model_path:
        # Find the most recent model of any algorithm
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
            print("❌ No saved models found")
            return
        
        # Use most recent
        all_models.sort(key=os.path.getmtime, reverse=True)
        model_path = all_models[0]
        print(f"🎯 Using most recent model: {model_path}")
    
    try:
        evaluator = UnifiedModelEvaluator(
            model_path=model_path,
            can_x_range=can_x_range,
            can_y_range=can_y_range,
            angle_range_deg=angle_range,
            distance_range=distance_range
        )
        print(f"✅ {evaluator.algorithm_type} Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    successes = 0
    total_rewards = []
    min_distances = []
    angle_errors = []
    
    print(f"\n🚀 Running {num_episodes} test episodes...")
    print("-" * 50)
    
    # Generate test scenarios
    test_scenarios = evaluator.generate_test_scenarios(num_tests=num_episodes, seed=42)
    
    for i, (scenario_name, can_x, can_y) in enumerate(test_scenarios):
        print(f"\n📍 Test {i+1}: {scenario_name}")
        print(f"   Can at: ({can_x:.2f}, {can_y:.2f})")
        
        result = evaluator.run_test_scenario(
            scenario_name=f"Quick Test {i+1}",
            can_x=can_x,
            can_y=can_y,
            max_steps=150,
            render=render,
            render_delay=0.02 if render else 0
        )
        
        if result['success']:
            successes += 1
        
        total_rewards.append(result['total_reward'])
        min_distances.append(result['min_distance_achieved'])
        angle_errors.append(result['avg_angle_error_deg'])
        
        if not verbose:
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            print(f"   {status} - Reward: {result['total_reward']:.2f}")
    
    # Summary
    success_rate = (successes / num_episodes) * 100
    avg_reward = np.mean(total_rewards)
    avg_min_distance = np.mean(min_distances)
    avg_angle_error = np.mean(angle_errors)
    
    print("\n" + "="*60)
    print(f"📊 QUICK EVALUATION RESULTS ({evaluator.algorithm_type})")
    print("="*60)
    print(f"Episodes:              {num_episodes}")
    print(f"Successes:             {successes}")
    print(f"Success Rate:          {success_rate:.1f}%")
    print(f"Avg Reward:            {avg_reward:.2f}")
    print(f"Reward Range:          {min(total_rewards):.2f} to {max(total_rewards):.2f}")
    print(f"Avg Min Distance:      {avg_min_distance:.3f}")
    print(f"Avg Angle Error:       {avg_angle_error:.1f}°")
    print(f"Test Configuration:")
    print(f"  Angle Range:         {angle_range[0]}° to {angle_range[1]}°")
    print(f"  Distance Range:      {distance_range[0]} to {distance_range[1]}")
    
    # Performance assessment
    if success_rate >= 80:
        print("🎉 Model performance: EXCELLENT")
    elif success_rate >= 60:
        print("👍 Model performance: GOOD")
    elif success_rate >= 40:
        print("⚠️  Model performance: FAIR")
    else:
        print("😟 Model performance: POOR - needs more training")
    
    evaluator.env.close()
    
    return {
        'algorithm': evaluator.algorithm_type,
        'success_rate': success_rate / 100,
        'avg_reward': avg_reward,
        'avg_min_distance': avg_min_distance,
        'avg_angle_error': avg_angle_error,
        'num_episodes': num_episodes
    }

def compare_algorithms():
    """Compare all available algorithms"""
    print("🔍 ALGORITHM COMPARISON MODE")
    print("=" * 80)
    
    # Find models for each algorithm
    algorithm_patterns = {
        'DDPG': 'ddpg_clawbot_model_*.pth',
        'TD3': 'td3_clawbot_model_*.pth',
        'SAC': 'sac_clawbot_model_*.pth', 
        'PPO': 'ppo_clawbot_model_*.pth'
    }
    
    available_algorithms = {}
    for algorithm, pattern in algorithm_patterns.items():
        models = glob.glob(pattern)
        if models:
            # Use most recent model
            latest_model = max(models, key=os.path.getmtime)
            available_algorithms[algorithm] = latest_model
    
    if len(available_algorithms) < 2:
        print("❌ Need at least 2 different algorithms trained for comparison")
        return
    
    print(f"📋 Found models for {len(available_algorithms)} algorithms:")
    for algorithm, model_path in available_algorithms.items():
        print(f"   {algorithm}: {model_path}")
    
    # Test each algorithm
    results = {}
    for algorithm, model_path in available_algorithms.items():
        print(f"\n{'='*60}")
        print(f"🤖 Testing {algorithm}")
        print('='*60)
        
        try:
            result = quick_test(
                model_path=model_path,
                num_episodes=8,  # Fewer episodes for comparison
                render=False,  # No rendering for batch comparison
                verbose=False
            )
            results[algorithm] = result
            print(f"   ✅ {algorithm} completed")
        except Exception as e:
            print(f"   ❌ {algorithm} failed: {e}")
    
    # Generate comparison report
    print(f"\n{'='*80}")
    print("📈 ALGORITHM COMPARISON REPORT")
    print(f"{'='*80}")
    
    if len(results) < 2:
        print("❌ Not enough successful evaluations for comparison")
        return
    
    print(f"{'Algorithm':<10} {'Success%':<10} {'Avg Reward':<12} {'Avg Distance':<13} {'Angle Error°':<12}")
    print("-" * 70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['success_rate'], reverse=True)
    
    for algorithm, result in sorted_results:
        success_pct = result['success_rate'] * 100
        print(f"{algorithm:<10} {success_pct:<10.1f} {result['avg_reward']:<12.2f} {result['avg_min_distance']:<13.3f} {result['avg_angle_error']:<12.1f}")
    
    # Highlight best performer
    best_algorithm = sorted_results[0][0]
    best_result = sorted_results[0][1]
    
    print(f"\n🏆 Best Performing Algorithm: {best_algorithm}")
    print(f"   Success Rate: {best_result['success_rate']*100:.1f}%")
    print(f"   Avg Reward: {best_result['avg_reward']:.2f}")
    print(f"   Avg Distance: {best_result['avg_min_distance']:.3f}")

def main():
    """Main function with command line interface"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        compare_algorithms()
        return
    
    # Default parameters
    episodes = 5
    render = True
    model_path = None
    
    # Parse command line arguments
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--episodes':
            episodes = int(sys.argv[i+1])
            i += 2
        elif arg == '--no-render':
            render = False
            i += 1
        elif arg == '--model':
            model_path = sys.argv[i+1]
            i += 2
        else:
            # Assume it's a model path
            model_path = arg
            i += 1
    
    print("🧪 Quick Evaluation Tool (All Algorithms)")
    print("=" * 50)
    print(f"Episodes: {episodes}")
    print(f"Rendering: {render}")
    if model_path:
        print(f"Model: {model_path}")
    
    quick_test(model_path=model_path, num_episodes=episodes, render=render)

if __name__ == "__main__":
    main()