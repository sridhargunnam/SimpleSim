#!/usr/bin/env python3
"""
Quick Model Evaluation Script
Run specific test scenarios or a quick evaluation
"""

from model_evaluator import ModelEvaluator
import sys
import os
import numpy as np

def quick_test(model_path=None, angle_range=(-90, 90), distance_range=(0.4, 0.75), 
               can_x_range=(-0.75, 0.75), can_y_range=(0.4, 0.75)):
    """Run a few quick tests to check model performance
    
    Args:
        model_path: Path to model file (auto-detected if None)
        angle_range: (min_angle, max_angle) in degrees for test scenarios
        distance_range: (min_dist, max_dist) for test scenarios
        can_x_range: (min_x, max_x) for training environment setup
        can_y_range: (min_y, max_y) for training environment setup
    """
    import glob
    
    if model_path and os.path.exists(model_path):
        print(f"🎯 Quick Evaluation with specified model: {model_path}")
    else:
        # Look for any model files (timestamped or old format)
        patterns = [
            'ddpg_clawbot_model_*.pth',
            'ddpg_clawbot_model.pth'
        ]
        
        all_models = []
        for pattern in patterns:
            all_models.extend(glob.glob(pattern))
        
        if not all_models:
            print("❌ No saved models found!")
            return
        
        # Use the most recent model
        model_path = max(all_models, key=os.path.getmtime)
        print(f"🚀 Quick Evaluation with most recent model: {model_path}")
    
    print(f"📐 Test Parameters:")
    print(f"   Angle range: {angle_range[0]}° to {angle_range[1]}°")
    print(f"   Distance range: {distance_range[0]} to {distance_range[1]}")
    print(f"   Training can X range: {can_x_range[0]} to {can_x_range[1]}")
    print(f"   Training can Y range: {can_y_range[0]} to {can_y_range[1]}")
    
    # Create evaluator with specified parameters
    evaluator = ModelEvaluator(
        model_path=model_path,
        can_x_range=can_x_range,
        can_y_range=can_y_range,
        angle_range_deg=angle_range,
        distance_range=distance_range
    )
    
    # Generate test scenarios using the evaluator's parameterized method
    quick_scenarios = evaluator.generate_test_scenarios(num_tests=3, seed=123)
    
    results = []
    for scenario_name, can_x, can_y in quick_scenarios:
        result = evaluator.run_test_scenario(
            scenario_name=scenario_name,
            can_x=can_x,
            can_y=can_y,
            max_steps=100,
            render=True,
            render_delay=0.08
        )
        results.append(result)
    
    # Quick summary
    successful = sum(1 for r in results if r['success'])
    grabbed = sum(1 for r in results if r['object_grabbed'])
    avg_reward = sum(r['total_reward'] for r in results) / len(results)
    
    print(f"\n🎯 Quick Summary:")
    print(f"   Success rate: {successful}/{len(results)}")
    print(f"   Grab rate: {grabbed}/{len(results)}")
    print(f"   Average reward: {avg_reward:.2f}")
    
    evaluator.env.close()

if __name__ == "__main__":
    # Check if a model file was specified
    model_file = sys.argv[1] if len(sys.argv) > 1 else None
    quick_test(model_file)
