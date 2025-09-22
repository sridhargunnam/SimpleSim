#!/usr/bin/env python3
"""
Test Different Configuration Scenarios
Easily test models with different angle and distance constraints
"""

from unified_evaluator import UnifiedModelEvaluator
from quick_eval import quick_test
import numpy as np

def test_forward_hemisphere():
    """Test with forward hemisphere only (-90° to +90°)"""
    print("🎯 Testing Forward Hemisphere Configuration")
    print("=" * 60)
    
    quick_test(
        angle_range=(-90, 90),
        distance_range=(0.15, 0.75),
        can_x_range=(-0.75, 0.75),
        can_y_range=(0.1, 0.75)
    )

def test_full_circle():
    """Test with full 360° range"""
    print("\n🎯 Testing Full Circle Configuration")
    print("=" * 60)
    
    quick_test(
        angle_range=(-180, 180),
        distance_range=(0.15, 0.75),
        can_x_range=(-0.75, 0.75),
        can_y_range=(-0.75, 0.75)  # Allow behind robot
    )

def test_close_range_only():
    """Test with close range scenarios only"""
    print("\n🎯 Testing Close Range Configuration")
    print("=" * 60)
    
    quick_test(
        angle_range=(-60, 60),
        distance_range=(0.1, 0.3),
        can_x_range=(-0.3, 0.3),
        can_y_range=(0.1, 0.3)
    )

def test_side_approaches():
    """Test with side approach scenarios (±60° to ±90°)"""
    print("\n🎯 Testing Side Approaches Configuration")
    print("=" * 60)
    
    quick_test(
        angle_range=(-90, -60),  # Left side only
        distance_range=(0.2, 0.6),
        can_x_range=(-0.75, 0.75),
        can_y_range=(0.1, 0.75)
    )
    
    quick_test(
        angle_range=(60, 90),   # Right side only
        distance_range=(0.2, 0.6),
        can_x_range=(-0.75, 0.75),
        can_y_range=(0.1, 0.75)
    )

def compare_configurations(model_path=None):
    """Compare model performance across different configurations"""
    print("🔍 CONFIGURATION COMPARISON")
    print("=" * 80)
    
    if not model_path:
        import glob
        import os
        patterns = [
            'ddpg_clawbot_model_*.pth',
            'td3_clawbot_model_*.pth',
            'sac_clawbot_model_*.pth',
            'ppo_clawbot_model_*.pth'
        ]
        all_models = []
        for pattern in patterns:
            all_models.extend(glob.glob(pattern))
        if all_models:
            model_path = max(all_models, key=os.path.getmtime)
        else:
            print("❌ No models found")
            return
    
    configurations = [
        {
            'name': 'Forward Hemisphere',
            'angle_range': (-90, 90),
            'distance_range': (0.4, 0.75),
            'can_x_range': (-0.75, 0.75),
            'can_y_range': (0.4, 0.75)
        },
        {
            'name': 'Close Range Only', 
            'angle_range': (-60, 60),
            'distance_range': (0.1, 0.3),
            'can_x_range': (-0.3, 0.3),
            'can_y_range': (0.1, 0.3)
        },
        {
            'name': 'Side Approaches',
            'angle_range': (-90, -60),
            'distance_range': (0.2, 0.6),
            'can_x_range': (-0.75, 0.75),
            'can_y_range': (0.1, 0.75)
        },
        {
            'name': 'Full Circle',
            'angle_range': (-180, 180),
            'distance_range': (0.15, 0.75),
            'can_x_range': (-0.75, 0.75),
            'can_y_range': (-0.75, 0.75)
        }
    ]
    
    results = []
    
    for config in configurations:
        print(f"\n📐 Testing {config['name']} Configuration:")
        print(f"   Angles: {config['angle_range'][0]}° to {config['angle_range'][1]}°")
        print(f"   Distance: {config['distance_range'][0]} to {config['distance_range'][1]}")
        
        try:
            evaluator = UnifiedModelEvaluator(
                model_path=model_path,
                **{k: v for k, v in config.items() if k != 'name'}
            )
            
            eval_result = evaluator.run_quick_evaluation(render=False)
            evaluator.env.close()
            
            results.append({
                'config_name': config['name'],
                'avg_reward': eval_result['avg_reward'],
                'avg_distance': eval_result['avg_final_distance'],
                'avg_angle_error': eval_result['avg_angle_error'],
                'success_rate': eval_result['success_rate']
            })
            
            print(f"   ✅ Avg Reward: {eval_result['avg_reward']:.1f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Summary comparison
    print(f"\n📊 Configuration Performance Summary:")
    print(f"{'Configuration':<20} {'Avg Reward':<12} {'Avg Distance':<13} {'Angle Error':<12} {'Success%'}")
    print("-" * 75)
    
    for result in results:
        print(f"{result['config_name']:<20} {result['avg_reward']:<12.1f} {result['avg_distance']:<13.3f} {result['avg_angle_error']:<12.1f} {result['success_rate']*100:.0f}%")

def main():
    """Main function with configuration options"""
    import sys
    
    if len(sys.argv) < 2:
        print("🤖 Configuration Tester")
        print("\nUsage:")
        print("  python test_configurations.py forward      # Test forward hemisphere")
        print("  python test_configurations.py full         # Test full 360°")
        print("  python test_configurations.py close        # Test close range")
        print("  python test_configurations.py side         # Test side approaches")
        print("  python test_configurations.py compare      # Compare all configs")
        print("\nExamples:")
        print("  python test_configurations.py forward")
        print("  python test_configurations.py compare ddpg_clawbot_model_checkpoint_xxx.pth")
        return
    
    command = sys.argv[1].lower()
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if command == 'forward':
        test_forward_hemisphere()
    elif command == 'full':
        test_full_circle()
    elif command == 'close':
        test_close_range_only()
    elif command == 'side':
        test_side_approaches()
    elif command == 'compare':
        compare_configurations(model_path)
    else:
        print(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    main()
