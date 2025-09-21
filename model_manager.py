#!/usr/bin/env python3
"""
Model Manager Utility
Helps manage saved DDPG models - list, compare, and clean up old models
"""

import os
import glob
import datetime
import torch
from typing import List, Dict

def list_all_models() -> List[Dict]:
    """List all available model files with details"""
    patterns = [
        'ddpg_clawbot_model_*.pth',
        'ddpg_clawbot_model.pth'
    ]
    
    all_models = []
    for pattern in patterns:
        all_models.extend(glob.glob(pattern))
    
    if not all_models:
        return []
    
    model_info = []
    for model_file in all_models:
        try:
            # Get file stats
            stat = os.stat(model_file)
            size_mb = stat.st_size / (1024 * 1024)
            mod_time = datetime.datetime.fromtimestamp(stat.st_mtime)
            
            # Try to load model metadata
            try:
                # Try loading with weights_only=True first (secure)
                try:
                    checkpoint = torch.load(model_file, map_location='cpu', weights_only=True)
                except Exception:
                    # Fall back to weights_only=False for compatibility
                    checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
                
                best_reward = checkpoint.get('best_total_reward', 'Unknown')
                last_episode = checkpoint.get('last_episode', 'Unknown')
                save_type = checkpoint.get('save_type', 'legacy')
                save_time = checkpoint.get('save_time', 'Unknown')
            except:
                best_reward = 'Unknown'
                last_episode = 'Unknown'
                save_type = 'legacy'
                save_time = 'Unknown'
            
            model_info.append({
                'filename': model_file,
                'size_mb': size_mb,
                'mod_time': mod_time,
                'best_reward': best_reward,
                'last_episode': last_episode,
                'save_type': save_type,
                'save_time': save_time
            })
        except Exception as e:
            print(f"⚠️  Error reading {model_file}: {e}")
    
    # Sort by modification time, newest first
    model_info.sort(key=lambda x: x['mod_time'], reverse=True)
    return model_info

def print_model_summary():
    """Print a detailed summary of all models"""
    models = list_all_models()
    
    if not models:
        print("❌ No saved models found.")
        return
    
    print("📋 DDPG Model Summary")
    print("=" * 80)
    print(f"{'#':<2} {'Filename':<35} {'Type':<12} {'Episode':<8} {'Reward':<8} {'Size':<7} {'Date'}")
    print("-" * 80)
    
    for i, model in enumerate(models, 1):
        filename = model['filename']
        if len(filename) > 34:
            filename = filename[:31] + "..."
        
        reward_str = f"{model['best_reward']:.1f}" if isinstance(model['best_reward'], (int, float)) else str(model['best_reward'])[:7]
        episode_str = str(model['last_episode'])[:7]
        date_str = model['mod_time'].strftime("%m/%d %H:%M")
        
        print(f"{i:<2} {filename:<35} {model['save_type']:<12} {episode_str:<8} {reward_str:<8} {model['size_mb']:.1f}MB {date_str}")

def compare_models(model1_path: str, model2_path: str):
    """Compare two models"""
    print(f"\n🔍 Comparing Models:")
    print(f"   Model 1: {model1_path}")
    print(f"   Model 2: {model2_path}")
    
    try:
        # Try loading with weights_only=True first (secure)
        try:
            checkpoint1 = torch.load(model1_path, map_location='cpu', weights_only=True)
            checkpoint2 = torch.load(model2_path, map_location='cpu', weights_only=True)
        except Exception:
            # Fall back to weights_only=False for compatibility
            checkpoint1 = torch.load(model1_path, map_location='cpu', weights_only=False)
            checkpoint2 = torch.load(model2_path, map_location='cpu', weights_only=False)
        
        reward1 = checkpoint1.get('best_total_reward', 'Unknown')
        reward2 = checkpoint2.get('best_total_reward', 'Unknown')
        episode1 = checkpoint1.get('last_episode', 'Unknown')
        episode2 = checkpoint2.get('last_episode', 'Unknown')
        
        print(f"\n📊 Performance Comparison:")
        print(f"   Model 1 - Best Reward: {reward1}, Episodes: {episode1}")
        print(f"   Model 2 - Best Reward: {reward2}, Episodes: {episode2}")
        
        if isinstance(reward1, (int, float)) and isinstance(reward2, (int, float)):
            if reward1 > reward2:
                print(f"   🏆 Model 1 performs better (+{reward1-reward2:.2f} reward)")
            elif reward2 > reward1:
                print(f"   🏆 Model 2 performs better (+{reward2-reward1:.2f} reward)")
            else:
                print(f"   🤝 Models have similar performance")
        
    except Exception as e:
        print(f"❌ Error comparing models: {e}")

def cleanup_old_models_by_pattern(pattern: str, keep_count: int = 2):
    """Clean up old model files matching a specific pattern"""
    files = glob.glob(pattern)
    if len(files) > keep_count:
        # Sort by modification time, newest first
        files.sort(key=os.path.getmtime, reverse=True)
        # Remove older files
        removed_count = 0
        for old_file in files[keep_count:]:
            try:
                os.remove(old_file)
                print(f"🗑️  Removed old model: {old_file}")
                removed_count += 1
            except OSError:
                pass
        return removed_count
    return 0

def cleanup_all_old_models(keep_count: int = 3, verbose: bool = True):
    """Clean up old models of all types, keeping only the most recent ones"""
    model_patterns = [
        'ddpg_clawbot_model_completed_*.pth',
        'ddpg_clawbot_model_interrupted_*.pth', 
        'ddpg_clawbot_model_checkpoint_*.pth'
    ]
    
    total_removed = 0
    for pattern in model_patterns:
        removed = cleanup_old_models_by_pattern(pattern, keep_count)
        total_removed += removed
    
    if total_removed > 0 and verbose:
        print(f"🧹 Cleanup complete: {total_removed} old model files removed (keeping {keep_count} most recent of each type)")
    
    return total_removed

def cleanup_old_models(keep_count: int = 3, dry_run: bool = True):
    """Clean up old model files, keeping only the most recent ones"""
    models = list_all_models()
    
    if len(models) <= keep_count:
        print(f"✅ Only {len(models)} models found, no cleanup needed (keeping {keep_count})")
        return
    
    models_to_remove = models[keep_count:]
    
    print(f"🗑️  Cleanup Plan (keeping {keep_count} most recent models):")
    print(f"   Models to keep: {len(models) - len(models_to_remove)}")
    print(f"   Models to remove: {len(models_to_remove)}")
    
    if dry_run:
        print("\n📋 Files that would be removed (DRY RUN):")
        for model in models_to_remove:
            print(f"   - {model['filename']} (reward: {model['best_reward']}, {model['mod_time'].strftime('%Y-%m-%d %H:%M')})")
        print("\n💡 Run with --execute to actually remove files")
    else:
        print("\n🗑️  Removing old models:")
        removed_count = 0
        for model in models_to_remove:
            try:
                os.remove(model['filename'])
                print(f"   ✅ Removed: {model['filename']}")
                removed_count += 1
            except Exception as e:
                print(f"   ❌ Failed to remove {model['filename']}: {e}")
        
        print(f"\n✅ Cleanup complete: {removed_count} files removed")

def main():
    """Main function with command line interface"""
    import sys
    
    if len(sys.argv) < 2:
        print("🤖 DDPG Model Manager")
        print("\nUsage:")
        print("  python model_manager.py list                    # List all models")
        print("  python model_manager.py cleanup [--execute]     # Clean up old models")
        print("  python model_manager.py compare <model1> <model2>  # Compare two models")
        print("\nExamples:")
        print("  python model_manager.py list")
        print("  python model_manager.py cleanup")
        print("  python model_manager.py cleanup --execute")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == 'list':
        print_model_summary()
    
    elif command == 'cleanup':
        execute = '--execute' in sys.argv
        cleanup_old_models(keep_count=3, dry_run=not execute)
    
    elif command == 'compare':
        if len(sys.argv) < 4:
            print("❌ Compare requires two model filenames")
            print("Usage: python model_manager.py compare <model1> <model2>")
            sys.exit(1)
        
        model1 = sys.argv[2]
        model2 = sys.argv[3]
        
        if not os.path.exists(model1):
            print(f"❌ Model file not found: {model1}")
            sys.exit(1)
        if not os.path.exists(model2):
            print(f"❌ Model file not found: {model2}")
            sys.exit(1)
        
        compare_models(model1, model2)
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: list, cleanup, compare")
        sys.exit(1)

if __name__ == "__main__":
    main()
