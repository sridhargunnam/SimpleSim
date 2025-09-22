# -*- coding: utf-8 -*-
"""
Video-Enabled Model Evaluator for All RL Algorithms
Evaluates saved models and records video of the evaluation process
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
import cv2
import datetime
from typing import Optional, List, Tuple

# Import network classes from unified_evaluator
from unified_evaluator import (
    QNetwork, PolicyNetwork, SACPolicyNetwork, ValueNetwork, PPOPolicyNetwork,
    UnifiedModelEvaluator
)

class VideoRecorder:
    """Simple video recorder for MuJoCo environments"""
    
    def __init__(self, output_path: str, fps: int = 30, width: int = 640, height: int = 480):
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = None
        self.frames = []
        
    def start_recording(self):
        """Start recording video"""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.video_writer = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.width, self.height))
        self.frames = []
        print(f"🎥 Started recording video: {self.output_path}")
        
    def capture_frame(self, env):
        """Capture a frame from the environment"""
        try:
            # Get RGB array from MuJoCo
            if hasattr(env, 'viewer') and env.viewer is not None and env.viewer.is_running():
                # Try to get pixels from the viewer
                pixels = env.viewer.read_pixels()
                if pixels is not None:
                    # Convert from RGB to BGR for OpenCV
                    frame = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                    # Resize if needed
                    if frame.shape[:2] != (self.height, self.width):
                        frame = cv2.resize(frame, (self.width, self.height))
                    self.frames.append(frame)
                    if self.video_writer:
                        self.video_writer.write(frame)
                    return True
        except Exception as e:
            print(f"⚠️  Frame capture failed: {e}")
        return False
        
    def stop_recording(self):
        """Stop recording and save video"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            
        # If direct recording failed, try to create video from frames
        if len(self.frames) > 0:
            try:
                writer = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.width, self.height))
                for frame in self.frames:
                    writer.write(frame)
                writer.release()
                print(f"✅ Video saved successfully: {self.output_path}")
                print(f"   Frames recorded: {len(self.frames)}")
                return True
            except Exception as e:
                print(f"❌ Failed to save video: {e}")
        
        print(f"⚠️  No frames captured for video recording")
        return False

class VideoEnabledEvaluator(UnifiedModelEvaluator):
    """Extended evaluator with video recording capabilities"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_recorder: Optional[VideoRecorder] = None
        
    def run_test_scenario_with_video(self, scenario_name: str, can_x: float, can_y: float,
                                   max_steps: int = 100, render: bool = True,
                                   render_delay: float = 0.1, 
                                   record_video: bool = True,
                                   video_path: Optional[str] = None) -> dict:
        """Run test scenario with optional video recording"""
        
        if record_video:
            if video_path is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_scenario = scenario_name.replace(" ", "_").replace("-", "").replace("(", "").replace(")", "")
                video_path = f"runlogs/evaluation_videos/{self.algorithm_type}_{safe_scenario}_{timestamp}.mp4"
            
            self.video_recorder = VideoRecorder(video_path, fps=int(1/render_delay) if render_delay > 0 else 30)
            self.video_recorder.start_recording()
        
        print(f"\n🧪 Running Test ({self.algorithm_type}): {scenario_name}")
        print(f"   Can position: ({can_x:.2f}, {can_y:.2f})")
        if record_video:
            print(f"   🎥 Recording video: {video_path}")
        
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
                
                # Capture frame for video if recording
                if record_video and self.video_recorder:
                    self.video_recorder.capture_frame(self.env)
                
                time.sleep(render_delay)
            
            obs = new_obs
            
            if terminal:
                print(f"   ✅ Episode terminated at step {step}")
                break
        
        # Stop video recording
        if record_video and self.video_recorder:
            success = self.video_recorder.stop_recording()
            if success:
                print(f"   🎥 Video saved: {video_path}")
        
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
            'success': grabbed or final_distance < 0.1,
            'video_path': video_path if record_video else None
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
    
    def run_comprehensive_evaluation_with_video(self, render: bool = True, render_delay: float = 0.08,
                                               record_video: bool = True, record_all_scenarios: bool = False):
        """Run comprehensive evaluation with video recording options"""
        print(f"🚀 Starting Comprehensive {self.algorithm_type} Model Evaluation with Video Recording")
        print("=" * 70)
        
        # Create video directory
        if record_video:
            os.makedirs("runlogs/evaluation_videos", exist_ok=True)
        
        # Print model info
        best_reward = self.model_data.get('best_total_reward', 'Unknown')
        last_episode = self.model_data.get('last_episode', 'Unknown')
        print(f"📊 Model Info: Best Training Reward: {best_reward}, Episode: {last_episode}")
        
        # Generate test scenarios
        test_scenarios = self.generate_test_scenarios(num_tests=12)
        results = []
        
        for i, (scenario_name, can_x, can_y) in enumerate(test_scenarios):
            # Record video for first few scenarios or all if requested
            should_record = record_video and (record_all_scenarios or i < 3)
            
            result = self.run_test_scenario_with_video(
                scenario_name=scenario_name,
                can_x=can_x, 
                can_y=can_y,
                max_steps=150,
                render=render,
                render_delay=render_delay,
                record_video=should_record
            )
            results.append(result)
            
            # Brief pause between scenarios
            if render:
                time.sleep(0.5)
        
        # Generate summary report
        self.generate_summary_report(results)
        
        # Print video summary
        if record_video:
            video_results = [r for r in results if r.get('video_path')]
            if video_results:
                print(f"\n🎥 Video Recording Summary:")
                print(f"   Videos recorded: {len(video_results)}")
                print(f"   Video directory: runlogs/evaluation_videos/")
                for result in video_results:
                    print(f"   - {result['scenario']}: {result['video_path']}")
        
        return results

def create_evaluation_summary_video(video_paths: List[str], output_path: str):
    """Create a summary video combining multiple scenario videos"""
    try:
        import cv2
        
        if not video_paths:
            print("❌ No video paths provided for summary")
            return False
        
        # Read first video to get dimensions
        cap = cv2.VideoCapture(video_paths[0])
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for video_path in video_paths:
            if os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                cap.release()
                
                # Add a brief pause between videos (black frames)
                black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                for _ in range(fps // 2):  # 0.5 second pause
                    out.write(black_frame)
        
        out.release()
        print(f"✅ Summary video created: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create summary video: {e}")
        return False

def main():
    """Main function with video recording support"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Video-Enabled Model Evaluator')
    parser.add_argument('model_path', nargs='?', help='Path to the model file')
    parser.add_argument('--no-video', action='store_true', help='Disable video recording')
    parser.add_argument('--record-all', action='store_true', help='Record all scenarios (not just first 3)')
    parser.add_argument('--no-render', action='store_true', help='Disable visual rendering')
    parser.add_argument('--render-delay', type=float, default=0.05, help='Delay between frames (default: 0.05)')
    
    args = parser.parse_args()
    
    # Check if a specific model was provided
    if args.model_path:
        if not os.path.exists(args.model_path):
            print(f"❌ Specified model file not found: {args.model_path}")
            sys.exit(1)
        model_path = args.model_path
        print(f"🎯 Using specified model: {model_path}")
    else:
        # Find latest model
        from unified_evaluator import list_all_algorithm_models
        available_models = list_all_algorithm_models()
        
        if not available_models:
            print("❌ No saved models found.")
            sys.exit(1)
        
        # Use the most recent model
        algorithm, model_path = available_models[0]
        print(f"\n🤖 Using most recent {algorithm} model: {model_path}")
    
    # Create video-enabled evaluator
    evaluator = VideoEnabledEvaluator(model_path)
    
    # Run evaluation with video recording
    results = evaluator.run_comprehensive_evaluation_with_video(
        render=not args.no_render,
        render_delay=args.render_delay,
        record_video=not args.no_video,
        record_all_scenarios=args.record_all
    )
    
    # Close environment
    evaluator.env.close()
    print("\n✅ Video-enabled evaluation completed!")
    
    # Create summary video if multiple videos were recorded
    video_results = [r for r in results if r.get('video_path') and os.path.exists(r['video_path'])]
    if len(video_results) > 1:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = f"runlogs/evaluation_videos/{evaluator.algorithm_type}_summary_{timestamp}.mp4"
        print(f"\n🎬 Creating summary video...")
        create_evaluation_summary_video([r['video_path'] for r in video_results], summary_path)

if __name__ == "__main__":
    main()
