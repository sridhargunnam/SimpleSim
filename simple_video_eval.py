#!/usr/bin/env python3
"""
Simple Video Recording Script for Model Evaluation
Uses screen recording tools to capture the evaluation
"""

import subprocess
import time
import os
import datetime
import sys

def run_evaluation_with_screen_recording(model_path: str, duration_minutes: int = 5):
    """Run model evaluation while recording the screen"""
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"runlogs/evaluation_videos/screen_recording_{timestamp}.mp4"
    log_filename = f"runlogs/model_review_20250922_110704/evaluation_{timestamp}.log"
    
    # Ensure directories exist
    os.makedirs("runlogs/evaluation_videos", exist_ok=True)
    os.makedirs("runlogs/model_review_20250922_110704", exist_ok=True)
    
    print(f"🎥 Starting screen recording evaluation")
    print(f"   Model: {model_path}")
    print(f"   Video will be saved to: {video_filename}")
    print(f"   Log will be saved to: {log_filename}")
    print(f"   Duration: {duration_minutes} minutes")
    print("\n" + "="*60)
    
    # Check if ffmpeg is available for screen recording
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        ffmpeg_available = True
    except:
        ffmpeg_available = False
    
    if ffmpeg_available:
        print("✅ FFmpeg available - will record screen")
        # Start screen recording with ffmpeg
        record_cmd = [
            'ffmpeg', '-y',  # Overwrite output file
            '-f', 'x11grab',  # Screen capture on Linux
            '-s', '1024x768',  # Resolution
            '-r', '30',  # Frame rate
            '-i', ':0.0',  # Display
            '-t', str(duration_minutes * 60),  # Duration in seconds
            '-c:v', 'libx264',  # Video codec
            '-preset', 'fast',  # Encoding speed
            video_filename
        ]
        
        print("🎬 Starting screen recording...")
        record_process = subprocess.Popen(record_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2)  # Let recording start
    else:
        print("⚠️  FFmpeg not available - evaluation will run without screen recording")
        record_process = None
    
    # Start the evaluation
    eval_cmd = [
        'python', 'unified_evaluator.py', model_path
    ]
    
    print("🚀 Starting model evaluation...")
    with open(log_filename, 'w') as log_file:
        eval_process = subprocess.Popen(eval_cmd, stdout=log_file, stderr=subprocess.STDOUT)
        
        # Wait for evaluation to complete or timeout
        try:
            eval_process.wait(timeout=duration_minutes * 60)
            print("✅ Evaluation completed successfully")
        except subprocess.TimeoutExpired:
            print("⏰ Evaluation timed out - terminating...")
            eval_process.terminate()
            eval_process.wait()
    
    # Stop screen recording
    if record_process:
        print("🛑 Stopping screen recording...")
        record_process.terminate()
        record_process.wait()
        
        if os.path.exists(video_filename):
            file_size = os.path.getsize(video_filename) / (1024 * 1024)  # MB
            print(f"✅ Screen recording saved: {video_filename} ({file_size:.1f}MB)")
        else:
            print("❌ Screen recording failed")
    
    # Show log content
    if os.path.exists(log_filename):
        print(f"\n📋 Evaluation log saved: {log_filename}")
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY (last 20 lines):")
        print("="*60)
        with open(log_filename, 'r') as f:
            lines = f.readlines()
            for line in lines[-20:]:
                print(line.rstrip())
    
    return video_filename if record_process else None, log_filename

def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_video_eval.py <model_path> [duration_minutes]")
        print("Example: python simple_video_eval.py ppo_clawbot_model_completed_20250922_092654.pth 3")
        sys.exit(1)
    
    model_path = sys.argv[1]
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    video_file, log_file = run_evaluation_with_screen_recording(model_path, duration)
    
    print(f"\n🎉 Evaluation complete!")
    print(f"📁 Files created:")
    if video_file:
        print(f"   🎥 Video: {video_file}")
    print(f"   📋 Log: {log_file}")

if __name__ == "__main__":
    main()
