#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Training Launcher for All RL Algorithms
Easily train DDPG, TD3, SAC, or PPO models
"""

import subprocess
import sys
import os
import time
import uuid
import datetime

def run_training(algorithm, redirect_output=True):
    """Run training for the specified algorithm"""
    
    script_map = {
        'ddpg': 'ddpg_clawbot.py',
        'td3': 'td3_clawbot.py', 
        'sac': 'sac_clawbot.py',
        'ppo': 'ppo_clawbot.py'
    }
    
    if algorithm.lower() not in script_map:
        print(f"❌ Unknown algorithm: {algorithm}")
        print(f"Available algorithms: {', '.join(script_map.keys())}")
        return False
    
    script_name = script_map[algorithm.lower()]
    
    if not os.path.exists(script_name):
        print(f"❌ Training script not found: {script_name}")
        return False
    
    print(f"🚀 Starting {algorithm.upper()} training...")
    print(f"📝 Using script: {script_name}")
    
    if redirect_output:
        # Create unique log files with timestamp and UUID
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        stdout_log = f"./runlogs/{algorithm.lower()}_training_{timestamp}_{unique_id}_stdout.log"
        stderr_log = f"./runlogs/{algorithm.lower()}_training_{timestamp}_{unique_id}_stderr.log"
        
        print(f"📋 Output will be logged to:")
        print(f"   STDOUT: {stdout_log}")
        print(f"   STDERR: {stderr_log}")
        
        # Start the training process with output redirection
        with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
            # Write header information to log files
            header = f"=== {algorithm.upper()} Training Started ===\n"
            header += f"Timestamp: {datetime.datetime.now().isoformat()}\n"
            header += f"Script: {script_name}\n"
            header += f"Working Directory: {os.getcwd()}\n"
            header += "=" * 50 + "\n\n"
            
            stdout_file.write(header)
            stderr_file.write(header)
            stdout_file.flush()
            stderr_file.flush()
            
            # Start the process
            process = subprocess.Popen(
                [sys.executable, script_name],
                stdout=stdout_file,
                stderr=stderr_file,
                universal_newlines=True
            )
            
            print(f"🔄 Training process started with PID: {process.pid}")
            print("💡 Tips:")
            print(f"   - Monitor progress: tail -f {stdout_log}")
            print(f"   - Check errors: tail -f {stderr_log}")
            print(f"   - Stop training: kill {process.pid} or Ctrl+C in the original terminal")
            
            try:
                # Wait for process to complete
                return_code = process.wait()
                
                if return_code == 0:
                    print(f"✅ {algorithm.upper()} training completed successfully!")
                else:
                    print(f"❌ {algorithm.upper()} training failed with return code: {return_code}")
                    print(f"   Check error log: {stderr_log}")
                
                return return_code == 0
                
            except KeyboardInterrupt:
                print(f"\n🛑 Training interrupted by user")
                process.terminate()
                
                # Wait a bit for graceful shutdown
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print("⚠️  Force killing process...")
                    process.kill()
                
                return False
    else:
        # Run without output redirection (direct to terminal)
        print("🔄 Starting training (output to terminal)...")
        try:
            result = subprocess.run([sys.executable, script_name])
            return result.returncode == 0
        except KeyboardInterrupt:
            print(f"\n🛑 Training interrupted by user")
            return False

def monitor_training_progress(log_file):
    """Monitor training progress from log file"""
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return
    
    print(f"📊 Monitoring training progress from: {log_file}")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        # Follow the log file like 'tail -f'
        subprocess.run(["tail", "-f", log_file])
    except KeyboardInterrupt:
        print("\n📈 Monitoring stopped")
    except FileNotFoundError:
        print("❌ 'tail' command not available. Reading file manually...")
        
        # Fallback: manual file monitoring
        try:
            with open(log_file, 'r') as f:
                f.seek(0, 2)  # Go to end of file
                while True:
                    line = f.readline()
                    if line:
                        print(line.strip())
                    else:
                        time.sleep(1)
        except KeyboardInterrupt:
            print("\n📈 Monitoring stopped")

def list_recent_training_logs():
    """List recent training log files"""
    import glob
    
    log_patterns = [
        './runlogs/*_training_*_stdout.log',
        './runlogs/*_training_*_stderr.log'
    ]
    
    all_logs = []
    for pattern in log_patterns:
        all_logs.extend(glob.glob(pattern))
    
    if not all_logs:
        print("📋 No training logs found in ./runlogs/")
        return
    
    # Sort by modification time, newest first
    all_logs.sort(key=os.path.getmtime, reverse=True)
    
    print("📋 Recent training logs:")
    for i, log_file in enumerate(all_logs[:10]):  # Show last 10
        mtime = os.path.getmtime(log_file)
        mod_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
        size_kb = os.path.getsize(log_file) / 1024
        log_type = "STDOUT" if "stdout" in log_file else "STDERR"
        print(f"   {i+1}. {os.path.basename(log_file)} ({log_type}, {mod_time}, {size_kb:.1f}KB)")

def main():
    """Main function with command line interface"""
    
    if len(sys.argv) < 2:
        print("🤖 RL Training Launcher")
        print("\nUsage:")
        print("  python train_rl.py <algorithm> [options]")
        print("\nAlgorithms:")
        print("  ddpg    - Deep Deterministic Policy Gradient")
        print("  td3     - Twin Delayed DDPG") 
        print("  sac     - Soft Actor-Critic")
        print("  ppo     - Proximal Policy Optimization")
        print("\nOptions:")
        print("  --no-redirect    - Output directly to terminal (default: redirect to logs)")
        print("  --monitor <log>  - Monitor training progress from log file")
        print("  --list-logs      - List recent training log files")
        print("\nExamples:")
        print("  python train_rl.py td3")
        print("  python train_rl.py sac --no-redirect")
        print("  python train_rl.py --list-logs")
        print("  python train_rl.py --monitor ./runlogs/td3_training_20250921_120000_abc123_stdout.log")
        sys.exit(1)
    
    # Parse arguments
    algorithm = None
    redirect_output = True
    monitor_log = None
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg == '--no-redirect':
            redirect_output = False
            i += 1
        elif arg == '--monitor':
            if i + 1 < len(sys.argv):
                monitor_log = sys.argv[i + 1]
                i += 2
            else:
                print("❌ --monitor requires a log file path")
                sys.exit(1)
        elif arg == '--list-logs':
            list_recent_training_logs()
            return
        elif arg.startswith('--'):
            print(f"❌ Unknown option: {arg}")
            sys.exit(1)
        else:
            algorithm = arg
            i += 1
    
    # Special commands
    if monitor_log:
        monitor_training_progress(monitor_log)
        return
    
    if not algorithm:
        print("❌ No algorithm specified")
        sys.exit(1)
    
    # Create runlogs directory if it doesn't exist
    os.makedirs('./runlogs', exist_ok=True)
    
    # Start training
    success = run_training(algorithm, redirect_output=redirect_output)
    
    if success:
        print(f"🎉 {algorithm.upper()} training completed successfully!")
    else:
        print(f"😞 {algorithm.upper()} training failed or was interrupted")
        
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
