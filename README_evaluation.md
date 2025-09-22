# RL Model Evaluation Guide

This directory contains comprehensive tools for evaluating your trained RL models (DQN, DDPG, TD3, SAC, PPO) on the ClawbotCan environment.

## Files Overview

### 1. `evaluate_dqn.py` - Comprehensive Evaluator
- **Purpose**: Full statistical analysis and performance evaluation
- **Features**: 
  - Detailed metrics (success rate, rewards, action distribution)
  - Success vs failure analysis
  - Optional visualization plots
  - JSON export of results
- **Usage**: `python evaluate_dqn.py`

### 2. `quick_test.py` - Quick Testing Tool
- **Purpose**: Rapid testing and debugging
- **Features**:
  - Fast evaluation with visual feedback
  - Simple pass/fail assessment
  - Customizable episode count and rendering
- **Usage**: 
  - `python quick_test.py` (5 episodes with rendering)
  - `python quick_test.py 10` (10 episodes with rendering)
  - `python quick_test.py 10 false` (10 episodes without rendering)

### 3. `unified_evaluator.py` - Multi-Algorithm Evaluator
- **Purpose**: Evaluate models from all RL algorithms (DDPG, TD3, SAC, PPO)
- **Features**: 
  - Auto-detects algorithm type from model file
  - Comprehensive test scenarios with multiple can positions
  - Statistical analysis and performance metrics
  - Visual rendering with configurable scenarios
- **Usage**: 
  - `python unified_evaluator.py` (auto-select latest model)
  - `python unified_evaluator.py model_file.pth` (specific model)

### 4. `video_evaluator.py` - Video Recording Evaluator
- **Purpose**: Record video of model evaluation for analysis
- **Features**:
  - Records MP4 videos of evaluation scenarios
  - Supports all RL algorithms (DDPG, TD3, SAC, PPO)
  - Configurable recording options (all scenarios or subset)
  - Creates summary videos combining multiple scenarios
- **Usage**:
  - `python video_evaluator.py model_file.pth` (record first 3 scenarios)
  - `python video_evaluator.py model_file.pth --record-all` (record all scenarios)
  - `python video_evaluator.py model_file.pth --no-video` (evaluation without recording)

### 5. `simple_video_eval.py` - Screen Recording Evaluator
- **Purpose**: Simple screen recording approach using external tools
- **Features**:
  - Uses FFmpeg for screen recording (if available)
  - Captures entire evaluation session
  - More reliable than direct frame capture
  - Automatic timeout and cleanup
- **Usage**:
  - `python simple_video_eval.py model_file.pth 5` (5-minute recording)
  - `python simple_video_eval.py model_file.pth 3` (3-minute recording)

### 6. `model_manager.py` - Model Management Tool
- **Purpose**: Manage, compare, and organize saved models
- **Features**:
  - List all available models with metadata
  - Compare performance between models
  - Clean up old model files
- **Usage**:
  - `python model_manager.py list` (show all models)
  - `python model_manager.py compare model1.pth model2.pth` (compare models)
  - `python model_manager.py cleanup --execute` (remove old models)

### 7. `reset_analysis.py` - Environment Analysis
- **Purpose**: Analyze environment reset distributions
- **Features**: Statistical analysis of initial states
- **Usage**: Called automatically from DQN.py or run standalone

## Evaluation Metrics

### Primary Metrics
- **Success Rate**: Percentage of episodes where object is successfully grabbed
- **Average Reward**: Mean total reward per episode
- **Average Steps**: Mean number of steps to completion/termination
- **Final Distance**: Distance to target at episode end

### Advanced Analysis
- **Action Distribution**: How often each action is chosen
- **Action Sequences**: Complete record of actions taken in each episode
- **Action Patterns**: Most common action sequences and transitions
- **Action Timing**: When different actions are typically used within episodes
- **Action Consistency**: How often the model switches between actions
- **Success vs Failure Comparison**: Action usage differences between successful and failed episodes
- **Temporal Analysis**: Performance trends over episodes

## Quick Start

### 1. List Available Models
```bash
# Show all trained models with performance info
python model_manager.py list

# Expected output:
# 📋 Model Summary
# # Filename                          Type        Episode  Reward   Size    Date
# 1 ppo_clawbot_model_completed...   completed   2025     9644.4   0.4MB   09/22 09:26
# 2 sac_clawbot_model_checkpoint...  checkpoint  1500     8234.1   0.4MB   09/22 10:43
```

### 2. Quick Evaluation (Recommended First Step)
```bash
# Auto-select latest model and run evaluation
python unified_evaluator.py

# Or specify a specific model
python unified_evaluator.py ppo_clawbot_model_completed_20250922_092654.pth

# Expected output:
# ✅ Loaded PPO model from ppo_clawbot_model_completed_20250922_092654.pth
# 🚀 Starting Comprehensive PPO Model Evaluation
# 📊 Model Info: Best Training Reward: 9644.36, Episode: 2025
# 🧪 Running Test (PPO): Far - Front Left (-23°)
# ...
# 📈 PPO EVALUATION SUMMARY REPORT
# 🎯 Overall Performance:
#    Tests passed: 8/12 (66.7%)
#    Objects grabbed: 3/12 (25.0%)
#    Average reward: 145.23
```

### 3. Video Recording Evaluation
```bash
# Record video of first 3 evaluation scenarios
python video_evaluator.py ppo_clawbot_model_completed_20250922_092654.pth

# Record all scenarios with video
python video_evaluator.py ppo_clawbot_model_completed_20250922_092654.pth --record-all

# Screen recording approach (more reliable)
python simple_video_eval.py ppo_clawbot_model_completed_20250922_092654.pth 5

# Videos saved to: runlogs/evaluation_videos/
```

### 3. Custom Evaluation
Modify the `main()` function in `evaluate_dqn.py` to customize:
```python
evaluation_results = evaluate_model(
    model_path="dqn_model.pth",
    num_episodes=50,        # Number of episodes
    render_episodes=5,      # How many to render
    verbose=True           # Detailed output
)
```

## Understanding Results

### Success Rate Interpretation
- **80-100%**: Excellent performance, model is well-trained
- **60-80%**: Good performance, may benefit from more training
- **40-60%**: Fair performance, needs improvement
- **<40%**: Poor performance, requires significant training

### Action Analysis
The evaluator tracks which actions the model chooses:
- **Action 0 (Stay)**: [0, 0, 0, 0] - No movement
- **Action 1 (Left-Right)**: [-1, 1, 0, 0] - Rotate left claw
- **Action 2 (Right-Left)**: [1, -1, 0, 0] - Rotate right claw

### Reward Analysis
- **Positive rewards**: Generally indicate good performance
- **Reward components**: Based on distance to target and angle alignment
- **Success bonus**: +50 points when object is grabbed

## Troubleshooting

### Model Loading Issues
```bash
❌ Model file not found: dqn_model.pth
```
**Solution**: Ensure you've trained and saved a model first by running `DQN.py`

### Import Errors
```bash
ModuleNotFoundError: No module named 'torch'
```
**Solution**: Install required dependencies:
```bash
pip install torch numpy
pip install matplotlib pandas scipy seaborn  # For visualization
```

### Rendering Issues
If rendering doesn't work:
1. Set `render_episodes=0` in the evaluator
2. Use `python quick_test.py 5 false` for no rendering
3. Check MuJoCo installation

## Advanced Usage

### Batch Evaluation
Test multiple model checkpoints:
```python
models = ["dqn_model_100.pth", "dqn_model_500.pth", "dqn_model_1000.pth"]
for model_path in models:
    results = evaluate_model(model_path, num_episodes=50, verbose=False)
    print(f"{model_path}: Success Rate = {results['statistics']['success_rate']:.1f}%")
```

### Custom Metrics
Add your own evaluation metrics by modifying the `calculate_statistics()` function in `evaluate_dqn.py`.

## Output Files
- `evaluation_results.json`: Complete results data including action sequences
- `evaluation_plots.png`: Performance visualization (if matplotlib available)
- `action_analysis_plots.png`: Comprehensive action analysis visualization (12 plots)

## Action Analysis Features

The enhanced evaluator now provides detailed action analysis:

### 📊 Action Visualizations (12 plots total)
1. **Action Sequences**: Time-series plots of successful vs failed episodes
2. **Action Frequency Heatmap**: How action usage changes across episodes
3. **Success vs Failure Distribution**: Action usage comparison
4. **Episode Length Distribution**: How long episodes typically last
5. **Action Transition Matrix**: What actions follow other actions
6. **Action Timing**: When actions are used within episodes
7. **Action Consistency**: How often models switch between actions
8. **Individual Episode Analysis**: Detailed view of specific interesting episodes

### 📈 Text Analysis Output
```
🎯 ACTION SEQUENCE ANALYSIS:
  Average Episode Length:    89.2 steps
  Average Action Switch Rate: 0.234 (0=no switches, 1=switch every step)
  Most Common Action Patterns (3-step):
    Stay → L-R → Stay       : 15.3% (45 times)
    L-R → Stay → L-R        : 12.8% (38 times)
    Stay → Stay → Stay      :  9.1% (27 times)

✅❌ SUCCESS vs FAILURE ACTION COMPARISON:
  Success Episodes - Action Distribution:
    Stay [0,0,0,0]         : 45.2%
    Left-Right [-1,1,0,0]  : 32.1%
    Right-Left [1,-1,0,0]  : 22.7%
  💡 Insight: Failed episodes switch actions 0.089 more often (less consistent)
```

## Video Recording Evaluation

### Overview
The video evaluation tools allow you to record your model's performance for detailed analysis and sharing.

### Method 1: Direct Video Recording (`video_evaluator.py`)
```bash
# Basic video recording (first 3 scenarios)
python video_evaluator.py ppo_clawbot_model_completed_20250922_092654.pth

# Record all 12 test scenarios
python video_evaluator.py ppo_clawbot_model_completed_20250922_092654.pth --record-all

# Disable video recording (evaluation only)
python video_evaluator.py ppo_clawbot_model_completed_20250922_092654.pth --no-video

# Adjust rendering speed
python video_evaluator.py ppo_clawbot_model_completed_20250922_092654.pth --render-delay 0.05
```

**Features:**
- Records individual MP4 files for each test scenario
- Auto-creates summary video combining multiple scenarios
- Supports all RL algorithms (detects automatically)
- Files saved to `runlogs/evaluation_videos/`

### Method 2: Screen Recording (`simple_video_eval.py`)
```bash
# 5-minute screen recording session
python simple_video_eval.py ppo_clawbot_model_completed_20250922_092654.pth 5

# 3-minute session
python simple_video_eval.py ppo_clawbot_model_completed_20250922_092654.pth 3
```

**Features:**
- More reliable than direct frame capture
- Records entire evaluation session including terminal output
- Requires FFmpeg (auto-detects availability)
- Single video file output

### Video Output Structure
```
runlogs/evaluation_videos/
├── PPO_Far_Front_Left_23°_20250922_111540.mp4
├── PPO_Far_Angled_Right_42°_20250922_111616.mp4
├── PPO_Medium_Side_Left_62°_20250922_111632.mp4
├── PPO_summary_20250922_111700.mp4
└── screen_recording_20250922_111800.mp4
```

### Video Analysis Tips
1. **Watch for consistency** - Does the robot behave similarly in similar scenarios?
2. **Check approach patterns** - How does the robot navigate to the target?
3. **Analyze failure cases** - What causes the robot to fail in certain scenarios?
4. **Compare algorithms** - Record videos from different models (PPO vs SAC vs TD3)

### Troubleshooting Video Recording
- **Segmentation faults**: Use screen recording method instead
- **Empty video files**: Ensure rendering is enabled (`--no-render` disabled)
- **FFmpeg not found**: Install with `sudo apt install ffmpeg` (Linux) or `brew install ffmpeg` (Mac)
- **Performance issues**: Increase `--render-delay` to reduce frame rate

## Algorithm Comparison

### Compare Multiple Models
```bash
# Quick comparison of latest models from each algorithm
python unified_evaluator.py ppo_clawbot_model_completed_20250922_092654.pth > ppo_results.txt
python unified_evaluator.py sac_clawbot_model_checkpoint_20250922_104358.pth > sac_results.txt
python unified_evaluator.py td3_clawbot_model_checkpoint_20250922_105131.pth > td3_results.txt

# Compare model metadata
python model_manager.py compare ppo_clawbot_model_completed_20250922_092654.pth sac_clawbot_model_checkpoint_20250922_104358.pth
```

### Performance Interpretation
- **PPO**: Often achieves highest training rewards, good for complex environments
  - ⚠️  **Real Robot Issue**: Works on real robot but lacks turning capability - only moves forward
- **SAC**: Excellent exploration, good final performance, entropy-based learning
- **TD3**: Stable training, good for continuous control, twin critics reduce overestimation
  - ⚠️  **Known Issue**: TD3 model tends to move backward while aligning angle instead of approaching the target object. This suggests a reward function problem that needs fixing.
- **DDPG**: Simpler algorithm, faster training, may be less stable

### 🤖 Real Robot vs Simulation Gap
**Critical Issues Discovered During Real Robot Testing**:
- **Limited Turning**: Models only move forward, cannot turn to reach angled targets
- **No Memory**: Cannot track objects that move out of field of view
- **Perfect Vision Dependency**: Training assumes unlimited FOV and perfect object detection
- **Missing Robustness**: No training with sensor failures or partial observations

See `KNOWN_ISSUES.md` and `IMPROVEMENT_ROADMAP.md` for detailed improvement plans.

## Tips
1. **Start with model listing** to see all available trained models
2. **Use unified_evaluator.py** for comprehensive performance analysis
3. **Record videos** of your best performing models for analysis
4. **Compare algorithms** by evaluating models from different training runs
5. **Use screen recording** if direct video capture fails
6. **Check logs** in `runlogs/` for detailed evaluation output
