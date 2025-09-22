# Video Evaluation Quick Reference

## 🎥 Record Model Evaluation Videos

### Method 1: Direct Video Recording (Advanced)
```bash
# Record first 3 scenarios with video
python video_evaluator.py ppo_clawbot_model_completed_20250922_092654.pth

# Record ALL scenarios with video
python video_evaluator.py ppo_clawbot_model_completed_20250922_092654.pth --record-all

# Evaluation only (no video)
python video_evaluator.py ppo_clawbot_model_completed_20250922_092654.pth --no-video

# Adjust frame rate
python video_evaluator.py ppo_clawbot_model_completed_20250922_092654.pth --render-delay 0.1
```

### Method 2: Screen Recording (Recommended - More Reliable)
```bash
# 5-minute screen recording session
python simple_video_eval.py ppo_clawbot_model_completed_20250922_092654.pth 5

# 3-minute session
python simple_video_eval.py ppo_clawbot_model_completed_20250922_092654.pth 3
```

### Method 3: Standard Evaluation (No Video)
```bash
# Auto-select latest model
python unified_evaluator.py

# Specific model
python unified_evaluator.py ppo_clawbot_model_completed_20250922_092654.pth
```

## 📋 Model Management

```bash
# List all available models
python model_manager.py list

# Compare two models
python model_manager.py compare model1.pth model2.pth

# Clean up old models
python model_manager.py cleanup --execute
```

## 📁 Output Locations

- **Videos**: `runlogs/evaluation_videos/`
- **Logs**: `runlogs/model_review_YYYYMMDD_HHMMSS/`
- **Models**: `*_clawbot_model_*.pth`

## 🎯 Quick Commands for Your Models

Based on your current models:

```bash
# PPO Model (Latest - Best Performance: 9644.36 reward)
python video_evaluator.py ppo_clawbot_model_completed_20250922_092654.pth
python simple_video_eval.py ppo_clawbot_model_completed_20250922_092654.pth 5

# SAC Model
python video_evaluator.py sac_clawbot_model_checkpoint_20250922_104358.pth
python simple_video_eval.py sac_clawbot_model_checkpoint_20250922_104358.pth 5

# TD3 Model (⚠️ Known Issue: moves backward while aligning angle)
python video_evaluator.py td3_clawbot_model_checkpoint_20250922_105131.pth
python simple_video_eval.py td3_clawbot_model_checkpoint_20250922_105131.pth 5
```

## 🔧 Troubleshooting

- **Segmentation fault**: Use `simple_video_eval.py` instead
- **Empty videos**: Ensure rendering is enabled (remove `--no-render`)
- **FFmpeg missing**: `sudo apt install ffmpeg` (Linux) or `brew install ffmpeg` (Mac)
- **Slow performance**: Increase `--render-delay` value

## 📊 Expected Performance (Your PPO Model)

- **Training Reward**: 9644.36 (Excellent!)
- **Episodes Trained**: 2025
- **Test Reward**: ~180 per scenario
- **Success Rate**: ~67% (8/12 scenarios)
- **Grab Rate**: ~25% (3/12 scenarios)

Your PPO model shows excellent performance - definitely worth recording videos of!
