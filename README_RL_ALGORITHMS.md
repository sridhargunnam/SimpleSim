# 🤖 Multi-Algorithm Reinforcement Learning Suite

This repository now supports **four state-of-the-art RL algorithms** for continuous control tasks, specifically designed for the clawbot can-grabbing environment.

## 🚀 Available Algorithms

| Algorithm | Type | Sample Efficiency | Stability | Best For |
|-----------|------|------------------|-----------|----------|
| **DDPG** | Off-Policy | Medium | Low | Baseline continuous control |
| **TD3** | Off-Policy | Medium | **High** | Improved DDPG with stability fixes |
| **SAC** | Off-Policy | **Very High** | **High** | Maximum performance & sample efficiency |
| **PPO** | On-Policy | Low | **Very High** | Maximum stability & reliability |

## 📁 File Structure

```
├── ddpg_clawbot.py          # Original DDPG implementation
├── td3_clawbot.py           # Twin Delayed DDPG 
├── sac_clawbot.py           # Soft Actor-Critic
├── ppo_clawbot.py           # Proximal Policy Optimization
├── train_rl.py              # 🆕 Unified training launcher
├── unified_evaluator.py     # 🆕 Universal model evaluator
├── quick_eval.py            # 🆕 Quick testing for all algorithms
├── test_configurations.py   # Updated configuration testing
├── model_manager.py         # Updated model management
├── mujoco_env.py            # Environment (unchanged)
└── runlogs/                 # 🆕 Training output logs
```

## 🏃 Quick Start

### 1. Training Models

**Easy way (recommended):**
```bash
# Train any algorithm with automatic log management
python train_rl.py td3     # Train TD3
python train_rl.py sac     # Train SAC  
python train_rl.py ppo     # Train PPO
python train_rl.py ddpg    # Train original DDPG
```

**Manual way:**
```bash
# Direct training (output to terminal)
python td3_clawbot.py      # Train TD3 directly
python sac_clawbot.py      # Train SAC directly
python ppo_clawbot.py      # Train PPO directly
```

### 2. Evaluating Models

**Universal evaluation (works with any algorithm):**
```bash
python unified_evaluator.py                    # Auto-detect latest model
python unified_evaluator.py your_model.pth     # Evaluate specific model
```

**Quick testing:**
```bash
python quick_eval.py                           # Quick test of latest model
python quick_eval.py --compare                 # Compare all algorithms
python quick_eval.py --episodes 10 --no-render # Batch testing
```

### 3. Configuration Testing

```bash
python test_configurations.py compare          # Test different scenarios
python test_configurations.py forward          # Test forward hemisphere only
python test_configurations.py close            # Test close range only
```

## 📊 Training Monitoring

### Real-time Monitoring
```bash
# Start training with logs
python train_rl.py sac

# In another terminal, monitor progress
python train_rl.py --monitor ./runlogs/sac_training_*_stdout.log

# Or use tail directly
tail -f ./runlogs/sac_training_*_stdout.log
```

### List Recent Training Runs
```bash
python train_rl.py --list-logs
```

## 🔧 Algorithm Details

### TD3 (Twin Delayed DDPG)
- **Improvements over DDPG:**
  - Two Q-networks reduce overestimation bias
  - Target policy smoothing for stability  
  - Delayed policy updates prevent instability
- **Best for:** Reliable improvement over DDPG
- **Training time:** ~2-4 hours for good performance

### SAC (Soft Actor-Critic)  
- **Key features:**
  - Entropy maximization for better exploration
  - Automatic temperature tuning
  - Superior sample efficiency
- **Best for:** Maximum performance with fewer samples
- **Training time:** ~1-3 hours for good performance

### PPO (Proximal Policy Optimization)
- **Key features:**
  - On-policy learning (no replay buffer)
  - Clipped surrogate objective for safety
  - Very stable, less sensitive to hyperparameters
- **Best for:** Rock-solid stability and reliability
- **Training time:** ~3-6 hours (more episodes needed)

## 📈 Performance Comparison

Run a comprehensive comparison:

```bash
# Train multiple algorithms
python train_rl.py td3 &
python train_rl.py sac &  
python train_rl.py ppo &
wait

# Compare their performance
python quick_eval.py --compare
```

Expected performance ranking (for this task):
1. **SAC** - Fastest learning, highest peak performance
2. **TD3** - Reliable and stable, good performance  
3. **PPO** - Most stable, moderate performance
4. **DDPG** - Baseline, can be unstable

## 🎛️ Hyperparameter Notes

### TD3 Hyperparameters
- `policy_noise = 0.2` - Noise added to target policy
- `noise_clip = 0.5` - Clip noise to this range  
- `policy_freq = 2` - Update policy every 2 Q-updates

### SAC Hyperparameters
- `target_entropy = -act_dim` - Automatic entropy tuning target
- `polyak = 0.995` - Slower target network updates
- Higher learning rate (3e-4) than DDPG/TD3

### PPO Hyperparameters
- `rollout_length = 2048` - Steps per update
- `clip_range = 0.2` - PPO clipping parameter
- `gae_lambda = 0.95` - Advantage estimation parameter

## 🔍 Model Management

```bash
# List all models (all algorithms)
python model_manager.py list

# Clean up old models
python model_manager.py cleanup --execute

# Compare specific models  
python model_manager.py compare model1.pth model2.pth
```

## 📝 Output Files

### Training Outputs
- **Models:** `{algorithm}_clawbot_model_{type}_{timestamp}.pth`
- **Logs:** `./runlogs/{algorithm}_training_{timestamp}_{uuid}_{stdout|stderr}.log`

### Model Contents
Each saved model contains:
- Algorithm-specific network states
- Optimizer states  
- Training metadata (episodes, best reward, etc.)
- Algorithm identifier for auto-detection

## 🚨 Troubleshooting

### Common Issues

1. **"No module named 'mujoco'"**
   ```bash
   pip install mujoco
   ```

2. **Training seems stuck**
   - Check logs: `tail -f ./runlogs/*_stdout.log`
   - TD3/SAC should show progress within 1000 episodes
   - PPO may take longer to show improvement

3. **Low performance**
   - Try SAC first (most sample efficient)
   - Ensure environment is working: test with quick_eval.py
   - Check if can spawning parameters are reasonable

4. **Memory issues**
   - Reduce rollout_length for PPO
   - Reduce replay buffer size (maxlen) for off-policy methods

### Performance Tips

1. **For fastest results:** Use SAC
2. **For most reliable training:** Use PPO  
3. **For DDPG improvement:** Use TD3
4. **For research/comparison:** Train all and compare

## 🎯 Next Steps

1. **Train your first model:**
   ```bash
   python train_rl.py sac
   ```

2. **Evaluate it:**
   ```bash
   python unified_evaluator.py
   ```

3. **Compare algorithms:**
   ```bash
   python quick_eval.py --compare
   ```

4. **Experiment with configurations:**
   ```bash
   python test_configurations.py compare
   ```

## 📚 References

- **DDPG:** Lillicrap et al. "Continuous Control with Deep Reinforcement Learning" (2015)
- **TD3:** Fujimoto et al. "Addressing Function Approximation Error in Actor-Critic Methods" (2018)  
- **SAC:** Haarnoja et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL" (2018)
- **PPO:** Schulman et al. "Proximal Policy Optimization Algorithms" (2017)

---

🎉 **Happy Training!** This suite gives you access to the most effective modern RL algorithms for continuous control. Start with SAC for quick results, or try PPO for maximum stability.
