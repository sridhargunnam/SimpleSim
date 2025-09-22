# Known Issues and Bugs

## 🐛 Current Issues

### TD3 Algorithm - Backward Movement Issue
**Status**: 🔴 **CRITICAL** - Needs Fix  
**Algorithm**: TD3 (Twin Delayed Deep Deterministic Policy Gradient)  
**Issue**: TD3 model moves backward while trying to align angle instead of approaching the target object  
**Impact**: Poor performance, counter-productive behavior  
**Discovered**: 2025-09-22 during video evaluation  

**Description**:
The TD3 model exhibits problematic behavior where it:
- Tries to align its angle with the target (correct)
- But moves backward/away from the target while doing so (incorrect)
- This results in the robot getting further from the object instead of closer
- Suggests an issue with the reward function or action space interpretation

**Potential Causes**:
1. **Reward Function Issue**: Distance reward component may be incorrectly implemented
2. **Action Space Confusion**: TD3 may be interpreting movement actions incorrectly
3. **Training Instability**: Twin critic networks may have learned suboptimal policy
4. **Environment State**: TD3 may be misinterpreting the observation space

**Suggested Fixes**:
1. **Review Reward Function**: Check distance calculation and ensure it encourages approach
2. **Debug Action Mapping**: Verify that positive/negative actions map correctly to forward/backward
3. **Retrain with Modified Reward**: Increase weight on distance reduction reward
4. **Add Debugging Output**: Log actions and rewards during evaluation to identify the issue

**Files to Investigate**:
- `td3_clawbot.py` - Main TD3 implementation
- `mujoco_env.py` - Environment reward function
- TD3 model files: `td3_clawbot_model_*.pth`

**Comparison**:
- ✅ PPO: Approaches target correctly (9644.36 reward)
- ✅ SAC: Expected to approach target correctly
- ❌ TD3: Moves backward while aligning
- ❓ DDPG: Behavior unknown

### SAC Algorithm - No Forward Locomotion / Claw Reward Exploitation
**Status**: 🔴 **CRITICAL** - Needs Fix  
**Algorithm**: SAC (Soft Actor-Critic)  
**Issue**: Orients toward the object but does not drive forward; repeatedly opens/closes the claw to farm reward  
**Impact**: Fails to approach objects; 0/12 test passes in evaluation, no successful grabs  
**Discovered**: 2025-09-22 during unified evaluation

**Description**:
- SAC consistently rotates to face the target but remains largely stationary.
- It oscillates the gripper (open/close) to collect reward signal without making translational progress.
- Evaluation summary indicates poor approach behavior (e.g., average final distance ≈ 0.63 m; 0/12 success).

**Potential Causes**:
1. Reward shaping allows gripper-related incentives without sufficient progress-based rewards.
2. Per-step distance decrease is under-weighted relative to movement penalties and time penalty.
3. Claw bonus not gated by proximity/alignment, enabling reward hacking from afar.
4. Temperature/entropy encourages dithering over committing to forward motion.

**Suggested Fixes**:
1. Gate grab/close bonuses on proximity (e.g., distance < 0.15 m) and good alignment (|dtheta| < 0.3).
2. Add step-wise progress shaping: positive reward for distance reduction; penalty for distance increase.
3. Add penalty for gripper actuation when far or misaligned (anti-spam term).
4. Soften forward-movement penalties when well aligned; encourage approach once |dtheta| is small.
5. Tune SAC temperature (alpha) or use auto-alpha with tighter target entropy to reduce dithering.

---

## 🔴 Critical Real-World Performance Issues

### PPO Real Robot Performance - Limited Turning Capability
**Status**: 🔴 **CRITICAL** - Real Robot Testing Issue  
**Algorithm**: PPO (and likely all current algorithms)  
**Issue**: PPO model only moves forward toward object but doesn't turn properly  
**Impact**: Cannot navigate to objects that require turning maneuvers  
**Discovered**: 2025-09-22 during real robot evaluation  
**Root Cause**: Training environment may not emphasize turning behavior sufficiently

### Lack of Memory/State History Architecture
**Status**: 🔴 **CRITICAL** - Architecture Limitation  
**Issue**: Current RL models have no memory of previous states  
**Impact**: Cannot locate objects when they move out of field of view  
**Requirement**: Need LSTM/GRU or Transformer-based architecture for sequential memory  

### Unrestricted Field of View During Training
**Status**: 🔴 **CRITICAL** - Training Environment Issue  
**Issue**: Training assumes perfect object visibility at all times  
**Impact**: Models fail when objects are occluded or out of view in real world  
**Requirement**: Restrict/limit field of view during training to match real robot constraints

### Missing Partial Observability Training
**Status**: 🔴 **CRITICAL** - Training Robustness Issue  
**Issue**: No training scenarios with missing/corrupted object detection  
**Impact**: Models fail when sensors don't detect objects (common in real world)  
**Requirement**: Randomly mask object observations during training to force reliance on memory

## 🟡 Minor Issues

### Video Recording Segmentation Faults
**Status**: 🟡 **WORKAROUND AVAILABLE**  
**Component**: `video_evaluator.py`  
**Issue**: Direct frame capture causes segmentation faults  
**Workaround**: Use `simple_video_eval.py` with screen recording instead  
**Impact**: Video recording requires alternative method  

---

## ✅ Resolved Issues

*No resolved issues yet*

---

## 🚀 Required Improvements Plan

### 1. Improve Turning Capability
**Priority**: 🔴 **HIGH**  
**Implementation**:
- **Reward Function Enhancement**: Increase reward for turning actions when object is at an angle
- **Training Scenarios**: Add more scenarios where object requires significant turning to reach
- **Action Space Review**: Ensure turning actions are properly weighted and encouraged
- **Environment Modifications**: Create training scenarios that require complex navigation paths

**Files to Modify**:
- `mujoco_env.py` - Enhance reward function for turning behavior
- All algorithm files (`ppo_clawbot.py`, `sac_clawbot.py`, etc.) - Review action space
- Training scenarios - Add angular approach requirements

### 2. Add Memory/Sequential Architecture
**Priority**: 🔴 **CRITICAL**  
**Implementation**:
- **LSTM/GRU Integration**: Replace feedforward networks with recurrent architectures
- **Transformer Option**: Consider attention-based models for longer memory
- **State History Buffer**: Maintain history of last N observations and actions
- **Memory-Based Reward**: Reward successful navigation when object is not visible

**New Files Needed**:
- `lstm_ppo_clawbot.py` - LSTM-enhanced PPO implementation
- `gru_sac_clawbot.py` - GRU-enhanced SAC implementation  
- `memory_utils.py` - Utilities for state history management

### 3. Restrict Field of View During Training
**Priority**: 🔴 **HIGH**  
**Implementation**:
- **Limited Vision Environment**: Create environment variant with restricted FOV
- **Occlusion Simulation**: Add random obstacles that block object visibility
- **Sensor Noise**: Add realistic sensor limitations and noise
- **Partial Observability**: Train with intermittent object detection

**Environment Enhancements**:
- `limited_fov_env.py` - New environment with realistic vision constraints
- Modify `mujoco_env.py` to support vision masking parameters
- Add FOV angle limitations and distance restrictions

### 4. Partial Observability Training
**Priority**: 🔴 **CRITICAL**  
**Implementation**:
- **Observation Masking**: Randomly set object observations to "unknown/missing"
- **Sensor Failure Simulation**: Simulate realistic sensor failures during training
- **Memory-Dependent Navigation**: Force robot to rely on previous observations
- **Robust State Estimation**: Train models to maintain object location estimates

**Training Enhancements**:
- Add `--partial-obs` flag to all training scripts
- Implement observation masking probability (e.g., 20% of timesteps)
- Create "blind navigation" training phases
- Add state estimation accuracy metrics

## 📝 Investigation Notes

### Real Robot vs Simulation Gap Analysis
**Findings from Real Robot Testing**:
- ✅ PPO moves toward object (basic navigation works)
- ❌ No turning behavior (major limitation)
- ❌ Loses track when object moves out of view
- ❌ Cannot handle partial occlusion or sensor noise

**Simulation Limitations Identified**:
- Perfect object detection at all times
- Unlimited field of view
- No sensor noise or failures
- Simplified turning dynamics

### TD3 Behavior Analysis
- **Training Reward**: Unknown (need to check model metadata)
- **Evaluation Behavior**: Moves backward + angle alignment
- **Comparison Needed**: Record video to confirm behavior vs other algorithms
- **Next Steps**: 
  1. Record TD3 evaluation video
  2. Compare with PPO video
  3. Analyze reward function in `mujoco_env.py`
  4. Check TD3 action interpretation

### Implementation Priority Order
1. **Immediate**: Fix TD3 backward movement issue
2. **Short-term**: Enhance turning capability in reward function
3. **Medium-term**: Implement LSTM/GRU memory architecture
4. **Medium-term**: Add partial observability training
5. **Long-term**: Create comprehensive real-world training environment

### Debugging Commands
```bash
# Test current models on real robot scenarios
python unified_evaluator.py ppo_clawbot_model_completed_20250922_092654.pth

# Record TD3 behavior for analysis
python simple_video_eval.py td3_clawbot_model_checkpoint_20250922_105131.pth 3

# Compare with working PPO model
python simple_video_eval.py ppo_clawbot_model_completed_20250922_092654.pth 3

# Analyze model metadata
python model_manager.py compare td3_clawbot_model_checkpoint_20250922_105131.pth ppo_clawbot_model_completed_20250922_092654.pth

# Test turning scenarios (when implemented)
python unified_evaluator.py --test-turning-only ppo_clawbot_model_completed_20250922_092654.pth
```
