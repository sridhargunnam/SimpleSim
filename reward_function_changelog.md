# Reward Function Change Log

## Version History

### v1.0 - Original Organic Reward (Lost)
- **Status**: Lost during development
- **Issue**: Robot learned well in simulation but moved straight in real world
- **Behavior**: Had some working version but wasn't committed

### v2.0 - Structured Phase-Based Reward (2025-09-24)
- **File**: `mujoco_env.py` lines 152-207
- **Goal**: Systematic Align → Approach → Grasp learning
- **Changes**:
  - Binary phase switching with `ALIGNMENT_THRESHOLD = 0.2` (11.5°)
  - ALIGN phase: Only reward alignment, ignore distance
  - APPROACH phase: Only reward proximity, penalize misalignment
  - GRASP phase: Huge success reward (1000.0)

**Code**:
```python
if abs(dtheta) > ALIGNMENT_THRESHOLD:
    # ALIGN: alignment_reward = 10.0 * exp(-2.0 * abs(dtheta)) - 0.5
else:
    # APPROACH: proximity_reward = 20.0 / (1.0 + 10.0 * distance)
    #           alignment_penalty = -5.0 * abs(dtheta)
```

**Result**: Robot learned to align (~15°) but got stuck, never transitioned to approach

### v2.1 - Lenient Threshold Fix (2025-09-24)
- **Change**: `ALIGNMENT_THRESHOLD = 0.2 → 0.3` (11.5° → 17.2°)
- **Goal**: Allow robot to transition to APPROACH phase at its current 15° performance
- **Result**: ✅ Robot now transitions to APPROACH phase
- **New Issue**: Robot doesn't move forward aggressively in APPROACH phase

## Current Issues to Address

### Issue: Weak Forward Movement in APPROACH Phase
**Symptoms**: 
- Robot reaches APPROACH phase (✅)
- Proximity rewards only 2.12-3.10 (weak)
- Final distances show minimal improvement
- No successful object grabs

**Root Cause**: No explicit reward for forward movement actions

## Next Version Plans

### v2.2 - Forward Movement Incentive (2025-09-24) - IMPLEMENTED
**Goal**: Add explicit forward action rewards in APPROACH phase
**Changes**:
- Added `forward_bonus = 5.0 * max(0, action[0])` in APPROACH phase
- Updated logging to show forward_bonus component
- Changed log message to "APPROACH PHASE v2.2" for tracking

**Code**:
```python
# In APPROACH phase:
proximity_reward = 20.0 / (1.0 + 10.0 * distance)
alignment_penalty = -5.0 * abs(dtheta)
forward_bonus = 5.0 * max(0, action[0])  # NEW: Reward forward movement
total_reward = proximity_reward + alignment_penalty + forward_bonus - 0.1
```

**Expected Result**: Robot should move forward more aggressively when aligned

**ACTUAL RESULT (Episode 1631)**: ✅ SUCCESS! 
- 2.25x better training rewards (9822 vs 4300)
- 36% closer final distances (0.450m vs 0.710m) 
- Best test achieved 0.296m distance
- Forward bonus actively working (4.87-4.96 values)
- Robot consistently reaches APPROACH phase
- **Status**: Continue V2.2 training - showing excellent progress!

### v3.0 - Graduated Reward System (Future)
**Goal**: Replace binary phases with smooth transitions
**Features**:
- Multiple threshold levels
- Mixed-phase rewards
- Distance-dependent adjustments
- Action-specific bonuses
**Reference**: See `improved_reward_function.py`

## Training Performance Log

| Version | Phase Transition | Forward Movement | Success Rate | Notes |
|---------|------------------|------------------|--------------|-------|
| v2.0    | ❌ Stuck in ALIGN | ❌ No forward    | 0%          | Threshold too strict |
| v2.1    | ✅ Reaches APPROACH | ❌ FAILED - moves away | 0%  | REWARD INVERSION: 3000+ episodes, robot learned to move away |
| v2.2    | ✅ Reaches APPROACH | ✅ **SUCCESS** | Real: 6.5in | **COMPLETE**: 14.6x training reward, real robot 6.5in approach |

## Testing Protocol

For each version:
1. Run evaluation: `python unified_evaluator.py model.pth`
2. Check phase transitions in logs
3. Monitor reward components
4. Track final distances and success rates
5. Document behavioral changes

## Backup Strategy

- Always commit working versions
- Keep previous reward functions as comments
- Test threshold changes incrementally
- Document exact parameters that work
