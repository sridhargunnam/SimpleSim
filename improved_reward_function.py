"""
IMPROVED REWARD FUNCTION FOR BETTER ALIGN -> APPROACH TRANSITION

This addresses the specific issue where robots learn to align but not approach.
Key improvements:
1. Graduated thresholds instead of binary switching
2. Mixed-phase rewards encourage both behaviors
3. Explicit forward movement incentives
4. Distance-based adjustments
"""

import numpy as np

def improved_reward(self, state, action, info=None):
    """
    GRADUATED REWARD FOR SMOOTH PHASE TRANSITIONS
    
    Instead of binary ALIGN/APPROACH phases, this creates smooth transitions
    that encourage both alignment refinement AND forward movement.
    """
    distance, dtheta, objectGrabbed = state
    
    # === GRADUATED THRESHOLDS ===
    ROUGH_ALIGNMENT = 0.5    # ~29 degrees - start caring about alignment
    GOOD_ALIGNMENT = 0.3     # ~17 degrees - start rewarding approach
    PRECISE_ALIGNMENT = 0.1  # ~6 degrees - maximum approach rewards
    CLOSE_DISTANCE = 0.15    # 15cm - start preparing to grab
    
    # =================================================
    # PHASE 3: GRASP (Success!)
    # =================================================
    if objectGrabbed:
        return 1000.0
    
    # =================================================
    # PHASE 2.5: CLOSE APPROACH (Prepare to grab)
    # =================================================
    if distance <= CLOSE_DISTANCE:
        # Close to target - focus on precise alignment and gentle approach
        alignment_reward = 15.0 * np.exp(-5.0 * abs(dtheta))
        proximity_reward = 30.0 / (1.0 + 20.0 * distance)
        grab_preparation = 5.0 * max(0, action[3])  # Reward claw opening
        return alignment_reward + proximity_reward + grab_preparation - 0.2
    
    # =================================================
    # PHASE 2: APPROACH (Well aligned, move forward)
    # =================================================
    elif abs(dtheta) <= PRECISE_ALIGNMENT:
        # Well aligned - strongly reward forward movement
        proximity_reward = 25.0 / (1.0 + 10.0 * distance)
        forward_bonus = 8.0 * max(0, action[0])  # Strong forward movement reward
        alignment_maintenance = -3.0 * abs(dtheta)  # Small penalty for misalignment
        return proximity_reward + forward_bonus + alignment_maintenance - 0.1
    
    # =================================================
    # PHASE 1.5: MIXED (Somewhat aligned, encourage both)
    # =================================================
    elif abs(dtheta) <= GOOD_ALIGNMENT:
        # Good alignment - reward both alignment refinement AND approach
        alignment_reward = 12.0 * np.exp(-3.0 * abs(dtheta))
        proximity_reward = 8.0 / (1.0 + 8.0 * distance)
        forward_bonus = 3.0 * max(0, action[0])  # Gentle forward encouragement
        
        # Balance between alignment and approach based on current alignment quality
        alignment_weight = abs(dtheta) / GOOD_ALIGNMENT  # 0-1, higher when less aligned
        approach_weight = 1.0 - alignment_weight
        
        return (alignment_weight * alignment_reward + 
                approach_weight * (proximity_reward + forward_bonus)) - 0.3
    
    # =================================================
    # PHASE 1: ALIGN (Poorly aligned, focus on turning)
    # =================================================
    elif abs(dtheta) <= ROUGH_ALIGNMENT:
        # Rough alignment - primarily focus on alignment but hint at approach
        alignment_reward = 10.0 * np.exp(-2.0 * abs(dtheta))
        distance_awareness = 1.0 / (1.0 + distance)  # Gentle distance awareness
        return alignment_reward + distance_awareness - 0.5
    
    # =================================================
    # PHASE 0: LOST (Very misaligned, pure alignment)
    # =================================================
    else:
        # Very misaligned - pure alignment focus
        return 8.0 * np.exp(-1.5 * abs(dtheta)) - 1.0

def debug_reward_phases():
    """Test function to visualize reward behavior across different states"""
    import matplotlib.pyplot as plt
    
    angles = np.linspace(0, np.pi/2, 50)  # 0 to 90 degrees
    distances = [0.2, 0.5, 0.8]
    
    for dist in distances:
        rewards = []
        phases = []
        for angle in angles:
            # Simulate the reward function
            state = (dist, angle, False)
            action = [0.1, 0, 0, 0]  # Small forward action
            reward = improved_reward(None, state, action)
            rewards.append(reward)
            
            # Determine phase
            if angle <= 0.1:
                phase = "APPROACH"
            elif angle <= 0.3:
                phase = "MIXED"
            elif angle <= 0.5:
                phase = "ALIGN"
            else:
                phase = "LOST"
            phases.append(phase)
        
        plt.plot(angles * 180/np.pi, rewards, label=f'Distance: {dist}m')
    
    plt.xlabel('Angle Error (degrees)')
    plt.ylabel('Reward')
    plt.title('Improved Reward Function - Smooth Phase Transitions')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage in training:
# Replace the reward function in mujoco_env.py with improved_reward
