from collections import deque, namedtuple
import os
import mujoco
import mujoco.viewer
import numpy as np

# this is the manually implemented mujoco, it seems to work on pendulum

def quaternion_to_euler(q):
    # q should be in [x, y, z, w] format
    w, x, y, z = q

    # Roll (X-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (Y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))  # Clamp to avoid NaNs

    # Yaw (Z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw  # in radians

def bug_fix_angles(qpos, idx=None):
  """Fix angles to be in the range [-pi, pi]."""
  if idx is None:
    idx = list(range(len(qpos)))
  for i in idx:
    qpos[i] = np.mod(qpos[i] + np.pi, 2 * np.pi) - np.pi
  return qpos

class ClawbotCan:
  def __init__(self, mujoco_model_path: str="env/clawbot.xml", 
               can_x_range: tuple=(-0.75, 0.75),
               can_y_range: tuple=(0.4, 0.75),
               min_distance: float=0.4):
    """Initialize ClawbotCan environment
    
    Args:
        mujoco_model_path: Path to the MuJoCo XML model
        can_x_range: (min_x, max_x) range for can X position
        can_y_range: (min_y, max_y) range for can Y position  
        min_distance: Minimum distance between robot and can
    """
    with open(mujoco_model_path, 'r') as fp:
      model_xml = fp.read()
    self.model = mujoco.MjModel.from_xml_string(model_xml)
    self.data = mujoco.MjData(self.model)
    self.time_duration = 0.05
    
    # Store positioning parameters
    self.can_x_range = can_x_range
    self.can_y_range = can_y_range
    self.min_distance = min_distance

    # Fix: Identify hinge joint indices for safe angle wrapping
    self.hinge_joint_qpos_indices = []
    for i in range(self.model.njnt):
        if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
            qpos_start = self.model.jnt_qposadr[i]
            qpos_size = 1  # Hinge joints have 1 DOF
            self.hinge_joint_qpos_indices.extend(range(qpos_start, qpos_start + qpos_size))

    self.sensor_names = [self.model.sensor_adr[i] for i in range(self.model.nsensor)]
    self.actuator_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]
    self.body_names = self.model.names.decode('utf-8').split('\x00')[1:]

    self._steps = 0
    self.max_steps = 1500  # Increased from 1000 to allow more time for complex maneuvers
    self.observation_space = namedtuple('Box', ['high', 'low', 'shape'])
    # self.observation_space.shape = (self.model.nsensor,)
    self.observation_space.shape = (3,)
    self.observation_space.low = np.full(self.observation_space.shape, float('-inf')).tolist()
    self.observation_space.high = np.full(self.observation_space.shape, float('inf')).tolist()
    self.action_space = namedtuple('Box', ['high', 'low', 'shape'])
    self.action_space.shape = (self.model.nu,)
    self.action_space.low  = self.model.actuator_ctrlrange[:,0].tolist()
    self.action_space.high = self.model.actuator_ctrlrange[:,1].tolist()

    self.viewer = None
    self.prev_action = np.array([0.0, 0.0, 0.0, 0.0]) # ramping
    # Track last distance and action to shape progress and stability rewards
    self.last_distance = None
    self.no_progress_steps = 0
    self.last_action = np.array([0.0, 0.0, 0.0, 0.0])

  def _calc_state(self):
      # Calculate reward and termination
      # Get sensor indices by name
      touch_lc_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_lc")
      touch_rc_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_rc")

      # Read values from the sensordata array
      touch_lc_value = self.data.sensordata[touch_lc_id]
      touch_rc_value = self.data.sensordata[touch_rc_id]

      # print(f"Left claw touch force: {touch_lc_value}")
      # print(f"Right claw touch force: {touch_rc_value}")

      objectGrabbed = touch_lc_value or touch_rc_value

      # Can position
      can1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "can1")
      can1_pos = self.data.xpos[can1_id]  # [x, y, z] in world frame

      # Claw position
      claw_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "virtual_claw_target")
      claw_pos = self.data.xpos[claw_id]  # [x, y, z] in world frame

      dx = can1_pos[0] - claw_pos[0]  # Direction FROM claw TO can
      dy = can1_pos[1] - claw_pos[1]  # Direction FROM claw TO can
      dz = can1_pos[2] - claw_pos[2]
      distance = np.sqrt(dx * dx + dy * dy + dz * dz)
      heading = np.arctan2(dy, dx)  # Angle to target
      # print("Distance:", dist, "Heading:", heading)

      roll, pitch, yaw = quaternion_to_euler(self.data.xquat[claw_id])
      # print("Yaw:", yaw)
      # yaw 0 is East (positive X), so adjust to make 0 = North (positive Y)
      robot_heading = yaw + np.pi/2  # Convert from East=0 to North=0
      
      dtheta = bug_fix_angles([robot_heading - heading])[0]
      # print("Dtheta:", dtheta)

      # Additional state information for reward calculation
      # Check if can is behind the claw (negative Y relative to robot)
      can_behind_claw = dy < -0.1  # Can is behind the claw position
      
      # Return enhanced state with positional information
      return np.array([distance, dtheta, objectGrabbed]), {
          'dx': dx, 'dy': dy, 'dz': dz, 
          'claw_pos': claw_pos, 'can_pos': can1_pos,
          'can_behind_claw': can_behind_claw,
          'heading': heading, 'robot_heading': robot_heading
      }

  def reward(self, state, action, info=None):
    distance, dtheta, objectGrabbed = state
    
    # Get additional state information for enhanced reward calculation
    if info is None:
        # If no info provided, calculate it (for backward compatibility)
        _, info = self._calc_state()
    
    can_behind_claw = info.get('can_behind_claw', False)
    dy = info.get('dy', 0)
    
    # Check if claw is touching the can (but not grabbing)
    # Note: objectGrabbed is already calculated from the same touch sensors in _calc_state
    # We need to detect "touching but not grabbing" state
    is_touching_can = objectGrabbed  # For now, treat any touch as potential sweeping
    # TODO: Could be refined to detect partial vs full contact
    
    # Enhanced "Rotation Always First" Reward Function
    # Key insight: Rotation priority increases when closer (less room for error)
    
    # Calculate rotation priority multiplier based on distance
    # Closer = higher rotation priority (more critical alignment)
    rotation_priority = 1.0 + (1.0 - min(distance, 1.0))  # Range: 1.0 to 2.0
    
    # Always prioritize rotation, but intensity varies by distance and angle
    rotation_reward = -rotation_priority * 8 * np.abs(dtheta)
    
    # Distance penalty with anti-sweeping and wrong-position detection
    if objectGrabbed:
        # When grabbed: No distance penalty (success!), encourage episode completion
        distance_penalty = 0  
        wrong_position_penalty = 0
    else:
        # Normal distance penalty when not grabbed
        distance_penalty = -2 * distance
        
        # Anti-sweeping: Penalty for being very close but not grabbing
        # This prevents robot from just pushing the can around
        if distance < 0.12 and not objectGrabbed:  # Very close but no grab
            wrong_position_penalty = -8.0  # Penalty for being too close without success
        else:
            wrong_position_penalty = 0
    
    # Severely penalize forward movement when ANY misalignment exists
    forward_movement = abs(action[0] + action[1]) / 2  # Average wheel speed
    angle_tolerance = 0.2  # ~11.5 degrees - very tight tolerance
    
    if abs(dtheta) > angle_tolerance:
        # Any misalignment > 11.5° prohibits forward movement
        movement_penalty = -25 * forward_movement if forward_movement > 0.1 else 0
    else:
        # Only when very well aligned, allow forward movement
        movement_penalty = 0
    
    approach_reward = rotation_reward + distance_penalty + movement_penalty + wrong_position_penalty
    
    # Progressive bonuses for good alignment
    if abs(dtheta) < 0.1:  # < 5.7° - excellent alignment
        alignment_bonus = 10.0
    elif abs(dtheta) < 0.3:  # < 17.2° - good alignment  
        alignment_bonus = 5.0
    elif abs(dtheta) < 0.5:  # < 28.6° - acceptable alignment
        alignment_bonus = 2.0
    else:
        alignment_bonus = 0.0
    
    # Close approach bonus (only when well aligned)
    if distance < 0.15 and abs(dtheta) < 0.3:
        close_approach_bonus = 8.0
    elif distance < 0.25 and abs(dtheta) < 0.2:
        close_approach_bonus = 4.0
    else:
        close_approach_bonus = 0.0
    
    # Object grabbed bonus (unchanged)
    grab_bonus = int(objectGrabbed) * 50
    
    # Progress and stability shaping
    # Encourage steady approach, penalize stagnation and moving away
    delta_distance = 0.0 if self.last_distance is None else (self.last_distance - distance)
    # Reward progress (moving closer) modestly
    progress_reward = 2.0 * max(delta_distance, 0.0)
    # Penalize moving away proportionally
    away_penalty = -4.0 * max(-delta_distance, 0.0)
    # Penalize sustained lack of progress to avoid dithering/jitter
    min_progress_threshold = 0.002  # 2 mm per step considered meaningful
    if delta_distance < min_progress_threshold and not objectGrabbed:
        self.no_progress_steps = min(self.no_progress_steps + 1, 50)
    else:
        self.no_progress_steps = 0
    stagnation_penalty = -0.2 * self.no_progress_steps
    
    # Penalize being close but not in a grabbable alignment window
    grabbable_penalty = 0.0
    if distance < 0.20 and abs(dtheta) > 0.30 and not objectGrabbed:
        grabbable_penalty = -6.0
    
    # Mild dynamic time penalty that increases as steps elapse
    time_penalty = -0.05 - 0.05 * (self._steps / self.max_steps)
    
    # Anti-jitter: penalize abrupt action changes (after built-in ramping)
    try:
        action_change = float(np.sum(np.abs(action - self.last_action[:len(action)])))
    except Exception:
        action_change = 0.0
    jitter_penalty = -0.5 * max(action_change - 0.3, 0.0)
    
    # Fix 1: Penalty for can being behind the claw (wrong position for grabbing)
    behind_claw_penalty = 0
    if can_behind_claw and distance < 0.3:  # Only penalize if close AND behind
        behind_claw_penalty = -15.0  # Heavy penalty for being in wrong position
    
    # Fix 2: Penalty for robot being stuck (not moving)
    stuck_penalty = 0
    movement_magnitude = abs(action[0]) + abs(action[1])  # Total wheel movement
    if movement_magnitude < 0.05:  # Very low movement
        # Check if robot should be moving (not well aligned and close)
        should_be_moving = (abs(dtheta) > 0.1 or distance > 0.2)
        if should_be_moving:
            stuck_penalty = -5.0  # Penalty for being stuck when should be active
    
    total_reward = (approach_reward + alignment_bonus + close_approach_bonus + 
                   grab_bonus + time_penalty + behind_claw_penalty + stuck_penalty +
                   progress_reward + away_penalty + stagnation_penalty + grabbable_penalty + jitter_penalty)
    
    # Update last_distance for next step
    self.last_distance = distance
    
    return total_reward

  def terminal(self, state, action):
    distance, dtheta, objectGrabbed = state
    # Fix: Much more lenient angle termination - only fail if facing completely wrong (>150°)
    is_facing_completely_wrong = abs(dtheta) > (5 * np.pi / 6)  # 150 degrees
    # Also terminate if robot gets stuck very far away
    is_too_far = distance > 2.0
    return self._steps >= 1500 or objectGrabbed or is_facing_completely_wrong or is_too_far

  def reset(self):
    self.prev_action = np.array([0.0, 0.0, 0.0, 0.0]) 
    """Reset the environment to its initial state."""
    self._steps = 0
    mujoco.mj_resetData(self.model, self.data)

    # set a new can position
    can1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "can1")
    can1_jntadr = self.model.body_jntadr[can1_id]
    can1_qposadr = self.model.jnt_qposadr[can1_jntadr]

    pos = (0, 0)
    while np.sqrt(pos[0] * pos[0] + pos[1] * pos[1]) < self.min_distance:
      pos = (np.random.uniform(*self.can_x_range), np.random.uniform(*self.can_y_range))
    self.data.qpos[can1_qposadr+0] = pos[0]
    self.data.qpos[can1_qposadr+1] = pos[1]

    # Fix: Only apply angle wrapping to hinge joints, not free joints
    bug_fix_angles(self.data.qpos, idx=self.hinge_joint_qpos_indices)
    mujoco.mj_forward(self.model, self.data)
    bug_fix_angles(self.data.qpos, idx=self.hinge_joint_qpos_indices)
    sensor_values = self.data.sensordata.copy()
    s, info = self._calc_state()
    # Initialize progress tracking
    self.last_distance = float(s[0])
    self.no_progress_steps = 0
    self.last_action = np.array([0.0, 0.0, 0.0, 0.0])
    return s, info

  def step(self, action, time_duration=0.05):
    # for now, disable arm, and restrict the claw to only move in the open direction
    action[2] = 0 # TODO: to disable the arm
    action[3] = action[3] / 2 - 0.5 # TODO: To restrict the claw to only move in the open direction

    # Option 2: Force rotation-only behavior when misaligned
    # Get current state to check alignment
    current_state, _ = self._calc_state()
    distance, dtheta, objectGrabbed = current_state
    
    if abs(dtheta) > 0.2:  # ANY misalignment > 11.5° - force rotation-only
        forward_component = (action[0] + action[1]) / 2
        if forward_component > 0.1:  # Agent trying to move forward when misaligned
            # Convert to pure rotation based on the intended direction
            rotation_strength = forward_component  # Use the forward intent as rotation strength
            turn_direction = 1 if dtheta > 0 else -1  # Turn toward target
            
            # Override actions to pure rotation
            action[0] = rotation_strength * turn_direction   # Left wheel
            action[1] = -rotation_strength * turn_direction  # Right wheel (opposite)
            
            # Debug info (can be removed later)
            # print(f"🔄 Forced rotation: dtheta={dtheta:.2f}, turn_dir={turn_direction}")

    # Preserve last executed (filtered) action for anti-jitter shaping
    try:
        self.last_action = self.prev_action.copy()
    except Exception:
        self.last_action = np.array([0.0, 0.0, 0.0, 0.0])
    
    self.prev_action = action = \
      np.clip(np.array(action) - self.prev_action, -0.25, 0.25) + self.prev_action
    for i, a in enumerate(action):
      self.data.ctrl[i] = a
    t = time_duration
    while t - self.model.opt.timestep > 0:
      t -= self.model.opt.timestep
      # Fix: Only apply angle wrapping to hinge joints, not free joints
      bug_fix_angles(self.data.qpos, idx=self.hinge_joint_qpos_indices)
      mujoco.mj_step(self.model, self.data)
      bug_fix_angles(self.data.qpos, idx=self.hinge_joint_qpos_indices)
    sensor_values = self.data.sensordata.copy()
    s, info = self._calc_state()
    obs = s
    self._steps += 1
    reward_value = self.reward(s, action, info)
    terminal_value = self.terminal(s, action)

    return obs, reward_value, terminal_value, info

  def render(self):
    """Render the environment."""
    if self.viewer is None:
      self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
    if self.viewer.is_running():
      self.viewer.sync()
    else:
      self.viewer.close()
      self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
  
  def close(self):
    """Close the environment and clean up resources."""
    if hasattr(self, 'viewer') and self.viewer is not None:
      try:
        self.viewer.close()
        self.viewer = None
      except:
        pass