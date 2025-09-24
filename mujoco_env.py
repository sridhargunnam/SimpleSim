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
               min_distance: float=0.4,
               curriculum_level: int=1):
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
    self.curriculum_level = curriculum_level
    self.episode_count = 0

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
      # Calculate reward and termination with real-world noise simulation
      # Get sensor indices by name
      touch_lc_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_lc")
      touch_rc_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_rc")

      # Read values from the sensordata array
      touch_lc_value = self.data.sensordata[touch_lc_id]
      touch_rc_value = self.data.sensordata[touch_rc_id]

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
      
      # Add realistic sensor noise (5% of reading + small constant)
      distance = np.sqrt(dx * dx + dy * dy + dz * dz)
      if hasattr(self, 'episode_count') and self.episode_count > 1000:  # Add noise after basic learning
          distance_noise = np.random.normal(0, distance * 0.02 + 0.005)  # 2% + 5mm noise
          distance = max(0.1, distance + distance_noise)  # Ensure positive
          
          # Add angle noise (±2 degrees typical for AprilTags)
          angle_noise = np.random.normal(0, np.radians(2))
          dx += np.random.normal(0, 0.01)  # 1cm position noise
          dy += np.random.normal(0, 0.01)
      
      heading = np.arctan2(dy, dx)  # Angle to target

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
    """
    STRUCTURED REWARD FOR SEQUENTIAL LEARNING
    
    This function explicitly guides the agent through three phases:
    1. ALIGN: If the angle to the target is large, only reward turning.
    2. APPROACH: If the angle is small, reward moving closer.
    3. GRASP: If close enough, give a massive bonus for grabbing.
    """
    distance, dtheta, objectGrabbed = state
    
    # --- Define thresholds for switching between phases ---
    ALIGNMENT_THRESHOLD = 0.3  # Radians (~17 degrees) - more lenient for transition
    APPROACH_THRESHOLD = 0.05  # Meters (5 cm)

    # =================================================
    # PHASE 3: GRASP (Highest Priority)
    # =================================================
    if objectGrabbed:
      return 1000.0  # Huge, definitive reward for success

    # =================================================
    # PHASE 1: ALIGN (When angle is large)
    # =================================================
    if abs(dtheta) > ALIGNMENT_THRESHOLD:
      # Reward is focused *only* on alignment. We use an exponential reward
      # to give a strong signal as the agent gets closer to the correct heading.
      # Distance is ignored here to prevent the agent from moving forward prematurely.
      alignment_reward = 10.0 * np.exp(-2.0 * abs(dtheta))
      
      # Optional: Log reward components for debugging (every 100 steps)
      if hasattr(self, '_steps') and self._steps % 100 == 0:
          print(f"Step {self._steps}: ALIGN PHASE - d={distance:.2f}, θ={np.degrees(dtheta):.1f}° | "
                f"alignment_reward={alignment_reward:.2f}")
      
      return alignment_reward - 0.5 # Small time penalty to encourage action

    # =================================================
    # PHASE 2: APPROACH (When aligned) - v2.2 with forward movement incentive
    # =================================================
    else: # abs(dtheta) <= ALIGNMENT_THRESHOLD
      # Once aligned, reward reducing distance AND forward movement actions
      proximity_reward = 20.0 / (1.0 + 10.0 * distance)
      
      # We still include a small penalty for losing alignment.
      # This ensures the robot stays pointed at the target as it moves.
      alignment_penalty = -5.0 * abs(dtheta)
      
      # NEW v2.2: Explicit forward movement incentive
      forward_bonus = 5.0 * max(0, action[0])  # Reward forward movement actions
      
      total_reward = proximity_reward + alignment_penalty + forward_bonus - 0.1 # Small time penalty
      
      # Optional: Log reward components for debugging (every 100 steps)
      if hasattr(self, '_steps') and self._steps % 100 == 0:
          print(f"Step {self._steps}: APPROACH PHASE v2.2 - d={distance:.2f}, θ={np.degrees(dtheta):.1f}° | "
                f"proximity={proximity_reward:.2f}, align_penalty={alignment_penalty:.2f}, "
                f"forward_bonus={forward_bonus:.2f}, total={total_reward:.2f}")
      
      return total_reward
  
  def _log_reward_components(self, distance, dtheta, proximity_reward, alignment_reward, 
                           progress_bonus, exploration_bonus, total_reward):
    """Log reward components for training analysis"""
    print(f"Step {self._steps}: d={distance:.2f}, θ={np.degrees(dtheta):.1f}° | "
          f"proximity={proximity_reward:.1f}, align={alignment_reward:.1f}, "
          f"progress={progress_bonus:.1f}, explore={exploration_bonus:.1f}, total={total_reward:.1f}")

  def _get_curriculum_position(self):
    """
    Progressive curriculum learning for object placement
    
    Level 1 (Episodes 0-500): Easy front positions (small angles)
    Level 2 (Episodes 500-1500): Medium angles 
    Level 3 (Episodes 1500-3000): Hard side positions
    Level 4 (Episodes 3000+): Full random (like real world)
    """
    current_level = min(4, (self.episode_count // 500) + 1)
    
    if current_level == 1:
        # Level 1: Easy front positions (-20° to +20°)
        angle_range = np.pi / 9  # ±20 degrees
        angle = np.random.uniform(-angle_range, angle_range)
        distance = np.random.uniform(0.4, 0.6)  # Close to medium distance
        pos = (distance * np.sin(angle), distance * np.cos(angle))
        
    elif current_level == 2:
        # Level 2: Medium angles (-45° to +45°)
        angle_range = np.pi / 4  # ±45 degrees  
        angle = np.random.uniform(-angle_range, angle_range)
        distance = np.random.uniform(0.4, 0.7)
        pos = (distance * np.sin(angle), distance * np.cos(angle))
        
    elif current_level == 3:
        # Level 3: Hard side positions (-80° to +80°)
        angle_range = 4 * np.pi / 9  # ±80 degrees
        angle = np.random.uniform(-angle_range, angle_range) 
        distance = np.random.uniform(0.4, 0.75)
        pos = (distance * np.sin(angle), distance * np.cos(angle))
        
    else:
        # Level 4: Full random (real-world preparation)
        pos = (0, 0)
        while np.sqrt(pos[0] * pos[0] + pos[1] * pos[1]) < self.min_distance:
            pos = (np.random.uniform(*self.can_x_range), np.random.uniform(*self.can_y_range))
    
    # Ensure position is within bounds and minimum distance
    pos = (np.clip(pos[0], self.can_x_range[0], self.can_x_range[1]),
           np.clip(pos[1], self.can_y_range[0], self.can_y_range[1]))
    
    if np.sqrt(pos[0]**2 + pos[1]**2) < self.min_distance:
        # Fallback to minimum distance if too close
        angle = np.random.uniform(-np.pi, np.pi)
        pos = (self.min_distance * np.sin(angle), self.min_distance * np.cos(angle))
        pos = (np.clip(pos[0], self.can_x_range[0], self.can_x_range[1]),
               np.clip(pos[1], self.can_y_range[0], self.can_y_range[1]))
    
    return pos


  def get_curriculum_info(self):
    """Get current curriculum level info for logging"""
    current_level = min(4, (self.episode_count // 500) + 1)
    level_names = {1: "Easy Front", 2: "Medium Angles", 3: "Hard Sides", 4: "Full Random"}
    return f"Level {current_level}: {level_names[current_level]} (Episode {self.episode_count})"

  def terminal(self, state, action):
    distance, dtheta, objectGrabbed = state
    # Terminate only on success, timeout, or if robot gets completely lost
    is_too_far = distance > 2.0 # Terminate if it gets lost
    # The new reward function will penalize bad alignment, so we don't need to terminate.
    return self._steps >= 1500 or objectGrabbed or is_too_far

  def reset(self):
    self.prev_action = np.array([0.0, 0.0, 0.0, 0.0]) 
    """Reset the environment to its initial state."""
    self._steps = 0
    mujoco.mj_resetData(self.model, self.data)

    # set a new can position
    can1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "can1")
    can1_jntadr = self.model.body_jntadr[can1_id]
    can1_qposadr = self.model.jnt_qposadr[can1_jntadr]

    # Curriculum-based object placement for progressive learning
    pos = self._get_curriculum_position()
    self.data.qpos[can1_qposadr+0] = pos[0]
    self.data.qpos[can1_qposadr+1] = pos[1]
    self.episode_count += 1

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
    # REMOVED: Hardcoded action restrictions - agent now has full control
    # The new structured reward function will teach proper arm and claw usage

    # REMOVED: Forced rotation logic - let the model learn to turn naturally
    # The new reward function will incentivize proper turning behavior
    # without hardcoded action overrides.

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