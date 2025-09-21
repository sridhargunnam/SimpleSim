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

    self.sensor_names = [self.model.sensor_adr[i] for i in range(self.model.nsensor)]
    self.actuator_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]
    self.body_names = self.model.names.decode('utf-8').split('\x00')[1:]

    self._steps = 0
    self.max_steps = 1000
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

      return np.array([distance, dtheta, objectGrabbed]), np.concatenate([np.array([dtheta, dx, dy]), claw_pos], -1)

  def reward(self, state, action):
    distance, dtheta, objectGrabbed = state
    return -distance - np.abs(dtheta) + int(objectGrabbed) * 50

  def terminal(self, state, action):
    _, __, objectGrabbed = state
    return self._steps >= 1000 or objectGrabbed or np.cos(state[1]) < 0

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

    bug_fix_angles(self.data.qpos)
    mujoco.mj_forward(self.model, self.data)
    bug_fix_angles(self.data.qpos)
    sensor_values = self.data.sensordata.copy()
    return self._calc_state()

  def step(self, action, time_duration=0.05):
    # for now, disable arm, and restrict the claw to only move in the open direction
    action[2] = 0 # TODO: to disable the arm
    action[3] = action[3] / 2 - 0.5 # TODO: To restrict the claw to only move in the open direction

    self.prev_action = action = \
      np.clip(np.array(action) - self.prev_action, -0.25, 0.25) + self.prev_action
    for i, a in enumerate(action):
      self.data.ctrl[i] = a
    t = time_duration
    while t - self.model.opt.timestep > 0:
      t -= self.model.opt.timestep
      bug_fix_angles(self.data.qpos)
      mujoco.mj_step(self.model, self.data)
      bug_fix_angles(self.data.qpos)
    sensor_values = self.data.sensordata.copy()
    s, info = self._calc_state()
    obs = s
    self._steps += 1
    reward_value = self.reward(s, action)
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