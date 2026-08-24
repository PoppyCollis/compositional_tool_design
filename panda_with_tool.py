"""PandaWithTool: a Panda arm with a tau-conditioned tool welded into its
(fixed-shut) grip, implemented directly against panda-gym's PyBulletRobot.
Not a Panda subclass: Panda's __init__ hardcodes the URDF path, gripper
joints, and ee_link, all of which are invalid once the tool is spliced in.
"""
import numpy as np
from gymnasium import spaces
from panda_gym.envs.core import PyBulletRobot

import panda_with_tool_urdf
from utils.helpers import get_link_index_by_name

JOINT_INDICES = np.array([0, 1, 2, 3, 4, 5, 6])
JOINT_FORCES = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
NEUTRAL_JOINT_VALUES = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79])
MAX_JOINT_DELTA = 0.05  # rad, per control step


class PandaWithTool(PyBulletRobot):
    """Panda robot with a tool of given tau welded rigidly into its grip.

    Args:
        sim (PyBullet): Simulation instance.
        tau: (l1, l2, theta) tool design parameters.
        base_position (np.ndarray, optional): Base position, as (x, y, z).
    """

    def __init__(self, sim, tau, base_position=None):
        base_position = base_position if base_position is not None else np.zeros(3)
        self.tau = tau
        urdf_path = panda_with_tool_urdf.write_panda_with_tool_urdf(tau)
        action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        super().__init__(
            sim,
            body_name="panda_with_tool",
            file_name=urdf_path,
            base_position=base_position,
            action_space=action_space,
            joint_indices=JOINT_INDICES,
            joint_forces=JOINT_FORCES,
        )

    def setup(self):
        """Called once by PyBulletRobot.__init__, right after the URDF loads."""
        body_id = self.sim._bodies_idx[self.body_name]
        self.tool_tip_link = get_link_index_by_name(body_id, "tool_tip")
        panda_with_tool_urdf.disable_finger_tool_collision(body_id)

    def set_action(self, action):
        action = np.clip(np.array(action).copy(), self.action_space.low, self.action_space.high)
        current_angles = np.array([self.get_joint_angle(joint=i) for i in range(7)])
        target_angles = current_angles + action * MAX_JOINT_DELTA
        self.control_joints(target_angles=target_angles)

    def get_obs(self):
        return np.concatenate([self.get_ee_position(), self.get_ee_velocity()])

    def reset(self):
        self.set_joint_angles(NEUTRAL_JOINT_VALUES)

    def get_ee_position(self):
        """Returns the tool-tip position as (x, y, z)."""
        return self.get_link_position(self.tool_tip_link)

    def get_ee_velocity(self):
        """Returns the tool-tip velocity as (vx, vy, vz)."""
        return self.get_link_velocity(self.tool_tip_link)
