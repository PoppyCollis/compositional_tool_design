"""PandaWithTool: a Panda arm with a tau-conditioned tool welded into its
(fixed-shut) grip, driven as a planar rigid body.

The action is 3-dimensional -- (dx, dy, dyaw) of the hand -- with height, roll and
pitch pinned, so the tool is always low and parallel to the ground. An IK layer
turns those three numbers into the seven joint targets; the policy never sees the
joint manifold. See se2.py for the geometry this rests on.

Implemented directly against panda-gym's PyBulletRobot, not as a Panda subclass:
Panda's __init__ hardcodes the URDF path, gripper joints, and ee_link, all of which
are invalid once the tool is spliced in.
"""
import numpy as np
from gymnasium import spaces
from panda_gym.envs.core import PyBulletRobot

import panda_with_tool_urdf
import se2
from config import ArmConfig, SE2Config, TaskConfig
from utils.helpers import get_link_index_by_name

JOINT_INDICES = np.array([0, 1, 2, 3, 4, 5, 6])
JOINT_FORCES = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
NEUTRAL_JOINT_VALUES = np.array(ArmConfig.NEUTRAL_JOINT_VALUES)

# Null-space arguments for calculateInverseKinematics, precomputed once. PyBullet
# only honours them if all four are supplied.
JOINT_LOWER = [lo for lo, _ in ArmConfig.JOINT_LIMITS]
JOINT_UPPER = [hi for _, hi in ArmConfig.JOINT_LIMITS]
JOINT_RANGES = [hi - lo for lo, hi in ArmConfig.JOINT_LIMITS]
REST_POSES = list(ArmConfig.NEUTRAL_JOINT_VALUES)


class PandaWithTool(PyBulletRobot):
    """Panda robot with a tool of given tau welded rigidly into its grip.

    Controlled in SE(2): the action is (dx, dy, dyaw) of the panda_hand frame,
    each in [-1, 1] and scaled by SE2Config.POS_SCALE / YAW_SCALE. The hand's
    height and its fingers-down orientation are constants, not action dimensions.

    Args:
        sim (PyBullet): Simulation instance.
        tau: (l1, l2, phi) tool design parameters.
        base_position (np.ndarray, optional): Base position, as (x, y, z).
        np_random (np.random.Generator, optional): Source of randomness for reset
            sampling. Defaults to a fresh unseeded generator; the env layer can
            replace it via set_np_random so episodes seed reproducibly.
    """

    def __init__(self, sim, tau, base_position=None, np_random=None):
        base_position = base_position if base_position is not None else np.zeros(3)
        self.tau = tau
        self.np_random = np_random if np_random is not None else np.random.default_rng()

        # The workspace was swept with the base at the origin, so a robot placed
        # elsewhere carries the same rectangle, translated.
        self.workspace = SE2Config.WORKSPACE.translate(base_position[:2])
        self.reset_workspace = self.workspace.shrink(SE2Config.RESET_MARGIN)
        # Where the hand may go and what the observation is normalised against are
        # two different boxes. The tool exists to put the tip *outside* the first, so
        # normalising on it would send exactly the designs under evaluation into the
        # network's extrapolation region. See TaskConfig.SCENE_BOX.
        self.scene = TaskConfig.SCENE_BOX.translate(base_position[:2])

        urdf_path = panda_with_tool_urdf.write_panda_with_tool_urdf(tau)
        action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
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
        self.hand_link = get_link_index_by_name(body_id, "panda_hand")
        panda_with_tool_urdf.disable_finger_tool_collision(body_id)
        # Seeded here so set_action is safe even if called before the first reset;
        # reset overwrites it. At this point the arm is still in the URDF's default
        # pose, so the first clip will pull the target into the workspace.
        self._target = self.get_hand_se2()

    def set_np_random(self, np_random):
        """Replace the reset sampler's source of randomness."""
        self.np_random = np_random

    # -- control ----------------------------------------------------------------

    def set_action(self, action):
        """Nudge the hand in the plane and re-solve the arm for the new pose.

        The nudge accumulates onto an internal target, which is then clipped into
        the workspace and held within MAX_LAG of where the hand actually is. Both
        clips happen *before* the IK solver sees the target: afterwards is too
        late, because the solver will already have bought the extra reach by
        tilting the wrist, and the tool is no longer flat.

        Args:
            action (np.ndarray): (dx, dy, dyaw), each in [-1, 1].
        """
        action = np.clip(np.array(action, dtype=float), self.action_space.low, self.action_space.high)

        target_xy = self.workspace.clip(
            self._target[:2] + action[:2] * SE2Config.POS_SCALE
        )
        # Yaw is a clipped dimension like x and y, not a free one: the tool points
        # along the hand's +x axis, so yaw near +-pi aims it back at the robot's own
        # base, which the wrist cannot reach (ArmConfig.JOINT_LIMITS). The sweep
        # only certifies the workspace across +-YAW_LIMIT, so the controller must
        # stay inside the range the workspace was measured under.
        target_yaw = float(np.clip(
            se2.wrap_angle(self._target[2] + action[2] * SE2Config.YAW_SCALE),
            -SE2Config.YAW_LIMIT, SE2Config.YAW_LIMIT,
        ))

        self._target = self._clamp_lag(target_xy, target_yaw)
        self.control_joints(target_angles=self._solve_ik(self._target[:2], self._target[2]))

    def _clamp_lag(self, target_xy, target_yaw):
        """Pull a target back so it never leads the measured hand by more than MAX_LAG.

        Clipping to the workspace stops the target escaping the table; this stops it
        escaping the *arm*. When the tool is blocked by an object rather than a
        boundary the hand stalls while the target keeps accumulating, and without
        this the arm would lunge for a target far away the instant the obstruction
        cleared.

        Args:
            target_xy: Proposed target position as (x, y).
            target_yaw (float): Proposed target yaw.

        Returns:
            np.ndarray: The clamped target as (x, y, yaw).
        """
        x, y, yaw = self.get_hand_se2()
        offset = np.asarray(target_xy) - np.array([x, y])
        lag = np.linalg.norm(offset)
        if lag > SE2Config.MAX_LAG:
            target_xy = np.array([x, y]) + offset * (SE2Config.MAX_LAG / lag)

        yaw_offset = se2.wrap_angle(target_yaw - yaw)
        if abs(yaw_offset) > SE2Config.MAX_YAW_LAG:
            target_yaw = se2.wrap_angle(yaw + np.sign(yaw_offset) * SE2Config.MAX_YAW_LAG)

        return np.array([target_xy[0], target_xy[1], target_yaw])

    def _solve_ik(self, target_xy, target_yaw, rounds=1):
        """Joint angles putting the hand at (x, y, HAND_Z) with the given yaw.

        Args:
            target_xy: Target hand position in the plane, as (x, y).
            target_yaw (float): Target hand yaw, in radians.
            rounds (int): IK iterations. calculateInverseKinematics iterates from
                the current joint state, so one call suffices for a per-step nudge
                but not for a large jump; reset and the workspace sweep re-seed and
                repeat SE2Config.IK_ROUNDS times.

        Returns:
            np.ndarray: The 7 arm joint angles, within the URDF's joint limits.

        Calls PyBullet directly rather than through panda-gym's inverse_kinematics
        wrapper, which passes no joint limits. Without them the solver winds
        panda_joint7 straight past its +-2.9671 stop for yaws near +-pi; the motors
        then saturate at the stop when stepping, and the hand quietly settles
        somewhere other than where it was told to go. See ArmConfig.JOINT_LIMITS.
        """
        position = np.array([target_xy[0], target_xy[1], SE2Config.HAND_Z])
        orientation = se2.hand_quat(target_yaw)
        body_id = self.sim._bodies_idx[self.body_name]
        for _ in range(rounds):
            # The fingers are welded fixed in the spliced URDF, so this body has 7
            # movable DoFs and IK returns 7 values -- not the 9 panda-gym's Panda
            # slices down from. The slice is a no-op here, kept as documentation.
            angles = np.array(self.sim.physics_client.calculateInverseKinematics(
                bodyUniqueId=body_id,
                endEffectorLinkIndex=self.hand_link,
                targetPosition=position,
                targetOrientation=orientation,
                lowerLimits=JOINT_LOWER,
                upperLimits=JOINT_UPPER,
                jointRanges=JOINT_RANGES,
                restPoses=REST_POSES,
            ))[:7]
            if rounds > 1:
                self.set_joint_angles(angles)
        return angles

    def reset(self):
        """Teleport the arm to a random flat pose drawn from inside the workspace.

        Sampled from the inset box so an episode never starts pressed against a
        wall with half its action range dead. Seeded from the neutral pose each
        time, because IK is iterative and path-dependent -- without the reseed the
        starting pose would depend on wherever the previous episode ended.
        """
        xy = self.reset_workspace.sample(self.np_random)
        yaw = self.np_random.uniform(-SE2Config.YAW_LIMIT, SE2Config.YAW_LIMIT)
        self.set_se2(xy, yaw)

    def set_se2(self, xy, yaw):
        """Teleport the hand to an explicit flat pose. Bypasses the clipper.

        Args:
            xy: Hand position in the plane, as (x, y).
            yaw (float): Hand yaw, in radians.
        """
        self.set_joint_angles(NEUTRAL_JOINT_VALUES)
        self.set_joint_angles(self._solve_ik(xy, yaw, rounds=SE2Config.IK_ROUNDS))
        # The integrated target starts where the arm was actually placed, not where
        # it was asked to go, so the first action does not inherit the IK residual.
        self._target = self.get_hand_se2()

    # -- state ------------------------------------------------------------------

    def _link_frame(self, link):
        """Returns a link's URDF frame pose as (position, quaternion).

        Deliberately not panda-gym's get_link_position / get_link_orientation: those
        read getLinkState indices 0 and 1, which are the link's *centre of mass*,
        whereas calculateInverseKinematics targets the URDF *link frame* (indices 4
        and 5). For panda_hand the two are 4 cm apart, so mixing them would feed the
        controller a pose 4 cm below the one it commands and the height would never
        settle. They coincide for tool_tip, whose inertial origin is at 0, which is
        why get_ee_position can keep using the inherited accessor.
        """
        state = self.sim.physics_client.getLinkState(self.sim._bodies_idx[self.body_name], link)
        return np.array(state[4]), np.array(state[5])

    def get_hand_rotation(self):
        """Returns the hand's 3x3 world rotation matrix."""
        _, quat = self._link_frame(self.hand_link)
        return np.array(self.sim.physics_client.getMatrixFromQuaternion(quat)).reshape(3, 3)

    def get_hand_position(self):
        """Returns the hand's link-frame world position as (x, y, z)."""
        return self._link_frame(self.hand_link)[0]

    def get_hand_se2(self):
        """Returns the measured hand pose as (x, y, yaw)."""
        position = self.get_hand_position()
        yaw = se2.yaw_from_matrix(self.get_hand_rotation())
        return np.array([position[0], position[1], yaw])

    def get_hand_tilt(self):
        """Angle between the hand's +z axis and world -z, in radians.

        Zero when the tool is exactly parallel to the ground. Not part of the
        observation -- it is a constant by construction -- but the drift tests and
        the workspace sweep assert on it, since this is the quantity that grows
        silently when IK trades orientation for reach.
        """
        return se2.tilt_from_matrix(self.get_hand_rotation())

    def get_obs(self):
        """Observation of the arm: slices 0:9 of the full state vector.

        Returns:
            np.ndarray: 9 values --
                hand position, scene-normalised (2),
                hand yaw as (cos, sin) so it is continuous across +-pi (2),
                hand planar velocity and yaw rate, scaled (3),
                tool-tip position, scene-normalised (2).

        The object and task blocks (slices 9:21) are appended by the env layer; the
        full layout is documented in initial_state.py, which reproduces this vector
        analytically. The two stay in step by calling the same Box.normalise_point on
        the same box, not by an assertion -- keep it that way.

        The tip is computed in closed form from the hand pose and tau rather than
        queried by FK, which makes the design's only effect on this MDP explicit.
        """
        x, y, yaw = self.get_hand_se2()
        # Centre-of-mass velocity, unlike the pose above. It differs from the link
        # frame's by omega x r, which is negligible here (the offset is along the
        # hand's own z, parallel to the only axis it spins about) and in any case
        # this term is a conditioning signal for the policy, not a control input.
        velocity = self.get_link_velocity(self.hand_link)
        yaw_rate = self.sim.get_link_angular_velocity(self.body_name, self.hand_link)[2]
        tip = se2.tip_from_hand((x, y, yaw), self.tau)
        return np.concatenate([
            self.scene.normalise_point((x, y)),
            [np.cos(yaw), np.sin(yaw)],
            np.asarray(velocity)[:2] / SE2Config.VEL_SCALE,
            [yaw_rate / SE2Config.VEL_SCALE],
            self.scene.normalise_point(tip),
        ])

    def get_ee_position(self):
        """Returns the tool-tip position as (x, y, z)."""
        return self.get_link_position(self.tool_tip_link)

    def get_ee_velocity(self):
        """Returns the tool-tip velocity as (vx, vy, vz)."""
        return self.get_link_velocity(self.tool_tip_link)
