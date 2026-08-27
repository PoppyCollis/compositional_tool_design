"""Pure SE(2) math for the flat, fixed-height tool controller. No PyBullet, no I/O.

The arm is driven as a planar rigid body: the ``panda_hand`` frame is held at a
constant world height with its +z axis pointing straight down (fingers down), and
only ``(x, y, yaw)`` are commanded. Concretely the hand's world rotation is always

    R(psi) = Rz(psi) @ Rx(pi) = [[ cos psi,  sin psi,  0],
                                 [ sin psi, -cos psi,  0],
                                 [       0,        0, -1]]

so hand +x maps to ``(cos psi, sin psi, 0)``, hand +y to ``(sin psi, -cos psi, 0)``
(note the handedness flip from the pi roll), and hand +z to world -z.

Tool tip in the hand frame. ``tool_urdf.TOOL_MOUNT_RPY`` welds the tool with
``R_mount = Ry(pi/2) @ Rz(pi/2) = [[0,0,1],[1,0,0],[0,1,0]]``, so the tool's long
axis (tool frame +z) maps to hand +x and its bend plane (tool frame x-z) maps to the
hand's x-y plane. Composing that with ``tool_geometry.tip_position`` and the TCP
offset gives

    tip_hand = (l1 + l2*cos(phi),  l2*sin(phi),  TCP_OFFSET_Z)

whose z component is independent of tau -- which is what makes a single fixed tool
height legitimate for every design, and what puts the elbow's bend in the ground
plane rather than out of it. See ``tip_offset`` and ``tip_from_hand``.

This module stays free of config imports for the same reason ``tool_geometry`` does:
the functions are pure, and callers supply the bounds.
"""
from typing import NamedTuple

import numpy as np

import tool_geometry as geom


class Box(NamedTuple):
    """An axis-aligned rectangle in the table plane, in world metres.

    Serves as the single source of truth for the reachable workspace: the same
    instance is read by the per-step clipper, the reset sampler, and the
    observation normaliser (see ``config.SE2Config.WORKSPACE``).
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def centre(self):
        """Centre of the box as ``(x, y)``."""
        return np.array([(self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0])

    @property
    def half_extents(self):
        """Half-width of the box along each axis, as ``(hx, hy)``."""
        return np.array([(self.x_max - self.x_min) / 2.0, (self.y_max - self.y_min) / 2.0])

    def clip(self, xy):
        """Clip a point into the box.

        Applied to the commanded target *before* it reaches the IK solver. Clipping
        afterwards would be too late: the solver has already traded orientation for
        reach to chase an out-of-bounds target, and the tool is no longer flat.

        Args:
            xy: Point as ``(x, y)``.

        Returns:
            np.ndarray: The clipped point as ``(x, y)``.
        """
        x, y = float(xy[0]), float(xy[1])
        return np.array([
            min(max(x, self.x_min), self.x_max),
            min(max(y, self.y_min), self.y_max),
        ])

    def contains(self, xy, tol=0.0):
        """Whether a point lies in the box, optionally with a tolerance band."""
        x, y = float(xy[0]), float(xy[1])
        return (
            self.x_min - tol <= x <= self.x_max + tol
            and self.y_min - tol <= y <= self.y_max + tol
        )

    def shrink(self, margin):
        """Return a box inset by ``margin`` on every side.

        Used both for the safety margin on the swept workspace and for the tighter
        box episodes are reset into, so an episode never starts already pressed
        against a wall.

        Raises:
            ValueError: If the margin would collapse the box.
        """
        out = Box(self.x_min + margin, self.x_max - margin, self.y_min + margin, self.y_max - margin)
        if out.x_min >= out.x_max or out.y_min >= out.y_max:
            raise ValueError(f"margin {margin} collapses {self}")
        return out

    def translate(self, xy):
        """Return the box shifted by ``(dx, dy)``.

        The swept workspace is measured with the robot base at the origin; a robot
        placed elsewhere carries the same box, translated.
        """
        dx, dy = float(xy[0]), float(xy[1])
        return Box(self.x_min + dx, self.x_max + dx, self.y_min + dy, self.y_max + dy)

    def normalise(self, xy):
        """Map a point to ``[-1, 1]`` per axis, with the box centre at the origin.

        Gives the policy a consistent input scale wherever the box happens to sit in
        the robot's frame. Points outside the box map outside ``[-1, 1]``, which is
        deliberate -- the tool tip routinely reaches past the hand's own bounds, and
        that is information rather than an error.

        Args:
            xy: Point as ``(x, y)``.

        Returns:
            np.ndarray: Normalised coordinates as ``(nx, ny)``.
        """
        return (np.asarray(xy, dtype=float)[:2] - self.centre) / self.half_extents

    def sample(self, np_random):
        """Draw a point uniformly from the box.

        Args:
            np_random (np.random.Generator): Source of randomness.

        Returns:
            np.ndarray: Sampled point as ``(x, y)``.
        """
        return np.array([
            np_random.uniform(self.x_min, self.x_max),
            np_random.uniform(self.y_min, self.y_max),
        ])


def wrap_angle(angle):
    """Wrap an angle into ``(-pi, pi]``."""
    return float(-np.mod(-angle + np.pi, 2.0 * np.pi) + np.pi)


def hand_quat(yaw):
    """Quaternion for the fingers-down hand orientation at a given yaw.

    The target rotation is the fixed-axis rpy ``(pi, 0, yaw)``, i.e.
    ``R = Rz(yaw) @ Ry(0) @ Rx(pi)``. Composing the corresponding unit quaternions
    collapses to a closed form, which keeps this module free of PyBullet:

        q = qz(yaw) (x) qx(pi) = (cos(yaw/2), sin(yaw/2), 0, 0)   in (x, y, z, w)

    At ``yaw = 0`` this is ``(1, 0, 0, 0)``, the constant panda-gym uses for its
    straight-down end-effector target. ``tests/test_se2.py`` checks the closed form
    against ``pybullet.getQuaternionFromEuler``.

    Args:
        yaw (float): Rotation about the world z axis, in radians.

    Returns:
        np.ndarray: Quaternion as ``(x, y, z, w)``, PyBullet's ordering.
    """
    half = 0.5 * float(yaw)
    return np.array([np.cos(half), np.sin(half), 0.0, 0.0])


def yaw_from_matrix(rotation):
    """Extract the hand's yaw from its world rotation matrix.

    Uses ``atan2(R[1,0], R[0,0])`` rather than an Euler decomposition. In the
    fingers-down pose the roll sits exactly on the ``+-pi`` boundary and flips sign
    between representations, whereas ``R[0,0] = cos(yaw)`` and ``R[1,0] = sin(yaw)``
    hold identically (see the module docstring), so this is unambiguous. Pitch is
    ~0 throughout, so there is no gimbal lock to guard against.

    Args:
        rotation: 3x3 world rotation matrix of the hand.

    Returns:
        float: Yaw in ``(-pi, pi]``.
    """
    r = np.asarray(rotation, dtype=float).reshape(3, 3)
    return float(np.arctan2(r[1, 0], r[0, 0]))


def tilt_from_matrix(rotation):
    """Angle between the hand's +z axis and world -z, in radians.

    Zero when the fingers point exactly straight down and the tool is exactly
    parallel to the ground. This is the quantity that silently grows when the IK
    solver trades orientation for reach near the edge of the workspace, so it is
    what the workspace sweep screens on and what the drift test asserts.

    The hand's +z axis in world coordinates is the third column of the matrix, so
    this is ``atan2(|(r02, r12)|, -r22)``. Deliberately not ``arccos(-r22)``, which
    is the same angle but ill-conditioned at exactly the pose we care most about:
    ``arccos`` has a square-root singularity at its endpoints, so near-perfect
    alignment turns a 1e-16 rounding error in ``r22`` into a 1e-8 error in the
    angle. The ``atan2`` form stays accurate all the way down to zero.

    Args:
        rotation: 3x3 world rotation matrix of the hand.

    Returns:
        float: Tilt in ``[0, pi]``.
    """
    r = np.asarray(rotation, dtype=float).reshape(3, 3)
    return float(np.arctan2(np.hypot(r[0, 2], r[1, 2]), -r[2, 2]))


def tip_offset(tau):
    """Tool-tip position in the ``panda_hand`` frame, as ``(x, y, z)``.

    Equals ``(l1 + l2*cos(phi), l2*sin(phi), TCP_OFFSET_Z)`` -- see the module
    docstring for the derivation from ``tool_urdf.TOOL_MOUNT_RPY``. The z component
    is returned as 0 here because the TCP offset is not this module's to know; the
    caller adds it. Only the in-plane part depends on tau.

    Args:
        tau: Design parameters ``(l1, l2, phi)``.

    Returns:
        np.ndarray: In-plane tip offset in the hand frame, as ``(x, y)``.
    """
    l1, l2, phi = geom._unpack(tau)
    return np.array([l1 + l2 * np.cos(phi), l2 * np.sin(phi)])


def tip_polar(tau):
    """Tool-tip offset from the hand in polar form, as ``(radius, bearing)``.

    The same offset ``tip_offset`` returns, re-expressed so the world tip reads as
    a single rotation of the hand's yaw::

        tip_from_hand((x, y, psi), tau) == (x, y) + radius * (cos(psi + bearing),
                                                              sin(psi + bearing))

    Concretely ``radius = sqrt(l1^2 + l2^2 + 2*l1*l2*cos(phi))`` and
    ``bearing = -atan2(l2*sin(phi), l1 + l2*cos(phi))``, though it is computed from
    ``tip_offset`` rather than restated, so there is one source of truth.

    This is the form the workspace geometry needs. Yaw is clipped to
    ``+-SE2Config.YAW_LIMIT``, so the tip's bearing from the hand is confined to
    ``[bearing - YAW_LIMIT, bearing + YAW_LIMIT]``: tau does not merely set how far
    the tip reaches, it *rotates the half-plane of directions it can reach in*. A
    straight rod has ``bearing = 0`` and can never point its tip backwards, whatever
    its length. Note also that the radius is exact, not an upper bound -- the tip
    lies on a circle about the hand, not in a disk -- so a long tool cannot reach a
    target close to the hand.

    Args:
        tau: Design parameters ``(l1, l2, phi)``.

    Returns:
        tuple: ``(radius, bearing)`` in metres and radians, the bearing in
            ``(-pi, pi]`` and measured in the hand's yawed frame.
    """
    ox, oy = tip_offset(tau)
    return float(np.hypot(ox, oy)), float(np.arctan2(-oy, ox))


def tip_from_hand(hand_se2, tau):
    """Tool-tip position in the table plane, in closed form.

    Saves an FK query per step, and more importantly makes the tip's dependence on
    the design explicit: this is the only place tau enters the SE(2) problem.

    The y component picks up a sign flip relative to the raw hand-frame offset
    because the fingers-down orientation carries a pi roll, so hand +y maps to
    world ``(sin psi, -cos psi, 0)``. Equivalently the world offset is
    ``Rz(psi) @ (ox, -oy)``.

    Args:
        hand_se2: Hand pose as ``(x, y, yaw)``.
        tau: Design parameters ``(l1, l2, phi)``.

    Returns:
        np.ndarray: Tool-tip position as ``(x, y)``.
    """
    x, y, yaw = (float(v) for v in hand_se2)
    ox, oy = tip_offset(tau)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([x + c * ox + s * oy, y + s * ox - c * oy])
