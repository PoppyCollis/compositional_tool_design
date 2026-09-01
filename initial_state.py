"""The initial-state map ``h``: a design becomes the state the value function scores.

::

    xi ~ p(xi | g)          reset randomness: hand pose, object start pose
    x1 = h(tau, g, xi)      deterministic, differentiable in tau

This is the bridge from design to value. ``V`` scores states; ``h`` turns a tool into
one, so the design objective is ``f(tau, g) = E_xi[ V(h(tau, g, xi), tau, g) ]`` and

    grad_tau E = -[ dV/dtau + (dV/dx1) . (dh/dtau) ]

The second term is the whole reason this module is written in torch rather than read
off the simulator: PyBullet's kinematics are not differentiable, and only the tool
part needs to be. With the arm teleported to ``xi``'s pose the hand is a constant, and
``tau`` enters ``x1`` through exactly two slices -- the tip position and the
``obj - tip`` vector that hangs off it. ``test_initial_state.py`` asserts that the
gradient is identically zero everywhere else.

``xi`` is not part of ``g``: the hand pose fails the membership rule (it enters no
reward term and is plainly visible in ``x1``), and the object's start pose is drawn
from a distribution keyed by ``task_id`` (``task.object_start``) rather than carried
as a field. ``xi`` is also independent of ``tau``, which is what lets the
``dh/dtau`` path survive the expectation over resets.


Observation layout
------------------

The single description of ``x_t``, 21 dims. ``P(p) = (p - c)/s`` and ``D(v) = v/s``
with ``c``, ``s`` from ``TaskConfig.SCENE_BOX``; see ``se2.Box.scale`` for why the
divisor is one scalar rather than one per axis.

======  ==========================================  =============
slice   contents                                    tau-dependent
======  ==========================================  =============
0:2     P(hand xy)                                  no
2:4     cos psi, sin psi                            no
4:7     hand vx, vy, psi-dot / VEL_SCALE            no (0 at reset)
7:9     P(tip xy)                                   **yes**
9:11    P(object xy)                                no
11:13   D(object - tip)   -- reach/contact phase    **yes**
13:15   D(target - object) -- transport phase       no
15:17   P(target xy)                                no
17:20   task one-hot                                no
20:21   r_obj / s                                   no
======  ==========================================  =============

Slices 0:9 are ``PandaWithTool.get_obs``; 15:21 are ``task.encode``. This module
reproduces the first block analytically, so the two must agree -- they do because both
call ``Box.normalise_point`` on the same box, which is the only thing keeping them in
step until the env layer adds a sim-vs-analytic assertion.

``obj - tip`` and ``target - obj``, not ``target - tip``: the reward is
``-[w_reach*d(tip, obj) + w_trans*d(obj, target)]``, so those two are the terms the
policy is actually differentiating. Their sum, ``target - tip``, points somewhere
useful in neither sweeping nor pushing. For reaching the object is at the target, so
the first coincides with ``target - tip`` and the second is identically zero.
"""
from typing import NamedTuple

import numpy as np
import torch

import se2
import task as task_mod
from config import SE2Config, TaskConfig

# Block widths. ROBOT_DIM must match PandaWithTool.get_obs, which this module cannot
# import (it would pull PyBullet and panda-gym into a pure-torch module). The env
# layer is where that gets asserted.
ROBOT_DIM = 9
OBS_DIM = ROBOT_DIM + 2 + 2 + 2 + task_mod.ENCODING_DIM  # 21

HAND_XY = slice(0, 2)
HAND_YAW = slice(2, 4)
HAND_VEL = slice(4, 7)
TIP_XY = slice(7, 9)
OBJ_XY = slice(9, 11)
D_OBJ_TIP = slice(11, 13)
D_TARGET_OBJ = slice(13, 15)
TASK_BLOCK = slice(15, 21)

# The only two slices carrying a design gradient.
TAU_SLICES = (TIP_XY, D_OBJ_TIP)


class Xi(NamedTuple):
    """Reset randomness. Independent of ``tau``, and not part of ``g``."""

    hand_se2: np.ndarray  # (3,) hand pose (x, y, yaw)
    obj_xy: np.ndarray    # (2,) object start position


def sample_xi(np_random, task):
    """Draw ``xi ~ p(xi | g)``.

    The hand pose is drawn exactly as ``PandaWithTool.reset`` draws it -- uniform over
    the workspace inset by ``RESET_MARGIN``, yaw uniform over ``+-YAW_LIMIT`` -- so
    the designer's expectation over resets is taken against the same law the policy
    was trained under. If those two ever diverge the design objective is scoring
    states the value function never saw.

    Args:
        np_random (np.random.Generator): Source of randomness.
        task (task.Task): The task instance ``g``.

    Returns:
        Xi: The sampled reset.
    """
    xy = SE2Config.WORKSPACE.shrink(SE2Config.RESET_MARGIN).sample(np_random)
    yaw = np_random.uniform(-SE2Config.YAW_LIMIT, SE2Config.YAW_LIMIT)
    return Xi(
        hand_se2=np.array([xy[0], xy[1], yaw]),
        obj_xy=task_mod.object_start(task, np_random),
    )


def tip_offset_torch(tau):
    """Tool-tip offset in the ``panda_hand`` frame, as ``(..., 2)``. Differentiable.

    The torch mirror of ``se2.tip_offset``: ``(l1 + l2*cos(phi), l2*sin(phi))``. Kept
    to that one line rather than reimplemented, and pinned to the numpy original by a
    test, because this is the entire gradient path from design to state.

    Unlike ``tool_geometry``, this accepts a batched ``tau``. That rule exists because
    geometry feeds URDF generation, which builds one body at a time and would silently
    drop all but the first row; nothing here touches a URDF, and the Langevin chain
    runs a batch of designs against one ``g``.

    Args:
        tau: Design parameters ``(l1, l2, phi)``, shape ``(..., 3)``.

    Returns:
        torch.Tensor: In-plane tip offset in the hand frame, shape ``(..., 2)``.
    """
    l1, l2, phi = tau[..., 0], tau[..., 1], tau[..., 2]
    return torch.stack([l1 + l2 * torch.cos(phi), l2 * torch.sin(phi)], dim=-1)


def tip_from_hand_torch(hand_se2, tau):
    """Tool-tip position in the table plane, as ``(..., 2)``. Differentiable in ``tau``.

    The torch mirror of ``se2.tip_from_hand``, including the sign flip on the offset's
    y component: the fingers-down pose carries a pi roll, so hand +y maps to world
    ``(sin psi, -cos psi, 0)`` and the world offset is ``Rz(psi) @ (ox, -oy)``.

    Args:
        hand_se2: Hand pose ``(x, y, yaw)``, a length-3 sequence or tensor.
        tau: Design parameters, shape ``(..., 3)``.

    Returns:
        torch.Tensor: Tool-tip position ``(x, y)``, shape ``(..., 2)``.
    """
    ox, oy = tip_offset_torch(tau).unbind(dim=-1)
    x, y, yaw = (float(v) for v in hand_se2)
    c, s = np.cos(yaw), np.sin(yaw)
    return torch.stack([x + c * ox + s * oy, y + s * ox - c * oy], dim=-1)


def h(tau, task, xi):
    """``x1 = h(tau, g, xi)``: the state a design lands the episode in.

    Args:
        tau: Design parameters ``(l1, l2, phi)``, shape ``(3,)`` or ``(B, 3)``. May
            carry ``requires_grad``; the returned tensor is differentiable in it.
        task (task.Task): The task instance ``g``.
        xi (Xi): The reset draw, shared across the batch (``xi`` is independent of
            ``tau``, so one draw scores every design on the same reset).

    Returns:
        torch.Tensor: Shape ``(OBS_DIM,)`` or ``(B, OBS_DIM)``, matching ``tau``'s
            rank, in the layout documented at the top of this module.
    """
    tau = torch.as_tensor(tau)
    if tau.shape[-1] != 3:
        raise ValueError(f"tau must have a trailing dimension of 3, got {tuple(tau.shape)}")
    if tau.ndim > 2:
        raise ValueError(f"tau must be (3,) or (B, 3), got {tuple(tau.shape)}")
    single = tau.ndim == 1
    if single:
        tau = tau.unsqueeze(0)
    batch = tau.shape[0]

    box = TaskConfig.SCENE_BOX
    kwargs = {"dtype": tau.dtype, "device": tau.device}

    def const(values):
        """A tau-independent block, broadcast over the batch."""
        return torch.as_tensor(np.asarray(values, dtype=float), **kwargs).expand(batch, -1)

    hand_xy = np.asarray(xi.hand_se2, dtype=float)[:2]
    yaw = float(xi.hand_se2[2])
    obj = np.asarray(xi.obj_xy, dtype=float)[:2]
    target = np.asarray(task.target, dtype=float)[:2]

    # The two tau-dependent blocks. Everything else below is a constant of the reset.
    tip = tip_from_hand_torch(xi.hand_se2, tau)
    tip_n = (tip - const(box.centre)) / box.scale
    d_obj_tip = (const(obj) - tip) / box.scale

    blocks = [
        const(box.normalise_point(hand_xy)),
        const([np.cos(yaw), np.sin(yaw)]),
        # Zero: the arm is teleported to xi's pose, not driven there.
        const(np.zeros(3)),
        tip_n,
        const(box.normalise_point(obj)),
        d_obj_tip,
        const(box.normalise_delta(target - obj)),
        const(task_mod.encode(task)),
    ]
    out = torch.cat(blocks, dim=-1)
    return out.squeeze(0) if single else out
