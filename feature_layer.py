"""The parameter-free feature layer: the tool's dimensions, read back off ``x``.

::

    tau -> h -> x1 -> features -> V

Fixed arithmetic, no parameters. ``x`` already contains the tool -- the hand, the
elbow and the tip are all in it (``initial_state``'s layout) -- but only behind a
rotation by the hand yaw. ``features`` undoes that rotation and appends what comes
out, so neither network has to learn trigonometry it can be handed for free.
Everything here is a function of ``x``, so it adds no ``tau`` input and reopens
nothing in ``networks_and_design_gradient.md`` §1.

It is the first line of every network's ``forward``, actor and critic alike, and is
**not** a gym ``ObservationWrapper``. A wrapper would compute the features in numpy
for training, and the designer would then need a second, torch copy to differentiate
through -- two implementations of one formula, which is the drift this codebase keeps
paying for. Here there is one copy, and it is torch.

It is also part of the frozen normalisation contract (``h_initial_state_map.md``): the
designer reapplies it against a ``V`` trained much earlier, so it changes under the
same rule as ``TaskConfig.SCENE_BOX``, which is never.


Feature layout
--------------

29 dims. ``s`` is ``SCENE_BOX.scale``; every length stays on the scene-box metric, and
``phi`` -- the one quantity with no length units -- is divided by ``PHI_MAX``.

======  ==========================================
slice   contents
======  ==========================================
0:24    ``x``, unchanged
24:26   ``o_x / s, o_y / s`` -- tip offset, hand frame
26:27   ``l1 / s``
27:28   ``l2 / s``
28:29   ``phi / PHI_MAX``
======  ==========================================

``(o_x, o_y)`` is the sufficient statistic for reaching; ``l1, l2, phi`` are inert
there and are what sweeping and pushing will need.


Smoothness
----------

The design gradient is read through this layer, so it must be smooth on every state
that can actually occur. It is:

- The rotation into the hand frame is products and sums.
- ``l1`` is a projection onto hand ``+x``, which is linear, rather than a length, so it
  has no kink at zero.
- ``hypot`` and ``atan2`` misbehave only at the origin, and their input is the second
  segment, elbow -> tip, whose length is ``l2 >= L_MIN = 0.1 m``.
- ``atan2`` jumps at ``+-pi``, but ``|phi| <= PHI_MAX = 1.9`` never gets there.
"""
import torch

import initial_state as ini
from config import DesignPriorConfig

FEATURE_DIM = ini.OBS_DIM + 5  # 29

TIP_OFFSET = slice(24, 26)  # (o_x, o_y) / s
DESIGN = slice(26, 29)      # (l1 / s, l2 / s, phi / PHI_MAX)


def features(x):
    """Append the tool's hand-frame dimensions to an observation. Differentiable.

    Inverts ``se2.tip_from_hand`` and ``se2.elbow_from_hand``: rotating
    ``elbow - hand`` and ``tip - hand`` by ``Rz(psi)^T`` gives ``(l1, 0)`` and
    ``(o_x, -o_y)``, and the elbow -> tip vector is then ``l2 * (cos phi, sin phi)``.
    Recovery is exact up to rounding (``tests/test_feature_layer.py``).

    Pure torch, with no in-place writes, so autograd runs from ``V`` back through this
    into ``h`` and on to ``tau``.

    Args:
        x (torch.Tensor): Observation in ``initial_state``'s layout, shape
            ``(..., OBS_DIM)``.

    Returns:
        torch.Tensor: Shape ``(..., FEATURE_DIM)``, in ``x``'s dtype and on its
            device: ``x`` followed by ``o_x/s, o_y/s, l1/s, l2/s, phi/PHI_MAX``.

    Raises:
        ValueError: If the trailing dimension of ``x`` is not ``OBS_DIM``.
    """
    if x.shape[-1] != ini.OBS_DIM:
        raise ValueError(f"x must have a trailing dimension of {ini.OBS_DIM}, got {tuple(x.shape)}")

    # Every slice keeps its trailing axis, so each feature below is (..., 1) and the
    # output is one torch.cat whatever the batch rank.
    c, s = x[..., ini.HAND_YAW].split(1, dim=-1)  # cos psi, sin psi
    # Both P(), so the centring cancels: (elbow - hand)/scale and (tip - hand)/scale.
    d_elbow = x[..., ini.ELBOW_XY] - x[..., ini.HAND_XY]
    d_tip = x[..., ini.TIP_XY] - x[..., ini.HAND_XY]

    # Rotate into the hand frame. The y row inverts tip_from_hand's Rz(psi) @
    # (ox, -oy); the elbow has no y component to flip.
    l1 = c * d_elbow[..., 0:1] + s * d_elbow[..., 1:2]
    ox = c * d_tip[..., 0:1] + s * d_tip[..., 1:2]
    oy = s * d_tip[..., 0:1] - c * d_tip[..., 1:2]
    l2 = torch.hypot(ox - l1, oy)
    phi = torch.atan2(oy, ox - l1)

    return torch.cat([x, ox, oy, l1, l2, phi / DesignPriorConfig.PHI_MAX], dim=-1)
