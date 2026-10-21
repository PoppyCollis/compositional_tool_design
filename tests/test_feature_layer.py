"""Tests for feature_layer.py, the fixed arithmetic between x and the networks. No sim."""
import numpy as np
import pytest
import torch

import feature_layer as fl
import initial_state as ini
import se2
import task
from config import DesignPriorConfig, TaskConfig

L_MIN, L_MAX = DesignPriorConfig.L_MIN, DesignPriorConfig.L_MAX
PHI_MAX = DesignPriorConfig.PHI_MAX
SCALE = TaskConfig.SCENE_BOX.scale

# The straight rods and the box corners. phi = +-PHI_MAX is as close as atan2 gets to
# its jump, l2 = L_MIN as close as hypot gets to its singularity.
TAUS = [
    (L_MAX, L_MAX, 0.0),
    (L_MIN, L_MIN, 0.0),
    (L_MIN, L_MAX, PHI_MAX),
    (L_MAX, L_MIN, -PHI_MAX),
    (L_MIN, L_MIN, PHI_MAX),
    (L_MAX, L_MAX, -PHI_MAX),
]

# Recovery tolerance by dtype. Measured worst over 200 resets x 64 designs, corners
# included: 7.8e-16 (values) and 3.8e-15 (Jacobian) in float64, 4.1e-7 and 2.4e-6 in
# float32. float64 is exact; float32 -- what the networks run in -- bottoms out near
# 1e-6, because a 0.1 m segment is read as a difference of O(1) coordinates.
ATOL = {torch.float64: 1e-12, torch.float32: 1e-5}
DTYPES = pytest.mark.parametrize("dtype", list(ATOL), ids=["float64", "float32"])


def draws(seed=0, resets=5, n=16):
    """``(g, xi, taus)``: one reset each, scoring the corners plus ``n`` uniform designs.

    Drawn the way the env and the designer will draw them, so every ``x`` built from
    these is a state that can actually occur.
    """
    generator = np.random.default_rng(seed)
    lo, hi = [L_MIN, L_MIN, -PHI_MAX], [L_MAX, L_MAX, PHI_MAX]
    for _ in range(resets):
        g = task.sample_task(generator)
        xi = ini.sample_xi(generator, g)
        yield g, xi, np.concatenate([TAUS, generator.uniform(lo, hi, size=(n, 3))])


# -- shape and layout ---------------------------------------------------------


@DTYPES
def test_features_appends_five_columns_and_passes_x_through(dtype):
    g, xi, taus = next(draws())
    x = ini.h(torch.tensor(taus, dtype=dtype), g, xi)
    before = x.clone()
    out = fl.features(x)

    assert fl.FEATURE_DIM == 29
    assert out.shape == (len(taus), fl.FEATURE_DIM)
    assert out.dtype == dtype
    assert torch.equal(out[..., : ini.OBS_DIM], x)
    assert torch.equal(x, before)  # no in-place writes


def test_features_rank_follows_x():
    """``h`` returns ``(OBS_DIM,)`` for a single design; so must this."""
    g, xi, taus = next(draws())
    x = ini.h(torch.tensor(taus[0], dtype=torch.float64), g, xi)
    assert fl.features(x).shape == (fl.FEATURE_DIM,)


def test_features_rejects_the_wrong_width():
    for width in [ini.OBS_DIM - 3, ini.OBS_DIM + 1]:
        with pytest.raises(ValueError):
            fl.features(torch.zeros(2, width))


# -- recovery -----------------------------------------------------------------


@DTYPES
def test_tau_is_exactly_recoverable_from_x1(dtype):
    """The claim decision 1 of networks_and_design_gradient.md rests on.

    l1 from hand -> elbow, (l2, phi) from elbow -> tip, all in the hand frame. The
    arithmetic is ``feature_layer.features``; it used to live inline here.
    """
    for g, xi, taus in draws():
        x1 = ini.h(torch.tensor(taus, dtype=dtype), g, xi)
        got = fl.features(x1)[..., fl.DESIGN].double().numpy()
        assert np.allclose(got, taus / [SCALE, SCALE, PHI_MAX], rtol=0, atol=ATOL[dtype])


@DTYPES
def test_tip_offset_block_is_the_closed_form(dtype):
    """``(o_x, o_y)``, the reach statistic, against the audited numpy original."""
    for g, xi, taus in draws():
        x1 = ini.h(torch.tensor(taus, dtype=dtype), g, xi)
        got = fl.features(x1)[..., fl.TIP_OFFSET].double().numpy()
        expected = np.array([se2.tip_offset(tau) for tau in taus]) / SCALE
        assert np.allclose(got, expected, rtol=0, atol=ATOL[dtype])


# -- gradients ----------------------------------------------------------------


def test_features_passes_gradcheck_at_valid_states():
    """Smooth wherever a state can occur, the corners included."""
    for g, xi, taus in draws(resets=3, n=4):
        x = ini.h(torch.tensor(taus, dtype=torch.float64), g, xi).requires_grad_(True)
        assert torch.autograd.gradcheck(fl.features, (x,))


@DTYPES
def test_the_design_jacobian_through_h_is_diagonal(dtype):
    """``d features(h(tau))[DESIGN] / d tau == diag(1/s, 1/s, 1/PHI_MAX)``, by autograd.

    The design block is ``tau`` rescaled, so its Jacobian is the rescaling and nothing
    else. A detach anywhere on ``tau -> h -> features`` zeroes it; a sign wrong in the
    rotation flips the ``phi`` entry.
    """
    expected = np.diag([1 / SCALE, 1 / SCALE, 1 / PHI_MAX])
    for g, xi, taus in draws(resets=3, n=4):
        for tau in taus:
            jac = torch.autograd.functional.jacobian(
                lambda t: fl.features(ini.h(t, g, xi))[fl.DESIGN],
                torch.tensor(tau, dtype=dtype),
            )
            assert np.allclose(jac.double().numpy(), expected, rtol=0, atol=ATOL[dtype])
