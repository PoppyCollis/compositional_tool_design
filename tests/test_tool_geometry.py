import numpy as np
import pytest

import panda_with_tool_urdf
import tool_geometry as geom
import tool_urdf
from config import DesignPriorConfig, GripperConfig


@pytest.mark.parametrize("l1,l2", [(0.2, 0.3), (0.15, 0.5), (0.5, 0.15)])
@pytest.mark.parametrize("phi,expected_d2", [
    (0.0, np.array([0.0, 0.0, 1.0])),        # straight rod
    (np.pi / 2, np.array([1.0, 0.0, 0.0])),  # right angle
    (-np.pi / 2, np.array([-1.0, 0.0, 0.0])),  # mirror image
])
def test_tip_position(l1, l2, phi, expected_d2):
    expected = l1 * np.array([0.0, 0.0, 1.0]) + l2 * expected_d2
    tip = geom.tip_position(l1, l2, phi)
    np.testing.assert_allclose(tip, expected, atol=1e-10)


@pytest.mark.parametrize("l1,l2", [(0.2, 0.3), (0.15, 0.15), (0.5, 0.5)])
def test_mass(l1, l2):
    expected = geom.RHO * geom.W * geom.H * (l1 + l2)
    assert geom.mass(l1, l2) == pytest.approx(expected)


@pytest.mark.parametrize("l1,l2,phi", [
    (0.2, 0.3, 0.0),
    (0.2, 0.3, np.pi / 2),
    (0.2, 0.3, 1.9),
    (0.15, 0.5, -1.0),
])
def test_inertia_symmetric_positive_definite(l1, l2, phi):
    I = geom.inertia_tensor(l1, l2, phi)
    np.testing.assert_allclose(I, I.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(I)
    assert np.all(eigvals > 0)


def test_inertia_degenerates_to_single_box_as_l2_vanishes():
    l1, phi = 0.3, 1.2
    I_full = geom.inertia_tensor(l1, 1e-9, phi)

    m1 = geom.RHO * geom.W * geom.H * l1
    I_single = geom._box_inertia_local(m1, (geom.W, geom.H, l1))
    # box1 centroid is the combined COM in this limit, so no parallel-axis shift needed
    np.testing.assert_allclose(I_full, I_single, atol=1e-6)


def test_phi_max_keeps_link2_clear_of_the_gripper():
    """PHI_MAX exists so link 2 can never reach back into the fingers. Check the
    worst corner of the design box: shortest handle, longest head."""
    mount_z, _ = panda_with_tool_urdf._finger_span()
    tcp_z = tool_urdf.TCP_OFFSET[2]
    l1, l2 = DesignPriorConfig.L_MIN, DesignPriorConfig.L_MAX

    for phi in np.linspace(-DesignPriorConfig.PHI_MAX, DesignPriorConfig.PHI_MAX, 200):
        tip_z = geom.tip_position(l1, l2, phi)[2]
        assert tcp_z + tip_z > mount_z

    # The bound must do real work: derive the exact angle at which the tip meets
    # the mount rather than hardcoding a probe, so this keeps testing something
    # if L_MIN/L_MAX/TCP_OFFSET_Z ever move.
    critical_phi = np.arccos((mount_z - tcp_z - l1) / l2)
    assert DesignPriorConfig.PHI_MAX < critical_phi
    assert tcp_z + geom.tip_position(l1, l2, critical_phi + 0.05)[2] < mount_z


def test_gripper_config_matches_the_actual_urdf():
    """GripperConfig restates finger geometry that really lives in the URDF and
    the finger mesh; drift between the two would silently invalidate PHI_MAX."""
    mount_z, tip_z = panda_with_tool_urdf._finger_span()
    assert mount_z == pytest.approx(GripperConfig.FINGER_MOUNT_Z, abs=1e-4)
    assert tip_z == pytest.approx(GripperConfig.FINGERTIP_Z, abs=1e-4)
    assert GripperConfig.FINGER_MOUNT_Z < GripperConfig.TCP_OFFSET_Z < GripperConfig.FINGERTIP_Z


def test_interior_angle_is_the_supplement_of_phi():
    """theta is the interior elbow angle: straight rod at phi=0 is theta=pi."""
    assert geom.interior_angle(0.0) == pytest.approx(np.pi)
    assert geom.interior_angle(np.pi / 2) == pytest.approx(np.pi / 2)
    # folding link 2 back shrinks the interior angle, and the angle between the
    # two link directions is exactly theta
    for phi in (0.0, 0.7, 1.9, -1.9):
        d1, d2 = geom.link_directions(phi)
        assert np.arccos(np.dot(-d1, d2)) == pytest.approx(abs(geom.interior_angle(phi)))


@pytest.mark.parametrize("bad", [
    np.zeros((64, 3)),      # a batch: would silently collapse to row 0
    np.zeros((1, 3)),       # even a batch of one
    np.zeros(4),            # extra element: would be silently dropped
    np.zeros(2),
])
def test_unpack_rejects_anything_that_is_not_one_design(bad):
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        geom._unpack(bad)


def test_unpack_accepts_a_single_design():
    import torch
    assert geom._unpack((0.2, 0.3, 0.1)) == pytest.approx((0.2, 0.3, 0.1))
    assert geom._unpack(torch.tensor([0.2, 0.3, 0.1])) == pytest.approx((0.2, 0.3, 0.1))
