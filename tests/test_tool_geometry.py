import numpy as np
import pytest

import tool_geometry as geom


@pytest.mark.parametrize("l1,l2", [(0.2, 0.3), (0.15, 0.5), (0.5, 0.15)])
@pytest.mark.parametrize("theta,expected_d2", [
    (0.0, np.array([0.0, 0.0, -1.0])),
    (np.pi / 2, np.array([1.0, 0.0, 0.0])),
    (np.pi, np.array([0.0, 0.0, 1.0])),
])
def test_tip_position(l1, l2, theta, expected_d2):
    expected = l1 * np.array([0.0, 0.0, 1.0]) + l2 * expected_d2
    tip = geom.tip_position(l1, l2, theta)
    np.testing.assert_allclose(tip, expected, atol=1e-10)


@pytest.mark.parametrize("l1,l2", [(0.2, 0.3), (0.15, 0.15), (0.5, 0.5)])
def test_mass(l1, l2):
    expected = geom.RHO * geom.W * geom.H * (l1 + l2)
    assert geom.mass(l1, l2) == pytest.approx(expected)


@pytest.mark.parametrize("l1,l2,theta", [
    (0.2, 0.3, 0.0),
    (0.2, 0.3, np.pi / 2),
    (0.2, 0.3, np.pi),
    (0.15, 0.5, 1.0),
])
def test_inertia_symmetric_positive_definite(l1, l2, theta):
    I = geom.inertia_tensor(l1, l2, theta)
    np.testing.assert_allclose(I, I.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(I)
    assert np.all(eigvals > 0)


def test_inertia_degenerates_to_single_box_as_l2_vanishes():
    l1, theta = 0.3, 1.2
    I_full = geom.inertia_tensor(l1, 1e-9, theta)

    m1 = geom.RHO * geom.W * geom.H * l1
    I_single = geom._box_inertia_local(m1, (geom.W, geom.H, l1))
    # box1 centroid is the combined COM in this limit, so no parallel-axis shift needed
    np.testing.assert_allclose(I_full, I_single, atol=1e-6)
