"""Analytic tests for se2.py. No sim, no PyBullet body."""
import numpy as np
import pybullet as p
import pytest

import se2

BOX = se2.Box(x_min=0.1, x_max=0.5, y_min=-0.2, y_max=0.4)
YAWS = [-np.pi, -2.7, -1.0, -0.001, 0.0, 0.001, 1.0, 2.7, np.pi]


def test_box_clip_inside_is_identity():
    assert np.allclose(BOX.clip((0.3, 0.1)), (0.3, 0.1))


def test_box_clip_pulls_outside_points_to_the_edge():
    assert np.allclose(BOX.clip((-5.0, 5.0)), (BOX.x_min, BOX.y_max))
    assert np.allclose(BOX.clip((5.0, -5.0)), (BOX.x_max, BOX.y_min))


# -- the isotropic observation map --------------------------------------------


def test_box_scale_is_the_longest_half_extent():
    assert BOX.scale == pytest.approx(max(BOX.half_extents))


def test_normalise_point_is_isotropic():
    """One normalised unit means the same metres in every direction.

    The property the whole scalar-divisor decision exists for: dh/dtau is in metres,
    so a per-axis divisor would give two designs producing physically equal tip
    displacements unequal design gradients, purely from the box's aspect ratio.
    """
    step = 0.07
    c = BOX.centre
    dx = BOX.normalise_point(c + np.array([step, 0.0])) - BOX.normalise_point(c)
    dy = BOX.normalise_point(c + np.array([0.0, step])) - BOX.normalise_point(c)
    assert np.linalg.norm(dx) == pytest.approx(np.linalg.norm(dy))


def test_normalise_point_centres_and_bounds_the_box():
    assert np.allclose(BOX.normalise_point(BOX.centre), (0.0, 0.0))
    for corner in [(BOX.x_min, BOX.y_min), (BOX.x_max, BOX.y_max)]:
        assert np.max(np.abs(BOX.normalise_point(corner))) <= 1.0 + 1e-12


def test_normalise_point_is_symmetric_but_need_not_fill_the_unit_square():
    """The short axis stops short of +-1. That is the cost of isotropy, not a bug."""
    lo = BOX.normalise_point((BOX.x_min, BOX.y_min))
    hi = BOX.normalise_point((BOX.x_max, BOX.y_max))
    assert np.allclose(lo, -hi)
    short = int(np.argmin(BOX.half_extents))
    assert abs(hi[short]) < 1.0


def test_normalise_delta_does_not_centre():
    assert np.allclose(BOX.normalise_delta((0.0, 0.0)), (0.0, 0.0))
    assert np.allclose(BOX.normalise_delta((BOX.scale, 0.0)), (1.0, 0.0))


def test_normalise_delta_matches_a_difference_of_normalised_points():
    a, b = np.array([0.2, 0.1]), np.array([0.45, -0.15])
    assert np.allclose(
        BOX.normalise_delta(b - a),
        BOX.normalise_point(b) - BOX.normalise_point(a),
    )


def test_box_shrink_insets_every_side():
    small = BOX.shrink(0.05)
    assert small.x_min == pytest.approx(BOX.x_min + 0.05)
    assert small.x_max == pytest.approx(BOX.x_max - 0.05)
    assert small.y_min == pytest.approx(BOX.y_min + 0.05)
    assert small.y_max == pytest.approx(BOX.y_max - 0.05)


def test_box_shrink_rejects_a_collapsing_margin():
    with pytest.raises(ValueError):
        BOX.shrink(1.0)


def test_box_translate_moves_the_box_rigidly():
    moved = BOX.translate((-0.6, 0.1))
    assert moved.x_max - moved.x_min == pytest.approx(BOX.x_max - BOX.x_min)
    assert moved.x_min == pytest.approx(BOX.x_min - 0.6)
    assert moved.y_min == pytest.approx(BOX.y_min + 0.1)


def test_box_sample_stays_inside_and_is_reproducible():
    a = [BOX.sample(np.random.default_rng(0)) for _ in range(50)]
    b = [BOX.sample(np.random.default_rng(0)) for _ in range(50)]
    assert np.allclose(a, b)
    assert all(BOX.contains(xy) for xy in a)


@pytest.mark.parametrize("angle", [-3 * np.pi, -np.pi, 0.0, np.pi, 3 * np.pi, 7.0])
def test_wrap_angle_lands_in_the_canonical_interval(angle):
    wrapped = se2.wrap_angle(angle)
    assert -np.pi < wrapped <= np.pi + 1e-12
    assert np.isclose(np.cos(wrapped), np.cos(angle))
    assert np.isclose(np.sin(wrapped), np.sin(angle))


@pytest.mark.parametrize("yaw", YAWS)
def test_hand_quat_matches_pybullet(yaw):
    """The closed form must agree with getQuaternionFromEuler([pi, 0, yaw])."""
    mine = se2.hand_quat(yaw)
    theirs = np.array(p.getQuaternionFromEuler([np.pi, 0.0, yaw]))
    # q and -q are the same rotation.
    assert np.allclose(mine, theirs, atol=1e-12) or np.allclose(mine, -theirs, atol=1e-12)


@pytest.mark.parametrize("yaw", YAWS)
def test_yaw_round_trips_through_the_rotation_matrix(yaw):
    r = np.array(p.getMatrixFromQuaternion(se2.hand_quat(yaw))).reshape(3, 3)
    assert se2.wrap_angle(se2.yaw_from_matrix(r) - yaw) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("yaw", YAWS)
def test_fingers_down_pose_has_zero_tilt(yaw):
    r = np.array(p.getMatrixFromQuaternion(se2.hand_quat(yaw))).reshape(3, 3)
    assert se2.tilt_from_matrix(r) == pytest.approx(0.0, abs=1e-9)


def test_tilt_grows_with_a_deliberate_lean():
    """The quantity the sweep screens on must actually respond to a lean."""
    for lean in (0.05, 0.2, 0.5):
        quat = p.getQuaternionFromEuler([np.pi - lean, 0.0, 0.3])
        r = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        assert se2.tilt_from_matrix(r) == pytest.approx(lean, abs=1e-9)


def test_tip_offset_is_the_straight_rod_sum_when_phi_is_zero():
    assert np.allclose(se2.tip_offset((0.3, 0.2, 0.0)), (0.5, 0.0))


def test_tip_offset_folds_into_the_bend_at_a_right_angle():
    assert np.allclose(se2.tip_offset((0.3, 0.2, np.pi / 2)), (0.3, 0.2), atol=1e-12)


def test_tip_offset_mirrors_with_the_sign_of_phi():
    plus = se2.tip_offset((0.3, 0.2, 1.1))
    minus = se2.tip_offset((0.3, 0.2, -1.1))
    assert plus[0] == pytest.approx(minus[0])
    assert plus[1] == pytest.approx(-minus[1])


def test_tip_from_hand_reduces_to_the_offset_at_the_origin():
    ox, oy = se2.tip_offset((0.3, 0.2, 0.7))
    assert np.allclose(se2.tip_from_hand((0.0, 0.0, 0.0), (0.3, 0.2, 0.7)), (ox, -oy))


def test_tip_from_hand_translates_with_the_hand():
    tau = (0.3, 0.2, 0.7)
    base = se2.tip_from_hand((0.0, 0.0, 0.4), tau)
    moved = se2.tip_from_hand((1.5, -0.6, 0.4), tau)
    assert np.allclose(moved - base, (1.5, -0.6))


def test_tip_from_hand_keeps_the_tool_length_under_rotation():
    """Yaw may only rotate the tip about the hand, never stretch the tool."""
    tau = (0.3, 0.2, 0.7)
    reach = np.linalg.norm(se2.tip_offset(tau))
    for yaw in YAWS:
        tip = se2.tip_from_hand((0.45, -0.1, yaw), tau)
        assert np.linalg.norm(tip - np.array([0.45, -0.1])) == pytest.approx(reach)


def test_elbow_from_hand_lies_along_the_hands_own_x_axis():
    """The elbow's hand-frame offset is (l1, 0), so it never picks up a y component.

    This is where elbow_from_hand and tip_from_hand deliberately differ: the pi roll
    of the fingers-down pose flips the offset's y, and the elbow has none to flip.
    At yaw = pi/2 the hand's +x axis is world +y, so the displacement is +l1 * yhat --
    a sign flip would send it to -l1 * yhat and stay undetected by a length check.
    """
    tau = (0.3, 0.2, 0.7)
    hand = np.array([0.45, -0.1])
    for yaw, direction in [(0.0, (1.0, 0.0)), (np.pi / 2, (0.0, 1.0)),
                           (-np.pi / 2, (0.0, -1.0)), (np.pi, (-1.0, 0.0))]:
        elbow = se2.elbow_from_hand((hand[0], hand[1], yaw), tau)
        assert np.allclose(elbow - hand, 0.3 * np.asarray(direction), atol=1e-12)


def test_elbow_from_hand_ignores_l2_and_phi():
    """l1 is the only design parameter the elbow reads."""
    base = se2.elbow_from_hand((0.45, -0.1, 0.4), (0.3, 0.2, 0.7))
    for tau in [(0.3, 0.1, -1.9), (0.3, 0.2, 0.0), (0.3, 0.15, 1.9)]:
        assert np.allclose(se2.elbow_from_hand((0.45, -0.1, 0.4), tau), base)


def test_a_straight_rod_puts_the_elbow_between_the_hand_and_the_tip():
    """phi = 0 collapses the polyline to a segment of length l1 + l2."""
    tau = (0.3, 0.2, 0.0)
    hand_se2 = (0.45, -0.1, 0.4)
    hand = np.array(hand_se2[:2])
    elbow = se2.elbow_from_hand(hand_se2, tau)
    tip = se2.tip_from_hand(hand_se2, tau)
    assert np.linalg.norm(elbow - hand) == pytest.approx(0.3)
    assert np.linalg.norm(tip - elbow) == pytest.approx(0.2)
    assert np.linalg.norm(tip - hand) == pytest.approx(0.5)
