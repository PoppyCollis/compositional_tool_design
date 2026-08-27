"""Analytic tests for task_space.py and se2.tip_polar. No sim, no PyBullet body."""
import numpy as np
import pytest

import se2
import task_space as ts
from config import DesignPriorConfig, SE2Config

BOX = se2.Box(x_min=0.1, x_max=0.5, y_min=-0.2, y_max=0.4)
SYMMETRIC_BOX = se2.Box(x_min=0.38, x_max=0.62, y_min=-0.38, y_max=0.38)
PSI = np.pi / 2

L_MIN, L_MAX = DesignPriorConfig.L_MIN, DesignPriorConfig.L_MAX
PHI_MAX = DesignPriorConfig.PHI_MAX

TAUS = [
    (L_MAX, L_MAX, 0.0),
    (L_MAX, L_MAX, np.pi / 2),
    (L_MIN, L_MAX, PHI_MAX),
    (L_MAX, L_MIN, -PHI_MAX),
    (L_MIN, L_MIN, PHI_MAX),
]


# -- box_distance -------------------------------------------------------------

def test_box_distance_is_zero_inside_and_on_the_boundary():
    inside = np.array([BOX.centre, (BOX.x_min, BOX.y_min), (BOX.x_max, BOX.y_max)])
    assert np.allclose(ts.box_distance(inside, BOX), 0.0)


def test_box_distance_off_an_edge_is_the_perpendicular_gap():
    assert ts.box_distance((BOX.x_max + 0.07, 0.0), BOX) == pytest.approx(0.07)
    assert ts.box_distance((0.3, BOX.y_min - 0.02), BOX) == pytest.approx(0.02)


def test_box_distance_off_a_corner_is_the_diagonal_not_an_axis_gap():
    """The clamp-then-norm form has to pick the vertex, not the nearer edge."""
    q = (BOX.x_max + 0.03, BOX.y_max + 0.04)
    assert ts.box_distance(q, BOX) == pytest.approx(0.05)


def test_box_distance_broadcasts_over_a_grid():
    _, _, points = ts.grid((0.0, 0.6), (-0.3, 0.5), 0.05)
    distances = ts.box_distance(points, BOX)
    assert distances.shape == points.shape[:-1]


# -- se2.tip_polar ------------------------------------------------------------

@pytest.mark.parametrize("tau", TAUS)
def test_tip_polar_round_trips_tip_offset(tau):
    radius, bearing = se2.tip_polar(tau)
    ox, oy = se2.tip_offset(tau)
    assert radius == pytest.approx(np.hypot(ox, oy))
    assert np.allclose([radius * np.cos(bearing), -radius * np.sin(bearing)], [ox, oy])


@pytest.mark.parametrize("tau", TAUS)
@pytest.mark.parametrize("yaw", [-1.5, -0.4, 0.0, 0.4, 1.5])
def test_tip_polar_reproduces_tip_from_hand(tau, yaw):
    """The whole point of the polar form: the tip is one rotation of the yaw."""
    hand = np.array([0.42, -0.11])
    radius, bearing = se2.tip_polar(tau)
    expected = hand + radius * np.array([np.cos(yaw + bearing), np.sin(yaw + bearing)])
    assert np.allclose(se2.tip_from_hand((hand[0], hand[1], yaw), tau), expected)


@pytest.mark.parametrize("phi", [0.0, 0.7, np.pi / 2, PHI_MAX])
def test_symmetric_tool_has_the_closed_form_polar(phi):
    """l1 == l2 == L collapses to radius = 2*L*cos(phi/2), bearing = -phi/2."""
    radius, bearing = se2.tip_polar((L_MAX, L_MAX, phi))
    assert radius == pytest.approx(2 * L_MAX * np.cos(phi / 2))
    assert bearing == pytest.approx(-phi / 2)


def test_a_straight_rod_points_dead_ahead_whatever_its_length():
    for length in (L_MIN, L_MAX):
        radius, bearing = se2.tip_polar((length, length, 0.0))
        assert radius == pytest.approx(2 * length)
        assert bearing == pytest.approx(0.0)


# -- tip_reachable ------------------------------------------------------------

@pytest.mark.parametrize("tau", TAUS)
def test_every_pose_the_controller_can_hold_lands_a_tip_the_closed_form_accepts(tau):
    """No false negatives, checked exactly: enumerate legal (hand, yaw) poses, put
    the tip where se2.tip_from_hand puts it, and require tip_reachable to accept it.

    Grid-free on purpose -- snapping a tip to the nearest cell moves it by up to
    half a diagonal, which is enough to carry a boundary tip out of the region and
    make a correct implementation look wrong.
    """
    rng = np.random.default_rng(0)
    hands = np.stack([rng.uniform(BOX.x_min, BOX.x_max, 4000),
                      rng.uniform(BOX.y_min, BOX.y_max, 4000),
                      rng.uniform(-PSI, PSI, 4000)], axis=-1)
    tips = np.array([se2.tip_from_hand(h, tau) for h in hands])
    assert np.all(ts.tip_reachable(tips, BOX, tau, PSI, tol=1e-9))


@pytest.mark.parametrize("tau", TAUS)
def test_the_closed_form_accepts_nothing_a_brute_force_hand_sweep_misses(tau):
    """No false positives: brute-force the poses onto a grid and require every
    analytically-reachable cell to have been hit.

    Cells on the region boundary are exempt -- the brute force samples poses, so it
    necessarily leaves a sliver of the edge unvisited that the continuous form
    includes. The interior has no such excuse.
    """
    resolution = 0.01
    xs, ys, points = ts.grid((-0.2, 1.2), (-0.9, 0.9), resolution)
    analytic = ts.tip_reachable(points, BOX, tau, PSI, tol=0.0)

    hand_x = np.arange(BOX.x_min, BOX.x_max + 1e-9, resolution / 4)
    hand_y = np.arange(BOX.y_min, BOX.y_max + 1e-9, resolution / 4)
    yaws = np.linspace(-PSI, PSI, 361)
    radius, bearing = se2.tip_polar(tau)
    offsets = radius * np.stack([np.cos(yaws + bearing), np.sin(yaws + bearing)], -1)
    hands = np.stack(np.meshgrid(hand_x, hand_y, indexing="ij"), -1).reshape(-1, 2)
    tips = (hands[:, None, :] + offsets).reshape(-1, 2)

    brute = np.zeros_like(analytic)
    i = np.rint((tips[:, 0] - xs[0]) / resolution).astype(int)
    j = np.rint((tips[:, 1] - ys[0]) / resolution).astype(int)
    keep = (i >= 0) & (i < len(xs)) & (j >= 0) & (j < len(ys))
    brute[i[keep], j[keep]] = True

    missed = (analytic & ~_boundary(analytic)) & ~brute
    assert missed.sum() == 0, f"{missed.sum()} interior cells the brute force never hit"


def _boundary(mask):
    """Cells of a True region that touch a False cell (4-connected)."""
    padded = np.pad(mask, 1, constant_values=False)
    neighbours = (padded[:-2, 1:-1] & padded[2:, 1:-1]
                  & padded[1:-1, :-2] & padded[1:-1, 2:])
    return mask & ~neighbours


@pytest.mark.parametrize("phi", [0.3, 1.0, np.pi / 2, PHI_MAX])
def test_mirrored_designs_give_mirrored_regions(phi):
    """+phi and -phi are mirror-image tools, so over a y-symmetric box their reach
    regions are y-mirrors of each other. Catches a dropped sign in the bearing."""
    _, ys, points = ts.grid((0.0, 1.1), (-0.9, 0.9), 0.02)
    assert np.allclose(ys, -ys[::-1]), "grid must be y-symmetric for this test"
    left = ts.tip_reachable(points, SYMMETRIC_BOX, (L_MIN, L_MAX, phi), PSI, tol=0.03)
    right = ts.tip_reachable(points, SYMMETRIC_BOX, (L_MIN, L_MAX, -phi), PSI, tol=0.03)
    assert np.array_equal(left, right[:, ::-1])


def test_a_longer_tool_cannot_reach_a_target_close_to_the_hand():
    """The tip is on a circle of radius R, not in a disk of radius R.

    Reach is not monotone in tool length: the longest design in the prior only just
    fails to touch the workspace centre, while a short one has no trouble.
    """
    centre = SE2Config.WORKSPACE.centre
    longest = (L_MAX, L_MAX, 0.0)
    shortest = (L_MIN, L_MIN, 0.0)
    assert se2.tip_polar(longest)[0] > se2.tip_polar(shortest)[0]
    assert not ts.tip_reachable(centre, SE2Config.WORKSPACE, longest, PSI, tol=0.0)
    assert ts.tip_reachable(centre, SE2Config.WORKSPACE, shortest, PSI, tol=0.0)


def test_the_tolerance_dilates_the_region_monotonically():
    _, _, points = ts.grid((0.0, 1.2), (-0.9, 0.9), 0.02)
    tau = (L_MIN, L_MAX, 0.5)
    tight = ts.tip_reachable(points, BOX, tau, PSI, tol=0.0)
    loose = ts.tip_reachable(points, BOX, tau, PSI, tol=0.03)
    assert np.all(loose[tight]), "dilating by r_obj must not drop any cell"
    assert loose.sum() > tight.sum()


def test_a_tighter_yaw_limit_never_grows_the_region():
    _, _, points = ts.grid((0.0, 1.2), (-0.9, 0.9), 0.02)
    tau = (L_MAX, L_MIN, 1.0)
    narrow = ts.tip_reachable(points, BOX, tau, np.pi / 4, tol=0.0)
    wide = ts.tip_reachable(points, BOX, tau, PSI, tol=0.0)
    assert np.all(wide[narrow])


def _agree_off_the_boundary(a, b):
    """Whether two masks are the same set except at cells touching an edge.

    reach_mask quantises each arc offset to the nearest cell, and a grid built by
    np.arange is only symmetric to ~1e-15, which a `<=` comparison can resolve. Both
    move the boundary by at most a cell, and neither is a bug in the set itself.
    """
    edge = _boundary(a) | _boundary(b) | _boundary(~a) | _boundary(~b)
    return not np.any((a ^ b) & ~edge)


# -- reach_mask ---------------------------------------------------------------

@pytest.mark.parametrize("tau", TAUS)
def test_reach_mask_agrees_with_tip_reachable_away_from_the_boundary(tau):
    """The fast rasteriser and the exact predicate must be the same set.

    reach_mask quantises each arc offset to the nearest cell, so the two can only
    be required to agree away from the edge; every disagreement must be a cell that
    touches the boundary of one region or the other.
    """
    xs, ys, points = ts.grid((0.0, 1.1), (-0.85, 0.85), 0.005)
    fast = ts.reach_mask(xs, ys, SE2Config.WORKSPACE, tau, PSI, tol=0.03)
    exact = ts.tip_reachable(points, SE2Config.WORKSPACE, tau, PSI, tol=0.03)
    assert _agree_off_the_boundary(fast, exact)


@pytest.mark.parametrize("tau", TAUS)
def test_reach_mask_area_matches_tip_reachable(tau):
    """Boundary quantisation must not bias the region's size either way."""
    xs, ys, points = ts.grid((0.0, 1.1), (-0.85, 0.85), 0.005)
    fast = ts.reach_mask(xs, ys, SE2Config.WORKSPACE, tau, PSI, tol=0.03)
    exact = ts.tip_reachable(points, SE2Config.WORKSPACE, tau, PSI, tol=0.03)
    assert fast.sum() == pytest.approx(exact.sum(), rel=0.01)


def test_reach_mask_handles_an_offset_that_shifts_the_grid_clean_off_itself():
    """A tool whose reach exceeds the mapped extent must not index out of bounds."""
    xs, ys, _ = ts.grid((0.4, 0.5), (-0.05, 0.05), 0.01)
    mask = ts.reach_mask(xs, ys, SE2Config.WORKSPACE, (L_MAX, L_MAX, 0.0), PSI, tol=0.0)
    assert mask.shape == (len(xs), len(ys))


# -- hand_reachable -----------------------------------------------------------

def test_hand_reachable_is_the_box_dilated_by_the_radius():
    radius = 0.075
    assert ts.hand_reachable(BOX.centre, BOX, radius)
    assert ts.hand_reachable((BOX.x_max + radius - 1e-6, 0.0), BOX, radius)
    assert not ts.hand_reachable((BOX.x_max + radius + 1e-6, 0.0), BOX, radius)


# -- hand_pose_for_tip --------------------------------------------------------

@pytest.mark.parametrize("tau", TAUS)
def test_hand_pose_for_tip_agrees_with_tip_reachable_and_lands_on_target(tau):
    """The witness and the verdict must never disagree, and the witness must work."""
    tol = 0.03
    _, _, points = ts.grid((0.0, 1.2), (-0.9, 0.9), 0.05)
    reachable = ts.tip_reachable(points, BOX, tau, PSI, tol=tol)
    flat = points.reshape(-1, 2)
    for q, expected in zip(flat, reachable.reshape(-1)):
        pose = ts.hand_pose_for_tip(q, BOX, tau, PSI, tol=tol)
        assert (pose is not None) == bool(expected), f"disagreement at {q}"
        if pose is None:
            continue
        assert BOX.contains(pose[:2], tol=1e-9)
        assert abs(pose[2]) <= PSI + 1e-9
        assert np.linalg.norm(se2.tip_from_hand(pose, tau) - q) <= tol + 1e-6


def test_hand_pose_for_tip_returns_none_far_outside_the_region():
    assert ts.hand_pose_for_tip((5.0, 5.0), BOX, TAUS[0], PSI, tol=0.03) is None


def test_hand_pose_for_tip_prefers_a_pose_with_margin():
    """Of the bearings that work it should not pick one pinned to a wall."""
    tau = (L_MIN, L_MIN, 0.0)
    radius, _ = se2.tip_polar(tau)
    q = BOX.centre + np.array([radius, 0.0])  # reachable from a whole arc of hands
    pose = ts.hand_pose_for_tip(q, BOX, tau, PSI, tol=0.0)
    wall = min(pose[0] - BOX.x_min, BOX.x_max - pose[0],
               pose[1] - BOX.y_min, BOX.y_max - pose[1])
    assert wall > 1e-3


# -- coverage -----------------------------------------------------------------

def test_coverage_is_the_fraction_of_designs_that_reach():
    xs, ys, _ = ts.grid((0.0, 1.2), (-0.9, 0.9), 0.01)
    taus = TAUS[:3]
    frac = ts.coverage(xs, ys, BOX, taus, PSI, tol=0.03)
    expected = sum(ts.reach_mask(xs, ys, BOX, t, PSI, tol=0.03) for t in taus) / len(taus)
    assert np.allclose(frac, expected)
    assert frac.min() >= 0.0 and frac.max() <= 1.0


def test_coverage_of_a_single_design_is_its_own_indicator():
    xs, ys, _ = ts.grid((0.0, 1.2), (-0.9, 0.9), 0.01)
    frac = ts.coverage(xs, ys, BOX, [TAUS[0]], PSI, tol=0.03)
    assert np.array_equal(frac.astype(bool), ts.reach_mask(xs, ys, BOX, TAUS[0], PSI, tol=0.03))


def test_coverage_of_mirrored_designs_is_symmetric():
    """Over a y-symmetric box, a prior closed under phi -> -phi maps symmetrically."""
    xs, ys, _ = ts.grid((0.0, 1.1), (-0.9, 0.9), 0.01)
    taus = [(L_MIN, L_MAX, 1.0), (L_MIN, L_MAX, -1.0), (L_MAX, L_MIN, 0.4), (L_MAX, L_MIN, -0.4)]
    frac = ts.coverage(xs, ys, SYMMETRIC_BOX, taus, PSI, tol=0.03)
    for level in np.unique(frac):
        assert _agree_off_the_boundary(frac >= level, frac[:, ::-1] >= level)


def test_coverage_rejects_an_empty_design_set():
    xs, ys, _ = ts.grid((0.0, 0.2), (-0.1, 0.1), 0.1)
    with pytest.raises(ValueError):
        ts.coverage(xs, ys, BOX, [], PSI)


# -- largest_rectangle / box_for_mask -----------------------------------------

def test_largest_rectangle_of_an_all_true_mask_is_the_whole_mask():
    assert ts.largest_rectangle(np.ones((4, 6), dtype=bool)) == (0, 3, 0, 5)


def test_largest_rectangle_of_an_empty_mask_is_none():
    assert ts.largest_rectangle(np.zeros((4, 6), dtype=bool)) is None


def test_largest_rectangle_prefers_area_over_either_dimension():
    """A 2x5 block beats a 4x2 block, which a greedy row-or-column scan would miss."""
    mask = np.zeros((6, 8), dtype=bool)
    mask[0:4, 0:2] = True   # area 8
    mask[0:2, 3:8] = True   # area 10
    assert ts.largest_rectangle(mask) == (0, 1, 3, 7)


def test_largest_rectangle_ignores_a_hole():
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False
    i0, i1, j0, j1 = ts.largest_rectangle(mask)
    assert not (i0 <= 2 <= i1 and j0 <= 2 <= j1)
    assert (i1 - i0 + 1) * (j1 - j0 + 1) == 10  # the 2x5 half either side of the hole


def test_box_for_mask_returns_the_rectangle_in_metres():
    xs, ys, _ = ts.grid((0.0, 0.4), (-0.2, 0.2), 0.1)
    mask = np.zeros((len(xs), len(ys)), dtype=bool)
    mask[1:3, 2:4] = True
    box = ts.box_for_mask(mask, xs, ys)
    assert (box.x_min, box.x_max) == pytest.approx((xs[1], xs[2]))
    assert (box.y_min, box.y_max) == pytest.approx((ys[2], ys[3]))


def test_box_for_mask_of_an_empty_mask_is_none():
    xs, ys, _ = ts.grid((0.0, 0.4), (-0.2, 0.2), 0.1)
    assert ts.box_for_mask(np.zeros((len(xs), len(ys)), dtype=bool), xs, ys) is None


# -- grid ---------------------------------------------------------------------

def test_grid_indexes_row_as_x_column_as_y():
    xs, ys, points = ts.grid((0.0, 0.2), (-0.1, 0.1), 0.1)
    assert points.shape == (len(xs), len(ys), 2)
    assert np.allclose(points[2, 0], (xs[2], ys[0]))
    assert np.allclose(points[0, 2], (xs[0], ys[2]))


def test_grid_includes_the_upper_bound():
    xs, ys, _ = ts.grid((0.0, 0.3), (-0.2, 0.2), 0.1)
    assert xs[-1] == pytest.approx(0.3)
    assert ys[-1] == pytest.approx(0.2)
