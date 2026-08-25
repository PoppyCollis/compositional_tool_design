"""In-sim tests for the SE(2) controller: does the arm actually stay flat?

First coverage of the panda-gym integration layer, which previously had none.
"""
import contextlib
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from panda_gym.pybullet import PyBullet

import panda_with_tool_urdf
import se2
from config import ArmConfig, DesignPriorConfig, SE2Config
from panda_with_tool import PandaWithTool

# Straight, right-angled and both bound extremes, matching spawn_demo_step3.py.
TAUS = [
    (0.3, 0.3, 0.0),
    (0.3, 0.3, np.pi / 2),
    (0.15, 0.5, DesignPriorConfig.PHI_MAX),
    (0.5, 0.15, -DesignPriorConfig.PHI_MAX),
]

# Settled: what the pose must be once the arm has stopped moving.
HEIGHT_TOL = 0.001            # m
TILT_TOL = np.deg2rad(1.0)    # rad

# In transit: the arm lags its target while accelerating, and a joint-space blend
# between two flat poses is not itself flat. Measured worst across all four designs
# and every manoeuvre is 2.2 mm and 0.13 deg, during a spin started from a corner of
# the workspace -- the case se2_demo.py exercises. 3 mm is still an eighth of the
# tool's 20 mm cross-section, so the tool comes nowhere near the table.
# SE2Config.POS_SCALE and YAW_SCALE are what keep this small; see the note there.
MOVING_HEIGHT_TOL = 0.003
MOVING_TILT_TOL = np.deg2rad(0.5)


@pytest.fixture(scope="module")
def sim():
    simulation = PyBullet(render_mode="rgb_array")
    yield simulation
    simulation.close()


@pytest.fixture
def robot(sim):
    """A fresh robot for tests that do not care which tau is loaded.

    Function-scoped so it cannot collide with the tests that load their own tau:
    only one PandaWithTool may exist at a time (see _remove).
    """
    with loaded(sim, TAUS[0]) as bot:
        yield bot


def _remove(sim):
    """Drop the loaded robot from the world.

    PandaWithTool always registers under the same body_name, so constructing a
    second one silently overwrites sim._bodies_idx while leaving the first body in
    the simulation. Without this the parametrised tests stack a dozen overlapping
    Pandas that collide with each other, which is both wrong and very slow.
    """
    name = "panda_with_tool"
    if name in sim._bodies_idx:
        sim.physics_client.removeBody(sim._bodies_idx.pop(name))


@contextlib.contextmanager
def loaded(sim, tau, seed=0):
    """A robot with the given tau, removed from the world afterwards."""
    _remove(sim)
    try:
        yield PandaWithTool(sim, tau, np_random=np.random.default_rng(seed))
    finally:
        _remove(sim)


# -- the geometric claims the whole design rests on ---------------------------


@pytest.mark.parametrize("tau", TAUS)
@pytest.mark.parametrize("yaw", [-1.5, -0.4, 0.0, 0.9, 1.5])
def test_closed_form_tip_matches_pybullet_fk(sim, tau, yaw):
    """se2.tip_from_hand must agree with the sim, or get_obs is lying to the policy."""
    with loaded(sim, tau) as robot:
        robot.set_se2((0.45, 0.0), yaw)
        predicted = se2.tip_from_hand(robot.get_hand_se2(), tau)
        assert np.allclose(predicted, robot.get_ee_position()[:2], atol=1e-4)


@pytest.mark.parametrize("tau", TAUS)
def test_tool_height_does_not_depend_on_the_design(sim, tau):
    """The claim that licenses a single fixed TOOL_Z for every tau.

    tip_hand z is TCP_OFFSET_Z regardless of (l1, l2, phi), so pinning the hand's
    height pins the tool's -- see se2.py's module docstring.
    """
    with loaded(sim, tau) as robot:
        for yaw in (-1.2, 0.0, 1.2):
            robot.set_se2((0.45, 0.05), yaw)
            assert robot.get_ee_position()[2] == pytest.approx(SE2Config.TOOL_Z, abs=HEIGHT_TOL)


@pytest.mark.parametrize("tau", TAUS)
def test_the_elbow_bends_in_the_ground_plane(sim, tau):
    """phi must hook sideways, not up or down.

    Both tool segments have to sit at the tool height; if the mount rotation were
    wrong the bend would leave the plane and phi would be useless for hooking.
    """
    with loaded(sim, tau) as robot:
        robot.set_se2((0.45, 0.0), 0.0)
        body_id = sim._bodies_idx[robot.body_name]
        elbow_z = sim.physics_client.getLinkState(body_id, robot.tool_tip_link)[4][2]
        assert elbow_z == pytest.approx(SE2Config.TOOL_Z, abs=HEIGHT_TOL)


# -- the controller -----------------------------------------------------------


def test_action_space_is_three_dimensional(robot):
    assert robot.action_space.shape == (3,)


def test_obs_is_finite_and_the_expected_width(robot):
    robot.reset()
    obs = robot.get_obs()
    assert obs.shape == (9,)
    assert np.all(np.isfinite(obs))


def test_obs_normalises_the_hand_into_the_unit_square(robot):
    """The hand is always inside the box, so its normalised position must be too."""
    for _ in range(20):
        robot.reset()
        assert np.all(np.abs(robot.get_obs()[:2]) <= 1.0 + 1e-6)


def test_reset_starts_inside_the_inner_box(robot):
    for _ in range(30):
        robot.reset()
        x, y, yaw = robot.get_hand_se2()
        assert robot.reset_workspace.contains((x, y), tol=0.005)
        assert abs(yaw) <= SE2Config.YAW_LIMIT + 1e-3


def test_reset_is_reproducible_under_a_fixed_seed(sim):
    poses = []
    for _ in range(2):
        with loaded(sim, TAUS[0], seed=7) as robot:
            poses.append([robot.reset() or robot.get_hand_se2() for _ in range(5)])
    assert np.allclose(poses[0], poses[1])


def test_an_action_moves_the_hand_the_way_it_was_asked(sim, robot):
    """+x action moves +x, +y moves +y, +yaw turns anticlockwise."""
    for axis, index in (("x", 0), ("y", 1)):
        robot.set_se2((0.5, 0.0), 0.0)
        before = robot.get_hand_se2()
        action = np.zeros(3)
        action[index] = 1.0
        for _ in range(10):
            robot.set_action(action)
            sim.step()
        after = robot.get_hand_se2()
        assert after[index] > before[index] + 0.01, f"{axis} action did not move {axis}"

    robot.set_se2((0.5, 0.0), 0.0)
    for _ in range(10):
        robot.set_action(np.array([0.0, 0.0, 1.0]))
        sim.step()
    assert robot.get_hand_se2()[2] > 0.01


# -- the test that catches an improvised tilt ---------------------------------


WALLS = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+yaw": (0.0, 0.0, 1.0),
    "-yaw": (0.0, 0.0, -1.0),
}


@pytest.mark.parametrize("name,action", WALLS.items())
def test_stays_flat_when_driven_into_a_wall(sim, robot, name, action):
    """Full throttle into each boundary for 300 steps, checking every step.

    This is the whole point of clipping before the solver rather than after. An
    unclipped target would have the IK buy extra reach by leaning the wrist, the
    tool would stop being parallel to the ground, and nothing would raise an error
    -- the flat-2D problem would quietly stop being flat.
    """
    robot.set_se2((0.5, 0.0), 0.0)
    action = np.array(action)
    for step in range(300):
        robot.set_action(action)
        sim.step()
        tip_z = robot.get_ee_position()[2]
        assert tip_z == pytest.approx(SE2Config.TOOL_Z, abs=MOVING_HEIGHT_TOL), \
            f"{name}: tool height drifted to {tip_z:.5f} at step {step}"
        assert robot.get_hand_tilt() < MOVING_TILT_TOL, \
            f"{name}: tool tilted {np.degrees(robot.get_hand_tilt()):.3f} deg at step {step}"

    # Pressed against the boundary and no longer moving, it must be properly flat,
    # not merely within the in-transit allowance.
    assert robot.get_ee_position()[2] == pytest.approx(SE2Config.TOOL_Z, abs=HEIGHT_TOL)
    assert robot.get_hand_tilt() < TILT_TOL


@pytest.mark.parametrize("name,action", WALLS.items())
def test_the_hand_never_leaves_the_box(sim, robot, name, action):
    robot.set_se2((0.5, 0.0), 0.0)
    for _ in range(300):
        robot.set_action(np.array(action))
        sim.step()
    x, y, yaw = robot.get_hand_se2()
    assert robot.workspace.contains((x, y), tol=0.005), f"{name}: escaped to {x:.3f},{y:.3f}"
    assert abs(yaw) <= SE2Config.YAW_LIMIT + np.deg2rad(1.0), f"{name}: yaw reached {yaw:.3f}"


@pytest.mark.parametrize("start", [
    (0.39, -0.36, 0.0), (0.61, 0.36, -1.0), (0.39, 0.36, 1.0), (0.61, -0.36, 0.0),
])
@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_stays_flat_spinning_in_the_corners(sim, robot, start, direction):
    """Spinning is the worst case for flatness, and the corners are the worst place.

    The wall tests all start from the middle of the workspace; this covers the case
    se2_demo.py found, where a spin begun from a corner peaks higher than a spin
    begun from the centre.
    """
    robot.set_se2(start[:2], start[2])
    for step in range(120):
        robot.set_action(np.array([0.0, 0.0, direction]))
        sim.step()
        tip_z = robot.get_ee_position()[2]
        assert tip_z == pytest.approx(SE2Config.TOOL_Z, abs=MOVING_HEIGHT_TOL), \
            f"height {tip_z:.5f} at step {step} from {start}"
        assert robot.get_hand_tilt() < MOVING_TILT_TOL


@pytest.mark.parametrize("start", [(0.5, 0.0, 0.0), (0.42, -0.3, 1.0), (0.6, 0.3, -1.2)])
def test_a_zero_action_holds_position(sim, robot, start):
    """Doing nothing must mean staying put.

    Guards the integrated target. Re-deriving the target from the measured pose
    each step feeds the IK solver's residual back into its own input, and the error
    compounds: that arrangement crept up to 25 cm over these same 300 steps.
    """
    robot.set_se2(start[:2], start[2])
    before = robot.get_hand_se2()
    for _ in range(300):
        robot.set_action(np.zeros(3))
        sim.step()
    after = robot.get_hand_se2()
    assert np.linalg.norm(after[:2] - before[:2]) < 0.005
    assert abs(se2.wrap_angle(after[2] - before[2])) < np.deg2rad(1.0)


def test_the_target_cannot_outrun_a_blocked_hand(sim, robot):
    """With the hand pinned, the target must stall within MAX_LAG of it.

    A wall is handled by the workspace clip; this is the other case -- an object in
    the way -- where nothing bounds the target except this clamp.
    """
    robot.set_se2((0.5, 0.0), 0.0)
    frozen = np.array([robot.get_joint_angle(joint=i) for i in range(7)])
    for _ in range(200):
        robot.set_action(np.array([1.0, 0.0, 0.0]))
        robot.set_joint_angles(frozen)  # stand in for an immovable obstruction
    lag = np.linalg.norm(robot._target[:2] - robot.get_hand_se2()[:2])
    assert lag <= SE2Config.MAX_LAG + 1e-6


def test_joints_stay_within_their_limits_under_load(sim, robot):
    """IK is given the limits; the motors enforce them anyway. Neither should bind."""
    limits = np.array(ArmConfig.JOINT_LIMITS)
    for action in WALLS.values():
        robot.set_se2((0.5, 0.0), 0.0)
        for _ in range(200):
            robot.set_action(np.array(action))
            sim.step()
            angles = np.array([robot.get_joint_angle(joint=i) for i in range(7)])
            assert np.all(angles >= limits[:, 0] - 1e-3)
            assert np.all(angles <= limits[:, 1] + 1e-3)


# -- constants vs. the live URDF ----------------------------------------------


def test_joint_limits_match_the_live_urdf():
    """ArmConfig.JOINT_LIMITS is duplicated from the URDF; catch it drifting.

    Same guard _check_tcp_offset gives the gripper constants.
    """
    panda_with_tool_urdf.ensure_assets()
    root = ET.parse(panda_with_tool_urdf.PANDA_URDF_PATH).getroot()
    live = [
        (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
        for name in (f"panda_joint{i}" for i in range(1, 8))
        for j in root.iter("joint") if j.get("name") == name
    ]
    assert np.allclose(live, ArmConfig.JOINT_LIMITS)
