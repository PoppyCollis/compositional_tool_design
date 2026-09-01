"""In-sim tests for the reach env, and the sim-vs-analytic assertion it exists for.

The load-bearing test is test_reset_obs_matches_the_analytic_initial_state_map. Every
other test here guards a property the design objective quietly depends on: that the
episode runs to a fixed horizon, that the reward is on metres, and that the reset
distribution the env draws from is the one initial_state.sample_xi documents.
"""
import numpy as np
import pytest
import torch

import initial_state
import se2
import task as task_mod
from config import DesignPriorConfig, SE2Config, TaskConfig
from reach_env import ReachEnv

# Corners of the design prior: straight, right-angled, and both fold extremes.
# Not tests/test_se2_control.TAUS, whose 0.3-0.5 m links predate the current
# DesignPriorConfig bounds of [0.1, 0.2] and would test designs p(tau) never draws.
TAUS = [
    (0.1, 0.1, 0.0),
    (0.15, 0.15, np.pi / 2),
    (0.1, 0.2, DesignPriorConfig.PHI_MAX),
    (0.2, 0.1, -DesignPriorConfig.PHI_MAX),
]

# Slices h reproduces from constants of the reset. They must agree to the float32
# cast the observation makes and nothing more; measured worst is 3.0e-8.
#
# D_TARGET_OBJ is vacuously zero here: task.object_start pins the reaching object at
# the target, so the slice is identically (0, 0) and this check cannot fail for the
# right reason. It is kept because the layout is final before sweep and push exist,
# which are what will actually exercise it -- do not read a pass as evidence.
EXACT_SLICES = {
    "HAND_VEL": initial_state.HAND_VEL,
    "OBJ_XY": initial_state.OBJ_XY,
    "D_TARGET_OBJ": initial_state.D_TARGET_OBJ,
    "TASK_BLOCK": initial_state.TASK_BLOCK,
    # Both sides are literally 0.0: reset zeroes _elapsed before building the
    # observation, and h emits the start of an episode by construction.
    "PHASE": initial_state.PHASE,
}
EXACT_TOL = 1e-6

# Slices derived from the hand pose. h uses the *commanded* xi.hand_se2; the env
# reads the *measured* pose back, and set_se2's IK lands short of the command. This
# is the arm's tracking residual, not a geometry error -- the same trap
# reach_sweep.py --verify carries a note about.
#
# Measured 2026-09-01 over 4 designs x 25 resets: hand position residual 1.74 mm,
# yaw residual 0.037 deg, giving a worst observation difference of 1.50e-3 once
# divided by SCENE_BOX.scale (which is exactly 1.0 m). The yaw residual reaches the
# tip through the tool's lever arm and contributes ~2.6e-4 at the longest design, so
# position dominates. Tolerance is 2x the measured worst.
#
# What that buys, checked by injecting errors at 3e-3 (2026-09-01): a 5 mm tip
# displacement and a tau whose l1 is 5 mm wrong are both caught; 2 mm of either is
# not. The floor is the arm's residual, not a slack tolerance -- tightening it below
# ~3e-3 makes the test flap on IK noise instead of catching anything more.
#
# ELBOW_XY rides on the same residual as TIP_XY and sits on a shorter lever arm
# (l1 <= 0.2 m against the tip's full R), so the yaw residual reaches it less; the
# hand's position error dominates it just as it does the tip's.
POSE_SLICES = {
    "HAND_XY": initial_state.HAND_XY,
    "HAND_YAW": initial_state.HAND_YAW,
    "ELBOW_XY": initial_state.ELBOW_XY,
    "TIP_XY": initial_state.TIP_XY,
    "D_OBJ_TIP": initial_state.D_OBJ_TIP,
}
POSE_TOL = 3e-3


@pytest.fixture(scope="module")
def env():
    """One env for the tests that do not care which design is loaded."""
    instance = ReachEnv(TAUS[0])
    yield instance
    instance.close()


def reach_task(target):
    """A REACH task g with an explicit target, bypassing p(g)'s uniform draw."""
    w_reach, w_trans, rho = task_mod.TASK_PARAMS[task_mod.TaskType.REACH]
    return task_mod.Task(
        task_id=task_mod.TaskType.REACH,
        target=np.asarray(target, dtype=float),
        r_obj=TaskConfig.R_OBJ,
        rho=rho,
        w_reach=w_reach,
        w_trans=w_trans,
    )


@pytest.mark.parametrize("tau", TAUS)
def test_reset_obs_matches_the_analytic_initial_state_map(tau):
    """env.reset() is initial_state.h evaluated at the same (tau, g, xi).

    The whole design objective rests on this: f(tau, g) = E_xi[V(h(tau, g, xi))]
    scores states the policy is supposed to have been trained on, and until now the
    two agreed only because both call Box.normalise_point on the same box.
    """
    rng = np.random.default_rng(0)
    environment = ReachEnv(tau)
    try:
        for _ in range(10):
            task = task_mod.sample_task(rng)
            xi = initial_state.sample_xi(rng, task)
            obs, _ = environment.reset(seed=0, options={"task": task, "xi": xi})
            expected = initial_state.h(
                torch.as_tensor(np.asarray(tau, dtype=float)), task, xi
            ).numpy()

            for name, sl in EXACT_SLICES.items():
                np.testing.assert_allclose(
                    obs[sl].astype(float), expected[sl], atol=EXACT_TOL,
                    err_msg=f"{name} should be a constant of the reset, not a measurement",
                )
            for name, sl in POSE_SLICES.items():
                np.testing.assert_allclose(
                    obs[sl].astype(float), expected[sl], atol=POSE_TOL,
                    err_msg=f"{name} disagrees by more than the arm's IK residual",
                )
    finally:
        environment.close()


def test_reset_pose_distribution_matches_panda_with_tool_reset(env):
    """The env bypasses robot.reset(); the two draws must still share a support.

    reset() places the arm with sample_xi + set_se2 rather than robot.reset(), so
    that the sim reset and the xi h is evaluated at are the same draw. That leaves
    PandaWithTool.reset -- still used by the demos and sweeps -- as the one place the
    distribution is written twice.
    """
    assert env.robot.reset_workspace == SE2Config.WORKSPACE.shrink(SE2Config.RESET_MARGIN)

    rng = np.random.default_rng(0)
    task = task_mod.sample_task(rng)
    for _ in range(200):
        xi = initial_state.sample_xi(rng, task)
        assert env.robot.reset_workspace.contains(xi.hand_se2[:2], tol=1e-12)
        assert abs(xi.hand_se2[2]) <= SE2Config.YAW_LIMIT


def test_episode_runs_to_horizon_without_terminating(env):
    """Fixed HORIZON, no early termination, and the observation stays in its box.

    Terminating on success would truncate the accumulating negative reward, making V
    jump at the rho boundary; V is read as an energy across designs, so the returns
    have to stay comparable. See TaskConfig.HORIZON.
    """
    env.action_space.seed(0)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)

    assert obs[initial_state.PHASE] == pytest.approx(0.0)

    for step in range(1, TaskConfig.HORIZON + 1):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert terminated is False
        assert truncated == (step == TaskConfig.HORIZON)
        assert env.observation_space.contains(obs), f"observation left its box at step {step}"
        assert reward <= 0.0
        # The phase is what makes a return-to-go from this state well defined; it has
        # to advance with the step count, not sit at whatever reset left it.
        assert obs[initial_state.PHASE] == pytest.approx(
            step / TaskConfig.HORIZON, abs=1e-6
        )


def test_success_is_reported_in_info_but_never_terminates(env):
    """A tip sitting on the target scores is_success without ending the episode."""
    rng = np.random.default_rng(1)
    xi_pose = initial_state.sample_xi(rng, reach_task(np.zeros(2)))
    on_target = se2.tip_from_hand(xi_pose.hand_se2, env.tau)

    task = reach_task(on_target)
    xi = initial_state.Xi(hand_se2=xi_pose.hand_se2, obj_xy=task_mod.object_start(task, rng))
    _, info = env.reset(seed=0, options={"task": task, "xi": xi})
    assert info["is_success"], "tip placed on the target should score a success"

    _, _, terminated, truncated, info = env.step(np.zeros(3))
    assert info["is_success"]
    assert terminated is False
    assert truncated is False


def test_far_target_is_not_a_success(env):
    """The corner of SCENE_BOX is reachable by no design, so it must score False."""
    task = reach_task([TaskConfig.SCENE_BOX.x_max, TaskConfig.SCENE_BOX.y_max])
    _, info = env.reset(seed=0, options={"task": task})
    assert not info["is_success"]


def test_reward_is_computed_on_metres_not_normalised_units(env):
    """The env's reward equals task.reward on the physical tip and object positions.

    rho is a distance in metres; a reward accidentally computed on scene-normalised
    coordinates would be off by SCENE_BOX.scale and put the success tolerance and
    the objective on different footings.
    """
    env.action_space.seed(2)
    env.reset(seed=2)
    for _ in range(20):
        _, reward, _, _, _ = env.step(env.action_space.sample())
        tip = se2.tip_from_hand(env.robot.get_hand_se2(), env.tau)
        obj = np.asarray(env.sim.get_base_position("object"), dtype=float)[:2]
        assert reward == pytest.approx(task_mod.reward(env._task, tip, obj))
        # Reaching is w_trans = 0, so the reward is exactly minus the tip-object gap.
        assert reward == pytest.approx(-np.linalg.norm(tip - obj))


def test_obs_dim_and_robot_dim_agree_with_a_sim_up(env):
    """initial_state's declared widths match what the sim actually produces.

    initial_state.py cannot import PandaWithTool without pulling PyBullet into a
    pure-torch module, so ROBOT_DIM is a declaration there and can only be checked
    here. ReachEnv.__init__ raises on a mismatch; this pins it in the suite too.
    """
    assert len(env.robot.get_obs()) == initial_state.ROBOT_DIM
    assert len(env._get_obs()) == initial_state.OBS_DIM
    assert env.observation_space.shape == (initial_state.OBS_DIM,)


def test_reset_is_deterministic_given_a_seed(env):
    """Same seed, same episode: p(g) and p(xi | g) both draw from env.np_random."""
    first, _ = env.reset(seed=7)
    second, _ = env.reset(seed=7)
    np.testing.assert_array_equal(first, second)


def test_passes_the_gymnasium_env_checker():
    """Smoke test against gymnasium's own API conformance checks."""
    from gymnasium.utils.env_checker import check_env

    environment = ReachEnv(TAUS[0])
    try:
        check_env(environment, skip_render_check=True)
    finally:
        environment.close()


def test_two_fresh_envs_do_not_share_an_episode_stream():
    """Unseeded envs must draw independently, or parallel workers replay each other.

    gymnasium only reseeds ``np_random`` when ``seed is not None``, so a literal seed
    in ``__init__`` would leave every identically-constructed env on one stream --
    N SubprocVecEnv workers running the same episodes until something seeded them.
    """
    first, second = ReachEnv(TAUS[0]), ReachEnv(TAUS[0])
    try:
        assert not np.array_equal(first.reset()[0], second.reset()[0])
        assert np.array_equal(first.reset(seed=3)[0], second.reset(seed=3)[0])
    finally:
        first.close()
        second.close()
