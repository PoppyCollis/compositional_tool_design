"""Analytic tests for initial_state.py, the map h. No sim, no PyBullet body."""
import numpy as np
import pytest
import torch

import initial_state as ini
import se2
import task
import tool_geometry as geom
import task_space as ts
from config import DesignPriorConfig, SE2Config, TaskConfig

L_MIN, L_MAX = DesignPriorConfig.L_MIN, DesignPriorConfig.L_MAX
PHI_MAX = DesignPriorConfig.PHI_MAX

TAUS = [
    (L_MAX, L_MAX, 0.0),
    (L_MAX, L_MAX, np.pi / 2),
    (L_MIN, L_MAX, PHI_MAX),
    (L_MAX, L_MIN, -PHI_MAX),
    (L_MIN, L_MIN, 0.7),
]

BOX = TaskConfig.SCENE_BOX


def rng(seed=0):
    return np.random.default_rng(seed)


def fixture(seed=0):
    """A (task, xi) pair, drawn the way the env will draw them."""
    generator = rng(seed)
    g = task.sample_task(generator)
    return g, ini.sample_xi(generator, g)


def tensor(tau):
    return torch.tensor(tau, dtype=torch.float64, requires_grad=True)


# -- the torch kinematics mirror ----------------------------------------------


def test_tip_offset_matches_the_numpy_closed_form():
    for tau in TAUS:
        got = ini.tip_offset_torch(torch.tensor(tau, dtype=torch.float64)).numpy()
        assert np.allclose(got, se2.tip_offset(tau), atol=1e-12)


def test_tip_from_hand_matches_the_numpy_closed_form():
    """Binds the only differentiable path in the stack to the audited original."""
    for tau in TAUS:
        for hand in [(0.4, 0.0, 0.0), (0.5, -0.2, 1.1), (0.62, 0.38, -np.pi / 2)]:
            got = ini.tip_from_hand_torch(hand, torch.tensor(tau, dtype=torch.float64))
            assert np.allclose(got.numpy(), se2.tip_from_hand(hand, tau), atol=1e-12)


def test_elbow_from_hand_matches_the_numpy_closed_form():
    """The second differentiable path, pinned to its numpy original like the first."""
    for tau in TAUS:
        for hand in [(0.4, 0.0, 0.0), (0.5, -0.2, 1.1), (0.62, 0.38, -np.pi / 2)]:
            got = ini.elbow_from_hand_torch(hand, torch.tensor(tau, dtype=torch.float64))
            assert np.allclose(got.numpy(), se2.elbow_from_hand(hand, tau), atol=1e-12)


def test_tip_kinematics_accept_a_batch_unlike_tool_geometry():
    taus = torch.tensor(TAUS, dtype=torch.float64)
    got = ini.tip_offset_torch(taus).numpy()
    assert got.shape == (len(TAUS), 2)
    for row, tau in zip(got, TAUS):
        assert np.allclose(row, se2.tip_offset(tau), atol=1e-12)


# -- shape and layout ---------------------------------------------------------


def test_h_returns_the_documented_width():
    g, xi = fixture()
    assert ini.h(tensor(TAUS[0]), g, xi).shape == (ini.OBS_DIM,)
    assert ini.OBS_DIM == 24


def test_h_rank_follows_tau():
    g, xi = fixture()
    batched = ini.h(torch.tensor(TAUS, dtype=torch.float64), g, xi)
    assert batched.shape == (len(TAUS), ini.OBS_DIM)


def test_h_rejects_malformed_tau():
    g, xi = fixture()
    for bad in [torch.zeros(2), torch.zeros(4), torch.zeros(2, 2, 3)]:
        with pytest.raises(ValueError):
            ini.h(bad, g, xi)


def test_task_block_is_exactly_the_task_encoding():
    g, xi = fixture()
    x1 = ini.h(tensor(TAUS[0]), g, xi).detach().numpy()
    assert np.allclose(x1[ini.TASK_BLOCK], task.encode(g))


def test_hand_and_velocity_blocks_are_the_reset_pose_at_rest():
    g, xi = fixture()
    x1 = ini.h(tensor(TAUS[0]), g, xi).detach().numpy()
    assert np.allclose(x1[ini.HAND_XY], BOX.normalise_point(xi.hand_se2[:2]))
    assert np.allclose(x1[ini.HAND_YAW], [np.cos(xi.hand_se2[2]), np.sin(xi.hand_se2[2])])
    assert np.allclose(x1[ini.HAND_VEL], 0.0)


def test_tip_block_is_the_scene_normalised_closed_form():
    """The join between this module and se2/panda_with_tool: one map, one box."""
    g, xi = fixture()
    for tau in TAUS:
        x1 = ini.h(tensor(tau), g, xi).detach().numpy()
        expected = BOX.normalise_point(se2.tip_from_hand(xi.hand_se2, tau))
        assert np.allclose(x1[ini.TIP_XY], expected, atol=1e-12)


def test_elbow_block_is_the_scene_normalised_closed_form():
    """The elbow is what makes tau recoverable, so it must be the same elbow."""
    g, xi = fixture()
    for tau in TAUS:
        x1 = ini.h(tensor(tau), g, xi).detach().numpy()
        expected = BOX.normalise_point(se2.elbow_from_hand(xi.hand_se2, tau))
        assert np.allclose(x1[ini.ELBOW_XY], expected, atol=1e-12)


def test_the_tool_is_a_two_segment_polyline_of_the_right_shape():
    """hand -> elbow -> tip, in metres: |l1|, |l2| and the interior angle at the bend.

    There is no elbow *link* in the URDF -- the tool is one link carrying two boxes --
    so this is the geometric pin the sim cannot provide. It is also the statement the
    recoverability argument rests on: these three numbers are exactly tau.
    """
    g, xi = fixture()
    for tau in TAUS:
        l1, l2, phi = tau
        x1 = ini.h(tensor(tau), g, xi).detach().numpy()
        hand = x1[ini.HAND_XY] * BOX.scale
        elbow = x1[ini.ELBOW_XY] * BOX.scale
        tip = x1[ini.TIP_XY] * BOX.scale

        first, second = elbow - hand, tip - elbow
        assert np.linalg.norm(first) == pytest.approx(l1, abs=1e-12)
        assert np.linalg.norm(second) == pytest.approx(l2, abs=1e-12)

        # atan2, not arccos: the deflection is |phi|, TAUS carries a straight rod at
        # phi=0, and arccos has a square-root singularity exactly there -- 1e-16 in
        # the cosine becomes 1e-8 in the angle. Same argument as se2.tilt_from_matrix.
        cross = first[0] * second[1] - first[1] * second[0]
        deflection = np.arctan2(abs(cross), first @ second)
        assert deflection == pytest.approx(abs(phi), abs=1e-12)
        assert np.pi - deflection == pytest.approx(geom.interior_angle(phi), abs=1e-12)


def test_a_straight_rod_puts_the_elbow_on_the_hand_tip_line():
    """phi = 0 is the degenerate case the interior-angle check above cannot see."""
    g, xi = fixture()
    tau = (0.15, 0.15, 0.0)
    x1 = ini.h(tensor(tau), g, xi).detach().numpy()
    hand, elbow, tip = x1[ini.HAND_XY], x1[ini.ELBOW_XY], x1[ini.TIP_XY]
    first, second = elbow - hand, tip - hand
    cross = first[0] * second[1] - first[1] * second[0]
    assert cross == pytest.approx(0.0, abs=1e-12)


def test_tau_is_exactly_recoverable_from_x1():
    """The claim decision 1 of networks_and_design_gradient.md rests on.

    l1 from hand -> elbow, (l2, phi) from elbow -> tip, all in the hand frame. This
    arithmetic moves into the parameter-free feature layer when that module lands
    (section 6 of the same doc) -- it should be lifted from here, not written twice.
    """
    g, xi = fixture()
    for tau in TAUS:
        x1 = ini.h(tensor(tau), g, xi).detach().numpy()
        c, s = x1[ini.HAND_YAW]
        d_elbow = (x1[ini.ELBOW_XY] - x1[ini.HAND_XY]) * BOX.scale
        d_tip = (x1[ini.TIP_XY] - x1[ini.HAND_XY]) * BOX.scale

        # Rotate into the hand frame. The y row inverts tip_from_hand's Rz(psi) @
        # (ox, -oy); the elbow has no y component to flip.
        l1 = c * d_elbow[0] + s * d_elbow[1]
        ox = c * d_tip[0] + s * d_tip[1]
        oy = s * d_tip[0] - c * d_tip[1]
        l2 = np.hypot(ox - l1, oy)
        phi = np.arctan2(oy, ox - l1)

        assert np.allclose([l1, l2, phi], tau, atol=1e-12)


def test_the_phase_block_is_zero_because_h_is_the_start_of_an_episode():
    g, xi = fixture()
    x1 = ini.h(tensor(TAUS[0]), g, xi).detach().numpy()
    assert np.allclose(x1[ini.PHASE], 0.0)


def test_relative_blocks_are_the_two_reward_terms():
    g, xi = fixture()
    for tau in TAUS:
        x1 = ini.h(tensor(tau), g, xi).detach().numpy()
        tip = se2.tip_from_hand(xi.hand_se2, tau)
        assert np.allclose(x1[ini.D_OBJ_TIP], BOX.normalise_delta(xi.obj_xy - tip))
        assert np.allclose(x1[ini.D_TARGET_OBJ], BOX.normalise_delta(g.target - xi.obj_xy))


def test_reaching_pins_the_object_at_the_target_so_transport_is_zero():
    g, xi = fixture()
    x1 = ini.h(tensor(TAUS[0]), g, xi).detach().numpy()
    assert np.allclose(x1[ini.OBJ_XY], x1[ini.TASK_BLOCK][:2])
    assert np.allclose(x1[ini.D_TARGET_OBJ], 0.0)


def test_batched_h_agrees_row_wise_with_single_designs():
    g, xi = fixture()
    batched = ini.h(torch.tensor(TAUS, dtype=torch.float64), g, xi).detach().numpy()
    for row, tau in zip(batched, TAUS):
        assert np.allclose(row, ini.h(tensor(tau), g, xi).detach().numpy())


# -- gradients ----------------------------------------------------------------


def jacobian(tau, g, xi):
    return torch.autograd.functional.jacobian(
        lambda t: ini.h(t, g, xi), tensor(tau)
    ).numpy()


def test_the_tip_is_the_designs_only_channel_into_x1():
    """Everything outside slices 7:9, 9:11 and 13:15 is a constant of the reset."""
    g, xi = fixture()
    for tau in TAUS:
        rows = np.abs(jacobian(tau, g, xi)).sum(axis=1)
        design_rows = np.zeros(ini.OBS_DIM, dtype=bool)
        for block in ini.TAU_SLICES:
            design_rows[block] = True
        assert np.all(rows[~design_rows] == 0.0)


def test_the_design_gradient_is_finite_and_non_trivial():
    """A collapsed dh/dtau would make everything downstream of it decoration."""
    g, xi = fixture()
    for tau in TAUS:
        jac = jacobian(tau, g, xi)
        assert np.all(np.isfinite(jac))
        assert np.abs(jac[ini.TIP_XY]).max() > 1e-3


def test_the_elbow_makes_dh_dtau_rank_three():
    """Without it, tau reaches x1 only through the 3->2 tip-offset map.

    The elbow adds d(elbow)/d(l1) in exactly the direction that map is blind to, so
    tau -> x1 becomes injective and l1 gets a gradient component of its own. See
    ai_docs/networks_and_design_gradient.md section 2.
    """
    g, xi = fixture()
    for tau in TAUS:
        jac = jacobian(tau, g, xi)
        assert np.linalg.matrix_rank(jac, tol=1e-9) == 3

        elbow = jac[ini.ELBOW_XY]
        # l1 only: the elbow sits at (l1, 0) in the hand frame whatever l2 and phi do.
        assert np.abs(elbow[:, 0]).max() > 1e-3
        assert np.all(elbow[:, 1:] == 0.0)


def test_design_gradient_matches_central_finite_differences():
    g, xi = fixture()
    eps = 1e-6
    for tau in TAUS:
        jac = jacobian(tau, g, xi)
        for k in range(3):
            step = np.zeros(3)
            step[k] = eps
            plus = ini.h(tensor(np.array(tau) + step), g, xi).detach().numpy()
            minus = ini.h(tensor(np.array(tau) - step), g, xi).detach().numpy()
            assert np.allclose(jac[:, k], (plus - minus) / (2 * eps), atol=1e-7)


def test_gradient_flows_through_a_prior_sample():
    """The shape ToolPrior actually hands the Langevin chain."""
    from tool_design_prior import ToolPrior

    g, xi = fixture()
    taus = ToolPrior(device="cpu").sample(8)
    # Squared, not summed: the tip enters slice 9:11 as +tip/s and slice 13:15 as
    # -tip/s, so a plain sum cancels the design out exactly. A value head is not a
    # sum over the observation, but it is worth knowing the redundancy has that
    # degenerate direction in it.
    (ini.h(taus, g, xi) ** 2).sum().backward()
    assert taus.grad is not None and torch.isfinite(taus.grad).all()
    assert taus.grad.abs().max() > 0


# -- ties back to the reach geometry ------------------------------------------


def test_a_reachable_target_can_be_driven_to_zero_tip_error():
    """h and task_space describe the same tip, reached from the same workspace."""
    g, xi = fixture()
    for tau in TAUS:
        pose = ts.hand_pose_for_tip(
            g.target, SE2Config.WORKSPACE, tau, SE2Config.YAW_LIMIT, tol=g.r_obj
        )
        if pose is None:
            continue
        x1 = ini.h(tensor(tau), g, xi._replace(hand_se2=pose), ).detach().numpy()
        error = np.linalg.norm(x1[ini.TIP_XY] - x1[ini.TASK_BLOCK][:2]) * BOX.scale
        assert error <= g.r_obj + 1e-6


def test_xi_is_drawn_from_the_same_law_the_policy_resets_under():
    reset_box = SE2Config.WORKSPACE.shrink(SE2Config.RESET_MARGIN)
    g = task.sample_task(rng())
    for seed in range(200):
        xi = ini.sample_xi(rng(seed), g)
        assert reset_box.contains(xi.hand_se2[:2])
        assert abs(xi.hand_se2[2]) <= SE2Config.YAW_LIMIT
