"""Analytic tests for task.py, the generative model p(g). No sim, no PyBullet body."""
import numpy as np
import pytest

import task
from config import TaskConfig


def rng(seed=0):
    return np.random.default_rng(seed)


# -- sampling -----------------------------------------------------------------


def test_sampled_targets_lie_in_the_target_box():
    box = TaskConfig.TARGET_BOX
    targets = np.array([task.sample_task(rng(s)).target for s in range(500)])
    assert np.all(targets[:, 0] >= box.x_min) and np.all(targets[:, 0] <= box.x_max)
    assert np.all(targets[:, 1] >= box.y_min) and np.all(targets[:, 1] <= box.y_max)


def test_sampling_is_reproducible_under_a_seeded_generator():
    assert np.allclose(task.sample_task(rng(7)).target, task.sample_task(rng(7)).target)


def test_sampling_does_not_reject_unreachable_targets():
    """Unreachable targets are intentional: p(g) is uniform over the whole box.

    Checked by hitting a corner the arm plus any tool cannot get near -- x near 0 is
    inside the robot's own base -- which rejection sampling would never produce.
    """
    xs = np.array([task.sample_task(rng(s)).target[0] for s in range(2000)])
    assert xs.min() < 0.15  # tool-reachable x starts at 0.150


def test_reach_parameters_come_from_the_table():
    g = task.sample_task(rng())
    w_reach, w_trans, rho = task.TASK_PARAMS[task.TaskType.REACH]
    assert (g.w_reach, g.w_trans, g.rho) == (w_reach, w_trans, rho)
    assert g.r_obj == TaskConfig.R_OBJ


def test_unbuilt_task_types_raise_rather_than_inherit_reach_weights():
    for unbuilt in (task.TaskType.SWEEP, task.TaskType.PUSH):
        with pytest.raises(KeyError):
            task.sample_task(rng(), task_id=unbuilt)


# -- object start -------------------------------------------------------------


def test_reach_object_starts_at_the_target():
    g = task.sample_task(rng())
    assert np.allclose(task.object_start(g, rng()), g.target)


def test_object_start_is_a_copy_not_the_target_array():
    """The object pose is state; g must not be mutable through it."""
    g = task.sample_task(rng())
    start = task.object_start(g, rng())
    start[0] += 1.0
    assert not np.allclose(start, g.target)


def test_unbuilt_task_types_have_no_start_distribution():
    g = task.sample_task(rng())._replace(task_id=task.TaskType.SWEEP)
    with pytest.raises(NotImplementedError):
        task.object_start(g, rng())


# -- encoding -----------------------------------------------------------------


def test_encode_width_is_final_before_sweep_and_push_exist():
    """Adding a task type later must not change the observation size."""
    assert task.N_TASK_TYPES == 3
    assert len(task.encode(task.sample_task(rng()))) == task.ENCODING_DIM == 6


def test_encode_carries_the_normalised_target_and_a_one_hot():
    g = task.sample_task(rng())
    code = task.encode(g)
    assert np.allclose(code[:2], TaskConfig.SCENE_BOX.normalise_point(g.target))
    assert np.allclose(code[2:5], [1.0, 0.0, 0.0])
    assert code[5] == pytest.approx(g.r_obj / TaskConfig.SCENE_BOX.scale)


def test_encode_omits_the_weights_and_tolerance():
    """They follow deterministically from the task id, which is already encoded."""
    g = task.sample_task(rng())
    other = g._replace(rho=0.5, w_reach=0.1, w_trans=0.9)
    assert np.allclose(task.encode(g), task.encode(other))


# -- reward and success -------------------------------------------------------


def test_reward_is_negative_tip_to_object_distance_for_reaching():
    g = task.sample_task(rng())
    tip = g.target + np.array([0.3, 0.4])
    assert task.reward(g, tip, g.target) == pytest.approx(-0.5)


def test_reward_ignores_the_transport_term_for_reaching():
    g = task.sample_task(rng())
    far = g.target + np.array([2.0, 0.0])
    assert task.reward(g, g.target, g.target) == pytest.approx(0.0)
    assert task.reward(g, far, far) == pytest.approx(-0.0)


def test_reach_success_is_tip_based_not_object_based():
    """The spec's unified rule d(obj_T, target) < rho is trivially true for reaching.

    The reaching object is pinned at the target and never moves, so an object-based
    rule would score every episode a success, including one where the tip never left
    the other side of the table.
    """
    g = task.sample_task(rng())
    obj = task.object_start(g, rng())
    assert not task.success(g, g.target + np.array([1.0, 0.0]), obj)
    assert task.success(g, g.target, obj)


def test_success_boundary_is_rho():
    g = task.sample_task(rng())
    obj = task.object_start(g, rng())
    inside = g.target + np.array([g.rho * 0.99, 0.0])
    outside = g.target + np.array([g.rho * 1.01, 0.0])
    assert task.success(g, inside, obj)
    assert not task.success(g, outside, obj)


def test_transport_tasks_score_on_the_object():
    g = task.sample_task(rng())._replace(task_id=task.TaskType.PUSH)
    assert task.success(g, g.target + np.array([1.0, 0.0]), g.target)
    assert not task.success(g, g.target, g.target + np.array([1.0, 0.0]))
