"""The task generative model ``p(g)``: what a task *is*, and how it is scored.

No torch, no PyBullet, no I/O -- the same discipline as ``se2`` and ``task_space``.

``g`` is the **target specification** and nothing else::

    g = ( task_id, p_target, r_obj, rho_target, w_reach, w_trans )

This deliberately departs from ``ai_docs/task_encoding_g.md`` as first written, which
also carried ``s_start``, the object's *start* region. Under the split used here
``x_t`` is the full currently-observable state, including the object's current pose
re-emitted every step, so the object's starting position lands in ``x_1`` for free and
does not need a second home in ``g``. What replaces ``s_start`` is ``object_start``
below: a distribution keyed by ``task_id``, drawn at reset, never a field of ``g``.
Keeping it out of ``g`` matters because a ``g`` that varied with the design would
break the ``p(tau | g, O=1)`` factorisation the whole method rests on.

``rho``, ``w_reach`` and ``w_trans`` are fields of the task rather than module
constants so the objective and the success metric are read from one object and cannot
drift apart. They are *derived* from ``task_id`` (see ``TASK_PARAMS``) rather than
sampled, and are excluded from ``encode`` because the id they follow from is already
in it.
"""
import enum
from typing import NamedTuple

import numpy as np

from config import TaskConfig


class TaskType(enum.IntEnum):
    """The three task instances of ``ai_docs/task_encoding_g.md``.

    Only ``REACH`` is implemented. ``SWEEP`` and ``PUSH`` are declared now so the
    one-hot in ``encode`` has its final width: adding a task type later would
    otherwise change the observation size and invalidate every trained ``V``.
    """

    REACH = 0
    SWEEP = 1
    PUSH = 2


N_TASK_TYPES = len(TaskType)

# (w_reach, w_trans, rho) per task type.
#
# SWEEP and PUSH are absent rather than filled in with placeholder weights: their
# start distributions do not exist yet, and a dict lookup that raises is a better
# failure than one that silently hands back reaching's (1, 0). The doc's constraint
# eps_p < eps_s applies when they are added.
TASK_PARAMS = {
    TaskType.REACH: (1.0, 0.0, TaskConfig.RHO_TARGET),
}


class Task(NamedTuple):
    """One task instance ``g``. Constant for the whole episode."""

    task_id: TaskType
    target: np.ndarray  # (2,) desired object location, robot base frame
    r_obj: float        # desired object, as a plan-view radius
    rho: float          # success tolerance
    w_reach: float      # weight on tip->object
    w_trans: float      # weight on object->target


def sample_task(np_random, task_id=TaskType.REACH):
    """Draw ``g ~ p(g)``.

    The target is uniform over ``TaskConfig.TARGET_BOX``, with **no** rejection
    against reachability. Nearly 40% of the box is reachable by no design in the
    prior; those episodes are kept on purpose, so the value function learns that some
    targets are hopeless whatever the tool, rather than only ever seeing solvable ones
    and having to extrapolate at design time. The cost is that an aggregate success
    rate over the box is close to meaningless -- evaluation has to stratify by
    ``task_space.coverage`` band. See ``TaskConfig.SCENE_BOX``.

    Args:
        np_random (np.random.Generator): Source of randomness, so episodes seed with
            the env (cf. ``PandaWithTool.set_np_random``).
        task_id (TaskType): Which task instance to draw.

    Returns:
        Task: The sampled ``g``.

    Raises:
        KeyError: If ``task_id`` has no entry in ``TASK_PARAMS``.
    """
    w_reach, w_trans, rho = TASK_PARAMS[TaskType(task_id)]
    return Task(
        task_id=TaskType(task_id),
        target=TaskConfig.TARGET_BOX.sample(np_random),
        r_obj=TaskConfig.R_OBJ,
        rho=rho,
        w_reach=w_reach,
        w_trans=w_trans,
    )


def object_start(task, np_random):
    """Where the object sits at reset. This is what ``s_start`` became.

    Keyed by ``task_id``, not carried in ``g``. For reaching the object is at the
    target and never moves, so the draw is deterministic and the object block of
    ``x_t`` is a constant equal to ``g.target`` -- redundant, and kept anyway so the
    observation layout is final before sweep and push exist.

    The reaching object is virtual: a ghost sphere with no collision body. A real one
    would be knocked away by any tool that reached it, punishing the policy for
    succeeding.

    Args:
        task (Task): The task instance.
        np_random (np.random.Generator): Source of randomness. Unused for reaching;
            taken anyway so sweep and push slot in without changing the signature.

    Returns:
        np.ndarray: Object start position as ``(x, y)``.

    Raises:
        NotImplementedError: For task types whose start distribution is not built.
    """
    if task.task_id is TaskType.REACH:
        return np.asarray(task.target, dtype=float)[:2].copy()
    raise NotImplementedError(
        f"no object start distribution for {task.task_id!r}; sweep and push turn on "
        f"the contact normal, not tip position -- see plan.md"
    )


def encode(task):
    """``g`` as a fixed-width vector, the tail block of the observation.

    Excludes ``rho``, ``w_reach`` and ``w_trans``: they are a deterministic function
    of ``task_id``, which is already here as a one-hot.

    Args:
        task (Task): The task instance.

    Returns:
        np.ndarray: ``(6,)`` -- normalised target (2), task one-hot (3), and the
            object radius on the same length scale as everything else (1).
    """
    one_hot = np.zeros(N_TASK_TYPES)
    one_hot[int(task.task_id)] = 1.0
    return np.concatenate([
        TaskConfig.SCENE_BOX.normalise_point(task.target),
        one_hot,
        [task.r_obj / TaskConfig.SCENE_BOX.scale],
    ])


ENCODING_DIM = 2 + N_TASK_TYPES + 1


def reward(task, tip_xy, obj_xy):
    """Dense per-step reward ``-[w_reach*d(tip, obj) + w_trans*d(obj, target)]``.

    In reaching the tip->object term *is* the task (``w_trans = 0``); in sweeping and
    pushing it is shaping only, and the doc's plan is to anneal it to zero. If
    ``dV/dtau`` collapses under that anneal, the design signal was the shaping term
    rather than the task.

    Args:
        task (Task): The task instance.
        tip_xy: Tool-tip position as ``(x, y)``, in metres.
        obj_xy: Object position as ``(x, y)``, in metres.

    Returns:
        float: The reward, always <= 0.
    """
    tip = np.asarray(tip_xy, dtype=float)[:2]
    obj = np.asarray(obj_xy, dtype=float)[:2]
    target = np.asarray(task.target, dtype=float)[:2]
    return float(
        -task.w_reach * np.linalg.norm(tip - obj)
        - task.w_trans * np.linalg.norm(obj - target)
    )


def success(task, tip_xy, obj_xy):
    """Whether the episode succeeded, ``O = 1``.

    Not one rule for all three task types. ``ai_docs/task_encoding_g.md`` originally
    gave ``d(obj_T, p_target) < rho`` throughout, which is *trivially true* for
    reaching: the reaching object is pinned at the target and never moves, so every
    episode would score a success. Reaching succeeds when the **tip** arrives;
    sweeping and pushing when the **object** does.

    Args:
        task (Task): The task instance.
        tip_xy: Tool-tip position as ``(x, y)``, in metres.
        obj_xy: Object position as ``(x, y)``, in metres.

    Returns:
        bool: True if within ``task.rho`` of the target.
    """
    target = np.asarray(task.target, dtype=float)[:2]
    achieved = tip_xy if task.task_id is TaskType.REACH else obj_xy
    return bool(np.linalg.norm(np.asarray(achieved, dtype=float)[:2] - target) < task.rho)
