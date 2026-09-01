"""Visual and statistical check of ReachEnv: does a greedy policy reach the target?

There is no trained policy yet, so this drives a scripted one -- steer the hand so
the tool tip closes on the target, using the closed-form tip offset to pick the yaw.
It is not a good policy and is not meant to be; it exists so the ghost markers, the
reset spread and the reward scale are checkable by eye, and so the success rate has
a floor to compare a learned policy against later.

Success rates here are *stratified by reachability*, not aggregated. Nearly 40% of
TARGET_BOX is reachable by no design in the prior (see task.sample_task), so a single
aggregate number over uniform targets mostly measures that mixing ratio rather than
anything about the tool or the policy.

Usage:
    python reach_env_demo.py             # GUI
    python reach_env_demo.py --headless  # numbers only, no window
"""
import sys
import time

import numpy as np

import se2
import task as task_mod
import task_space
from config import DesignPriorConfig, SE2Config, TaskConfig
from reach_env import ReachEnv

TAUS = [
    (0.1, 0.1, 0.0),                                # short straight rod
    (0.2, 0.2, 0.0),                                # longest reach in the prior
    (0.1, 0.2, DesignPriorConfig.PHI_MAX),          # deepest hook
]

EPISODES = 20


def greedy_action(env):
    """Steer the tip at the target: yaw to aim the tool, then translate the hand.

    The tip sits at ``hand + R*u(yaw + alpha)`` (se2.tip_polar), so the hand yaw that
    points the tip at a target from the current hand position is
    ``bearing(target - hand) - alpha``. Turn towards that, and walk the hand along
    the residual. Both channels are saturated to +-1 and the env scales them.
    """
    hand = env.robot.get_hand_se2()
    target = np.asarray(env._task.target, dtype=float)[:2]
    radius, alpha = se2.tip_polar(env.tau)

    to_target = target - hand[:2]
    desired_yaw = np.arctan2(to_target[1], to_target[0]) - alpha
    dyaw = se2.wrap_angle(desired_yaw - hand[2]) / SE2Config.YAW_SCALE

    # Walk the hand in until the tip's own reach covers the rest of the gap.
    gap = np.linalg.norm(to_target) - radius
    direction = to_target / max(np.linalg.norm(to_target), 1e-9)
    step = direction * gap / SE2Config.POS_SCALE
    return np.clip([step[0], step[1], dyaw], -1.0, 1.0)


def run_episode(env, task, gui):
    """One episode under the greedy policy. Returns (return, success, final gap)."""
    env.reset(options={"task": task})
    total, success = 0.0, False
    for _ in range(TaskConfig.HORIZON):
        _, reward, _, truncated, info = env.step(greedy_action(env))
        total += reward
        success |= bool(info["is_success"])
        if gui:
            time.sleep(1.0 / 120.0)
        if truncated:
            break
    gap = np.linalg.norm(env._tip_xy() - np.asarray(task.target, dtype=float)[:2])
    return total, success, gap


def reachable_targets(tau, n, np_random):
    """Draw n targets this design can actually put its tip on.

    Rejection against task_space.tip_reachable, the same exact per-point test
    reach_sweep.py verifies against PyBullet FK.
    """
    out = []
    while len(out) < n:
        point = TaskConfig.TARGET_BOX.sample(np_random)
        if task_space.tip_reachable(
            point[None, :], SE2Config.WORKSPACE, tau, SE2Config.YAW_LIMIT
        )[0]:
            out.append(point)
    return out


def demo_tau(tau, gui):
    """Run both target strata for one design and print what happened."""
    env = ReachEnv(tau, render_mode="human" if gui else "rgb_array")
    # Seed once: envs are entropy-seeded at construction so parallel workers do not
    # share a stream, which makes an unseeded demo irreproducible. Later resets go
    # unseeded and continue this stream, so the whole run is fixed by this one call.
    env.reset(seed=0)
    np_random = np.random.default_rng(0)
    radius, alpha = se2.tip_polar(tau)
    print(f"\ntau = {tuple(round(v, 3) for v in tau)}   "
          f"tip radius {radius:.3f} m, bearing offset {np.rad2deg(alpha):+.1f} deg")

    strata = {
        "reachable": reachable_targets(tau, EPISODES, np_random),
        "uniform  ": [TaskConfig.TARGET_BOX.sample(np_random) for _ in range(EPISODES)],
    }
    for label, targets in strata.items():
        results = [run_episode(env, task_mod.Task(
            task_id=task_mod.TaskType.REACH, target=t, r_obj=TaskConfig.R_OBJ,
            rho=TaskConfig.RHO_TARGET, w_reach=1.0, w_trans=0.0), gui) for t in targets]
        returns, successes, gaps = zip(*results)
        print(f"  {label}  success {np.mean(successes):5.0%}   "
              f"mean return {np.mean(returns):8.2f}   "
              f"median final gap {np.median(gaps):.3f} m")
    env.close()


def main():
    gui = "--headless" not in sys.argv
    for tau in TAUS:
        demo_tau(tau, gui)


if __name__ == "__main__":
    main()
