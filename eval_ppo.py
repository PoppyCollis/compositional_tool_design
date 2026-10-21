"""Roll out a trained PPO policy on ReachEnv and report success by coverage band.

An aggregate success rate over ``p(g)`` mostly measures the mixing ratio: ~40% of
``TARGET_BOX`` is reachable by no design in the prior and ~11% by every design, and only
the band between carries design signal (``ai_docs/task_encoding_g.md``). So each
episode's target is labelled by the fraction of prior designs that can reach it
(``task_space.coverage``, the same 2000-design map ``reach_sweep.py`` builds), and
separately by whether the episode's own tool can reach it (``task_space.tip_reachable``).
The floor to beat on the reachable stratum is the greedy controller in
``reach_env_demo.py --headless``.

The policy is run stochastically by default: that is the policy ``V`` scores, and the
one the designer's energy describes. ``--deterministic`` gives the mean action.

Usage:
    python eval_ppo.py --model runs/<run>/final.zip [--episodes 400] [--deterministic]
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse

import numpy as np
from stable_baselines3 import PPO

import task_space
from config import SE2Config, TaskConfig
from reach_env import ReachEnv
from tool_design_prior import ToolPrior

# As reach_sweep.py: 5 mm grid, 2000 prior designs, touch tolerance R_OBJ.
RESOLUTION = 0.005
N_PRIOR = 2000
BANDS = (
    ("no design     (f = 0)", lambda f: f <= 0.0),
    ("some designs  (0 < f < 1)", lambda f: (f > 0.0) & (f < 1.0)),
    ("every design  (f = 1)", lambda f: f >= 1.0),
)


def coverage_lookup(seed=0):
    """A function ``target -> coverage fraction``, from the prior's reach map.

    Targets outside the grid are reachable by no design (the union of all reachable
    object centres is inside it, see ``TaskConfig.SCENE_BOX``), so they read as 0.
    """
    import torch
    torch.manual_seed(seed)
    taus = ToolPrior(device="cpu").sample(N_PRIOR).detach().numpy()
    box = TaskConfig.TARGET_BOX
    xs, ys, _ = task_space.grid((box.x_min, box.x_max), (box.y_min, box.y_max), RESOLUTION)
    cov = task_space.coverage(xs, ys, SE2Config.WORKSPACE, taus, SE2Config.YAW_LIMIT,
                              tol=TaskConfig.R_OBJ)

    def lookup(target):
        i = int(np.clip(np.rint((target[0] - xs[0]) / RESOLUTION), 0, len(xs) - 1))
        j = int(np.clip(np.rint((target[1] - ys[0]) / RESOLUTION), 0, len(ys) - 1))
        return float(cov[i, j])

    return lookup


def run_episode(env, model, deterministic):
    """One episode. Returns (tau, target, return, success)."""
    obs, info = env.reset()
    tau, target = info["tau"], info["target"]
    total, success, terminated = 0.0, False, False
    while not terminated:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, _, info = env.step(action)
        total += reward
        success |= bool(info["is_success"])
    return tau, target, total, success


def report(label, returns, successes):
    n = len(returns)
    if n == 0:
        print(f"  {label:<28}  n={n:4d}")
        return
    print(f"  {label:<28}  n={n:4d}   success {np.mean(successes):5.0%}   "
          f"mean return {np.mean(returns):8.2f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    model = PPO.load(args.model, device="cpu")
    lookup = coverage_lookup()
    env = ReachEnv(rebuild_every=1)
    env.reset(seed=args.seed)  # fixes tool, task and pose for the whole run
    try:
        rows = [run_episode(env, model, args.deterministic) for _ in range(args.episodes)]
    finally:
        env.close()

    taus, targets, returns, successes = (np.array(col) for col in zip(*rows))
    fractions = np.array([lookup(t) for t in targets])
    own_tool = np.array([
        task_space.tip_reachable(t[None, :], SE2Config.WORKSPACE, tuple(tau),
                                 SE2Config.YAW_LIMIT, tol=TaskConfig.R_OBJ)[0]
        for t, tau in zip(targets, taus)
    ])

    mode = "deterministic" if args.deterministic else "stochastic"
    print(f"\n{args.model}  ({mode} actions, {args.episodes} episodes, tau ~ prior)")
    print("by coverage band of the target (fraction of prior designs that can reach it):")
    for label, member in BANDS:
        mask = member(fractions)
        report(label, returns[mask], successes[mask])
    print("by whether this episode's own tool can reach the target:")
    report("reachable by own tool", returns[own_tool], successes[own_tool])
    report("not reachable by own tool", returns[~own_tool], successes[~own_tool])
    report("all", returns, successes)


if __name__ == "__main__":
    main()
