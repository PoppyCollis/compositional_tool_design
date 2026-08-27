"""Visualise the measured SE(2) workspace (see workspace_sweep.py).

Re-runs the same grid sweep workspace_sweep.py uses to derive
SE2Config.WORKSPACE, then renders it as a top-down map: green where the arm
can hold the tool flat at every yaw in +-SE2Config.YAW_LIMIT, with the raw
largest-clean-rectangle and the margin-inset box actually used at runtime
drawn on top.

Usage:
    python plot_workspace.py [--save workspace.png]
"""
import argparse

import numpy as np
import pybullet as p
from panda_gym.pybullet import PyBullet

import se2
from config import SE2Config
from panda_with_tool import PandaWithTool
from utils.plots import plot_workspace
from workspace_sweep import SWEEP_TAU, X_RANGE, Y_RANGE, RESOLUTION, N_YAW, sweep, box_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default=None, help="Path to save the figure to.")
    args = parser.parse_args()  

    sim = PyBullet(render_mode="rgb_array")
    robot = PandaWithTool(sim, SWEEP_TAU)

    xs = np.arange(X_RANGE[0], X_RANGE[1] + 1e-9, RESOLUTION)
    ys = np.arange(Y_RANGE[0], Y_RANGE[1] + 1e-9, RESOLUTION)
    yaws = np.linspace(-np.pi, np.pi, N_YAW, endpoint=False)

    print(f"sweeping {len(xs)}x{len(ys)} cells at {len(yaws)} yaws "
          f"({len(xs) * len(ys) * len(yaws)} IK solves)")
    masks = sweep(robot, xs, ys, yaws)
    p.disconnect()

    raw, inset, _ = box_for(masks, yaws, xs, ys, SE2Config.YAW_LIMIT)
    keep = np.abs(np.array([se2.wrap_angle(y) for y in yaws])) <= SE2Config.YAW_LIMIT + 1e-9
    mask = masks[keep].all(axis=0)

    boxes = []
    if raw is not None:
        boxes.append((raw, "largest clean rectangle", dict(color="tab:blue", linestyle="--")))
    if inset is not None:
        boxes.append((inset, "SWEEP_MARGIN inset", dict(color="tab:orange")))
    boxes.append((SE2Config.WORKSPACE, "SE2Config.WORKSPACE (configured)",
                  dict(color="tab:red", linewidth=1.5, linestyle=":")))

    ax = plot_workspace(
        xs, ys, mask, boxes=boxes,
        title=f"Measured SE(2) workspace (yaw within +-{np.degrees(SE2Config.YAW_LIMIT):.0f} deg)",
    )

    if args.save:
        ax.figure.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved to {args.save}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
