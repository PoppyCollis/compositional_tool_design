"""Visualise where an object may sit for the reach task (see reach_sweep.py).

Two panels, both over the same table grid:

  left   per-design reach regions, outlined, over the hand's own workspace and the
         bare-arm band. Shows what tau buys: the outlines are not nested, because
         the tip lies on a circle of radius R rather than in a disk, so a longer
         tool trades a near-target blind zone for outer reach.

  right  coverage -- the fraction of ToolPrior designs that can reach each cell,
         with the discriminating band outlined. The 0 < f < 1 band is where the
         design is load-bearing; at f = 1 every tool succeeds and p(tau | g, O=1)
         cannot separate from the prior, which is the failure mode the diagnostic in
         ai_docs/task_encoding_g.md looks for.

No rectangle is fitted to the band on purpose: it is an annular shell with the near-
field notch cut out of it, so the largest box inside it is a sliver of the outer edge
that would sample "long tools win" and nothing else. Sample s_start by rejection
against task_space.tip_reachable instead.

Pure geometry -- no PyBullet, so this runs in a second and needs no sweep.

Usage:
    python plot_reach_space.py [--save reach_space.png]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

import task_space as ts
from config import SE2Config, TaskConfig
from reach_sweep import (BANDS, CANONICAL_TAUS, N_PRIOR, RESOLUTION, X_RANGE,
                         Y_RANGE, bare_arm_mask)
from tool_design_prior import ToolPrior
from utils.plots import plot_workspace

# The band outlined on the coverage panel.
HIGHLIGHT_BAND = "discriminating    (0.2 <= f <= 0.8)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default=None, help="Path to save the figure to.")
    parser.add_argument("--n-prior", type=int, default=N_PRIOR,
                        help="Designs to sample for the coverage panel.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    xs, ys, points = ts.grid(X_RANGE, Y_RANGE, RESOLUTION)
    no_tool = bare_arm_mask(points)
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    # -- left: per-design regions ---------------------------------------------
    colours = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, len(CANONICAL_TAUS)))
    contours = []
    for (label, tau), colour in zip(CANONICAL_TAUS, colours):
        mask = ts.reach_mask(xs, ys, SE2Config.WORKSPACE, tau, SE2Config.YAW_LIMIT,
                             tol=TaskConfig.R_OBJ)
        contours.append((mask, f"{label}  tau={tuple(round(v, 2) for v in tau)}",
                         dict(colors=[colour], linewidths=1.4)))

    plot_workspace(
        xs, ys, no_tool, contours=contours,
        boxes=[(SE2Config.WORKSPACE, "SE2Config.WORKSPACE (hand)",
                dict(color="black", linestyle="--", linewidth=1.5))],
        title=(f"Reach region per design  (r_obj = {TaskConfig.R_OBJ} m)\n"
               f"shaded: reachable by the bare gripper, so no design signal"),
        ax=axes[0], cmap="Greys", vmax=3,
        legend_kwargs=dict(loc="lower center", fontsize=7, ncol=2, framealpha=0.9),
    )

    # -- right: coverage over the prior ---------------------------------------
    torch_taus = ToolPrior().sample(args.n_prior).detach().cpu().numpy()
    coverage = ts.coverage(xs, ys, SE2Config.WORKSPACE, torch_taus,
                           SE2Config.YAW_LIMIT, tol=TaskConfig.R_OBJ)

    label, lo, hi = next(b for b in BANDS if b[0] == HIGHLIGHT_BAND)
    band = (coverage >= lo) & (coverage <= hi)

    plot_workspace(
        xs, ys, coverage,
        boxes=[(SE2Config.WORKSPACE, "SE2Config.WORKSPACE (hand)",
                dict(color="black", linestyle="--", linewidth=1.5))],
        contours=[(band, f"{label.split('(')[0].strip()} band",
                   dict(colors=["red"], linestyles="dotted"))],
        title=(f"Fraction of {args.n_prior} ToolPrior designs that can reach\n"
               f"f = 1 carries no design signal; the band between does"),
        ax=axes[1], cmap="viridis", vmax=1.0, colorbar="fraction of designs",
        legend_kwargs=dict(loc="lower center", fontsize=7, framealpha=0.9),
    )

    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
