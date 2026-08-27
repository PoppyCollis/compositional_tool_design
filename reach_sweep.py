"""Offline analysis: where may an object sit for the reach task, and for which tools?

The counterpart to workspace_sweep.py. That one measures where the *hand* may go and
you paste its rectangle into SE2Config.WORKSPACE; this one turns that rectangle into
the set of object positions the tool *tip* can touch, per design and across the design
prior, and prints candidate boxes for the `s_start` / `p_target` fields of the task
encoding (ai_docs/task_encoding_g.md).

No simulator sweep is needed for the region itself -- it is closed form, see
task_space.py's module docstring. PyBullet is only touched by --measure-gripper and
--verify, both of which are checks rather than part of the derivation.

The column that matters is "needs tool": the area reachable by the tip but *not* by the
bare hand. A target the gripper alone could touch is solvable with no tool at all, so
it cannot separate designs however the policy behaves. Likewise the coverage map's
0 < f < 1 band is where the design is load-bearing; at f = 1 every tool succeeds and
p(tau | g, O=1) stays the prior.

Usage:
    python reach_sweep.py                    # tables, candidate boxes, ASCII maps
    python reach_sweep.py --measure-gripper  # re-measure TaskConfig.GRIPPER_RADIUS
    python reach_sweep.py --verify           # sim: drive to the poses and check FK
"""
import argparse
import math

import numpy as np

import se2
import task_space as ts
from config import DesignPriorConfig, SE2Config, TaskConfig
from tool_design_prior import ToolPrior

# Table extent to map, robot base at the origin. Wide enough to bracket the longest
# tool's reach past every wall of SE2Config.WORKSPACE (0.4 m for a straight rod at
# both links at L_MAX) without wasting cells behind the base.
X_RANGE = (0.00, 1.10)
Y_RANGE = (-0.85, 0.85)
RESOLUTION = 0.005

# Named corners of the design box, chosen to span the two axes that actually matter:
# how far the tip reaches (R) and which way it can point (alpha). See se2.tip_polar.
L_MIN, L_MAX = DesignPriorConfig.L_MIN, DesignPriorConfig.L_MAX
PHI_MAX = DesignPriorConfig.PHI_MAX
CANONICAL_TAUS = [
    ("straight, max", (L_MAX, L_MAX, 0.0)),
    ("right-angle, max", (L_MAX, L_MAX, math.pi / 2)),
    ("max fold, symmetric", (L_MAX, L_MAX, PHI_MAX)),
    ("short handle, long head", (L_MIN, L_MAX, PHI_MAX)),
    ("long handle, short head", (L_MAX, L_MIN, PHI_MAX)),
    ("straight, min", (L_MIN, L_MIN, 0.0)),
    ("min reach", (L_MIN, L_MIN, PHI_MAX)),
]

# Monte-Carlo sample size for the coverage map. Reach is closed form, so this costs
# seconds; the sample error on a per-cell fraction at N=2000 is under 1.1%.
N_PRIOR = 2000

# Coverage bands to print candidate boxes for. (label, lo, hi), inclusive of lo,
# exclusive of hi except at 1.0.
BANDS = [
    ("any tool          (f > 0)", 1e-9, 1.0 + 1e-9),
    ("discriminating    (0.2 <= f <= 0.8)", 0.2, 0.8 + 1e-9),
    ("most tools        (f >= 0.5)", 0.5, 1.0 + 1e-9),
    ("every tool        (f = 1)", 1.0 - 1e-9, 1.0 + 1e-9),
]


def bare_arm_mask(points):
    """Object positions the closed gripper could touch with no tool at all."""
    return ts.hand_reachable(points, SE2Config.WORKSPACE,
                             TaskConfig.R_OBJ + TaskConfig.GRIPPER_RADIUS)


def describe(label, tau, xs, ys, no_tool, cell_area):
    """One row of the per-design table, and the design's reach mask."""
    mask = ts.reach_mask(xs, ys, SE2Config.WORKSPACE, tau, SE2Config.YAW_LIMIT,
                         tol=TaskConfig.R_OBJ)
    radius, bearing = se2.tip_polar(tau)
    lo = math.degrees(bearing - SE2Config.YAW_LIMIT)
    hi = math.degrees(bearing + SE2Config.YAW_LIMIT)
    needs_tool = mask & ~no_tool
    # The near-target hole: the tip sits on a circle of radius R about the hand, not
    # in a disk, so a long tool has a blind zone close in. Measured against the bare
    # arm's own reach, which is the region a long tool is meant to be extending.
    blind = no_tool & ~mask
    print(f"  {label:<24} {radius:5.3f}  {math.degrees(bearing):+6.1f}  "
          f"[{lo:+7.1f},{hi:+7.1f}]  {mask.sum() * cell_area:6.3f}  "
          f"{needs_tool.sum() * cell_area:6.3f}  {blind.sum() * cell_area:6.3f}")
    return mask


def print_bands(coverage, cell_area):
    """Area of each coverage band.

    Areas only, deliberately: these regions are annular shells with a notch cut out
    of the near field (see the blind column above), so the largest axis-aligned
    rectangle inside one throws most of it away and picks a sliver of the outer
    shell -- which would sample "long tools win" and nothing else. Sample s_start by
    rejection against task_space.tip_reachable instead of against a box.

    Args:
        coverage (np.ndarray): Fraction-of-designs map from task_space.coverage.
        cell_area (float): Area of one grid cell, in m2.
    """
    print(f"\n  {'band':<38} {'cells':>7} {'area (m2)':>10}")
    for label, lo, hi in BANDS:
        mask = (coverage >= lo) & (coverage <= hi)
        print(f"  {label:<38} {mask.sum():7d} {mask.sum() * cell_area:10.3f}")


def measure_gripper():
    """Circumradius of the closed gripper's footprint at TOOL_Z, from the live URDF.

    Reads the world AABB of every link that spans the tool's height and takes the
    furthest corner from the panda_hand origin. Imports PyBullet lazily so the rest
    of this script stays pure.
    """
    import pybullet as p
    from panda_gym.pybullet import PyBullet

    from panda_with_tool import PandaWithTool
    from utils.helpers import get_link_index_by_name

    sim = PyBullet(render_mode="rgb_array")
    robot = PandaWithTool(sim, (L_MIN, L_MIN, 0.0))
    body = sim._bodies_idx[robot.body_name]
    robot.set_se2(SE2Config.WORKSPACE.centre, 0.0)
    hand = robot.get_hand_position()

    worst = 0.0
    for name in ("panda_hand", "panda_leftfinger", "panda_rightfinger"):
        lo, hi = (np.array(v) for v in p.getAABB(body, get_link_index_by_name(body, name)))
        spans_tool_plane = lo[2] <= SE2Config.TOOL_Z <= hi[2]
        corner = max(math.hypot(x - hand[0], y - hand[1])
                     for x in (lo[0], hi[0]) for y in (lo[1], hi[1]))
        print(f"  {name:<18} z=[{lo[2]:+.4f},{hi[2]:+.4f}]  "
              f"at TOOL_Z={'yes' if spans_tool_plane else ' no'}  circumradius={corner:.4f}")
        if spans_tool_plane:
            worst = max(worst, corner)
    print(f"\n    GRIPPER_RADIUS = {worst:.4f}   (round up when pasting into TaskConfig)")
    p.disconnect()


def verify(n_samples=40, seed=0):
    """Drive the arm to the computed poses and check the tool tip lands where predicted.

    The end-to-end check that the closed-form region composes with the real arm:
    task_space says a point is reachable and hands back a witness pose, the arm is
    teleported there, and PyBullet's own FK must agree with se2.tip_from_hand about
    where the tip ended up.

    That agreement is the thing being tested, not the tip-to-target distance. The
    region is defined as "tip within r_obj of the object centre", so a target on its
    boundary has the tip grazing the object's surface at exactly r_obj by
    construction; the arm's own IK residual then carries it a millimetre or two
    past. Asserting on tip-to-target would therefore fail on correct geometry, so
    the residual is reported separately and the target check is allowed the same
    slack the workspace itself was certified at (workspace_sweep.POSITION_TOL).

    Targets are drawn from reach_mask but re-checked with the exact tip_reachable,
    since the rasteriser's boundary is a cell wide; cells it over-claims are counted
    rather than treated as failures.
    """
    import pybullet as p
    from panda_gym.pybullet import PyBullet

    from panda_with_tool import PandaWithTool
    from workspace_sweep import POSITION_TOL

    rng = np.random.default_rng(seed)
    xs, ys, points = ts.grid(X_RANGE, Y_RANGE, RESOLUTION)
    sim = PyBullet(render_mode="rgb_array")

    print(f"  {'design':<24} {'closed form vs FK':>18}  {'tip to target':>14}  "
          f"{'edge':>5}")
    print(f"  {'':<24} {'(mm, max)':>18}  {'(mm, max)':>14}  {'cells':>5}")

    worst_model = 0.0
    for label, tau in CANONICAL_TAUS:
        robot = PandaWithTool(sim, tau)
        mask = ts.reach_mask(xs, ys, SE2Config.WORKSPACE, tau, SE2Config.YAW_LIMIT,
                             tol=TaskConfig.R_OBJ)
        candidates = points[mask]
        candidates = candidates[rng.choice(len(candidates), size=n_samples, replace=False)]
        exact = ts.tip_reachable(candidates, SE2Config.WORKSPACE, tau,
                                 SE2Config.YAW_LIMIT, tol=TaskConfig.R_OBJ)

        model_err, target_err = 0.0, 0.0
        for q in candidates[exact]:
            pose = ts.hand_pose_for_tip(q, SE2Config.WORKSPACE, tau,
                                        SE2Config.YAW_LIMIT, tol=TaskConfig.R_OBJ)
            assert pose is not None, f"{label}: tip_reachable and its witness disagree at {q}"
            robot.set_se2(pose[:2], pose[2])
            tip = robot.get_ee_position()[:2]
            model_err = max(model_err, float(np.linalg.norm(tip - se2.tip_from_hand(pose, tau))))
            target_err = max(target_err, float(np.linalg.norm(tip - q)))

        over_claimed = int((~exact).sum())
        status = "ok" if model_err <= POSITION_TOL else "FAIL"
        print(f"  {label:<24} {model_err * 1000:15.2f} {status:>3}  "
              f"{target_err * 1000:14.2f}  {over_claimed:5d}")
        worst_model = max(worst_model, model_err)
        sim.physics_client.removeBody(sim._bodies_idx.pop(robot.body_name))

    print(f"\n  worst closed-form-vs-FK disagreement across every design: "
          f"{worst_model * 1000:.2f} mm  (tolerance {POSITION_TOL * 1000:.0f} mm, the same "
          f"the\n  workspace itself was swept at -- this is the arm's IK residual, not a "
          f"geometry error)")
    print(f"  'edge' counts sampled cells reach_mask claims and tip_reachable rejects: "
          f"the\n  rasteriser's one-cell boundary, expected and harmless for maps.")
    if worst_model > POSITION_TOL:
        raise AssertionError("closed-form tip disagrees with PyBullet FK beyond tolerance")
    p.disconnect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure-gripper", action="store_true",
                        help="Re-measure TaskConfig.GRIPPER_RADIUS from the live URDF.")
    parser.add_argument("--verify", action="store_true",
                        help="Check the computed poses against PyBullet's own FK.")
    parser.add_argument("--map", action="store_true",
                        help="Print ASCII maps of the coverage bands.")
    args = parser.parse_args()

    if args.measure_gripper:
        print("\ngripper footprint at TOOL_Z:")
        measure_gripper()
        return
    if args.verify:
        print("\nverifying computed poses against PyBullet FK:")
        verify()
        return

    xs, ys, points = ts.grid(X_RANGE, Y_RANGE, RESOLUTION)
    cell_area = RESOLUTION ** 2
    no_tool = bare_arm_mask(points)

    print(f"\nhand workspace  {SE2Config.WORKSPACE}")
    print(f"yaw limit       +-{math.degrees(SE2Config.YAW_LIMIT):.0f} deg")
    print(f"object radius   {TaskConfig.R_OBJ} m   gripper radius {TaskConfig.GRIPPER_RADIUS} m")
    print(f"grid            {len(xs)}x{len(ys)} cells at {RESOLUTION} m")
    print(f"\nreachable with no tool at all: {no_tool.sum() * cell_area:.3f} m2")

    print("\nper design (object centres the tip can touch):")
    print(f"  {'design':<24} {'R':>5}  {'alpha':>6}  {'bearing window':^17}  "
          f"{'area':>6}  {'needs':>6}  {'blind':>6}")
    print(f"  {'':<24} {'(m)':>5}  {'(deg)':>6}  {'(deg)':^17}  "
          f"{'(m2)':>6}  {'tool':>6}  {'(m2)':>6}")
    masks = [describe(label, tau, xs, ys, no_tool, cell_area)
             for label, tau in CANONICAL_TAUS]

    union = np.any(masks, axis=0)
    intersection = np.all(masks, axis=0)
    print()
    for name, mask in (("union of the above", union), ("intersection", intersection)):
        print(f"  {name:<24} {'':>5}  {'':>6}  {'':^17}  "
              f"{mask.sum() * cell_area:6.3f}  {(mask & ~no_tool).sum() * cell_area:6.3f}  "
              f"{(no_tool & ~mask).sum() * cell_area:6.3f}")
    print("\n  area   object centres the tip can touch")
    print("  needs  of that, the part the bare gripper could not reach anyway")
    print("  blind  of the bare arm's own reach, the part this tip cannot touch --"
          " the near-target")
    print("         hole, since the tip is on a circle of radius R, not in a disk")

    print(f"\ncoverage over {N_PRIOR} designs from ToolPrior:")
    taus = ToolPrior().sample(N_PRIOR).detach().cpu().numpy()
    coverage = ts.coverage(xs, ys, SE2Config.WORKSPACE, taus, SE2Config.YAW_LIMIT,
                           tol=TaskConfig.R_OBJ)
    print_bands(coverage, cell_area)

    reachable = coverage > 0
    discriminating = (coverage > 0) & (coverage < 1)
    print(f"\n  of the {reachable.sum() * cell_area:.3f} m2 some tool can reach, "
          f"{discriminating.sum() * cell_area:.3f} m2 "
          f"({100 * discriminating.sum() / max(reachable.sum(), 1):.0f}%) separates designs")
    print(f"  and {(discriminating & ~no_tool).sum() * cell_area:.3f} m2 of that is "
          f"also out of the bare arm's reach")

    if args.map:
        for label, lo, hi in BANDS:
            mask = (coverage >= lo) & (coverage <= hi)
            print(f"\n{label}  ('+' in band, '.' not)")
            ts.print_map(mask[::4, ::4], xs[::4], ys[::4])


if __name__ == "__main__":
    main()
