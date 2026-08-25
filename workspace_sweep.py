"""Offline calibration: measure where the arm can actually hold the tool flat.

Run once, paste the printed rectangle into config.SE2Config.WORKSPACE. Not on the
training path.

The point is to define the wall ourselves rather than let the IK solver improvise
one. calculateInverseKinematics never refuses an unreachable target -- it returns
its best effort, typically buying the extra centimetre by tilting the wrist a few
degrees. That silently breaks the flat-2D abstraction the whole speedup rests on.
So: sweep a grid, keep only cells where the achieved pose is genuinely flat at
*every* yaw, take the largest clean rectangle, and inset it for margin.

Usage:
    python workspace_sweep.py
"""
import numpy as np
import pybullet as p
from panda_gym.pybullet import PyBullet

import se2
from config import ArmConfig, SE2Config
from panda_with_tool import PandaWithTool

# The hand target is tau-independent (se2.py module docstring), so any design does.
SWEEP_TAU = (0.3, 0.3, 0.0)

# Sweep extent, robot base at the origin. Generous enough to bracket the Panda's
# ~0.85 m reach without wasting cells behind the base.
X_RANGE = (0.10, 0.90)
Y_RANGE = (-0.50, 0.50)
RESOLUTION = 0.01

# A cell counts only if every yaw in the allowed range passes. A rectangle that
# works at yaw=0 is worthless the moment the policy spins the tool. Swept over the
# full circle so any candidate SE2Config.YAW_LIMIT can be scored from one run.
N_YAW = 12

POSITION_TOL = 0.005          # m, how far the achieved hand may sit from the target
HEIGHT_TOL = 0.002            # m, deviation from SE2Config.HAND_Z
TILT_TOL = np.deg2rad(1.0)    # rad, deviation from exactly fingers-down
YAW_TOL = np.deg2rad(5.0)     # rad, deviation from the commanded yaw

JOINT_LIMITS = np.array(ArmConfig.JOINT_LIMITS)


def cell_is_clean(robot, x, y, yaw):
    """Whether the arm can hold the tool flat at this pose, within tolerance.

    Args:
        robot (PandaWithTool): The robot, which is teleported as a side effect.
        x (float): Target hand x.
        y (float): Target hand y.
        yaw (float): Target hand yaw.

    Returns:
        bool: True if position, height, tilt and joint limits all pass.
    """
    robot.set_se2((x, y), yaw)

    angles = np.array([robot.get_joint_angle(joint=i) for i in range(7)])
    if np.any(angles < JOINT_LIMITS[:, 0]) or np.any(angles > JOINT_LIMITS[:, 1]):
        return False

    achieved = robot.get_hand_position()
    if np.linalg.norm(achieved[:2] - np.array([x, y])) > POSITION_TOL:
        return False
    if abs(achieved[2] - SE2Config.HAND_Z) > HEIGHT_TOL:
        return False
    if robot.get_hand_tilt() > TILT_TOL:
        return False

    achieved_yaw = se2.yaw_from_matrix(robot.get_hand_rotation())
    return abs(se2.wrap_angle(achieved_yaw - yaw)) <= YAW_TOL


def sweep(robot, xs, ys, yaws):
    """Per-yaw feasibility masks over the grid.

    Kept per-yaw rather than pre-intersected so candidate yaw ranges can be
    compared from a single sweep. The intersection over all yaws is dominated by
    the worst one -- yaws pointing the tool back at the robot's own base are much
    harder to reach than yaws pointing outward -- so which range to allow is a
    decision this data should make, not a guess.

    Returns:
        np.ndarray: Boolean array of shape (len(yaws), len(xs), len(ys)).
    """
    masks = np.zeros((len(yaws), len(xs), len(ys)), dtype=bool)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, yaw in enumerate(yaws):
                masks[k, i, j] = cell_is_clean(robot, x, y, yaw)
        print(f"  x={x:+.2f}  ({i + 1}/{len(xs)})", end="\r", flush=True)
    print(" " * 40, end="\r")
    return masks


def largest_rectangle(mask):
    """Largest all-True axis-aligned rectangle in a boolean mask.

    Standard maximal-rectangle-in-histogram scan: build the run of consecutive True
    cells ending at each row, then for each row solve the largest-rectangle-in-a-
    histogram problem with a monotonic stack. O(n*m).

    Args:
        mask (np.ndarray): 2D boolean array.

    Returns:
        tuple: (i0, i1, j0, j1) inclusive index bounds, or None if mask is all False.
    """
    n_rows, n_cols = mask.shape
    heights = np.zeros(n_cols, dtype=int)
    best = (0, None)

    for i in range(n_rows):
        heights = np.where(mask[i], heights + 1, 0)
        stack = []  # (start_col, height), heights strictly increasing
        for j in range(n_cols + 1):
            h = heights[j] if j < n_cols else 0
            start = j
            while stack and stack[-1][1] >= h:
                col, height = stack.pop()
                area = height * (j - col)
                if area > best[0]:
                    best = (area, (i - height + 1, i, col, j - 1))
                start = col
            if h > 0:
                stack.append((start, h))

    return best[1]


def print_map(mask, xs, ys, rect):
    """ASCII map of the mask with the chosen rectangle overlaid.

    matplotlib is not installed and utils/plots.py is empty; a new dependency is
    not worth it for a script run once.
    """
    i0, i1, j0, j1 = rect if rect else (-1, -2, -1, -2)
    print(f"\n  rows = x from {xs[0]:+.2f} to {xs[-1]:+.2f}, "
          f"cols = y from {ys[0]:+.2f} to {ys[-1]:+.2f}")
    print("  '#' clean and inside the rectangle, '+' clean, '.' not clean\n")
    for i in range(len(xs)):
        row = "".join(
            ("#" if i0 <= i <= i1 and j0 <= j <= j1 else "+") if mask[i, j] else "."
            for j in range(len(ys))
        )
        print(f"  x={xs[i]:+.2f} |{row}|")


def box_for(masks, yaws, xs, ys, half_width):
    """Largest clean rectangle when yaw is restricted to +-half_width.

    Returns:
        tuple: (raw_box, inset_box, n_cells), or (None, None, 0) if nothing is
            clean across the whole yaw range. inset_box is None when the margin
            would collapse the rectangle.
    """
    keep = np.abs(np.array([se2.wrap_angle(y) for y in yaws])) <= half_width + 1e-9
    mask = masks[keep].all(axis=0)
    rect = largest_rectangle(mask)
    if rect is None:
        return None, None, 0
    i0, i1, j0, j1 = rect
    raw = se2.Box(x_min=xs[i0], x_max=xs[i1], y_min=ys[j0], y_max=ys[j1])
    try:
        inset = raw.shrink(SE2Config.SWEEP_MARGIN)
    except ValueError:
        inset = None
    return raw, inset, int(mask.sum())


def main():
    sim = PyBullet(render_mode="rgb_array")
    robot = PandaWithTool(sim, SWEEP_TAU)

    xs = np.arange(X_RANGE[0], X_RANGE[1] + 1e-9, RESOLUTION)
    ys = np.arange(Y_RANGE[0], Y_RANGE[1] + 1e-9, RESOLUTION)
    yaws = np.linspace(-np.pi, np.pi, N_YAW, endpoint=False)

    print(f"sweeping {len(xs)}x{len(ys)} cells at {len(yaws)} yaws "
          f"({len(xs) * len(ys) * len(yaws)} IK solves)")
    masks = sweep(robot, xs, ys, yaws)

    print("\ncells clean per yaw:")
    for yaw, m in zip(yaws, masks):
        print(f"  yaw={yaw:+.2f}: {m.sum()}")

    print("\nlargest clean rectangle by allowed yaw range:")
    print(f"  {'+-yaw':>8}  {'cells':>6}  {'area':>7}  rectangle after inset")
    candidates = [np.pi / 4, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 3 * np.pi / 4, np.pi]
    for half_width in candidates:
        raw, inset, n = box_for(masks, yaws, xs, ys, half_width)
        if inset is None:
            print(f"  {np.degrees(half_width):7.0f}d  {n:6d}  {'--':>7}  (collapses under margin)")
            continue
        area = (inset.x_max - inset.x_min) * (inset.y_max - inset.y_min)
        print(f"  {np.degrees(half_width):7.0f}d  {n:6d}  {area:7.4f}  "
              f"x[{inset.x_min:+.3f},{inset.x_max:+.3f}] y[{inset.y_min:+.3f},{inset.y_max:+.3f}]")

    chosen = SE2Config.YAW_LIMIT
    raw, inset, _ = box_for(masks, yaws, xs, ys, chosen)
    keep = np.abs(np.array([se2.wrap_angle(y) for y in yaws])) <= chosen + 1e-9
    print_map(masks[keep].all(axis=0), xs, ys, largest_rectangle(masks[keep].all(axis=0)))

    if inset is None:
        print("\nNothing clean across the configured yaw range. Lower SE2Config.YAW_LIMIT.")
        return

    print(f"\nat the configured SE2Config.YAW_LIMIT = {np.degrees(chosen):.0f} deg")
    print(f"largest clean rectangle: {raw}")
    print(f"inset by SWEEP_MARGIN={SE2Config.SWEEP_MARGIN}:\n")
    print(f"    WORKSPACE = se2.Box(x_min={inset.x_min:.3f}, x_max={inset.x_max:.3f}, "
          f"y_min={inset.y_min:.3f}, y_max={inset.y_max:.3f})")

    p.disconnect()


if __name__ == "__main__":
    main()
