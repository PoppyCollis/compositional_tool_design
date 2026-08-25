"""Plotting helpers. Pure matplotlib -- no pybullet/robot imports here, so this
stays usable from a script that has already disconnected the sim.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_workspace(xs, ys, mask, boxes=(), robot_origin=(0.0, 0.0), title=None, ax=None):
    """Top-down map of a measured SE(2) workspace.

    Args:
        xs (np.ndarray): Swept x coordinates, shape (n_x,).
        ys (np.ndarray): Swept y coordinates, shape (n_y,).
        mask (np.ndarray): Boolean array, shape (n_x, n_y). True where the arm
            can hold the tool flat -- e.g. workspace_sweep.sweep()'s per-yaw
            masks reduced with .all(axis=0) over the allowed yaw range.
        boxes (list): (se2.Box, label, matplotlib Rectangle kwargs) triples to
            draw on top, e.g. the raw largest-clean-rectangle and the margin-
            inset SE2Config.WORKSPACE actually used at runtime.
        robot_origin (tuple): (x, y) of the robot base, marked with an "x".
        title (str): Optional axes title.
        ax (matplotlib.axes.Axes): Axes to draw into; a new figure is made if
            omitted.

    Returns:
        matplotlib.axes.Axes: The axes drawn into.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    x_step = xs[1] - xs[0] if len(xs) > 1 else 0.01
    y_step = ys[1] - ys[0] if len(ys) > 1 else 0.01
    extent = (ys[0] - y_step / 2, ys[-1] + y_step / 2,
              xs[0] - x_step / 2, xs[-1] + x_step / 2)
    ax.imshow(mask, origin="lower", extent=extent, aspect="equal",
              cmap="Greens", vmin=0, vmax=1)

    for box, label, kwargs in boxes:
        style = dict(fill=False, linewidth=2)
        style.update(kwargs)
        rect = patches.Rectangle(
            (box.y_min, box.x_min), box.y_max - box.y_min, box.x_max - box.x_min,
            label=label, **style,
        )
        ax.add_patch(rect)

    ax.scatter(*robot_origin[::-1], marker="x", color="black", s=80, zorder=5,
               label="robot base")

    ax.set_xlabel("y (m)")
    ax.set_ylabel("x (m)")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    return ax
