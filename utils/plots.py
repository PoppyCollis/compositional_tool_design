"""Plotting helpers. Pure matplotlib -- no pybullet/robot imports here, so this
stays usable from a script that has already disconnected the sim.
"""
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def plot_workspace(xs, ys, mask, boxes=(), contours=(), robot_origin=(0.0, 0.0),
                   title=None, ax=None, cmap="Greens", vmax=1, colorbar=None,
                   legend_kwargs=None):
    """Top-down map of a measured region over the table.

    Draws one filled field (a boolean feasibility mask, or a continuous one such as
    task_space.coverage's fraction-of-designs map), with any number of rectangles
    and mask outlines on top.

    Note the axes are transposed relative to the arrays: x runs up the figure and y
    across it, matching how the robot's own workspace reads when you look down at
    the table. Masks are indexed ``mask[i, j]`` for ``(xs[i], ys[j])``, the
    convention task_space.grid produces.

    Args:
        xs (np.ndarray): Swept x coordinates, shape (n_x,).
        ys (np.ndarray): Swept y coordinates, shape (n_y,).
        mask (np.ndarray): Array of shape (n_x, n_y) -- boolean, or continuous in
            [0, vmax]. E.g. workspace_sweep.sweep()'s per-yaw masks reduced with
            .all(axis=0), or task_space.coverage().
        boxes (list): (se2.Box, label, matplotlib Rectangle kwargs) triples to
            draw on top, e.g. the raw largest-clean-rectangle and the margin-
            inset SE2Config.WORKSPACE actually used at runtime.
        contours (list): (mask, label, matplotlib kwargs) triples drawn as outlines,
            for overlaying several regions that are not rectangles -- e.g. one reach
            region per tool design.
        robot_origin (tuple): (x, y) of the robot base, marked with an "x".
        title (str): Optional axes title.
        ax (matplotlib.axes.Axes): Axes to draw into; a new figure is made if
            omitted.
        cmap (str): Colormap for the filled field.
        vmax (float): Value mapped to the top of the colormap.
        colorbar (str): Label for a colorbar; omitted if None.
        legend_kwargs (dict): Overrides for the legend call, e.g. a different
            location or column count when many outlines crowd the default corner.

    Returns:
        matplotlib.axes.Axes: The axes drawn into.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    x_step = xs[1] - xs[0] if len(xs) > 1 else 0.01
    y_step = ys[1] - ys[0] if len(ys) > 1 else 0.01
    extent = (ys[0] - y_step / 2, ys[-1] + y_step / 2,
              xs[0] - x_step / 2, xs[-1] + x_step / 2)
    image = ax.imshow(mask, origin="lower", extent=extent, aspect="equal",
                      cmap=cmap, vmin=0, vmax=vmax)
    if colorbar:
        ax.figure.colorbar(image, ax=ax, label=colorbar, fraction=0.046, pad=0.04)

    for box, label, kwargs in boxes:
        style = dict(fill=False, linewidth=2)
        style.update(kwargs)
        rect = patches.Rectangle(
            (box.y_min, box.x_min), box.y_max - box.y_min, box.x_max - box.x_min,
            label=label, **style,
        )
        ax.add_patch(rect)

    # contour takes Z indexed [row, col] = [y_axis, x_axis] of the *plot*, which is
    # (xs, ys) here -- the same transpose the extent above applies. A ContourSet is
    # not a legend handle, so each outline also gets an empty proxy line, translated
    # from contour's plural kwarg spellings into plot's singular ones.
    for outline, label, kwargs in contours:
        style = dict(levels=[0.5], linewidths=1.2, colors=["tab:blue"])
        style.update(kwargs)
        ax.contour(ys, xs, np.asarray(outline, dtype=float), **style)
        proxy = {"color": np.atleast_1d(style["colors"])[0],
                 "linewidth": np.atleast_1d(style["linewidths"])[0]}
        if "linestyles" in style:
            proxy["linestyle"] = np.atleast_1d(style["linestyles"])[0]
        ax.plot([], [], label=label, **proxy)

    ax.scatter(*robot_origin[::-1], marker="x", color="black", s=80, zorder=5,
               label="robot base")

    ax.set_xlabel("y (m)")
    ax.set_ylabel("x (m)")
    if title:
        ax.set_title(title)
    ax.legend(**{"loc": "upper right", "fontsize": 8, **(legend_kwargs or {})})
    return ax
