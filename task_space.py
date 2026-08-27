"""Where an *object* may sit, as opposed to where the *hand* may go. No PyBullet, no I/O.

``config.SE2Config.WORKSPACE`` answers the second question: it is the rectangle the
controller clips the hand into (``PandaWithTool.set_action``), measured by
``workspace_sweep.py``. The task layer needs the first: given a tool design, which
object positions can the tool tip actually touch? That is what ``s_start`` and
``p_target`` are drawn from in the task encoding (``ai_docs/task_encoding_g.md``).

The reach region is closed form, so nothing here needs a simulator. ``se2.tip_polar``
puts the tip at ``hand + R*u(psi + alpha)``, and the workspace sweep only accepted a
cell if it was clean at *every* yaw in ``+-YAW_LIMIT``, so every ``(hand, psi)`` in
``WORKSPACE x [-Psi, Psi]`` is available. Hence

    Reach(tau) = ( WORKSPACE (+) Arc(R, [alpha-Psi, alpha+Psi]) ) (+) Disk(r_obj)

whose membership test reduces to a 1-D scan over the arc, since the distance from a
point to an axis-aligned box is itself closed form. See ``tip_reachable``.

Two consequences of ``R`` being an *exact* radius rather than a bound, both of which
this module's callers keep tripping over and both of which have tests:

- The tip lies on a *circle* about the hand, not in a disk, so a long tool cannot
  reach a target near the hand. Reach feasibility is not monotone in tool length.
- ``alpha`` rotates the accessible half-plane of bearings. Length sets how far the
  tip goes; ``alpha`` sets which way it can go.

Like ``se2`` and ``tool_geometry`` this module imports no config: the functions are
pure and the caller supplies the box, the yaw limit and the object radius.

Caveat inherited from the rest of the stack: arm/tool self-collision is disabled
(see ``plan.md``), so for designs whose bearing reaches far backwards the region
computed here can include poses where the tool passes through the arm's own links.
Neither this map nor PyBullet will report it.
"""
import numpy as np

import se2


def grid(x_range, y_range, resolution):
    """Regular grid of candidate object positions over the table.

    Args:
        x_range: ``(x_min, x_max)`` inclusive, in metres.
        y_range: ``(y_min, y_max)`` inclusive, in metres.
        resolution (float): Cell size in metres.

    Returns:
        tuple: ``(xs, ys, points)`` where ``points`` has shape ``(len(xs), len(ys), 2)``
            so a mask over it indexes as ``mask[i, j]`` for ``(xs[i], ys[j])`` -- the
            same row-is-x convention ``workspace_sweep`` and ``utils.plots`` use.
    """
    xs = np.arange(x_range[0], x_range[1] + 1e-9, resolution)
    ys = np.arange(y_range[0], y_range[1] + 1e-9, resolution)
    points = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)
    return xs, ys, points


def box_distance(points, box):
    """Euclidean distance from each point to an ``se2.Box``, zero inside it.

    The standard axis-aligned form: clamp the per-axis overshoot at zero and take
    the norm, which is exact both outside the box and in the corner regions where
    the nearest feature is a vertex rather than an edge.

    Args:
        points: Array of points, shape ``(..., 2)``.
        box (se2.Box): The rectangle.

    Returns:
        np.ndarray: Distances, shape ``points.shape[:-1]``.
    """
    points = np.asarray(points, dtype=float)
    dx = np.maximum(np.maximum(box.x_min - points[..., 0], points[..., 0] - box.x_max), 0.0)
    dy = np.maximum(np.maximum(box.y_min - points[..., 1], points[..., 1] - box.y_max), 0.0)
    return np.hypot(dx, dy)


def _bearings(tau, yaw_limit, n_theta):
    """The arc of tip bearings the yaw limit allows, plus the tip radius.

    Returns:
        tuple: ``(radius, thetas)`` with ``thetas`` of shape ``(n_theta,)``.
    """
    radius, alpha = se2.tip_polar(tau)
    thetas = alpha + np.linspace(-yaw_limit, yaw_limit, n_theta)
    return radius, thetas


def tip_reachable(points, box, tau, yaw_limit, tol=0.0, n_theta=720):
    """Whether the tool tip can be brought within ``tol`` of each point.

    Evaluates ``exists theta in [alpha-Psi, alpha+Psi]: dist(q - R*u(theta), box) <= tol``
    on a uniform scan over the arc. The scan is a discretisation of a continuous
    condition, but a conservative one in a controlled way: consecutive samples are
    ``R*2*Psi/n_theta`` apart along the arc, which at the default is 0.35 mm for the
    longest tool in the design prior -- well under any tolerance the task uses.

    Args:
        points: Candidate object positions, shape ``(..., 2)``.
        box (se2.Box): Where the hand may go, i.e. ``SE2Config.WORKSPACE``.
        tau: Design parameters ``(l1, l2, phi)``.
        yaw_limit (float): Hand yaw is clipped to ``+-yaw_limit``.
        tol (float): Touch tolerance. Pass the object radius to get the set of object
            *centres* the tip can touch; pass 0 to get the bare tip locus.
        n_theta (int): Samples along the bearing arc.

    Returns:
        np.ndarray: Boolean array of shape ``points.shape[:-1]``.
    """
    points = np.asarray(points, dtype=float)
    radius, thetas = _bearings(tau, yaw_limit, n_theta)
    offsets = radius * np.stack([np.cos(thetas), np.sin(thetas)], axis=-1)
    # (..., 1, 2) - (n_theta, 2) -> every candidate hand position for every bearing.
    hands = points[..., None, :] - offsets
    return np.any(box_distance(hands, box) <= tol, axis=-1)


def reach_mask(xs, ys, box, tau, yaw_limit, tol=0.0):
    """``tip_reachable`` over a whole regular grid, by dilation instead of scanning.

    Same set, different algorithm. ``tip_reachable`` asks, for each point
    independently, whether any bearing works -- fine for a handful of queries, but
    it costs ``n_cells * n_theta`` distance evaluations and the map scripts run it
    thousands of times. On a *regular* grid the Minkowski sum can be taken directly:
    rasterise ``box (+) Disk(tol)`` once, then OR in a copy of it shifted by each
    arc offset, which is a slice assignment rather than an arithmetic pass. Distinct
    offsets are deduplicated, so the work is set by the arc's length in cells (~250
    for the longest tool at 5 mm) rather than by any sampling density.

    Roughly 100x faster than the equivalent ``tip_reachable`` call, at the cost of
    quantising each offset to the nearest cell: the boundary can move by up to half
    a cell diagonal. Use this for maps and ``tip_reachable`` for exact queries; a
    test asserts they agree everywhere except within one cell of the boundary.

    Args:
        xs (np.ndarray): Uniformly spaced grid x coordinates.
        ys (np.ndarray): Uniformly spaced grid y coordinates.
        box (se2.Box): Where the hand may go.
        tau: Design parameters ``(l1, l2, phi)``.
        yaw_limit (float): Hand yaw is clipped to ``+-yaw_limit``.
        tol (float): Touch tolerance, as in ``tip_reachable``.

    Returns:
        np.ndarray: Boolean array of shape ``(len(xs), len(ys))``.
    """
    step_x = float(xs[1] - xs[0])
    step_y = float(ys[1] - ys[0])
    points = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)
    base = box_distance(points, box) <= tol

    # One sample per half-cell along the arc guarantees no offset is skipped; the
    # dedupe then collapses them down to the offsets that actually differ.
    radius, alpha = se2.tip_polar(tau)
    step = min(step_x, step_y)
    n_theta = max(2, int(np.ceil(4.0 * radius * yaw_limit / step)))
    thetas = alpha + np.linspace(-yaw_limit, yaw_limit, n_theta)
    offsets = np.unique(np.stack([
        np.rint(radius * np.cos(thetas) / step_x),
        np.rint(radius * np.sin(thetas) / step_y),
    ], axis=-1).astype(int), axis=0)

    n_i, n_j = base.shape
    out = np.zeros_like(base)
    for di, dj in offsets:
        src_i = slice(max(0, -di), n_i - max(0, di))
        dst_i = slice(max(0, di), n_i - max(0, -di))
        src_j = slice(max(0, -dj), n_j - max(0, dj))
        dst_j = slice(max(0, dj), n_j - max(0, -dj))
        if src_i.start >= src_i.stop or src_j.start >= src_j.stop:
            continue
        out[dst_i, dst_j] |= base[src_i, src_j]
    return out


def hand_reachable(points, box, radius):
    """Whether the bare hand could touch an object at each point.

    The counterfactual the tool has to beat: ``box (+) Disk(radius)`` with
    ``radius = r_obj + gripper_radius``. A target in here is solvable without a tool
    at all, so it carries no design signal however the policy behaves -- which is why
    the reach maps report the area *outside* this band separately.

    Args:
        points: Candidate object positions, shape ``(..., 2)``.
        box (se2.Box): Where the hand may go.
        radius (float): Object radius plus the gripper's planar footprint radius.

    Returns:
        np.ndarray: Boolean array of shape ``points.shape[:-1]``.
    """
    return box_distance(points, box) <= radius


def hand_pose_for_tip(point, box, tau, yaw_limit, tol=0.0, n_theta=720):
    """A hand pose putting the tool tip within ``tol`` of ``point``, or None.

    Inverts the same scan ``tip_reachable`` runs, and returns the witness rather
    than just the verdict: of the bearings that work, take the one whose hand
    position sits deepest inside the box, so the pose has margin on every side
    instead of being pressed against the wall the scan happened to find first.

    Args:
        point: Target object position as ``(x, y)``.
        box (se2.Box): Where the hand may go.
        tau: Design parameters ``(l1, l2, phi)``.
        yaw_limit (float): Hand yaw is clipped to ``+-yaw_limit``.
        tol (float): Touch tolerance, as in ``tip_reachable``.
        n_theta (int): Samples along the bearing arc.

    Returns:
        np.ndarray: Hand pose ``(x, y, yaw)``, or None if the point is out of reach.
            The yaw is guaranteed to lie within ``+-yaw_limit``.
    """
    point = np.asarray(point, dtype=float)[:2]
    radius, alpha = se2.tip_polar(tau)
    _, thetas = _bearings(tau, yaw_limit, n_theta)
    hands = point - radius * np.stack([np.cos(thetas), np.sin(thetas)], axis=-1)

    distances = box_distance(hands, box)
    feasible = np.flatnonzero(distances <= tol)
    if feasible.size == 0:
        return None

    # Deepest inside the box among the feasible bearings. Points strictly inside all
    # score distance 0, so break that tie on clearance from the nearest wall.
    clipped = np.stack([box.clip(hands[k]) for k in feasible])
    margin = np.minimum(
        np.minimum(clipped[:, 0] - box.x_min, box.x_max - clipped[:, 0]),
        np.minimum(clipped[:, 1] - box.y_min, box.y_max - clipped[:, 1]),
    )
    best = feasible[int(np.argmax(margin - distances[feasible]))]
    xy = box.clip(hands[best])
    return np.array([xy[0], xy[1], se2.wrap_angle(thetas[best] - alpha)])


def coverage(xs, ys, box, taus, yaw_limit, tol=0.0):
    """Fraction of the given designs that can reach each point.

    The map the task encoding actually needs. A cell at coverage 1 is reachable
    whatever tool is sampled, so conditioning ``p(tau | g, O=1)`` on a target there
    cannot separate designs; a cell at 0 is impossible for all of them. The
    discriminating band is strictly between, and that is where ``s_start`` and
    ``p_target`` belong if the diagnostic in ``ai_docs/task_encoding_g.md`` is to
    have anything to detect.

    Built on ``reach_mask`` rather than ``tip_reachable``: this is a grid-only
    quantity and the sample runs to thousands of designs.

    Args:
        xs (np.ndarray): Uniformly spaced grid x coordinates.
        ys (np.ndarray): Uniformly spaced grid y coordinates.
        box (se2.Box): Where the hand may go.
        taus: Iterable of ``(l1, l2, phi)`` designs, e.g. from ``ToolPrior.sample``.
        yaw_limit (float): Hand yaw is clipped to ``+-yaw_limit``.
        tol (float): Touch tolerance, as in ``tip_reachable``.

    Returns:
        np.ndarray: Fractions in ``[0, 1]``, shape ``(len(xs), len(ys))``.
    """
    taus = list(taus)
    if not taus:
        raise ValueError("coverage needs at least one design")
    total = np.zeros((len(xs), len(ys)), dtype=float)
    for tau in taus:
        total += reach_mask(xs, ys, box, tau, yaw_limit, tol=tol)
    return total / len(taus)


def largest_rectangle(mask):
    """Largest all-True axis-aligned rectangle in a boolean mask.

    Standard maximal-rectangle-in-histogram scan: build the run of consecutive True
    cells ending at each row, then for each row solve the largest-rectangle-in-a-
    histogram problem with a monotonic stack. O(n*m).

    Lives here rather than in ``workspace_sweep`` (where it started) so the pure
    analysis scripts can use it without importing PyBullet and panda-gym.

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


def print_map(mask, xs, ys, rect=None, legend=("#", "+", ".")):
    """ASCII map of a boolean mask, optionally with a chosen rectangle overlaid.

    matplotlib figures live in ``utils.plots``; this is for scripts whose output is
    meant to be read in a terminal or pasted into a commit message.

    Args:
        mask (np.ndarray): Boolean array of shape ``(len(xs), len(ys))``.
        xs (np.ndarray): Grid x coordinates.
        ys (np.ndarray): Grid y coordinates.
        rect: Optional ``(i0, i1, j0, j1)`` index bounds to mark differently.
        legend (tuple): Characters for (inside rect, True, False).
    """
    inside_char, true_char, false_char = legend
    i0, i1, j0, j1 = rect if rect else (-1, -2, -1, -2)
    print(f"  rows = x from {xs[0]:+.2f} to {xs[-1]:+.2f}, "
          f"cols = y from {ys[0]:+.2f} to {ys[-1]:+.2f}")
    for i in range(len(xs)):
        row = "".join(
            (inside_char if i0 <= i <= i1 and j0 <= j <= j1 else true_char)
            if mask[i, j] else false_char
            for j in range(len(ys))
        )
        print(f"  x={xs[i]:+.2f} |{row}|")
