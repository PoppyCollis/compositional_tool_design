"""Pure geometry/mass/inertia for a 2-link (l1, l2, phi) tool. No PyBullet, no I/O.

Tool frame: origin at the grip point (proximal end of link 1). Link 1 runs
along +z. phi is the deflection of link 2 from straight: phi=0 is a straight
rod (the longest tool), phi -> +-pi folds link 2 back alongside link 1. The
interior elbow angle is theta = pi - |phi|. The bend is fixed to the tool
frame's x-z plane (any other plane is reachable via a wrist rotation), so
+phi and -phi are mirror-image tools, equivalent under a wrist roll.

Nothing here enforces a range on phi -- these are pure functions and the
module stays free of config imports. Callers bound it: ToolPrior.sample and
ToolPrior.constrain hold phi within +-DesignPriorConfig.PHI_MAX so link 2
cannot reach back into the gripper (see config.py for that derivation).
"""
import numpy as np
import torch

W = 0.02  # cross-section width in m 
H = 0.02  # cross-section height in m
RHO = 1250.0  # solid-PLA density in kg/m^3


def _unpack(tau):
    """Detach a single (3,) tau and return three floats.

    Deliberately rejects batches. Geometry is per-design -- it feeds URDF
    generation, which builds one body at a time -- and a batched tau arriving
    here means a caller wired something up wrong. Flattening it would silently
    return the first row's (l1, l2, phi) and discard the rest.
    """
    if isinstance(tau, torch.Tensor):
        tau = tau.detach().cpu().numpy()
    tau = np.asarray(tau, dtype=float)
    if tau.shape != (3,):
        raise ValueError(
            f"tau must be a single (l1, l2, phi) design with shape (3,), got "
            f"shape {tau.shape}. Geometry is per-design; iterate over a batch "
            f"explicitly."
        )
    return float(tau[0]), float(tau[1]), float(tau[2])


def interior_angle(phi):
    """Interior elbow angle theta = pi - |phi|, the angle between the two links.

    theta=pi is a straight rod; theta -> 0 folds link 2 back alongside link 1.
    Takes |phi| because an angle between two segments is unsigned and lies in
    [0, pi]: +phi and -phi are mirror-image tools and share the same elbow.
    """
    return np.pi - np.abs(phi)


def link_directions(phi):
    """Unit direction vectors d1, d2 for link 1 and link 2 in the tool frame."""
    d1 = np.array([0.0, 0.0, 1.0])
    d2 = np.array([np.sin(phi), 0.0, np.cos(phi)])
    return d1, d2


def tip_position(l1, l2, phi):
    """Analytic tool-tip position p_tip = l1*d1 + l2*d2 in the tool frame."""
    d1, d2 = link_directions(phi)
    return l1 * d1 + l2 * d2


def box_specs(l1, l2, phi):
    """Size and pose (position, rpy) of each of the two collision/visual boxes."""
    d1, d2 = link_directions(phi)
    box1 = {
        "size": (W, H, l1),
        "pos": np.array([0.0, 0.0, l1 / 2.0]),
        "rpy": (0.0, 0.0, 0.0),
    }
    box2_pos = l1 * d1 + (l2 / 2.0) * d2
    box2 = {
        "size": (W, H, l2),
        "pos": box2_pos,
        "rpy": (0.0, phi, 0.0),
    }
    return {"box1": box1, "box2": box2}


def mass(l1, l2):
    """Total tool mass: constant density times total link length times cross-section."""
    return RHO * W * H * (l1 + l2)


def combined_com(l1, l2, phi):
    """Mass-weighted centroid of the two boxes, in the tool frame."""
    m1 = RHO * W * H * l1
    m2 = RHO * W * H * l2
    specs = box_specs(l1, l2, phi)
    com1 = specs["box1"]["pos"]
    com2 = specs["box2"]["pos"]
    total = m1 + m2
    return (m1 * com1 + m2 * com2) / total


def _box_inertia_local(m, size):
    """Inertia tensor of a solid box about its own centroid, box-aligned axes."""
    w, h, l = size
    ixx = (m / 12.0) * (h**2 + l**2)
    iyy = (m / 12.0) * (w**2 + l**2)
    izz = (m / 12.0) * (w**2 + h**2)
    return np.diag([ixx, iyy, izz])


def _rotation_from_rpy(rpy):
    """Rotation matrix for URDF-style fixed-axis roll-pitch-yaw (R = Rz*Ry*Rx)."""
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def inertia_tensor(l1, l2, phi):
    """Full symmetric 3x3 inertia tensor about the combined COM, in the tool frame."""
    specs = box_specs(l1, l2, phi)
    com = combined_com(l1, l2, phi)

    total = np.zeros((3, 3))
    for key, m_i in (("box1", RHO * W * H * l1), ("box2", RHO * W * H * l2)):
        box = specs[key]
        i_local = _box_inertia_local(m_i, box["size"])
        r = _rotation_from_rpy(box["rpy"])
        i_world = r @ i_local @ r.T

        offset = box["pos"] - com
        d2 = offset @ offset
        parallel_axis = m_i * (d2 * np.eye(3) - np.outer(offset, offset))

        total += i_world + parallel_axis

    return total


def tau_to_geometry(tau):
    """Convenience wrapper: unpack a torch tau tensor and return all derived quantities."""
    l1, l2, phi = _unpack(tau)
    return {
        "l1": l1,
        "l2": l2,
        "phi": phi,
        "tip_position": tip_position(l1, l2, phi),
        "box_specs": box_specs(l1, l2, phi),
        "mass": mass(l1, l2),
        "com": combined_com(l1, l2, phi),
        "inertia": inertia_tensor(l1, l2, phi),
    }
