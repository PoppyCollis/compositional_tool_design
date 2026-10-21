"""tau -> URDF for a Panda hand with the tool welded in and fingers welded shut
around it. Splices tool_urdf.py's tool link tree into the real Panda URDF
(copied out of pybullet_data into assets/, never mutated in site-packages).
"""
import atexit
import functools
import hashlib
import os
import shutil
import xml.etree.ElementTree as ET

import pybullet as p
import pybullet_data

import tool_geometry as geom
import tool_urdf
from utils.helpers import get_link_index_by_name

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
PANDA_ASSET_DIR = os.path.join(ASSETS_DIR, "franka_panda")
PANDA_URDF_PATH = os.path.join(PANDA_ASSET_DIR, "panda.urdf")
GENERATED_DIR = PANDA_ASSET_DIR  # generated URDFs must sit next to meshes/ for package:// paths to resolve

FINGER_JOINTS = ("panda_finger_joint1", "panda_finger_joint2")
FINGER_LINKS = ("panda_leftfinger", "panda_rightfinger")


def ensure_assets():
    """Copy pybullet_data's franka_panda/ (URDF + meshes) into assets/ if not already there."""
    if os.path.isdir(PANDA_ASSET_DIR):
        return
    src = os.path.join(pybullet_data.getDataPath(), "franka_panda")
    os.makedirs(ASSETS_DIR, exist_ok=True)
    shutil.copytree(src, PANDA_ASSET_DIR)


@functools.lru_cache(maxsize=1)
def _finger_span():
    """(mount_z, tip_z) of a gripper finger in the panda_hand frame, read from
    the URDF joint origin and the finger mesh's own extent."""
    ensure_assets()  # assets/ is gitignored; this may be the first thing to touch it
    root = ET.parse(PANDA_URDF_PATH).getroot()
    mount_z = next(
        float(j.find("origin").get("xyz").split()[2])
        for j in root.iter("joint") if j.get("name") == FINGER_JOINTS[0]
    )
    mesh_rel = next(
        link.find("collision/geometry/mesh").get("filename")
        for link in root.iter("link") if link.get("name") == FINGER_LINKS[0]
    ).replace("package://", "")
    with open(os.path.join(PANDA_ASSET_DIR, mesh_rel)) as f:
        finger_length = max(float(line.split()[3]) for line in f if line.startswith("v "))
    return mount_z, mount_z + finger_length


def _check_tcp_offset():
    """The tool is welded at the TCP, which is an external Franka constant
    (F_T_EE) that appears nowhere in the URDF. Guard that it still lands between
    the finger mount and the fingertip of whatever hand the URDF actually has,
    so swapping in a different gripper fails loudly instead of silently welding
    the tool to the wrong place."""
    mount_z, tip_z = _finger_span()
    tcp_z = tool_urdf.TCP_OFFSET[2]
    if not mount_z < tcp_z < tip_z:
        raise ValueError(
            f"TCP_OFFSET z={tcp_z} lies outside the gripper's finger span "
            f"[{mount_z:.4f}, {tip_z:.4f}]; the loaded hand is not a standard "
            f"Franka Hand, so the tool weld point is wrong."
        )


def _weld_fingers_shut(root):
    """Convert the two finger joints from prismatic to fixed, closed to exactly
    half the tool's cross-section width on each side."""
    half_width = geom.W / 2.0
    for name in root.iter("joint"):
        if name.get("name") not in FINGER_JOINTS:
            continue
        name.set("type", "fixed")
        origin = name.find("origin")
        x, y, z = (float(v) for v in origin.get("xyz").split())
        sign = 1.0 if name.get("name") == "panda_finger_joint1" else -1.0
        origin.set("xyz", tool_urdf._xyz_str((x, y + sign * half_width, z)))
        for tag in ("axis", "limit", "mimic"):
            child = name.find(tag)
            if child is not None:
                name.remove(child)


def build_panda_with_tool_urdf(tau):
    """Parse the real Panda URDF and splice in the tool, welded at the TCP,
    with the fingers welded shut around it. Returns the ElementTree (not its
    root Element -- note tool_urdf.build_tool_urdf returns an Element instead)."""
    ensure_assets()
    _check_tcp_offset()
    l1, l2, phi = geom._unpack(tau)

    tree = ET.parse(PANDA_URDF_PATH)
    root = tree.getroot()

    _weld_fingers_shut(root)

    tool_link, tip_link, tip_joint = tool_urdf.build_tool_link_elements(l1, l2, phi)
    root.append(tool_link)
    root.append(tip_link)
    root.append(tip_joint)

    weld_joint = ET.SubElement(root, "joint", {"name": "hand_to_tool", "type": "fixed"})
    ET.SubElement(weld_joint, "parent", {"link": "panda_hand"})
    ET.SubElement(weld_joint, "child", {"link": "tool"})
    ET.SubElement(weld_joint, "origin", {
        "xyz": tool_urdf._xyz_str(tool_urdf.TCP_OFFSET),
        "rpy": tool_urdf._xyz_str(tool_urdf.TOOL_MOUNT_RPY),
    })

    return tree


_generated_paths = set()


def discard_generated_urdf(path):
    """Delete one generated URDF the moment its body has loaded.

    This is the cleanup that actually runs during training. ``cleanup_generated_urdfs``
    below is an ``atexit`` hook, and ``atexit`` never fires in SubprocVecEnv workers
    (forked or forkserver processes exit without it), so an env that redraws tau every
    episode would leave one ~12 KB file per episode in GENERATED_DIR -- about 100k
    files per 10M env steps. Safe because PyBullet has already parsed the file, and
    any later load of the same tau rewrites it first (``write_panda_with_tool_urdf``).

    Args:
        path (str): The path ``write_panda_with_tool_urdf`` returned.
    """
    _generated_paths.discard(path)
    try:
        os.remove(path)
    except OSError:
        pass  # already gone


def cleanup_generated_urdfs():
    """Remove the URDFs this process generated and did not discard. Registered with
    atexit as a backstop for callers that load a tool and never rebuild (demos,
    sweeps, tests); the training env discards each file itself, see above."""
    while _generated_paths:
        try:
            os.remove(_generated_paths.pop())
        except OSError:
            pass  # already gone, or another process cleaned up first


atexit.register(cleanup_generated_urdfs)


def write_panda_with_tool_urdf(tau, path=None):
    """Write the spliced URDF to disk and return the path. If path is not given,
    derive a unique one from a hash of tau and track it for cleanup at exit.

    The hash matters for correctness: PyBullet caches URDFs by path, so reusing
    one filename across different tau silently reloads stale geometry. Because
    every distinct tau gets a distinct path, PyBullet's file cache is safe to
    leave enabled -- and it should be, since it holds the meshes, which are
    identical across all tau and cost ~6x the load time to re-parse."""
    tree = build_panda_with_tool_urdf(tau)
    if path is None:
        l1, l2, phi = geom._unpack(tau)
        key = hashlib.sha1(f"{l1}_{l2}_{phi}".encode()).hexdigest()[:12]
        path = os.path.join(GENERATED_DIR, f"panda_with_tool_{key}.urdf")
        _generated_paths.add(path)
    tree.write(path)
    return path


def disable_finger_tool_collision(body_id, physics_client_id=0):
    """Disable collision between each finger and the tool link, resolving link
    indices by name (never hardcoded — the spliced links invalidate any fixed
    index panda-gym assumes). Addressed to an explicit client, because body ids
    are per client and an env that reconnects to rebuild its tool cannot rely on
    being client 0."""
    tool_idx = get_link_index_by_name(body_id, "tool", physics_client_id)
    for finger_name in FINGER_LINKS:
        finger_idx = get_link_index_by_name(body_id, finger_name, physics_client_id)
        p.setCollisionFilterPair(
            body_id, body_id, finger_idx, tool_idx, enableCollision=0,
            physicsClientId=physics_client_id,
        )
