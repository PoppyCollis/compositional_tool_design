"""tau -> URDF for a Panda hand with the tool welded in and fingers welded shut
around it. Splices tool_urdf.py's tool link tree into the real Panda URDF
(copied out of pybullet_data into assets/, never mutated in site-packages).
"""
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
    with the fingers welded shut around it. Returns the ElementTree root."""
    ensure_assets()
    l1, l2, theta = geom._unpack(tau)

    tree = ET.parse(PANDA_URDF_PATH)
    root = tree.getroot()

    _weld_fingers_shut(root)

    tool_link, tip_link, tip_joint = tool_urdf.build_tool_link_elements(l1, l2, theta)
    root.append(tool_link)
    root.append(tip_link)
    root.append(tip_joint)

    weld_joint = ET.SubElement(root, "joint", {"name": "hand_to_tool", "type": "fixed"})
    ET.SubElement(weld_joint, "parent", {"link": "panda_hand"})
    ET.SubElement(weld_joint, "child", {"link": "tool"})
    ET.SubElement(weld_joint, "origin", {"xyz": tool_urdf._xyz_str(tool_urdf.TCP_OFFSET), "rpy": "0 0 0"})

    return tree


def write_panda_with_tool_urdf(tau, path=None):
    """Write the spliced URDF to disk and return the path. If path is not given,
    derive a unique one from a hash of tau (PyBullet caches URDFs by path, so
    reusing a filename across different tau silently reloads stale geometry)."""
    tree = build_panda_with_tool_urdf(tau)
    if path is None:
        l1, l2, theta = geom._unpack(tau)
        key = hashlib.sha1(f"{l1}_{l2}_{theta}".encode()).hexdigest()[:12]
        path = os.path.join(GENERATED_DIR, f"panda_with_tool_{key}.urdf")
    tree.write(path)
    return path


def disable_finger_tool_collision(body_id):
    """Disable collision between each finger and the tool link, resolving link
    indices by name (never hardcoded — the spliced links invalidate any fixed
    index panda-gym assumes)."""
    tool_idx = get_link_index_by_name(body_id, "tool")
    for finger_name in FINGER_LINKS:
        finger_idx = get_link_index_by_name(body_id, finger_name)
        p.setCollisionFilterPair(body_id, body_id, finger_idx, tool_idx, enableCollision=0)
