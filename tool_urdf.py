"""tau -> URDF XML for the 2-link tool. Standalone-loadable (tool as root link)."""
import xml.etree.ElementTree as ET

import tool_geometry as geom

TOOL_TIP_MASS = 1e-6  # massless-but-not-zero, so PyBullet accepts the link
TCP_OFFSET = (0.0, 0.0, 0.1034)  # panda_hand -> TCP, per the Franka spec


def _xyz_str(v):
    return f"{v[0]} {v[1]} {v[2]}"


def _add_box_geom_elements(parent, box, name_prefix):
    for tag in ("visual", "collision"):
        el = ET.SubElement(parent, tag, {"name": f"{name_prefix}_{tag}"})
        origin = ET.SubElement(el, "origin")
        origin.set("xyz", _xyz_str(box["pos"]))
        origin.set("rpy", _xyz_str(box["rpy"]))
        geometry = ET.SubElement(el, "geometry")
        ET.SubElement(geometry, "box", {"size": _xyz_str(box["size"])})


def build_tool_link_elements(l1, l2, theta):
    """Build the 'tool' and 'tool_tip' <link> elements plus the joint welding them,
    for the given (l1, l2, theta). Shared by the standalone tool URDF and by
    panda_with_tool_urdf.py, which splices these into the full Panda URDF.

    Returns (tool_link, tip_link, tip_joint) Elements, unattached to any tree.
    """
    specs = geom.box_specs(l1, l2, theta)
    m = geom.mass(l1, l2)
    com = geom.combined_com(l1, l2, theta)
    inertia = geom.inertia_tensor(l1, l2, theta)
    tip = geom.tip_position(l1, l2, theta)

    tool_link = ET.Element("link", {"name": "tool"})
    _add_box_geom_elements(tool_link, specs["box1"], "box1")
    _add_box_geom_elements(tool_link, specs["box2"], "box2")

    inertial = ET.SubElement(tool_link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": _xyz_str(com), "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": str(m)})
    ET.SubElement(inertial, "inertia", {
        "ixx": str(inertia[0, 0]), "ixy": str(inertia[0, 1]), "ixz": str(inertia[0, 2]),
        "iyy": str(inertia[1, 1]), "iyz": str(inertia[1, 2]), "izz": str(inertia[2, 2]),
    })

    tip_link = ET.Element("link", {"name": "tool_tip"})
    tip_inertial = ET.SubElement(tip_link, "inertial")
    ET.SubElement(tip_inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(tip_inertial, "mass", {"value": str(TOOL_TIP_MASS)})
    ET.SubElement(tip_inertial, "inertia", {
        "ixx": "1e-9", "ixy": "0", "ixz": "0", "iyy": "1e-9", "iyz": "0", "izz": "1e-9",
    })

    tip_joint = ET.Element("joint", {"name": "tool_to_tip", "type": "fixed"})
    ET.SubElement(tip_joint, "parent", {"link": "tool"})
    ET.SubElement(tip_joint, "child", {"link": "tool_tip"})
    ET.SubElement(tip_joint, "origin", {"xyz": _xyz_str(tip), "rpy": "0 0 0"})

    return tool_link, tip_link, tip_joint


def build_tool_urdf(l1, l2, theta):
    """Build the standalone URDF ElementTree for the tool ('tool' is the root link,
    loadable on its own)."""
    tool_link, tip_link, tip_joint = build_tool_link_elements(l1, l2, theta)

    robot = ET.Element("robot", {"name": "tool"})
    robot.append(tool_link)
    robot.append(tip_link)
    robot.append(tip_joint)
    return robot


def tau_to_urdf_string(tau):
    l1, l2, theta = geom._unpack(tau)
    robot = build_tool_urdf(l1, l2, theta)
    return ET.tostring(robot, encoding="unicode")


def write_urdf(tau, path):
    xml_str = tau_to_urdf_string(tau)
    with open(path, "w") as f:
        f.write(xml_str)
    return path
