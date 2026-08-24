"""Generic PyBullet helpers not tied to any one robot or tool."""
import pybullet as p


def get_link_index_by_name(body_id, name):
    """Resolve a link's index by its URDF name. Link indices shift whenever links
    are spliced into a URDF, so never hardcode them - look them up by name."""
    for i in range(p.getNumJoints(body_id)):
        if p.getJointInfo(body_id, i)[12].decode() == name:
            return i
    raise ValueError(f"No link named {name!r} in body {body_id}")
