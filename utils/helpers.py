"""Generic PyBullet helpers not tied to any one robot or tool."""
import pybullet as p


def get_link_index_by_name(body_id, name, physics_client_id=0):
    """Resolve a link's index by its URDF name. Link indices shift whenever links
    are spliced into a URDF, so never hardcode them - look them up by name.

    Takes the client id explicitly: body ids are per client, and the default of 0
    only answers for the right body while every client happens to have loaded the
    same URDF in the same order. An env that reconnects to rebuild its tool breaks
    that coincidence."""
    for i in range(p.getNumJoints(body_id, physicsClientId=physics_client_id)):
        if p.getJointInfo(body_id, i, physicsClientId=physics_client_id)[12].decode() == name:
            return i
    raise ValueError(f"No link named {name!r} in body {body_id}")
