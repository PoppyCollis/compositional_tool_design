import hashlib
import os

import numpy as np
import pybullet as p
import pytest

import tool_geometry as geom
import tool_urdf


@pytest.fixture(scope="module")
def physics_client():
    cid = p.connect(p.DIRECT)
    # Caching left ON deliberately: _urdf_path_for hashes tau into the filename,
    # so distinct designs never collide in PyBullet's path-keyed cache.
    yield cid
    p.disconnect(cid)


def _urdf_path_for(tmp_path, l1, l2, phi):
    key = hashlib.sha1(f"{l1}_{l2}_{phi}".encode()).hexdigest()[:12]
    return os.path.join(tmp_path, f"tool_{key}.urdf")


@pytest.mark.parametrize("l1,l2,phi", [
    (0.2, 0.3, np.pi / 2),
    (0.15, 0.5, 0.0),
    (0.5, 0.15, 1.9),
])
def test_tool_fk_and_mass_match_analytic(physics_client, tmp_path, l1, l2, phi):
    path = _urdf_path_for(tmp_path, l1, l2, phi)
    tool_urdf.write_urdf((l1, l2, phi), path)

    body_id = p.loadURDF(path, useFixedBase=False)

    link_names = {}
    for i in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, i)
        link_names[info[12].decode()] = i
    tip_link_idx = link_names["tool_tip"]

    link_state = p.getLinkState(body_id, tip_link_idx, computeForwardKinematics=True)
    fk_tip = np.array(link_state[4])  # worldLinkFramePosition

    analytic_tip = geom.tip_position(l1, l2, phi)
    np.testing.assert_allclose(fk_tip, analytic_tip, atol=1e-6)

    tool_link_idx = link_names.get("tool", -1)
    dynamics_info = p.getDynamicsInfo(body_id, tool_link_idx)
    sim_mass = dynamics_info[0]
    analytic_mass = geom.mass(l1, l2)
    assert sim_mass == pytest.approx(analytic_mass, rel=1e-6)

    p.removeBody(body_id)
