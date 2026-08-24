"""Sample tau from the prior, spawn a PandaWithTool via panda-gym, hold a pose,
and report analytic vs. simulated tip position, tool mass, and steady-state
wrist torques. No task, no reward, no learning - just the design -> env pipeline.
"""
import numpy as np
import pybullet as p
from panda_gym.pybullet import PyBullet

import tool_geometry as geom
import tool_urdf
from panda_with_tool import JOINT_FORCES, JOINT_INDICES, NEUTRAL_JOINT_VALUES, PandaWithTool
from tool_design_prior import ToolPrior
from utils.helpers import get_link_index_by_name

HOLD_STEPS = 200


def main():
    torch_tau = ToolPrior().sample(1)[0]
    tau = tuple(float(v) for v in torch_tau.detach())
    l1, l2, theta = tau
    print(f"tau = (l1={l1:.4f}, l2={l2:.4f}, theta={theta:.4f})")

    sim = PyBullet(render_mode="rgb_array")
    robot = PandaWithTool(sim, tau)
    robot.reset()

    for _ in range(HOLD_STEPS):
        robot.control_joints(NEUTRAL_JOINT_VALUES)
        sim.step()

    body_id = sim._bodies_idx[robot.body_name]

    fk_tip = robot.get_ee_position()

    hand_idx = get_link_index_by_name(body_id, "panda_hand")
    hand_pos, hand_orn = p.getLinkState(body_id, hand_idx)[4:6]
    R_hand = np.array(p.getMatrixFromQuaternion(hand_orn)).reshape(3, 3)
    analytic_tip_world = np.array(hand_pos) + R_hand @ (np.array(tool_urdf.TCP_OFFSET) + geom.tip_position(l1, l2, theta))

    print(f"FK tool_tip (world):       {fk_tip}")
    print(f"analytic tool_tip (world): {analytic_tip_world}")
    print(f"error: {np.linalg.norm(fk_tip - analytic_tip_world):.2e} m")

    tool_idx = get_link_index_by_name(body_id, "tool")
    sim_mass = p.getDynamicsInfo(body_id, tool_idx)[0]
    analytic_mass = geom.mass(l1, l2)
    print(f"tool mass: sim={sim_mass:.5f} kg, analytic={analytic_mass:.5f} kg")

    torques = [p.getJointState(body_id, i)[3] for i in JOINT_INDICES]
    print("steady-state joint torques (N*m):")
    for i, (t, limit) in enumerate(zip(torques, JOINT_FORCES)):
        print(f"  joint {i}: {t:7.3f} / {limit:.1f}")

    sim.close()


if __name__ == "__main__":
    main()
