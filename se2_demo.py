"""Visual check of the SE(2) controller: drive the tool around and watch it stay flat.

Runs a scripted trajectory -- trace the workspace, spin on the spot, then lean full
throttle into each wall -- for a few tau, drawing the workspace rectangle so the
boundary is visible. Reports the worst tool-height and tilt deviation seen, so the
run is meaningful headless as well as on screen.

Usage:
    python se2_demo.py             # GUI
    python se2_demo.py --headless  # numbers only, no window
"""
import sys
import time

import numpy as np
import pybullet as p
from panda_gym.pybullet import PyBullet

import se2
from config import DesignPriorConfig, SE2Config
from panda_with_tool import PandaWithTool

TAUS = [
    (0.1, 0.1, 0.0),                                # straight rod
    (0.1, 0.1, np.pi / 2),                          # right-angled hook
    (0.1, 0.2, DesignPriorConfig.PHI_MAX),         # short handle, long head, max fold
    (0.2, 0.1, -DesignPriorConfig.PHI_MAX),        # mirrored extreme
]

HOLD = 40      # steps per leg of the square
SPIN = 60      # steps of pure rotation
PRESS = 80     # steps leaning into each wall

LEGS = [
    ("+x", (1.0, 0.0, 0.0)),
    ("+y", (0.0, 1.0, 0.0)),
    ("-x", (-1.0, 0.0, 0.0)),
    ("-y", (0.0, -1.0, 0.0)),
]


def draw_workspace(sim, box, height):
    """Outline the workspace rectangle at the tool's height."""
    corners = [
        (box.x_min, box.y_min), (box.x_max, box.y_min),
        (box.x_max, box.y_max), (box.x_min, box.y_max),
    ]
    for start, end in zip(corners, corners[1:] + corners[:1]):
        sim.physics_client.addUserDebugLine(
            [start[0], start[1], height], [end[0], end[1], height],
            lineColorRGB=[1.0, 0.4, 0.0], lineWidth=2.0,
        )


def run_leg(sim, robot, action, steps, gui):
    """Apply one constant action, returning the worst height and tilt error seen."""
    worst_height, worst_tilt = 0.0, 0.0
    for _ in range(steps):
        robot.set_action(np.array(action, dtype=float))
        sim.step()
        worst_height = max(worst_height, abs(robot.get_ee_position()[2] - SE2Config.TOOL_Z))
        worst_tilt = max(worst_tilt, robot.get_hand_tilt())
        if gui:
            time.sleep(1.0 / 120.0)
    return worst_height, worst_tilt


def demo_tau(sim, tau, gui):
    """Run the full trajectory for one design and print what it did."""
    robot = PandaWithTool(sim, tau, np_random=np.random.default_rng(0))
    robot.set_se2(SE2Config.WORKSPACE.centre, 0.0)
    if gui:
        sim.physics_client.removeAllUserDebugItems()
        draw_workspace(sim, robot.workspace, SE2Config.TOOL_Z)

    reach = np.linalg.norm(se2.tip_offset(tau))
    print(f"\ntau = {tuple(round(v, 3) for v in tau)}   tip reaches {reach:.3f} m from the hand")

    worst_height, worst_tilt = 0.0, 0.0
    schedule = (
        [(f"square {name}", action, HOLD) for name, action in LEGS]
        + [("spin +", (0.0, 0.0, 1.0), SPIN), ("spin -", (0.0, 0.0, -1.0), SPIN)]
        + [(f"press {name}", action, PRESS) for name, action in LEGS]
    )
    for label, action, steps in schedule:
        height, tilt = run_leg(sim, robot, action, steps, gui)
        worst_height, worst_tilt = max(worst_height, height), max(worst_tilt, tilt)
        x, y, yaw = robot.get_hand_se2()
        tip = se2.tip_from_hand((x, y, yaw), tau)
        print(f"  {label:<12} hand=({x:+.3f},{y:+.3f}) yaw={np.degrees(yaw):+7.1f} deg  "
              f"tip=({tip[0]:+.3f},{tip[1]:+.3f})  |dz|={height * 1000:5.2f} mm  "
              f"tilt={np.degrees(tilt):5.3f} deg")

    print(f"  worst over the run: |dz| = {worst_height * 1000:.2f} mm, "
          f"tilt = {np.degrees(worst_tilt):.3f} deg")

    # The point of the exercise: turning the wrist must sweep the tip through a real
    # arc in the table plane, by the amount the closed form predicts.
    robot.set_se2(SE2Config.WORKSPACE.centre, -0.6)
    before = se2.tip_from_hand(robot.get_hand_se2(), tau)
    run_leg(sim, robot, (0.0, 0.0, 1.0), 30, gui)
    after = se2.tip_from_hand(robot.get_hand_se2(), tau)
    swept = np.linalg.norm(after - before)
    turned = abs(se2.wrap_angle(robot.get_hand_se2()[2] + 0.6))
    print(f"  yaw swept {np.degrees(turned):.1f} deg -> tip moved {swept:.3f} m "
          f"(chord of a {reach:.3f} m arc: {2 * reach * np.sin(turned / 2):.3f} m)")

    sim.physics_client.removeBody(sim._bodies_idx.pop(robot.body_name))
    return worst_height, worst_tilt


def main():
    gui = "--headless" not in sys.argv
    sim = PyBullet(render_mode="human" if gui else "rgb_array")
    if gui:
        sim.physics_client.resetDebugVisualizerCamera(
            cameraDistance=1.3, cameraYaw=50, cameraPitch=-40,
            cameraTargetPosition=[0.45, 0.0, 0.1],
        )

    worst = [demo_tau(sim, tau, gui) for tau in TAUS]
    height = max(h for h, _ in worst)
    tilt = max(t for _, t in worst)
    print(f"\nacross every design: worst |dz| = {height * 1000:.2f} mm, "
          f"worst tilt = {np.degrees(tilt):.3f} deg")

    p.disconnect()


if __name__ == "__main__":
    main()
