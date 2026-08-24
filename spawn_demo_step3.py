"""Step-3 visual check: spawn the Panda with the tool welded in, fingers closed
around it, across a few tau, and eyeball the weld position, finger closure,
and absence of interpenetration blowup. GUI only, no panda-gym yet.
"""
import time

import numpy as np
import pybullet as p

import panda_with_tool_urdf as pwt

TAUS = [
    (0.3, 0.3, np.pi / 2),   # bent at a right angle
    (0.3, 0.3, np.pi),       # straight
    (0.3, 0.3, 1e-3),        # folded back on itself
    (0.15, 0.5, 2.0),        # bound extremes
]

HOLD_SECONDS = 3.0


def main():
    p.connect(p.GUI)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.setGravity(0, 0, -9.81)
    p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.5])

    for tau in TAUS:
        path = pwt.write_panda_with_tool_urdf(tau)
        body = p.loadURDF(path, useFixedBase=True)
        pwt.disable_finger_tool_collision(body)
        print(f"tau={tau} -> {path}")

        t0 = time.time()
        while time.time() - t0 < HOLD_SECONDS:
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

        p.removeBody(body)

    p.disconnect()


if __name__ == "__main__":
    main()
