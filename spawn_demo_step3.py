"""Step-3 visual check: spawn the Panda with the tool welded in, fingers closed
around it, across a few tau, and eyeball the weld position, finger closure,
and absence of interpenetration blowup. GUI only, no panda-gym yet.
"""
import time

import numpy as np
import pybullet as p

import panda_with_tool_urdf as pwt

TAUS = [
    (0.3, 0.3, 0.0),         # straight (phi=0, longest tool)
    (0.3, 0.3, np.pi / 2),   # bent at a right angle
    (0.3, 0.3, -np.pi / 2),  # mirror image
    (0.15, 0.5, 1.9),        # bound extremes: shortest handle, longest head, max fold
    (0.5, 0.15, -1.9),       # bound extremes, mirrored
]

HOLD_SECONDS = 3.0


def main():
    p.connect(p.GUI)
    # File caching stays ON: write_panda_with_tool_urdf gives every distinct tau
    # its own hashed path, so there is no stale-reload risk, and the cache holds
    # the Panda meshes -- identical across all tau, ~6x the load time to re-parse.
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
