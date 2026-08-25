# Memory

## Current focus
Audit + hardening of the tau -> tool -> URDF -> PyBullet pipeline, ahead of PPO
co-design training.

## Key decisions

- **Parameterisation.** `tau = (l1, l2, phi)`. phi is deflection from straight
  (phi=0 = straight rod, longest tool). theta, the interior elbow angle, is
  `pi - |phi|` — unsigned, because +phi/-phi are mirror-image tools that share the
  same elbow. Exposed as `tool_geometry.interior_angle`.
- **phi bound = +-1.9 rad.** Derived in `config.py` from the worst corner of the
  design box (l1=L_MIN, l2=L_MAX): critical angle is
  `arccos((FINGER_MOUNT_Z - TCP_OFFSET_Z - L_MIN)/L_MAX)` = 1.971 rad, rounded down.
  Leaves 3.3 cm clearance above the finger mount. Tip-z-above-mount is a
  deliberately conservative proxy (at phi=1.9 the tip is 0.47 m out in x).
- **Gripper constants live in `config.GripperConfig`**, not scattered in
  `tool_urdf.py`. `_check_tcp_offset` re-validates mount < TCP < tip against the
  live URDF at every build.
- **Geometry is per-design, never batched.** `_unpack` raises on a `(B, 3)` tau.
  Parallelism for PPO comes from N separate PyBullet clients (SubprocVecEnv), each
  handling one tau — there is no batch dimension for the geometry layer to exploit.

## Measured performance (this machine)

| operation | cost |
|---|---|
| `p.stepSimulation` | 24.3 us |
| `tau_to_geometry` | 44.6 us |
| `build_panda_with_tool_urdf` | 316 us |
| `loadURDF`, caching off | 220 ms |
| `loadURDF`, caching on | 40 ms |
| `createMultiBody`, cached mesh shapes | **0.65 ms** |

Geometry is ~0.02% of a rollout — vectorising it is pointless. **`loadURDF` is the
bottleneck**: tau is redrawn every episode reset, so at 64 envs x 512 steps with
50-step episodes that is ~655 loads = 26.2 s against 0.80 s of stepping.

## Gotchas

- `assets/` is **gitignored** and regenerated from `pybullet_data` by
  `ensure_assets()`. Anything touching `PANDA_URDF_PATH` must call it first.
- PyBullet caches URDFs **by path**. Safe to leave caching on only because
  `write_panda_with_tool_urdf` hashes tau into the filename.
- Generated URDFs are ~12 KB each, one per unique tau. Cleaned up via `atexit`
  (`cleanup_generated_urdfs`); without that they'd reach GBs over a training run.
- Tests need `tests/conftest.py` to put the repo root on `sys.path` — otherwise
  bare `pytest` fails and only `python -m pytest` works.
