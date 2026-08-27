# Plan

## Done
- [x] Audit of the Sonnet phi-reparameterisation pass (see memory.md).
- [x] `_unpack` rejects batched/malformed tau instead of silently returning row 0.
- [x] `_finger_span` calls `ensure_assets()` — fresh clone no longer crashes.
- [x] Gripper constants hoisted into `config.GripperConfig`; phi-bound derivation
      corrected (the "~9 cm clearance" claim was wrong; it is 3.3 cm).
- [x] `interior_angle(phi) = pi - |phi|` added and covered by tests.
- [x] phi-bound test derives its own critical angle instead of hardcoding 2.2.
- [x] File caching re-enabled (220 ms -> 44 ms per load).
- [x] Generated URDFs cleaned up at exit.
- [x] `config.DEVICE` wired into `ToolPrior`; `tests/conftest.py` added.
- [x] 28 tests pass under both `pytest` and `python -m pytest`.
- [x] `spawn_demo_step3.py` resets the arm to `config.ArmConfig.NEUTRAL_JOINT_VALUES`
      (panda-gym's ready pose) after each `loadURDF`, instead of sitting at the URDF's
      all-zeros default (arm straight up). Single source of truth shared with
      `PandaWithTool.reset`.
- [x] Tool mount reoriented: `hand_to_tool` weld now pitches 90° so the tool
      extends outward, perpendicular to the fingers, instead of in line with
      them. See `[[memory.md]]` "Tool mount orientation". Azimuth intentionally
      left to the arm's own wrist joint rather than added as a new parameter.
- [x] Tool mount also rolled 90° about its own long axis (before the pitch),
      so the elbow's bend plane (phi) is spun 90° relative to the fingers
      while the tool still points outward the same way. `GripperConfig.
      TOOL_MOUNT_ROLL`, composed with the pitch via `tool_urdf._compose_mount_rpy`.
      See `[[memory.md]]`.

- [x] **SE(2) control.** Action is now `(dx, dy, dyaw)` of the hand, 3-DoF instead
      of 7, with height/roll/pitch pinned so the tool is always low and parallel to
      the ground. New `se2.py` (pure math), `SE2Config`, `workspace_sweep.py`
      (offline calibration), `se2_demo.py`, and 108 new tests. See `[[memory.md]]`
      for the design decisions and the four bugs found on the way: link-frame vs
      centre-of-mass readback, IK ignoring joint limits, measured-relative
      integration compounding solver residual into a 25 cm drift, and untrackable
      step sizes tilting the tool mid-motion.

- [x] **Reach-task object placement map.** New `task_space.py` (pure geometry: reach
      regions, coverage over the design prior, mask helpers moved out of
      `workspace_sweep.py`), `se2.tip_polar`, `config.TaskConfig`, `reach_sweep.py`
      (tables + candidate boxes + `--verify` against PyBullet FK + `--measure-gripper`),
      `plot_reach_space.py`, and 87 new tests. See `[[memory.md]]` for the two geometric
      facts it surfaced: alpha rotates the reachable half-plane independently of reach,
      and long tools have a near-target blind zone because the tip is on a circle rather
      than in a disk.

## Next
- [ ] **Choose `s_start` / `p_target` for reaching** off the coverage map — the maps are
      built, the constant is not. Prefer rejection sampling against
      `task_space.tip_reachable` over an `se2.Box`; the regions are annular and a
      rectangle throws most of them away (0.467 m2 band -> 0.075 m2 box).
- [ ] **Sweep and push regions.** Deferred from the reach work and genuinely harder: they
      turn on the *contact normal* (the unit vector from the closest point of the tool
      polyline to the object centre), not on tip position. To sweep, that normal must
      point inward, which forces tool material to sit *beyond* the object — the "get
      around it" condition. Needs the tool as a two-segment polyline (hand, `hand +
      l1*u(psi)`, tip) inflated by `tool_geometry.W/2`, plus a transport-ray check that
      the contact survives as the object crosses the reach boundary. Note a straight rod
      *can* hook off its own end cap, so a tip-position-only test scores it almost as well
      as an L and will not separate designs.
- [ ] Wrap `PandaWithTool` in a gymnasium env with a task + reward, on top of the
      SE(2) action space. Object and goal both sampled from `SE2Config.WORKSPACE`
      (the goal has to be somewhere the hand can actually reach). Observation
      normalised against the same box. No penalty for hitting the boundary.
- [ ] Stand up `SubprocVecEnv` with per-env tau; confirm scaling against the numbers above.
- [ ] Wire the PPO loop and the outer design-optimisation loop.

## Deferred — revisit only if profiling says so

**Replacing per-episode `loadURDF` with cached shapes + `createMultiBody`** is 62x
faster (0.65 ms vs 40 ms), but the 26 s/iteration figure motivating it assumes tau is
redrawn every reset — an assumption about a training loop that doesn't exist yet.
Profile first. If it does bite, hold tau fixed for K episodes per env before rewriting
anything: K=10 cuts the cost 10x for a few lines, and 64 envs still give 64 designs per
update. `createMultiBody` is the last resort — it drops link names, joint limits, and
off-diagonal inertia (silently), and needs a second construction path kept in sync.

**`workspace_sweep.py` accepts self-intersecting arm configurations.** It screens on
the achieved hand pose, not on whether the arm passes through itself getting there —
see the self-collision item below. Revisit together.

**Arm/tool self-collision is off.** `loadURDF` disables self-collision within a
multi-body by default, and nothing here passes `flags=p.URDF_USE_SELF_COLLISION`.
The tool is spliced into the *same* URDF as the arm (`panda_with_tool_urdf.py`), so
PyBullet never checks tool-vs-arm-link contacts — e.g. at `phi` extremes or awkward
arm poses the tool can pass straight through a forearm/link with no contact force,
no warning. Doesn't matter for the current step-3 visual check or for training
episodes that don't depend on physically valid self-contact. Will matter once
sampled tau/joint configs need to be physically plausible (e.g. rejecting
self-intersecting configs during prior sampling, or if a policy could exploit an
arm the tool can pass through). Fix: pass `URDF_USE_SELF_COLLISION` on load. Caveats
to check when this is picked up:
- PyBullet auto-excludes parent/child link pairs joined by a joint (so
  `panda_hand`<->`tool` won't spuriously collide), but not other non-adjacent pairs —
  may need more `setCollisionFilterPair` calls beyond `disable_finger_tool_collision`
  (e.g. `panda_hand`<->fingers).
- All-pairs self-collision checking is measurably more expensive per step; matters
  if this ends up inside the parallel-env training loop.
