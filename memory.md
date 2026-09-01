# Memory

## Current focus
SE(2) control, the reach placement maps, and now the task generative model `p(g)`
(`task.py`) and the initial-state map `h` (`initial_state.py`). Next: the gymnasium
env (ghost-sphere reach target) whose reset is the simulator side of `h`, and with it
the deferred sim-vs-analytic agreement test. Sweep and push regions are still open.

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
- **Tool mount orientation: 90° pitch + 90° roll, no azimuth param.** The
  `hand_to_tool` weld (`panda_with_tool_urdf.py`) used to have `rpy="0 0 0"`, so
  the tool extended in line with the fingers. It now pitches by
  `GripperConfig.TOOL_MOUNT_PITCH` (pi/2, about panda_hand's y-axis) so the tool
  extends perpendicular to the fingers — parallel to the ground when the
  fingers point straight down. On top of that it's rolled by
  `GripperConfig.TOOL_MOUNT_ROLL` (pi/2, about the tool's *own* long axis,
  tool-frame +z) before the pitch is applied — this leaves the pointing
  direction unchanged (rotating about +z fixes +z) and only spins the tool's
  x-z bend plane, i.e. which plane phi's elbow deflection bends in
  (`tool_geometry.py` docstring), relative to the fingers.
  `tool_urdf._compose_mount_rpy(pitch, roll)` composes `R = Ry(pitch) @
  Rz(roll)` and decomposes it back into a single rpy triple, since URDF's rpy
  is fixed-axis roll-pitch-yaw about the *parent's* original axes
  (`R = Rz(yaw)@Ry(pitch)@Rx(roll)`) — naively writing `(roll, pitch, 0)`
  would rotate about the parent's x-axis instead of the tool's own z-axis and
  swing the tool to point somewhere else. Deliberately no azimuth parameter on
  the weld: which horizontal direction the tool points is left to the arm's
  existing wrist joint (panda_joint7), which rotates panda_hand about its own
  z-axis. `tool_urdf.TOOL_MOUNT_RPY` is the single source of truth;
  `spawn_demo.py`'s analytic FK check applies it via `geom._rotation_from_rpy`
  before composing with `R_hand` — it only consumes the resulting rotation
  matrix, so it's agnostic to how TOOL_MOUNT_RPY was derived.

- **Control is SE(2), not joint space.** The action is `(dx, dy, dyaw)` of the
  *hand* (not the tool tip), with height/roll/pitch pinned. Commanding the hand
  keeps the action space, workspace and reset distribution identical for every
  τ; the tip's pose is then recovered in closed form, whereas the reverse — a
  τ-independent action space — is not recoverable if you command the tip.
- **Why the tip's height carries no τ dependence.** `TOOL_MOUNT_RPY` maps tool +z
  to hand +x and the φ-bend plane to the hand's x-y plane, so in the hand frame
  `tip = (l1 + l2·cos φ, l2·sin φ, TCP_OFFSET_Z)`. The z component is constant, so
  pinning the hand's height pins the tool's for every design, and φ hooks *in* the
  ground plane. `se2.py`'s docstring carries the derivation; three tests assert it
  rather than trusting it.
- **Integrated target + lag clamp, not measured-relative.** Re-reading the hand
  pose each step and adding the nudge sounds safer and does stop the target running
  past the workspace, but it feeds the IK residual back into its own input: 300
  steps of *zero* action crept the hand up to 25 cm. The target is now accumulated
  internally (a zero action leaves it exactly unchanged, drift ~1.7 mm) and clamped
  to `MAX_LAG` of the measured pose, which restores what re-reading was for —
  bounding the target when an *object* blocks the hand, as opposed to a boundary,
  which the workspace clip already handles.
- **Both clips happen before IK, never after.** Afterwards the solver has already
  bought the extra reach by tilting the wrist.
- **IK must be given the joint limits.** `calculateInverseKinematics` ignores them
  unless all four null-space arguments are passed, and winds `panda_joint7` past
  its ±2.9671 stop for yaws near ±π. PyBullet's motors *do* enforce limits when
  stepping, so the joint saturates and the hand quietly settles somewhere other
  than commanded. Costs ~1-2 mm of tracking accuracy, which buys repeatable
  redundancy resolution.
- **Read the link frame, not the centre of mass.** panda-gym's `get_link_position`
  / `get_link_orientation` return `getLinkState` indices 0/1 — the link's CoM —
  while IK targets the URDF link frame (indices 4/5). For `panda_hand` those are
  4 cm apart. They coincide for `tool_tip`, so `get_ee_position` is unaffected.
- **Workspace is measured, not guessed.** `workspace_sweep.py` grids the table at
  1 cm over 12 yaws, keeps only cells where the achieved pose is genuinely flat,
  takes the largest all-clean rectangle and insets it. Yaw is a clipped dimension
  like x and y: at ±90° the clean rectangle is 0.24 × 0.76 m, but demanding the
  full circle collapses it to 0.01 m² — the tool points along hand +x, so yaw near
  ±π aims it back at the robot's own base. One `SE2Config.WORKSPACE` is read by the
  clipper, the reset sampler and the observation normaliser.
- **No boundary penalty, deliberately.** Clipping already handles it — the arm
  stalls, the object does not move, the reward does not come. An explicit penalty
  makes the policy timid exactly where it needs to work.
- **`POS_SCALE` is 0.01 m/step, a fifth of panda-gym's.** panda-gym constrains only
  position, so its transient tracking error is free; here a lagging arm is a
  *tilted* one, because a joint-space blend between two flat IK solutions is not
  itself flat. At 0.05 the tool dived 41 mm and leaned 2.8° mid-motion; at 0.01 the
  worst transient across all designs is 2.2 mm and 0.13°.
- **Object placement is a different question from hand placement.** `SE2Config.WORKSPACE`
  is where the *hand* may go; `task_space.py` turns that into where an *object* may sit.
  For reaching it is closed form and needs no sim sweep:
  `Reach(tau) = (WORKSPACE (+) Arc(R, [alpha-Psi, alpha+Psi])) (+) Disk(r_obj)`, tested
  against PyBullet FK to 2.4 mm across every canonical design (`reach_sweep.py --verify`;
  that 2.4 mm is the arm's IK residual, not a geometry error).
- **`se2.tip_polar(tau) -> (R, alpha)` is the form that makes the design space legible.**
  `R = sqrt(l1^2+l2^2+2*l1*l2*cos phi)`, `alpha = -atan2(l2 sin phi, l1 + l2 cos phi)`, and
  the tip sits at `hand + R*u(yaw + alpha)`. Two consequences that drive everything:
  - **alpha rotates the accessible half-plane of tip bearings**, which is a *separate* axis
    from reach. Yaw is clipped to +-90 deg, so bearings span `[alpha-90, alpha+90]`. A
    straight rod has `alpha=0` and can never point its tip backwards however long it is —
    its region stops dead at the workspace's own inner edge (x=0.35). `(0.1, 0.2, 1.9)`
    reaches bearing -169 deg for only 0.193 m of reach and gets in to x=0.256.
  - **The tip is on a *circle* of radius R, not in a disk, so long tools have a
    near-target blind zone.** Reach is not monotone in length: `(0.2, 0.2, 0)` reaches
    0.400 m but cannot touch the workspace centre with `tol=0`, and is blind to 0.035 m2
    of the bare arm's own reach. The `blind` column of `reach_sweep.py` measures this.
- **The coverage map is the thing to sample `s_start` from.** `task_space.coverage` gives
  the fraction of `ToolPrior` designs reaching each cell. `f=1` cells carry no design
  signal (every tool succeeds, so `p(tau|g,O=1)` stays the prior) and `f=0` are
  impossible; the band between is where the §7 diagnostic in `task_encoding_g.md` can
  bite. Measured at r_obj=0.03: 1.328 m2 reachable by some tool, of which 1.089 m2 (82%)
  discriminates and 0.902 m2 of that is also out of the bare arm's reach.
- **`g` is the target spec only; `x_t` is everything observable.** Deviates from the
  first draft of `ai_docs/task_encoding_g.md`, which put `s_start` (the object start
  *region*) in `g` so `h(tau, g)` would be well-defined. Instead `x_t` re-emits the
  object's *current* pose every step, so the start position lands in `x1` for free and
  needs no second home. What replaced `s_start` is `task.object_start`: a distribution
  keyed by `task_id`, drawn at reset inside `h`. `g = (task_id, p_target, r_obj, rho,
  w_reach, w_trans)`. The docs were rewritten to match rather than left contradicting
  the code.
- **`p_target ~ U(SCENE_BOX)`, unreachable targets included on purpose.** No rejection
  sampling, which is what the previous plan called for. Over the box (x[0, 1.10],
  y[-1, 1]) that is 39.7% reachable by no prior design, 49.2% discriminating, 11.1%
  reachable by all. The value function should learn that some targets are hopeless
  whatever the tool rather than extrapolate to them at design time. Cost: an aggregate
  success rate over the box mostly measures the mixing ratio, so **evaluation has to
  stratify by `task_space.coverage` band**, and the §7 diagnostic must draw its three
  `g`s from the middle band.
- **One box, one scalar, frozen: `TaskConfig.SCENE_BOX`.** Every planar position in the
  observation -- hand, tip, object, target -- goes through
  `Box.normalise_point(p) = (p - c)/s`, and the relative features through
  `normalise_delta(v) = v/s`. Three separate reasons, all about the design gradient:
  - *Not `SE2Config.WORKSPACE`.* The tool exists to put the tip outside the arm's
    reach, so normalising on the hand's own rectangle pushes long tools past +-1 --
    exactly the designs the search evaluates land in the network's extrapolation
    region, and `dV/dx1` is half of `grad_tau E`. Same for the object, which in
    sweeping and pushing crosses the reach boundary in both directions.
  - *One scalar divisor, not one per axis.* `dh/dtau` is in metres; per-axis scaling
    would make one normalised unit 0.55 m in x and 1.0 m in y, so two designs with
    physically equal tip displacements would get unequal gradients from the box's
    aspect ratio alone. Price: the short axis normalises to +-0.55, not +-1.
  - *It must never move.* `h` reapplies this exact map at design time against a frozen
    `V`. A running normaliser in the RL stack would make it drift and silently
    invalidate that. The old per-axis `Box.normalise` was deleted once it had no
    production caller; `normalise_point`/`normalise_delta` are the only maps.
- **Relative features are `obj - tip` and `target - obj`, never `target - tip`.** They
  are the two reward terms; their sum points somewhere useful in neither sweeping nor
  pushing. Degenerate direction worth knowing: the tip enters slice 7:9 as `+tip/s` and
  11:13 as `-tip/s`, so any plain sum over the observation cancels the design out.
- **Reach success is tip-based, and the spec was wrong.** `task_encoding_g.md` gave
  `d(obj_T, p_target) < rho` for all three task types. The reaching object is pinned at
  the target and never moves, so that scores every episode a success -- including one
  where the tip never left the far side of the table. `task.success` branches on the
  task type; a test pins it.
- **`h` is torch-only, with no numpy twin.** Two implementations of one formula is the
  drift this codebase has already paid for. `tip_offset_torch` / `tip_from_hand_torch`
  are pinned to `se2.tip_offset` / `se2.tip_from_hand` by test instead. They accept a
  batched tau, unlike `tool_geometry` -- that rule exists because geometry feeds URDF
  generation one body at a time, and `h` touches no URDF.
- **`s_start` cannot depend on tau.** Superseded in form -- there is no `s_start` field
  any more -- but the reason survives and now applies to `task.object_start`: a
  tau-conditioned start distribution would make `g` depend on the design and break the
  `p(tau | g, O=1)` factorisation. Hence the regions are always aggregated across the
  prior, never taken per-design.
- **Two implementations of the same set, deliberately.** `tip_reachable` is exact and
  per-point (for env-time queries); `reach_mask` rasterises the Minkowski sum by integer
  cell shifts and is ~200-460x faster (for maps and the 2000-design coverage pass, 52 min
  -> 6 s). They agree except within one cell of the boundary, which a test asserts.
- **Do not fit a rectangle to these regions** — `reach_sweep.py` deliberately reports
  areas only. They are annular shells with the near-field notch cut out, so the largest
  inscribed box in the discriminating band is 0.075 m2 of a 0.471 m2 region, and it lands
  in the outer shell: sampling it would sample "long tools win" and nothing else. Sample
  `s_start` by rejection against `task_space.tip_reachable`. (`largest_rectangle` still
  lives in `task_space` because `workspace_sweep` needs it — the *hand* workspace really
  is a rectangle, since the controller clips to one.)

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

- `TaskConfig.GRIPPER_RADIUS = 0.045` is measured, not guessed: only the two *fingers*
  reach the tool plane (they span z=[0.0075, 0.0692]; `panda_hand` itself bottoms out at
  z=0.0538), circumradius 0.0429 m from the hand origin. Re-measure with
  `python reach_sweep.py --measure-gripper`. Rounded up on purpose — that errs towards
  calling a target hand-reachable, i.e. towards *under*-claiming the tool-required region.
- `reach_sweep.py --verify` must assert on **closed-form-vs-FK**, not tip-to-target. The
  region is "tip within r_obj of the centre", so a boundary target has the tip grazing the
  surface at exactly r_obj by construction and the IK residual carries it a couple of mm
  past. Asserting on tip-to-target fails on correct geometry.

- `assets/` is **gitignored** and regenerated from `pybullet_data` by
  `ensure_assets()`. Anything touching `PANDA_URDF_PATH` must call it first.
- PyBullet caches URDFs **by path**. Safe to leave caching on only because
  `write_panda_with_tool_urdf` hashes tau into the filename.
- Generated URDFs are ~12 KB each, one per unique tau. Cleaned up via `atexit`
  (`cleanup_generated_urdfs`); without that they'd reach GBs over a training run.
- Tests need `tests/conftest.py` to put the repo root on `sys.path` — otherwise
  bare `pytest` fails and only `python -m pytest` works.
