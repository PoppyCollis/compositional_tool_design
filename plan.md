# Plan

## Next

- [ ] **Add a support surface before sweeping or pushing.** `sim.create_plane` and
  `sim.create_table` are real collision bodies, not scenery — both wrap
  `create_box(ghost=False, mass=0.0)`, and `create_table` takes friction kwargs
  precisely for a dragged object. Gravity is already on, so a massive object needs
  support and sliding friction is most of what those tasks are. Deliberately absent
  now: reach is entirely kinematic (the object is a ghost, the sweeps screen on IK
  pose), so a static body the arm never touches changes nothing, while a naively
  sized table intersects the Panda's own base link — the base is at the origin and
  panda-gym's table top is at z=0 centred on `x_offset`. When it lands:
  - One `create_table`, friction set explicitly on both it and the object.
  - The object must be a box or short cylinder, **not** a sphere: PyBullet's default
    rolling friction is 0, so a sphere rolls forever instead of sliding.
  - First test: run `se2_demo.py`'s trajectory with the table present and assert
    zero tool-table contacts. The tool spans z in [0.01, 0.03] against measured
    transients of 2.2-3 mm, so the 1 cm clearance is real but has never been checked
    against an actual surface. Revisit with the self-collision item below.
- [x] **Observation layout 21 -> 24.** Landed: the elbow at `7:9`, `t / HORIZON` at
  `23:24`, `ROBOT_DIM` 9 -> 11, `TAU_SLICES` three wide, `dh/dtau` now rank 3.
- [ ] **The parameter-free feature layer, and the design-space encoding.** The rest of
  `ai_docs/networks_and_design_gradient.md` §6-§7. A module holding `features(x)` --
  fixed arithmetic recovering `(o_x, o_y, l1, l2, phi)` from `x` -- shared by actor,
  critic and designer, and frozen under the same rule as `SCENE_BOX`. Deferred to the
  PPO pass because both of its readers are the PPO pass. The recovery arithmetic
  already exists, tested, in
  `tests/test_initial_state.test_tau_is_exactly_recoverable_from_x1`: lift it, do not
  rewrite it. Drop `tau` from both networks at the same time (§1) -- there is nothing
  to drop it from yet.
- [ ] **Walk the ridge, once `V` exists.** Three design knobs, two outcomes that
  matter, so every tool sits on a curve of exactly-equivalent tools --
  `(0.120, 0.180, 0)` and `(0.180, 0.120, 0)` share a tip. Before the elbow they shared
  an observation too and the network was *forced* to score them equally; now their
  elbows are 6 cm apart, so flatness along the ridge is a test rather than a
  tautology, and a tilt is a preference invented from noise in the returns. See
  `ai_docs/networks_and_design_gradient.md` §4.
- [ ] **PPO observation normalisation must be off, or verifiably frozen.** If the
  implementation wraps envs in `VecNormalize` or equivalent by default, the running
  mean/var makes `x -> x_tilde` a moving map. `initial_state.h` reapplies that exact
  map at design time against a frozen `V`; a map that drifted during training makes the
  reapplication wrong, silently. Same for reward normalisation if `V`'s scale is to
  mean anything as an energy.
- [ ] **Evaluation must stratify by coverage band.** 39.7% of uniformly sampled targets
  are reachable by no design in the prior, 11.1% by every design, and only the 49.2%
  between carries design signal. An aggregate success rate over `TARGET_BOX` mostly
  measures the mixing ratio. `task_space.coverage` is the labeller, and the §7
  diagnostic in `ai_docs/task_encoding_g.md` must draw its three `g`s from the middle
  band or it fails for the wrong reason.
- [ ] **Sweep and push regions.** Deferred from the reach work and genuinely harder: they
  turn on the *contact normal* (the unit vector from the closest point of the tool
  polyline to the object centre), not on tip position. To sweep, that normal must
  point inward, which forces tool material to sit *beyond* the object — the "get
  around it" condition. Needs the tool as a two-segment polyline (hand, `hand +
      l1*u(psi)`, tip) inflated by `tool_geometry.W/2`, plus a transport-ray check that
  the contact survives as the object crosses the reach boundary. Note a straight rod
  *can* hook off its own end cap, so a tip-position-only test scores it almost as well
  as an L and will not separate designs.
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