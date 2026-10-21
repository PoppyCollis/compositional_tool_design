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
- [x] **The parameter-free feature layer.** Landed as `feature_layer.features`,
  `(B, 24) -> (B, 29)`: `x` followed by `(o_x, o_y, l1, l2)/s` and `phi/PHI_MAX`.
  Pure torch, so the first line of each network's `forward` -- not an
  `ObservationWrapper`.
- [x] **The design-space encoding.** Closed by the PPO pass: both networks take `x`
  only, through `networks.ToolFeatureExtractor` (the feature layer); there is no `tau`
  input to drop.
- [ ] **Walk the ridge, once `V` exists.** Three design knobs, two outcomes that
  matter, so every tool sits on a curve of exactly-equivalent tools --
  `(0.120, 0.180, 0)` and `(0.180, 0.120, 0)` share a tip. Before the elbow they shared
  an observation too and the network was *forced* to score them equally; now their
  elbows are 6 cm apart, so flatness along the ridge is a test rather than a
  tautology, and a tilt is a preference invented from noise in the returns. See
  `ai_docs/networks_and_design_gradient.md` §4.
- [x] **PPO observation normalisation is off.** `train_ppo.py` wraps nothing in
  `VecNormalize` and says why; rewards stay in metres at scale 1, recorded in each
  run's `config.json`. The horizon is `terminated` so SB3 does not bootstrap on top of
  the phase in the observation.
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
- [x] Stand up `SubprocVecEnv`, with tau redrawn every episode by client teardown
  (memory.md has the leak finding). Smoke run: 1400 env steps/s on 4 workers.
- [x] Wire the PPO loop (`train_ppo.py`); the first 16-env, 2M-step run is what to read.
- [ ] **Fit the final `V` separately.** PPO's critic is a training aid: fit for a few
  epochs on recent data, targets partly its own guesses (GAE). After PPO, freeze `pi`,
  collect rollouts with tau from the prior, and regress `V` on Monte-Carlo returns --
  exact targets here, since the horizon is fixed and time is in the state. Decide
  whether `V` scores the stochastic policy (SB3's `log_std_init=0` is std 1 on
  actions in [-1, 1]) or the deterministic one §7 evaluates; it must be the one that
  is run. The same data gives a held-out set for the §7 checks.
- [ ] **Reward scale as a designer temperature.** With rewards in metres a tool whose
  tip stops 5 cm short loses ~2-3.5 units of return, i.e. 1/7 to 1/30 of the density
  of one that reaches. Give the Langevin chain an explicit temperature rather than
  rescaling rewards.
- [ ] Wire the outer design-optimisation loop. The entry point for gradients is
  `model.policy.predict_values` on a torch `x` with `requires_grad` (`networks.py`);
  SB3's `predict` is `no_grad` and numpy.

## Deferred — revisit only if profiling says so

**Replacing per-episode `loadURDF` with cached shapes + `createMultiBody`** is 62x
faster (0.65 ms vs 40 ms). Measured against the real loop (2026-10-21): a reset with
rebuild is 49 ms against an 84 ms episode, so `rebuild_every=1` costs about a third of
a worker's time -- affordable, and `--rebuild-every 10` cuts it to ~5% for free if it
ever matters. `createMultiBody` stays the last resort: it drops link names, joint
limits, and off-diagonal inertia (silently), and needs a second construction path kept
in sync. Note the rebuild must go through a new client; `removeBody` + reload leaks.

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