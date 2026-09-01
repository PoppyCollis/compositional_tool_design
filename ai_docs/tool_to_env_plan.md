# Plan: Tool Design → PyBullet Environment

**Scope:** τ = (l₁, l₂, φ) in, Panda spawned in panda-gym with that tool rigidly welded into its grip, out. Nothing else. No learning, no gripper actuation, no attach/detach logic, no task, no reward.

> **Two conventions moved after this plan was executed.** The document is otherwise the
> build as shipped; these two are corrected in place below.
>
> **The third design parameter is `φ`, not `θ`.** `φ` is deflection from straight:
> `φ = 0` is a straight rod (the longest tool), `|φ|` growing folds link 2 back. It is
> drawn from a box, `φ ~ U[−PHI_MAX, PHI_MAX]` with `DesignPriorConfig.PHI_MAX = 1.9`,
> not from a circle. The interior angle this document called `θ` is `π − |φ|`, exposed
> as `tool_geometry.interior_angle`.
>
> **The `hand_to_tool` weld is no longer `rpy="0 0 0"`.** It carries
> `GripperConfig.TOOL_MOUNT_PITCH` and `TOOL_MOUNT_ROLL` (both π/2), composed by
> `tool_urdf._compose_mount_rpy`. The pitch swings the tool's long axis off the finger
> axis so it extends perpendicular to the fingers — parallel to the ground when the
> fingers point down — and the roll spins the tool about its own long axis so that
> `φ`'s bend plane is the *ground* plane. Together they are why the tip's height carries
> no τ dependence, which is what `se2.py` and the SE(2) controller are built on.

---

## 1. Open input

`DesignPriorConfig.L_MIN` / `L_MAX` are still needed. They set whether the tool meaningfully extends reach and whether the mass is trivial or wrist-breaking. As a reference point: with a 20 mm square cross-section and solid-PLA density, l₁ + l₂ = 1.0 m gives ≈0.5 kg, which is comfortably inside the Panda's 3 kg payload and produces under 2 N·m about the wrist joints. Values in the 0.15–0.5 m range per link are the sane band.

---

## 2. Geometry convention (pin this down first)

Everything downstream depends on these three choices. They are decisions, not defaults to be quietly changed later.

**Tool frame.** Origin at the grip point — the proximal end of l₁, coincident with the gripper's TCP. Link 1 extends along **+z** (the gripper's approach axis), so the tool points straight out of the hand.

**φ is deflection from straight.** φ = 0 means the two links are collinear (a straight rod, maximum reach); |φ| → π would fold link 2 back alongside link 1. The interior angle at the elbow is π − |φ|. Deflection is the parameterisation the prior and the networks use, because it is a bounded interval with no wrap and its zero is a physically meaningful tool rather than a degenerate one.

**Bend plane.** Fixed as the tool frame's x–z plane. A bend in any other plane is reachable by rotating joint 7, so this loses no generality. The weld's roll (see the header note) then puts that plane in the ground plane.

Direction of link 2, and the tip:

```
d1   = (0, 0, 1)
d2   = (sin φ, 0, cos φ)
p_tip = l1 · d1 + l2 · d2
```

Sanity: φ = 0 → d2 = (0,0,1), straight. φ = π/2 → d2 = (1,0,0), perpendicular. |φ| → π → d2 = (0,0,−1), folded. The prior stops at |φ| = 1.9, short of that last case.

**Symmetry note:** +φ and −φ are mirror images equivalent under a joint-7 rotation, so they share an interior angle — which is why `tool_geometry.interior_angle` is unsigned. The prior samples the signed range all the same; the mirror pair is genuinely two distinct tools once the arm's yaw is clipped to ±YAW_LIMIT.

---

## 3. Structural decisions

- **One rigid link, two collision shapes.** Both boxes live in a single `tool` link. PyBullet never collision-tests shapes within the same link, so the self-intersection at large |φ| is geometrically real but physically inert — no VHACD, no mesh repair, no degenerate-case special-casing.
- **Primitives, not meshes.** Two `<box>` elements. Optional OBJ exporter for rendering only, off the critical path.
- **Fingers welded to grip the box.** Convert the finger joints to `fixed` at exactly half the box width, so the render shows the tool actually held rather than floating between open fingers. Cosmetic — the weld carries all load. Disable finger↔tool collision.
- **Torch boundary is explicit.** τ arrives as a tensor with `requires_grad=True`; geometry code takes plain floats. Detach at the boundary in one clearly-named place. PyBullet is not differentiable and nothing downstream should imply otherwise.

---

## 4. Modules

### `tool_geometry.py` — pure, no PyBullet, no I/O

τ → box specs (size + pose in tool frame), tip position, mass, inertia. Every function analytically testable in isolation.

- Box *i* placed at its segment midpoint: box 1 at `(0, 0, l1/2)` with rpy `(0,0,0)`; box 2 at `l1·d1 + (l2/2)·d2` with rpy `(0, φ, 0)`.
- **Mass:** constant density ρ, `mᵢ = ρ · w · h · lᵢ`. Clamp total mass at a configurable ceiling by scaling ρ down and emitting a warning, so no sampled design becomes uncontrollable.
- **Inertia:** closed-form box tensor about each box's own centre → rotate into tool frame (`R I Rᵀ`) → parallel-axis shift to the combined COM → sum. The result is generally **non-diagonal**; URDF supports `ixy`/`ixz`/`iyz` and Bullet diagonalises internally, so emit the full tensor rather than dropping off-diagonal terms.

### `tool_urdf.py`

τ → URDF XML string, with an optional write-to-path. Emits the `tool` link (two visual + two collision boxes, one `<inertial>` at the combined COM), a massless-but-not-zero (`1e-6`) `tool_tip` frame at `p_tip` for FK queries, and the fixed joint welding `tool` to `panda_hand` at the TCP offset `(0, 0, 0.1034)`.

**Never hardcode link indices.** panda-gym hardcodes `ee_link = 11`; the added links invalidate that. Resolve every link by name at load time.

**Do not pass `CUF_MERGE_FIXED_LINKS`** — merging erases `tool_tip` and breaks FK.

### `panda_with_tool.py`

Copy `franka_panda/panda.urdf` and its meshes out of `pybullet_data` into `assets/` — never mutate the copy in `site-packages`. Splice the tool links into that copy.

Implement `PandaWithTool(PyBulletRobot)` directly against panda-gym's core class, using its `Panda` as a reference to copy from. Do not subclass `Panda`: its `__init__` hardcodes the URDF path, the gripper joints, and `ee_link`, and patching all three is more fragile than writing the ~100 lines. Joint indices 0–6 only, forces `[87, 87, 87, 87, 12, 12, 12]`, `get_ee_position()` resolving to `tool_tip`.

### `spawn_demo.py`

GUI script: sample τ from `ToolPrior`, build, spawn, hold a pose, and print analytic tip vs. PyBullet FK tip, total mass, and steady-state wrist torques.

---

## 5. The one gotcha that will cost you an afternoon

**PyBullet caches URDF files by path.** Regenerating a design and writing it to the same filename will silently reload the *previous* geometry. Either call `p.setPhysicsEngineParameter(enableFileCaching=0)` at startup, or write each design to a unique path keyed by a hash of τ. This bites exactly the sample-in-a-loop pattern this pipeline is built for, and it presents as "my tool parameters have no effect," which is easy to misdiagnose as a bug in the geometry code.

---

## 6. Tests

**Analytic (no sim):**
- `p_tip` matches hand-computed values at φ ∈ {0, ±π/2, ±PHI_MAX}.
- Total mass equals ρ·w·h·(l₁+l₂).
- Inertia tensor is symmetric positive-definite; degenerates correctly to the single-box closed form when l₂ → 0.

**In-sim:**
- PyBullet FK of `tool_tip` matches analytic `p_tip` within 1e-6.
- `getDynamicsInfo` mass matches; reconstruct the full tensor from the returned principal diagonal + inertial frame orientation and compare to the analytic one.
- Fingers and tool are not in collision; nothing else on the robot is either.

**Robustness sweep:**
- 100 τ sampled from `ToolPrior`: every one builds, loads, and steps 100 steps with no NaN and no velocity blowup.
- Explicit degenerate cases: φ = 0 (straight), φ = ±PHI_MAX (maximum fold, boxes closest to overlapping), l₁ and l₂ at both bounds.
- **Gravity hold test:** command a fully extended pose, step 2 s, assert steady-state joint error < 0.02 rad and report peak torque on joints 5–7 against their 12 N·m limit. This is the check that tells you whether a sampled design is physically holdable at all.

---

## 7. Build order

1. `tool_geometry.py` + its analytic tests — no sim, fast to iterate.
2. `tool_urdf.py` + tool-only load test (spawn the tool alone, floating, verify mass/inertia/FK).
3. Splice into the Panda URDF, verify finger weld and collision filtering visually.
4. `PandaWithTool` + panda-gym integration.
5. Robustness sweep over the prior.

Steps 1–2 are where the real content is, and both are testable without ever loading the Panda. Get them exactly right before touching panda-gym.
