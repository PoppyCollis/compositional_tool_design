# Value-gradient design search for tools: Project specification

Pilot project: deliberately minimal. Goal: build the pipeline for this simple case first.

Two phases, run in sequence:

1. Learn a policy and value function, with tools sampled from the prior.
2. Freeze the value function, treat it as an energy, and Langevin-sample tools for a given
   target location.

Reaching is the only *implemented* task type. Sweeping and pushing are pre-declared in
`task.TaskType` so the one-hot in `task.encode` already has its final width — adding a
type later would change the observation size and invalidate every trained `V` — but
they have no entry in `TASK_PARAMS` and raise rather than silently inheriting
reaching's weights. The task is described by `g` (`task_encoding_g.md`), whose only
free field for reaching is the target location.

---

## Setup

- Simulator: PyBullet
- Robot: Franka Panda, 7 DoF.
- Control mode: **SE(2) operational space.** The action is 3-dimensional — `(dx, dy, dyaw)` of the hand — accumulated onto an internal target, clipped into a measured workspace rectangle, and solved by IK into the 7 joint position targets that PyBullet's internal PD then tracks. Height, roll and pitch are constants, so the tool is always low and parallel to the ground. Superseded the original 7-DoF joint-delta action (`q_target = q_current + σ·a`): for a task where the tool must stay flat, six of those dimensions are spent rediscovering the joint manifold that keeps it flat. See `panda_with_tool.py`, `se2.py` and `workspace_sweep.py`.
- Reward: dense, negative distance from the tool tip to g — now in the table plane, since the tip's height is a constant independent of τ.
- Randomise the initial arm configuration q₀ — drawn as an SE(2) pose from inside the workspace rectangle, not as raw joint angles, so no episode starts pressed against a boundary. That draw is `ξ`, and `h` takes it as an argument rather than fixing it.
- Tools have mass: PyBullet gives this for free if the tool is an actual body in the scene rather than an analytic offset.
- **The simulator need not be differentiable, but the post-design state map `x₁ = h(τ, g, ξ)` must be.** Implement `h` in the autodiff framework instead of reading it off PyBullet. This is cheap because only the tool part needs gradients: with the arm teleported to `ξ`'s pose the hand is a constant, and

  ```
  x₁ tool tip = ( hand pose from ξ )  ∘  tool_kinematics(l₁, l₂, φ)
  ```

  See `initial_state.py` and `h_initial_state_map.md`.

---

## 1. Design space and prior

```
τ = (l₁, l₂, φ)
l₁, l₂ ~ U[L_MIN, L_MAX]           box, 0.1–0.2 m
φ      ~ U[−PHI_MAX, PHI_MAX]      box, ±1.9 rad — NOT a circle
```

`φ` is **deflection from straight**, not the interior angle at the elbow: `φ = 0` is a
straight rod, the longest tool a given `(l₁, l₂)` can make. The interior angle is
`π − |φ|` (`tool_geometry.interior_angle`, unsigned, because `+φ` and `−φ` are
mirror-image tools sharing an elbow). `DesignPriorConfig.PHI_MAX = 1.9` is derived in
`config.py` from the worst corner of the design box, where a larger deflection would
swing the tip up into the finger mount.

`φ` is a bounded interval, so it goes into the network raw. There is no `(cos φ, sin φ)`
encoding and nothing wraps.

The prior contributes nothing to the energy: uniform on a box is constant, and all
three parameters are now box-constrained. Constraints are enforced geometrically during
sampling instead — reflect `l₁`, `l₂` and `φ` at their bounds. Reflection is
measure-preserving, so the chain still targets the correct distribution restricted to
the design space.

---

## 2. State

```
x = ( arm state , tool state , object state , g )
```

`g` lives in the state, so **do not condition the policy on `g` separately** —
`π(a | x)` already sees it. It reaches the design objective through `x₁`.

The authoritative, slice-by-slice description of `x` is the observation layout table in
`h_initial_state_map.md`; `initial_state.py` is the implementation. `x_t` is the full
currently-observable state and re-emits the object's pose every step, which is why `g`
in `task_encoding_g.md` carries only the target specification and no start region.

The tool state — elbow *and* tip — is forward kinematics of the arm *and* the tool
geometry, so it is a function of `τ`. Both are in the observation, which makes `τ`
exactly recoverable from `x` and is what lets §4 drop the `τ` input.

---

## 3. Controller training

```
for iteration in 1..N:
    for each parallel episode in the batch:
        τ ~ prior
        g ~ p(g)
        ξ ~ p(ξ | g)
        x₁ = h(τ, g, ξ)
        roll out π(a | x)
    update π and V from the batch
```

Designs are drawn fresh from the prior every episode and never adapt, so a single batch
already spans the whole design space. `π` must become competent across all of it.

---

## 4. Value function

PPO. The critic is a direct value head `V_φ(x)`. Fit it by regression onto the returns
observed in the batch:

```
L(φ) = E [ ( V_φ(x_t) − R_t )² ]
```

**No `τ` input, to either network.** `τ` is recoverable from `x` (§2), so `V_φ(x)` is
the same function — but the true `V^π` is only defined on the surface where the two
agree, and a network given both inputs answers `∂V/∂τ` by extrapolating off that
surface, where no training sample ever lived. Two critics with identical training loss
would then hand back different design gradients. Dropping the input removes the split.
The full argument, and why mass and inertia are not an exception, is
`networks_and_design_gradient.md` §1.

---

## 5. Design objective

```
f_φ(τ, g) := V( h(τ, g, ξ) )        in expectation over ξ
E(τ, g)   = − f_φ(τ, g)
∇_τ E     = − (∂V/∂x)·(∂h/∂τ)
```

One term, not two. `∂V/∂τ` is **zero by construction** — `V` has no `τ` input (§4) —
rather than merely small, so the state path is the entire design gradient. `h` must be
differentiable in `τ`; the simulator need not be.

With the elbow in the observation, `∂h/∂τ` is rank 3: `τ` used to reach `x₁` only
through the 3→2 tip-offset map, and the elbow adds `∂(elbow)/∂l₁` in exactly the
direction that map is blind to. `τ ↦ x₁` is injective, so `l₁`, `l₂` and `φ` each get a
gradient component.

---

## 6. Designer

Value function frozen. Fresh chain per target `g`.

```
τ ~ prior
for k in 1..K:
    τ ← τ − η ∇_τ E(τ, g) + √(2η) ε ,   ε ~ N(0, I)
    reflect l₁, l₂ at [L_MIN, L_MAX]
    reflect φ at ±PHI_MAX
```

All three parameters are reflected at box bounds; nothing wraps and nothing is
backpropagated through a trigonometric encoding. Report the chain as samples from
`p(τ | g, 𝒪=1)`, not as a single optimum.

---

## 7. Evaluation

On held-out targets:

1. Run the designer → `τ*`
2. Execute `π(a | x)` in the simulator with the tool built from `τ*`
3. Record whether the tool tip reaches `g`

Baselines: best of `N` designs sampled from the prior under the same evaluation budget;
and exhaustive grid search over the 3D design space, which is cheap here and gives the
ceiling.

**Stratify by coverage band.** 39.7% of `TaskConfig.TARGET_BOX` is reachable by no
design in the prior and 11.1% by every design, so an aggregate success rate over the box
mostly measures the mixing ratio. Only the 49.2% in between carries design signal.
`task_space.coverage` is the labeller; `task_encoding_g.md` has the table.

**Grid `f_φ` over the design space** for a few held-out `g` and compare against true
simulator outcomes on the same grid. If the high-value regions disagree, the failures
are value-function failures, not sampler failures.

That is necessary and not sufficient — the sampler never reads the values, it reads the
gradient. Two further checks, from `networks_and_design_gradient.md` §4:

- **Validate the slope.** On the same grid, difference the *simulated* return between
  `τ` and `τ + δ` and compare against `∇_τ f_φ`. A surface accurate to a percent
  everywhere can still carry ripples whose slopes point the wrong way, and the slope is
  what the chain consumes.
- **Walk the ridge.** Three design knobs but only two outcomes that matter, so every
  tool sits on a curve of physically different, exactly-equivalent tools. Pick a design,
  move along that curve, plot `V`. It should be flat. A tilt is a preference the network
  invented from noise in the returns, and the chain will slide along it to a tool that
  is not actually better. This became a real test rather than a tautology the moment the
  elbow entered the observation and made those tools distinguishable.
