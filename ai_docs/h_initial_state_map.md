# §2c. Initial-state map `h`

```
ξ  ∼ p(ξ | g)              reset randomness: hand pose, object start pose
x₁ = h(τ, g, ξ)            deterministic, differentiable in τ
```

Implemented in `initial_state.py`. Bridges design to value: `V` scores states, `h`
turns a design into one.

```
elbow = hand_xy + l₁ · u(ψ)                 se2.elbow_from_hand
tip   = hand_xy + R(τ) · u(ψ + α(τ))        se2.tip_polar
```

No sign flip on `y` in the elbow, unlike `se2.tip_from_hand`: the elbow's offset in
the hand frame is `(l₁, 0)`, and the π roll of the fingers-down pose only touches the
`y` component. The elbow is in the observation so that `τ` is *exactly* recoverable
from `x_t` — `networks_and_design_gradient.md` §2.

`ξ` is not in `g`. The hand pose fails the membership rule — it enters no reward term
and is plainly visible in `x₁` — and it is drawn exactly as `PandaWithTool.reset`
draws it, uniform over the workspace inset by `RESET_MARGIN` with yaw uniform over
`±YAW_LIMIT`, so the designer's expectation is taken against the same law the policy
trained under. If those two diverge, the design objective is scoring states `V` never
saw.

The **object's start pose is also part of `ξ`**, drawn from a distribution keyed by
`task_id` (`task.object_start`). This file used to say "`s_start` is in `g`: the
distribution is task, the sample is not". That is the line the current design
overturns — see `task_encoding_g.md`. The distribution is still task, but it is keyed
by the task id rather than carried as a field, because `x_t` already re-emits the
object's current pose every step and one quantity should not live in two places. For
reaching the draw is deterministic: the object sits at `p_target` and never moves.

**Objective.** `f_ψ(τ, g) = E_ξ[ V_ψ( h(τ, g, ξ) ) ]` — the state bank. `V` is *not*
conditioned on `τ`, so `∂V/∂τ` is zero by construction and the state path is the whole
design gradient (`networks_and_design_gradient.md` §1). `ξ ⊥ τ`, so the `∂h/∂τ` path
survives the expectation (§5), and one `ξ` draw scores every design in a batch on the
same reset.

**Constraint.** PyBullet kinematics are not differentiable; `h` is implemented in
torch instead. Only the tool kinematics need gradients — see the layout below, where
`τ` enters exactly **three** slices: the elbow, the tip, and the `obj − tip` vector
that hangs off the tip. `initial_state.TAU_SLICES` and the test asserting the gradient
is identically zero everywhere else are written against that sentence.

Trajectory randomness (action noise, contact) is already marginalised inside `V^π`.

---

## Observation layout

The single description of `x_t`, 24 dims. `P(p) = (p − c)/s` and `D(v) = v/s`, with
`c` and `s` from `TaskConfig.SCENE_BOX`.

| slice | contents | τ-dependent |
| --- | --- | --- |
| `0:2` | `P(hand xy)` | no |
| `2:4` | `cos ψ, sin ψ` | no |
| `4:7` | hand `vx, vy, ψ̇` / `VEL_SCALE` | no (zero at reset) |
| `7:9` | `P(elbow xy)` | **yes** (`l₁` only) |
| `9:11` | `P(tip xy)` | **yes** |
| `11:13` | `P(object xy)` | no |
| `13:15` | `D(object − tip)` — reach/contact phase | **yes** |
| `15:17` | `D(target − object)` — transport phase | no |
| `17:19` | `P(target xy)` | no |
| `19:22` | task one-hot | no |
| `22:23` | `r_obj / s` | no |
| `23:24` | `t / HORIZON` | no |

Slices `0:11` are `PandaWithTool.get_obs` (`ROBOT_DIM` 9 → 11); `17:23` are
`task.encode`, unchanged and still contiguous, so `h` copies it wholesale. `h`
reproduces the first block analytically, and the two agree because both call
`se2.Box.normalise_point` on the same box — not because anything asserts it. The env
layer is where that assertion belongs.

`x_t` carries the object's current pose every step and `g` is appended to the same
vector, so `V(x)` already sees the target; do not condition the policy on `g`
separately (`VGDS_spec.md` §2).

**`obj − tip` and `target − obj`, not `target − tip`.** The reward is
`−[w_reach·d(tip,obj) + w_trans·d(obj,target)]`, so those two are the terms the policy
is differentiating. Their sum, `target − tip`, points somewhere useful in neither
sweeping nor pushing. For reaching the object is at the target, so the first coincides
with `target − tip` and the second is identically zero.

Note the redundancy has one degenerate direction: the tip enters `9:11` as `+tip/s` and
`13:15` as `−tip/s`, so any readout that is a plain sum over the observation cancels
the design out exactly.

### Episode phase

`t / HORIZON` is in the observation because `reach_env.step` runs a fixed `HORIZON`
with `terminated = False`. Return-to-go from a state then depends on how many steps
remain, so without a time index `V_ψ(x)` is fitting an average over `t`: the same state
visited at `t = 90` and at `t = 10` have returns an order of magnitude apart and one
regression target between them.

`h` emits `0`. The design objective evaluates `V` at `t = 0` and nowhere else, so with
the phase in the observation `V(x₁)` is exactly "expected return of a full episode with
this design", which is the quantity `VGDS_spec.md` §5 wants.

The other correct fix is bootstrapping on truncation — treat the time limit as
non-terminal and add `γ·V(x_T)` to the last target. **Pick one.** If the PPO
implementation wraps envs in anything that bootstraps at truncation by default, doing
both double-counts. See `networks_and_design_gradient.md` §3.

---

## The normalisation contract

One box, one scalar, frozen for the life of a trained `V`.

```
c = SCENE_BOX.centre                    # per-axis centring
s = max(SCENE_BOX.half_extents)         # ONE scalar divisor
```

Two decisions, each closing a specific failure:

**Not the hand's workspace.** The tool exists to put the tip *outside* the arm's
reach, so normalising on `SE2Config.WORKSPACE` would send long tools past ±1: exactly
the designs the search evaluates would land in the network's extrapolation region, and
`∂V/∂x₁` is half the design gradient. The same argument applies to the object, which in
sweeping and pushing crosses the reach boundary in both directions, so no sub-box
contains its trajectory. `SCENE_BOX` contains the whole tool-reachable set
(`x ∈ [0.150, 1.035]`, `y ∈ [−0.800, 0.795]` over 2000 prior designs) with padding.

**One scalar, not one divisor per axis.** `∂h/∂τ` is in metres. Under per-axis scaling
one normalised unit would mean 0.55 m in x and 1.0 m in y, so two designs producing
physically equal tip displacements would get unequal gradient magnitudes purely from
the box's aspect ratio. A single divisor keeps the Jacobian isotropic. The price is
that the shorter axis normalises to ±0.55 rather than filling ±1 — symmetric, just not
full scale.

**The map must never move.** `h` reapplies it at design time against a value function
trained much earlier. A running observation normaliser in the RL stack (`VecNormalize`
and friends) would make `x ↦ x̃` drift during training and silently invalidate that
reapplication. Turn it off, or verify it is frozen. The same holds for reward
normalisation if `V`'s scale is to mean anything as an energy.

### The feature layer is part of this contract

The parameter-free feature layer (`networks_and_design_gradient.md` §6, which carries
the code) recovers `(o_x, o_y, l₁, l₂, φ)` from `x` by fixed arithmetic and prepends
them to the network input, shared by actor and critic. Everything it computes is a
function of `x`, so it adds no `τ` input and reopens nothing in §1 — it only saves the
networks from learning a rotation they are handed for free.

Because the designer reapplies it against a `V` trained much earlier, it moves under
exactly the same rule as the box above: **frozen, and living in one module read by both
the RL stack and the designer**, never inlined in a network definition.

`l₁, l₂` come out on the scene-box metric (`l/s`), so every length in the system stays
on one scale; `φ` is the one quantity with no length units and is divided by
`PHI_MAX`. `SCENE_BOX.scale` is currently exactly 1.0 m, so `l/s` lands in `[0.1, 0.2]`
— small next to the state features. If that turns out to matter, rescale `l₁, l₂` to
`[−1, 1]` over the design box and freeze *that* instead; do not make it a fitted
statistic.
