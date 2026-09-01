# §2c. Initial-state map `h`

```
ξ  ∼ p(ξ | g)              reset randomness: hand pose, object start pose
x₁ = h(τ, g, ξ)            deterministic, differentiable in τ
```

Implemented in `initial_state.py`. Bridges design to value: `V` scores states, `h`
turns a design into one.

```
tip = hand_xy + R(τ) · u(ψ + α(τ))          se2.tip_polar
```

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

**Objective.** `f_ψ(τ, g) = E_ξ[ V_ψ( h(τ, g, ξ), τ, g ) ]` — the state bank. `ξ ⊥ τ`,
so the `∂h/∂τ` path survives the expectation (§5), and one `ξ` draw scores every
design in a batch on the same reset.

**Constraint.** PyBullet kinematics are not differentiable; `h` is implemented in
torch instead. Only the tool kinematics need gradients — see the layout below, where
`τ` enters exactly two slices.

Trajectory randomness (action noise, contact) is already marginalised inside `V^π`.

---

## Observation layout

The single description of `x_t`, 21 dims. `P(p) = (p − c)/s` and `D(v) = v/s`, with
`c` and `s` from `TaskConfig.SCENE_BOX`.

| slice | contents | τ-dependent |
| --- | --- | --- |
| `0:2` | `P(hand xy)` | no |
| `2:4` | `cos ψ, sin ψ` | no |
| `4:7` | hand `vx, vy, ψ̇` / `VEL_SCALE` | no (zero at reset) |
| `7:9` | `P(tip xy)` | **yes** |
| `9:11` | `P(object xy)` | no |
| `11:13` | `D(object − tip)` — reach/contact phase | **yes** |
| `13:15` | `D(target − object)` — transport phase | no |
| `15:17` | `P(target xy)` | no |
| `17:20` | task one-hot | no |
| `20:21` | `r_obj / s` | no |

Slices `0:9` are `PandaWithTool.get_obs`; `15:21` are `task.encode`. `h` reproduces
the first block analytically, and the two agree because both call
`se2.Box.normalise_point` on the same box — not because anything asserts it. The env
layer is where that assertion belongs.

`x_t` carries the object's current pose every step and `g` is appended to the same
vector, so `V(x, τ)` already sees the target; do not condition the policy on `g`
separately (`VGDS_spec.md` §2).

**`obj − tip` and `target − obj`, not `target − tip`.** The reward is
`−[w_reach·d(tip,obj) + w_trans·d(obj,target)]`, so those two are the terms the policy
is differentiating. Their sum, `target − tip`, points somewhere useful in neither
sweeping nor pushing. For reaching the object is at the target, so the first coincides
with `target − tip` and the second is identically zero.

Note the redundancy has one degenerate direction: the tip enters `7:9` as `+tip/s` and
`11:13` as `−tip/s`, so any readout that is a plain sum over the observation cancels
the design out exactly.

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
