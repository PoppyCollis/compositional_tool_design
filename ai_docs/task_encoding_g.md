# §2b. Task encoding

```
g = ( task_id,     —    which task instance: reach | sweep | push
      p_target,    ℝ²   target position, robot base frame
      r_obj,       ℝ    the desired object, as a plan-view radius
      ρ_target,    ℝ    success tolerance
      w_reach,     ℝ    weight on tip→object
      w_trans      ℝ    weight on object→target )
```

Implemented in `task.py`. `g` is the **target specification and nothing else**: which
task, where, which object.

**Membership rule.** `g` is what the target specification names; `x_t` is everything
currently observable. A quantity belongs in `g` if it is constant over the episode,
enters reward or dynamics, and is not recoverable from `x_t`.

**Where `s_start` went.** An earlier version of this file carried `s_start`, the
object's start region, so that `h(τ, g)` would be well-defined. It is gone. `x_t` is
the full observable state and re-emits the object's *current* pose every step, so the
object's starting position lands in `x₁` for free; a second copy in `g` would put one
quantity in two places. What replaces it is `task.object_start`, a distribution keyed
by `task_id` and drawn at reset inside `h`. That keeps `g` independent of the design,
which the `p(τ | g, 𝒪=1)` factorisation requires.

`m`, `μ` are gone for the same reason a reaching object has no mass: it is a ghost
sphere with no collision body. They return with the first real body, in sweeping.

`ρ_target`, `w_reach` and `w_trans` are *derived* from `task_id` (`task.TASK_PARAMS`)
rather than sampled. They are fields of `g` all the same, so the objective and the
success metric are read off one object and cannot drift apart — but they are excluded
from `task.encode`, since the id they follow from is already in it.

**That exclusion has an expiry.** It stops being correct the moment the `w_reach`
anneal below starts. An annealed weight is constant within an episode, enters the
reward, and is not recoverable from `x_t` — the membership rule above, verbatim — so it
belongs in the encoding. Reserving the two slots now is free; adding them after a
training run costs a retrain, which is the same argument that pre-declared the 3-wide
one-hot. See `networks_and_design_gradient.md` §8.

---

## Sampling

```
p_target ~ U(TaskConfig.TARGET_BOX)      x ∈ [0, 1.10], y ∈ [-1, 1]
```

No rejection against reachability. Measured over 2000 prior designs at `r_obj = 0.03`,
that box is

| band | share |
| --- | --- |
| reachable by no design (`f = 0`) | 39.7 % |
| discriminating (`0 < f < 1`) | 49.2 % |
| reachable by every design (`f = 1`) | 11.1 % |
| reachable by the bare gripper | 15.6 % |

The impossible 40% is deliberate: the value function should learn that some targets
are hopeless whatever the tool, rather than only ever seeing solvable ones and having
to extrapolate at design time. Two consequences follow and neither is optional.

- **Evaluation must stratify by band.** An aggregate success rate over the box mostly
  measures the mixing ratio. `task_space.coverage` is the labeller.
- **Only the middle band carries design signal.** At `f = 1` every tool succeeds, so
  `p(τ | g, 𝒪=1)` stays the prior; at `f = 0` nothing does.

---

## Reward

```
r_t = − [ w_reach · d(tip_t, obj_t)  +  w_trans · d(obj_t, p_target) ]
```

Dense, per step, over a **fixed horizon with no early termination on success**
(`TaskConfig.HORIZON`). Terminating early truncates the accumulating negative reward,
which makes `V` jump discontinuously at the `ρ` boundary and puts success and failure
returns on different scales. `V` is read as an energy over designs, so that
comparability is worth more than the wall-clock an early exit would save.

A fixed horizon with no termination is also why `t / HORIZON` is in the observation
(`h_initial_state_map.md`): without it, return-to-go from a state depends on how many
steps remain and `V` is fitting an average over `t`. Time-awareness and bootstrapping on
truncation are both correct fixes and only one may be used — doing both double-counts.

### Success

Not one rule for all three types:

```
reach:         𝒪 = 1  ⟺  d(tip_T,  p_target) < ρ_target
sweep, push:   𝒪 = 1  ⟺  d(obj_T,  p_target) < ρ_target
```

The unified object-based rule this file used to give is *trivially true* for reaching,
where the object is pinned at the target and never moves: every episode would score a
success, including one where the tip never left the far side of the table. See
`task.success`, and the test that pins it.

---

## Task instances

|       | object start (drawn in `h`) | `p_target`   | `w_reach` | `w_trans` | object                       |
| ----- | --------------------------- | ------------ | --------- | --------- | ---------------------------- |
| Reach | **at `p_target`**           | anywhere     | 1         | 0         | ghost sphere, no collision   |
| Sweep | out of reach                | in reach     | ε_s       | 1         | real                         |
| Push  | in reach                    | out of reach | ε_p       | 1         | real                         |

Only reach is implemented. Sweep and push are declared in `task.TaskType` so the
one-hot in `task.encode` has its final width — adding a type later would change the
observation size and invalidate every trained `V` — but they have no entry in
`TASK_PARAMS` and no start distribution, so both raise rather than silently inheriting
reaching's weights.

In reaching, tip→object *is* the task; in sweeping and pushing it is shaping only.
Sweep vs. push is not a field — it is the relation between the start distribution and
`p_target` across the workspace boundary.

The reaching object is virtual and immovable. If it were a real body, a tool that
reached it would knock it away and the agent would be punished for succeeding.

### Curriculum alternative.

Instead of a persistent shaping term, set w_reach = 0 and curriculum on the start
distribution: begin with the object at the workspace boundary, where the arm alone
nearly reaches it, and translate it outward (sweep) or inward (push) as success rate
crosses a threshold. This gives dense early signal without rewarding tools that merely
point at the object, and removes the need to tune ε_s, ε_p, or an annealing schedule.

Worth noting it moves the tuning rather than eliminating it — you now have an
advancement threshold and a step size, and the curriculum can stall if you widen too
fast. But the parameters are ones you can read off a success-rate curve, which the
shaping weights aren't.

---

## Constraints

- `p(g)` samples `w_reach` **per task type**. Require `ε_p < ε_s`: in pushing the
  gripper alone satisfies the tip term at `τ`-independent cost, so a large `w_reach`
  lets the policy park on the object and stall.
- Anneal `w_reach → 0`. If `∂f_ψ/∂τ` collapses under the anneal, the design signal was
  the shaping term, not the task. The anneal is also what forces `w_reach` and
  `w_trans` into `task.encode` — see the note above.
- Horizon must cover reach *plus* transport. With `w_trans` active the return no longer
  splits into transit + tail as assumed in §0.

---

## Diagnostic

Run the designer on one `g` of each type from the same frozen `V`. If the three
posteriors `p(τ | g, 𝒪=1)` do not visibly separate, `V` is not reading `g` — it has
found something generically good and the conditioning is cosmetic.

Cheap in a 3D design space, and it fails loudly.

Two ways it fails for the wrong reason, and both must be closed before the result
means anything.

**Draw the three `g`s from the `0 < f < 1` band.** At `f = 1` the posteriors *should*
coincide, so a diagnostic run on easy targets proves nothing.

**Read the posteriors in `(o_x, o_y)`, not in raw `τ`.** Three design knobs, two
outcomes that matter — how far out and at what bearing the tip sits — so every tool lies
on a curve of physically different, exactly-equivalent tools, and `p(τ | g, 𝒪=1)`
genuinely is a ridge rather than a mode. Plotted in `τ`, free diffusion along that ridge
smears three distinct posteriors into looking alike. Project onto the tip offset
`(o_x, o_y)` or its polar form `(R, α)` first. See `networks_and_design_gradient.md` §4.
