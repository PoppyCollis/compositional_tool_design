# §2b. Task encoding

```
g = ( p_target,    ℝ³   target position, base frame
      ρ_target,    ℝ    success tolerance
      s_start,     —    object start region (sampled at reset)
      w_reach,     ℝ    weight on tip→object
      w_trans,     ℝ    weight on object→target
      m, μ, r_obj  ℝ³   object mass, friction, radius )
```

**Membership rule.** Constant over the episode, enters reward or dynamics, and not
recoverable from `x_t` before first contact.

Object *current* pose is state, not `g`. Object *start* pose is `x₁ ∼ p(· | s_start)`;
`s_start` lives in `g` so that `h(τ, g)` is well-defined.

---

## Reward

```
r_t = − [ w_reach · d(tip_t, obj_t)  +  w_trans · d(obj_t, p_target) ]

𝒪 = 1   ⟺   d(obj_T, p_target) < ρ_target
```

`ρ_target` defines the success event used in §7 evaluation. Keep it a field of `g`, not a
hardcoded eval constant, so objective and metric cannot drift apart.

---

## Task instances

|       | `s_start`     | `p_target`   | `w_reach` | `w_trans` | object             |
| ----- | ------------- | ------------ | --------- | --------- | ------------------ |
| Reach | at `p_target` | in reach     | 1         | 0         | virtual, immovable |
| Sweep | out of reach  | in reach     | ε_s       | 1         | real               |
| Push  | in reach      | out of reach | ε_p       | 1         | real               |

In reaching, tip→object *is* the task; in sweeping and pushing it is shaping only. Sweep vs.
push is not a field — it is the relation between `s_start` and `p_target` across the
workspace boundary.

The reaching object is virtual and immovable. If it were a real body, a tool that reached it
would knock it away and the agent would be punished for succeeding.

### Curriculum alternative. 

Instead of a persistent shaping term, set w_reach = 0 and curriculum on s_start: begin with the object at the workspace boundary, where the arm alone nearly reaches it, and translate s_start outward (sweep) or inward (push) as success rate crosses a threshold. This gives dense early signal without rewarding tools that merely point at the object, and removes the need to tune ε_s, ε_p, or an annealing schedule.

Worth noting it moves the tuning rather than eliminating it — you now have an advancement threshold and a step size, and the curriculum can stall if you widen too fast. But the parameters are ones you can read off a success-rate curve, which the shaping weights aren't

---

## Constraints

- `p(g)` samples `w_reach` **per task type**. Require `ε_p < ε_s`: in pushing the gripper
  alone satisfies the tip term at `τ`-independent cost, so a large `w_reach` lets the policy
  park on the object and stall.
- Anneal `w_reach → 0`. If `∂f_ψ/∂τ` collapses under the anneal, the design signal was the
  shaping term, not the task.
- Horizon must cover reach *plus* transport. With `w_trans` active the return no longer
  splits into transit + tail as assumed in §0.

---

## Diagnostic

Run the designer on one `g` of each type from the same frozen `V`. If the three posteriors
`p(τ | g, 𝒪=1)` do not visibly separate, `V` is not reading `g` — it has found something
generically good and the conditioning is cosmetic.

Cheap in a 3D design space, and it fails loudly.