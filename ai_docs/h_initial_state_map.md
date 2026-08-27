# §2c. Initial-state map `h`

```
ξ  ∼ p(ξ | g)              reset randomness: q₀, object start pose from s_start, noise
x₁ = h(τ, g, ξ)            deterministic, differentiable in τ
```

Bridges design to value: `V` scores states, `h` turns a design into one.

```
tip pose = ( gripper pose at q₀ )  ∘  tool_kinematics(l₁, l₂, θ)
```

`ξ` is not in `g` — `q₀` fails the membership rule (no reward term, visible in `x₁`).
`s_start` is in `g`: the distribution is task, the sample is not.

**Objective.** `f_ψ(τ, g) = E_ξ[ V_ψ( h(τ, g, ξ), τ, g ) ]` — the state bank. `ξ ⊥ τ`, so
the `∂h/∂τ` path survives the expectation (§5).

**Constraint.** PyBullet kinematics are not differentiable; implement `h` in the autodiff
framework. Only `tool_kinematics` needs gradients.

Trajectory randomness (action noise, contact) is already marginalised inside `V^π`.
