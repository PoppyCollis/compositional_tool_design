# Value-gradient design search for tools: Project specification

Pilot project: deliberately minimal. Goal: build the pipeline for this simple case first.

Two phases, run in sequence:

1. Learn a tool-conditioned policy and value function, with tools sampled from the prior.
2. Freeze the value function, treat it as an energy, and Langevin-sample tools for a given
   target location.

Single task type (reaching), so there is no task-type variable. The task is fully described
by the target location `g`.

---

## 0. Inputs you must supply

- **Simulator** and the post-design state map `x₁ = h(τ, g)`.
- **Reward** `r(x, a, τ)`.
- **Target distribution** `p(g)`, and a held-out set of targets.

---

## 1. Design space and prior

```
τ = (l₁, l₂, θ)
l₁, l₂ ~ U[l_min, l_max]     (box)
θ      ~ U[0, 2π)            (circle)
```

Network input encoding: `(l₁, l₂, cos θ, sin θ)` — never raw `θ`.

The prior contributes nothing to the energy: uniform on the box is constant, uniform on the
circle is constant. Constraints are enforced geometrically during sampling instead —
reflect `l₁, l₂` at the bounds, wrap `θ` mod 2π. Both are measure-preserving, so the chain
still targets the correct distribution restricted to the design space.

---

## 2. State

```
x = ( arm state , tool end-effector state , g )
```

`g` lives in the state, so **do not condition the policy on `g` separately** — `π(a | x, τ)`
already sees it. It reaches the design objective through `x₁`.

The tool end-effector state is forward kinematics of the arm *and* the tool geometry, so it
is a function of `τ`.

---

## 3. Controller training

```
for iteration in 1..N:
    for each parallel episode in the batch:
        τ ~ prior
        g ~ p(g)
        x₁ = h(τ, g)
        roll out π(a | x, τ)
    update π and V from the batch
```

Designs are drawn fresh from the prior every episode and never adapt, so a single batch
already spans the whole design space. `π` must become competent across all of it.

---

## 4. Value function

PPO. The critic is a direct value head `V_φ(x, τ).` Fit it by regression onto the returns observed in the batch:

```
L(φ) = E [ ( V_φ(x_t, τ) − R_t )² ]
```

---

## 5. Design objective

```
f_φ(τ, g) := V( h(τ, g), τ )
E(τ, g)   = − f_φ(τ, g)
∇_τ E = − [ ∂V/∂τ  +  (∂V/∂x)·(∂h/∂τ) ]
```

The second term is non-zero because the tool end-effector state depends on `τ`. `h` must be differentiable in `τ`; the simulator need not be.

---

## 6. Designer

Value function frozen. Fresh chain per target `g`.

```
τ ~ prior
for k in 1..K:
    τ ← τ − η ∇_τ E(τ, g) + √(2η) ε ,   ε ~ N(0, I)
    reflect l₁, l₂ at the bounds
    θ ← θ mod 2π
```

Obtain the `θ` gradient component by backpropagating through the `(cos θ, sin θ)` encoding.
Report the chain as samples from `p(τ | g, 𝒪=1)`, not as a single optimum.

---

## 7. Evaluation

On held-out targets:

1. Run the designer → `τ*`
2. Execute `π(a | x, τ*)` in the simulator
3. Record whether the tool tip reaches `g`

Baselines: best of `N` designs sampled from the prior under the same evaluation b`udget; and
exhaustive grid search over the 3D design space, which is cheap here and gives the ceiling.

Grid `f φover )`he design space
for a few held-out `g` and compare against true simulator outcomes on the same grid. If the
high-value regions disagree, the failures are value-function failures, not sampler failures.