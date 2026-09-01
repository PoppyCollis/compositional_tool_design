# §4a. Networks and the design gradient

Settles four things left open by `VGDS_spec.md`
§4-§5, which was written before the observation layout existed.

Decisions, in one line each:

1. **The elbow joins the observation.** `tau` becomes exactly recoverable from `x_t`.
2. **Neither network is conditioned on `tau`.** `V_phi(x)`, `pi(a | x)`. The design
   gradient is the state path alone.
3. **Episode phase `t / HORIZON` joins the observation.** A fixed-horizon return is
   not well defined without it.
4. **A parameter-free feature layer** recovers the tool's dimensions from `x` so the
   networks do not have to learn trigonometry.

---

## 1. Why no `tau` input

`V` does not need to see the tool to have an opinion about it. The tool moves the tip,
the tip is in `x`, and `V` scores `x`. Hand at `(0.5, 0)` pointing along `+x`, target
at `(0.85, 0)`:

| tool              | tip         | distance |
| ----------------- | ----------- | -------- |
| `(0.15, 0.15, 0)` | `(0.80, 0)` | 0.050 m  |
| `(0.16, 0.15, 0)` | `(0.81, 0)` | 0.040 m  |

One centimetre more handle, one centimetre closer, and the return improves. That whole
effect is `(dV/dx)(dh/dtau)` -- the term `VGDS_spec.md` §5 already calls non-zero and
the reason `h` must be differentiable. It is a full 3-vector; `l1`, `l2` and `phi` each
get a component, and the Langevin chain of §6 runs unchanged.

What is dropped is the *other* term. `dV/dtau` asks: hold the picture of the world
frozen -- hand here, tip exactly there -- and now make the handle 1 cm longer. That is
a tool whose length disagrees with where its own tip is, which is not a state the
simulator can produce. Every training sample has the two in agreement, because both
came out of the same rollout. So the network answers that question by extrapolation,
smoothly and with nothing behind it, and the invented number lands in the design
gradient.

Formally, the true `V^pi` is only defined on the surface `{ (x, tau) : o(x) = o(tau) }`
where `o` is the tool's tip offset in the hand frame. Both terms of the two-path
gradient differentiate *off* that surface; only their sum stays on it. The sum is
therefore a difference of two quantities the data never constrained, and two networks
with identical training loss can hand back different `grad_tau f`. Removing the `tau`
input removes the split: one path, and it is the one anchored to the physics.

**This holds for sweeping and pushing too**, once the elbow is in. The tool is the
polyline `(hand, elbow, tip)`, and all of it is then in `x`.

**Mass and inertia are not an exception.** The tool is a rigid two-segment rod of fixed
density, so its mass and how it swings are determined by `l1, l2, phi` and nothing else
-- and those three are read off `x` (§2). Whether the network *learns* to use them is a
training question; it is not missing the information.

## 2. The elbow

`cos psi, sin psi` is not enough. What `x_t` currently pins down is
`o(x) = Rz(psi)^T (tip - hand) = (l1 + l2 cos phi, l2 sin phi)` -- two numbers out of
three. `l1` alone is unrecoverable, so the elbow's *direction* from the hand is known
(always along hand `+x`) but not its distance.

```
elbow = hand + l1 * u(psi)
```

No sign flip on `y`, unlike `se2.tip_from_hand`: the elbow offset in the hand frame is
`(l1, 0)`, and the pi roll only touches the `y` component.

Two consequences.

**`tau` becomes exactly recoverable** -- `l1` from `hand -> elbow`, `(l2, phi)` from
`elbow -> tip` -- which is what makes `V_phi(x) = V_phi(x, tau)` an identity rather than
an approximation.

**`dh/dtau` goes from rank 2 to rank 3.** `tau` reached `x_1` only through the 3->2 map
`o(tau)`; the elbow adds `d(elbow)/d(l1)`, in exactly the direction the tip map is blind
to. `tau |-> x_1` becomes injective.

The cost is in §4.

## 3. Episode phase

`reach_env.step` runs a fixed `HORIZON` with `terminated = False`, and `x_t` carries no
time index. Return-to-go from a state then depends on how many steps remain, so
`V_phi(x)` is fitting an average over `t` -- a state visited at `t = 90` and the same
state at `t = 10` have returns an order of magnitude apart and one regression target
between them.

The two standard fixes are time-awareness (put the remaining time in the observation)
and bootstrapping on truncation (treat the time limit as non-terminal and add
`gamma * V(x_T)` to the last target). Either is correct. We take the first, because the
design objective evaluates `V` at `t = 0` and nowhere else: with the phase in the
observation, `V(x_1)` is exactly "expected return of a full episode with this design",
which is the quantity §5 wants. `h` emits `0`.

If the PPO implementation wraps envs in anything that also bootstraps at truncation,
pick one; doing both double-counts.

## 4. Consequences to check, not assume

**The reach optimum is a ridge, not a point.** Three design knobs, two outcomes that
matter (how far out and at what bearing the tip sits), so every tool has a curve of
physically different, exactly-equivalent tools running through it:

```
(0.120, 0.180, 0.0000)  ->  tip offset (0.3000, 0.0000)
(0.180, 0.120, 0.0000)  ->  tip offset (0.3000, 0.0000)
(0.1500, 0.150, 0.9000) ->  tip offset (0.2432, 0.1175)
(0.1068, 0.180, 0.7115) ->  tip offset (0.2431, 0.1175)
```

`p(tau | g, O=1)` genuinely is a ridge. The §7 diagnostic must be read in `(R, alpha)`
or `(o_x, o_y)`, or free diffusion along the ridge will smear three distinct posteriors
into looking alike.

**Flatness along the ridge becomes a test rather than a tautology.** Without the elbow,
the two tools above produce identical observations, so the network is *forced* to score
them equally. With the elbow it can tell them apart -- the second elbow is 6 cm further
out -- and it still should not care, but nothing stops it learning a slope from noise in
the returns. So: pick a design, walk the ridge, plot `V`. A tilt is an invented
preference, and the chain will slide along it to a tool that is not actually better.
This is the price of decision 1, and it is cheap to check.

**Validate the slope, not the value surface.** §7 grids `f_phi` and compares high-value
regions against the simulator. Necessary, not sufficient: the sampler never reads the
values, it reads the gradient, and a surface can be accurate to a percent everywhere
while carrying ripples whose slopes point the wrong way. On the same grid, difference
the *simulated* return between `tau` and `tau + delta` and compare against
`grad_tau f_phi`. That tests what the chain consumes.

## 5. Observation layout

24 dims, up from 21. **The table lives in `h_initial_state_map.md`**, which has been
updated in place rather than duplicated here -- one layout, one owner, and
`VGDS_spec.md` §2 already names that file authoritative. What changed: `P(elbow xy)`
enters at `7:9`, everything below shifts by two, and `t / HORIZON` is appended at
`23:24`.

`0:11` is `PandaWithTool.get_obs` (`ROBOT_DIM` 9 -> 11); `17:23` is `task.encode`,
unchanged and still contiguous, so `h` copies it wholesale. Three `tau`-dependent
slices now, not two.

The degenerate direction called out in `h_initial_state_map.md` survives: the tip enters
`9:11` as `+tip/s` and `13:15` as `-tip/s`, so any readout that is a plain sum over the
observation cancels the design out exactly.

## 6. The feature layer

Fixed arithmetic, no parameters, computed from `x` and prepended to the network input.
Shared by actor and critic. It recovers the tool's dimensions, which are *in* `x` but
only behind a rotation the network would otherwise have to learn.

```python
def features(x):
    c, s = x[2], x[3]                      # cos psi, sin psi
    d_tip   = x[9:11]  - x[0:2]            # both P(), so centring cancels: (tip - hand)/s
    d_elbow = x[7:9]   - x[0:2]            #                              (elbow - hand)/s

    # Rotate into the hand frame. Inverts se2.tip_from_hand's Rz(psi) @ (ox, -oy).
    ox =  c * d_tip[0] + s * d_tip[1]      # = o_x / s
    oy =  s * d_tip[0] - c * d_tip[1]      # = o_y / s

    # The elbow offset is (l1, 0) in the hand frame, so one projection gives l1
    # directly -- and unlike a norm it is linear, with no kink at the origin.
    l1 = c * d_elbow[0] + s * d_elbow[1]   # = l1 / s

    vx, vy = ox - l1, oy                   # elbow -> tip, hand frame = (l2 cos phi, l2 sin phi)
    l2  = torch.hypot(vx, vy)              # = l2 / s;  l2 >= L_MIN, so atan2 is never at 0
    phi = torch.atan2(vy, vx)

    return torch.cat([x, ox, oy, l1, l2, phi / PHI_MAX], dim=-1)
```

Everything here is a function of `x`, so gradients still flow through `x` and only
through `x` -- the convenience of a `tau` input without reopening §1. Checked against
2000 random `(tau, hand pose)` draws: worst recovery error 1.3e-15, i.e. exact.

`(ox, oy)` is the sufficient statistic for reaching: for that task `V` depends on the
design through those two numbers and nothing else. `l1, l2, phi` are inert for reaching
and are what sweeping and pushing will need.

**This layer is part of the frozen normalisation contract** (`h_initial_state_map.md`).
The designer reapplies it against a `V` trained much earlier, so it lives in one module
read by both the RL stack and the designer, not inlined in a network definition.

`l1, l2` come out on the scene-box metric (`l/s`), not rescaled to the design box, so
every length in the whole system stays on one scale. `phi` is divided by `PHI_MAX`
because it is the one quantity with no length units. Note `SCENE_BOX.scale` is currently
exactly 1.0 m, so `l/s` lands in `[0.1, 0.2]` -- small next to the state features. If
that turns out to matter, rescale `l1, l2` to `[-1, 1]` over the design box and freeze
*that* instead; do not make it a fitted statistic.

## 7. What this changes

Code, done -- every item moved `OBS_DIM`, so they landed together:

- `se2.elbow_from_hand`, plus `initial_state.elbow_from_hand_torch`.
- `panda_with_tool.get_obs` emits the elbow. `ROBOT_DIM` 9 -> 11.
- `initial_state`: `OBS_DIM` 21 -> 24, slice constants renumbered, `PHASE` added,
  `TAU_SLICES` gains the elbow, `h` emits the elbow and a zero phase.
- `reach_env._get_obs` appends `t / HORIZON`; the sim-vs-analytic test covers both new
  slices, and `test_initial_state` now asserts `dh/dtau` is rank 3 and that `tau`
  round-trips out of `x_1` to 1e-12.

Code, still to do:

- New module for the feature layer (§6) and the design-space encoding. Deferred to the
  PPO pass deliberately: it is read by both the RL stack and the designer, and neither
  exists yet. The recovery arithmetic currently sits in
  `tests/test_initial_state.test_tau_is_exactly_recoverable_from_x1` and should be
  lifted from there rather than written twice.
- Drop `tau` from both networks (§1) -- vacuous until there are networks.

Docs, done:

- `VGDS_spec.md`: §4's `V_phi(x, tau)` became `V_phi(x)` and §2's `pi(a | x, tau)` lost
  its `tau`; §5 is now the single term `grad_tau E = -(dV/dx)(dh/dtau)`; §1 and §6 no
  longer describe the circular `theta` that `DesignPriorConfig.PHI_MAX` retired; §7
  gained the slope and ridge checks of §4 above.
- `h_initial_state_map.md`: owns the 24-dim layout table outright (§5 here is a pointer
  to it), states `f_psi(tau, g) = E_xi[V_psi(h(tau, g, xi))]`, says three `tau`-dependent
  slices rather than two, and carries the episode-phase and feature-layer sections.
- `task_encoding_g.md`: the diagnostic must be read in `(o_x, o_y)` because the
  posterior is a ridge, and §8's `w_reach`/`w_trans` expiry is written down where the
  anneal is specified.
- `tool_to_env_plan.md`: its `theta` (interior angle, on a circle) restated as `phi`
  (deflection, on a box), and the tool-mount weld's pitch and roll noted.

## 8. Still open

- **`w_reach`, `w_trans` slots in `task.encode`.** Derived from `task_id` today, so
  correctly excluded. That stops being true the moment `task_encoding_g.md`'s anneal
  starts: an annealed weight is constant within an episode, enters the reward, and is
  not recoverable from `x_t` -- the membership rule, verbatim. Reserving the slots is
  free now and costs a retrain later, the same argument that pre-declared the 3-wide
  one-hot.
- **Whether `h` should model tool inertia at all.** §1 says the information is in `x`;
  it says nothing about whether the analytic `h` should reflect it. It currently does
  not, and for a teleported reset with zero velocities there is nothing to reflect.