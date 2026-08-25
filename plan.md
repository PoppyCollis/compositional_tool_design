# Plan

## Done
- [x] Audit of the Sonnet phi-reparameterisation pass (see memory.md).
- [x] `_unpack` rejects batched/malformed tau instead of silently returning row 0.
- [x] `_finger_span` calls `ensure_assets()` — fresh clone no longer crashes.
- [x] Gripper constants hoisted into `config.GripperConfig`; phi-bound derivation
      corrected (the "~9 cm clearance" claim was wrong; it is 3.3 cm).
- [x] `interior_angle(phi) = pi - |phi|` added and covered by tests.
- [x] phi-bound test derives its own critical angle instead of hardcoding 2.2.
- [x] File caching re-enabled (220 ms -> 44 ms per load).
- [x] Generated URDFs cleaned up at exit.
- [x] `config.DEVICE` wired into `ToolPrior`; `tests/conftest.py` added.
- [x] 28 tests pass under both `pytest` and `python -m pytest`.

## Next
- [ ] Wrap `PandaWithTool` in a gymnasium env with a task + reward.
- [ ] Stand up `SubprocVecEnv` with per-env tau; confirm scaling against the numbers above.
- [ ] Wire the PPO loop and the outer design-optimisation loop.

## Deferred — revisit only if profiling says so

**Replacing per-episode `loadURDF` with cached shapes + `createMultiBody`** is 62x
faster (0.65 ms vs 40 ms), but the 26 s/iteration figure motivating it assumes tau is
redrawn every reset — an assumption about a training loop that doesn't exist yet.
Profile first. If it does bite, hold tau fixed for K episodes per env before rewriting
anything: K=10 cuts the cost 10x for a few lines, and 64 envs still give 64 designs per
update. `createMultiBody` is the last resort — it drops link names, joint limits, and
off-diagonal inertia (silently), and needs a second construction path kept in sync.
