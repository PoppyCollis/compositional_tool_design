# Core Behavioral Guidelines (Karpathy Style)

- **Think Before Coding**: Analyze edge cases and state assumptions before writing code.
- **Simplicity First**: Write clean, readable, minimal code. Avoid over-engineering or unnecessary abstractions.
- **Surgical Changes**: Make targeted, local edits. Do not modify unrelated files or global configurations without reason.
- **Goal-Driven Execution**: Focus on verifiable outcomes (e.g., matching a baseline metric or passing a test).

# Academic Constraints & Rules

- **Documentation**: Write Google-style docstrings for every public function, noting mathematical notation or paper references where applicable.

# Where things live

- `reach_env.py` — `ReachEnv`, the gymnasium env and the entry point for the RL side.
  One tool design per env; `reset(options={"task": g, "xi": xi})` pins the draw.
- `initial_state.py` — the 21-dim observation layout, and `h(tau, g, xi)`, the analytic
  twin of `ReachEnv.reset`. Change one and `tests/test_reach_env.py` will tell you.
- `task.py` — `p(g)`, the reward, and the success rule. `config.py` — every constant,
  each carrying its measurement date and the script that produced it.
- `plan.md` / `memory.md` — what is next, and the decisions already made with their
  reasons. Read both before starting.

# git

Do not write any git commands, dont prompt for commits.

