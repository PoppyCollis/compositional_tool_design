"""Train PPO on ReachEnv: N SubprocVecEnv workers, each drawing a fresh tool per episode.

This is phase 1 of ``ai_docs/VGDS_spec.md``: learn ``pi(a | x)`` and ``V(x)`` with tools
sampled from the prior. Three things are fixed here on purpose, and the designer
(phase 2) depends on all of them:

- **No VecNormalize, for observations or rewards.** ``initial_state.h`` reapplies the
  frozen ``SCENE_BOX`` map at design time against the ``V`` trained here; a running
  normaliser would make that map drift and silently invalidate the reapplication, and
  a reward normaliser would strip ``V`` of its meaning as an energy
  (``ai_docs/h_initial_state_map.md``, "The normalisation contract"). The env already
  emits normalised observations.
- **Rewards in metres, scale 1.** Recorded here rather than left implicit: the
  Langevin chain samples ``p(tau) ~ exp(V)``, so a reward scale is a temperature.
- **The horizon is a terminal, never a truncation**, so SB3 does not bootstrap
  ``gamma * V(x_T)`` on top of the phase already in the observation
  (``reach_env.ReachEnv.step``).

CPU only, by construction: CUDA is hidden from torch before it is imported, and the
worker count plus learner threads are chosen to leave cores free for other users.

Usage:
    python train_ppo.py                                  # PPOConfig defaults
    python train_ppo.py --n-envs 4 --total-timesteps 20000 --run-dir runs/smoke
    tensorboard --logdir runs
"""
import os

# Before torch or numpy import. CUDA_VISIBLE_DEVICES="" means even a CUDA build of
# torch sees no device; the thread caps are inherited by the SubprocVecEnv workers,
# so each env process stays on one core instead of spawning an idle BLAS pool.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import datetime
import json
import pathlib

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import networks
from config import PPOConfig
from reach_env import ReachEnv

# Logged per episode by Monitor, next to return and length. tau is what makes the
# per-design return recoverable once every episode has its own tool.
MONITOR_KEYWORDS = ("is_success", "tau", "target")


def parse_args(argv=None):
    """Command line over ``PPOConfig``; every default is a constant there."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-envs", type=int, default=PPOConfig.N_ENVS)
    parser.add_argument("--n-steps", type=int, default=PPOConfig.N_STEPS)
    parser.add_argument("--total-timesteps", type=int, default=PPOConfig.TOTAL_TIMESTEPS)
    parser.add_argument("--rebuild-every", type=int, default=PPOConfig.REBUILD_EVERY,
                        help="episodes per tool per env")
    parser.add_argument("--seed", type=int, default=PPOConfig.SEED)
    parser.add_argument("--torch-threads", type=int, default=PPOConfig.TORCH_THREADS)
    parser.add_argument("--run-dir", type=pathlib.Path, default=None,
                        help=f"default {PPOConfig.RUNS_DIR}/<timestamp>_reach")
    return parser.parse_args(argv)


def make_vec(n_envs, seed, rebuild_every, monitor_dir):
    """N ReachEnv workers behind a SubprocVecEnv, each wrapped in Monitor.

    ``make_vec_env`` seeds worker ``i`` with ``seed + i`` on its first reset, so the
    workers do not share an episode stream even though ``ReachEnv.__init__``'s own
    reset is unseeded.

    Args:
        n_envs (int): Worker count.
        seed (int): Base seed.
        rebuild_every (int): Episodes per tool per env.
        monitor_dir (pathlib.Path): Where ``Monitor`` writes ``<rank>.monitor.csv``.

    Returns:
        SubprocVecEnv: The batched env. Deliberately **not** wrapped in VecNormalize.
    """
    return make_vec_env(
        lambda: ReachEnv(rebuild_every=rebuild_every),
        n_envs=n_envs,
        seed=seed,
        monitor_dir=str(monitor_dir),
        monitor_kwargs=dict(info_keywords=MONITOR_KEYWORDS),
        vec_env_cls=SubprocVecEnv,
    )


def build_model(vec_env, n_steps, seed, run_dir):
    """PPO with the feature layer as its extractor and ``PPOConfig``'s hyperparameters."""
    return PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=networks.policy_kwargs(),
        n_steps=n_steps,
        batch_size=PPOConfig.BATCH_SIZE,
        n_epochs=PPOConfig.N_EPOCHS,
        learning_rate=PPOConfig.LEARNING_RATE,
        gamma=PPOConfig.GAMMA,
        gae_lambda=PPOConfig.GAE_LAMBDA,
        clip_range=PPOConfig.CLIP_RANGE,
        ent_coef=PPOConfig.ENT_COEF,
        seed=seed,
        device=PPOConfig.DEVICE,
        tensorboard_log=str(run_dir / "tb"),
        verbose=1,
    )


def main(argv=None):
    args = parse_args(argv)
    assert not torch.cuda.is_available(), "CUDA is visible; this run is meant to stay on CPU"
    torch.set_num_threads(args.torch_threads)

    run_dir = args.run_dir or pathlib.Path(PPOConfig.RUNS_DIR) / (
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_reach"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump({
            **{k: v for k, v in vars(PPOConfig).items() if k.isupper()},
            **{f"arg_{k}": str(v) for k, v in vars(args).items()},
            "reward_scale": 1.0,
            "vec_normalize": False,
        }, f, indent=2)

    cores = os.cpu_count()
    print(f"run dir      {run_dir}")
    print(f"device       {PPOConfig.DEVICE} (cuda visible: {torch.cuda.is_available()})")
    print(f"core budget  {args.n_envs} env workers + {args.torch_threads} learner threads "
          f"of {cores} cores")

    vec_env = make_vec(args.n_envs, args.seed, args.rebuild_every, run_dir / "monitor")
    try:
        model = build_model(vec_env, args.n_steps, args.seed, run_dir)
        checkpoint = CheckpointCallback(
            save_freq=max(1, PPOConfig.CHECKPOINT_EVERY // args.n_envs),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo",
        )
        model.learn(total_timesteps=args.total_timesteps, callback=checkpoint)
        model.save(str(run_dir / "final"))
        print(f"saved        {run_dir / 'final.zip'}")
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
