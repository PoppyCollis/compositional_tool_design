"""The SB3 glue: the feature layer as an extractor, and PPO running on ReachEnv.

The smoke test is the one that matters. It proves the extractor, the terminated-at-
horizon convention and the Monitor keywords all plug into stable-baselines3 without
error -- and that SB3 never sees a truncation to bootstrap from.
"""
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import feature_layer
import initial_state
import networks
from config import TaskConfig
from reach_env import ReachEnv


def test_extractor_is_the_feature_layer():
    """Same arithmetic, same output: one copy of the feature layer, read by both sides."""
    env = ReachEnv((0.15, 0.15, 0.5), rebuild_every=None)
    try:
        extractor = networks.ToolFeatureExtractor(env.observation_space)
        assert extractor.features_dim == feature_layer.FEATURE_DIM
        assert sum(p.numel() for p in extractor.parameters()) == 0
        obs = torch.as_tensor(np.stack([env.reset(seed=i)[0] for i in range(4)]))
        out = extractor(obs)
        assert out.shape == (4, feature_layer.FEATURE_DIM)
        torch.testing.assert_close(out, feature_layer.features(obs))
    finally:
        env.close()


def test_ppo_smoke_on_reach_env():
    """A couple of PPO updates on two envs, through Monitor, with the real extractor.

    Also pins the truncation convention from the SB3 side: at the horizon the vec env
    must report a done with ``TimeLimit.truncated`` False, so PPO's rollout collector
    takes the final reward as is and bootstraps nothing.
    """
    vec_env = DummyVecEnv([
        lambda: Monitor(ReachEnv(rebuild_every=1), info_keywords=("is_success", "tau", "target"))
        for _ in range(2)
    ])
    try:
        vec_env.seed(0)
        model = PPO(
            "MlpPolicy", vec_env, policy_kwargs=networks.policy_kwargs(),
            n_steps=32, batch_size=32, n_epochs=1, seed=0, device="cpu", verbose=0,
        )
        model.learn(total_timesteps=128)

        # The value head is reachable with gradients on, which is what the designer
        # needs (networks.py docstring).
        x = torch.as_tensor(vec_env.reset(), dtype=torch.float32).requires_grad_(True)
        value = model.policy.predict_values(x)
        assert value.shape == (2, 1)
        value.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

        # Drive one env to the horizon and read what SB3 reports.
        vec_env.reset()
        for _ in range(TaskConfig.HORIZON):
            _, _, dones, infos = vec_env.step(np.zeros((2, 3), dtype=np.float32))
        assert dones.all()
        for info in infos:
            assert info["TimeLimit.truncated"] is False
            assert "terminal_observation" in info
            assert info["terminal_observation"][initial_state.PHASE] == pytest.approx(1.0)
            assert "episode" in info and set(("is_success", "tau", "target")) <= set(info["episode"])
    finally:
        vec_env.close()
