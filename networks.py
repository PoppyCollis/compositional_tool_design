"""The feature layer as a stable-baselines3 features extractor, and the policy kwargs.

::

    x -> feature_layer.features -> [pi MLP | V MLP]

Both networks take ``x`` and nothing else -- no ``tau`` input
(``ai_docs/networks_and_design_gradient.md`` §1) -- and both start with
``feature_layer.features``, wrapped here as SB3's ``features_extractor``. It has no
parameters, so sharing it between actor and critic (SB3's default) is free, and the
design gradient is plain autograd through ``h -> features -> V``.

**For the designer:** SB3's ``model.predict`` runs under ``no_grad`` and on numpy. To
differentiate ``V`` with respect to a state, call the torch modules directly::

    x = initial_state.h(tau, task, xi).float()        # tau.requires_grad
    v = model.policy.predict_values(x.unsqueeze(0))   # features -> value MLP, grad on

``predict_values`` applies this extractor and then ``value_net``, with gradients
enabled unless the caller turns them off.
"""
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import feature_layer
from config import PPOConfig


class ToolFeatureExtractor(BaseFeaturesExtractor):
    """``feature_layer.features`` as an SB3 features extractor: ``(B, 24) -> (B, 29)``.

    Parameter-free, so it contributes nothing to the optimiser and nothing to keep in
    sync between the two heads. It is the one copy of the feature arithmetic the RL
    stack and the designer both read (``h_initial_state_map.md``, "The feature layer
    is part of this contract").

    Args:
        observation_space (gymnasium.spaces.Box): The env's ``(OBS_DIM,)`` box.
    """

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=feature_layer.FEATURE_DIM)

    def forward(self, observations):
        """Append the tool's hand-frame dimensions to a batch of observations.

        Args:
            observations (torch.Tensor): Shape ``(B, OBS_DIM)``.

        Returns:
            torch.Tensor: Shape ``(B, FEATURE_DIM)``.
        """
        return feature_layer.features(observations)


def policy_kwargs(net_arch=PPOConfig.NET_ARCH):
    """``policy_kwargs`` for ``PPO("MlpPolicy", ...)``: this extractor, two separate MLPs.

    Args:
        net_arch (tuple): Hidden widths, used for both the policy and value heads.

    Returns:
        dict: Keyword arguments for SB3's ``ActorCriticPolicy``.
    """
    return dict(
        features_extractor_class=ToolFeatureExtractor,
        net_arch=dict(pi=list(net_arch), vf=list(net_arch)),
    )
