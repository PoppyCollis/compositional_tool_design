"""The design prior ``p(tau)``: uniform on a box, in torch for the designer and in
numpy for the env.

One set of bounds, ``LOWER`` / ``UPPER``, read by both samplers. ``ToolPrior.sample``
draws with torch's generator, which a gymnasium ``reset(seed=...)`` cannot reach, so
the env draws its per-episode tool with ``sample_tau`` from its own ``np_random``
instead -- that is what lets one seed reproduce tool, task and start pose together.
"""
import numpy as np
import torch

from config import DEVICE, DesignPriorConfig

# The prior box, (l1, l2, phi). phi is an interval, not a circle: the wedge around
# +-pi is excluded by PHI_MAX, so there is no wrap-around anywhere.
LOWER = (DesignPriorConfig.L_MIN, DesignPriorConfig.L_MIN, -DesignPriorConfig.PHI_MAX)
UPPER = (DesignPriorConfig.L_MAX, DesignPriorConfig.L_MAX, DesignPriorConfig.PHI_MAX)


def sample_tau(np_random):
    """Draw one ``tau ~ p(tau)`` with a numpy generator.

    The env's reset draws its tool here rather than from ``ToolPrior.sample``: that
    uses ``torch.rand``, whose stream ``reset(seed=...)`` does not touch, so two resets
    with the same seed would agree on task and start pose but not on the tool.

    Args:
        np_random (np.random.Generator): Source of randomness, the env's own.

    Returns:
        tuple: ``(l1, l2, phi)`` as Python floats.
    """
    return tuple(float(v) for v in np_random.uniform(LOWER, UPPER))


class ToolPrior:

    def __init__(self, device=DEVICE):

        self.device = device
        self.lower_bounds = torch.tensor(LOWER, device=device)
        self.upper_bounds = torch.tensor(UPPER, device=device)

    def sample(self, n):
        """Sample from uniform prior: l1,l2 and phi all from a uniform box.

        phi (radians) is an interval, not a circle: the wedge around +-pi is
        excluded by PHI_MAX, so there is no wrap-around to handle."""
        
        u = torch.rand(n, self.lower_bounds.shape[0], device=self.device) # batch size n
        tau = self.lower_bounds + (self.upper_bounds - self.lower_bounds) * u
        tau.requires_grad_(True)
    
        return tau
    
    @staticmethod
    def _reflect(x, lo, hi):
        """Fold x into [lo, hi] via triangle-wave reflection (measure-preserving)."""
        span = hi - lo
        x = (x - lo) % (2 * span)
        x = torch.where(x > span, 2 * span - x, x)
        return x + lo

    def constrain(self, tau):
        """Enforce measure-preserving geometric constraints to tau: reflect all
        three coordinates. phi reflects rather than wraps because it lives on an
        interval, not a circle — see sample()."""
        l1, l2, phi = tau[..., 0], tau[..., 1], tau[..., 2]

        l1 = self._reflect(l1, self.lower_bounds[0], self.upper_bounds[0])
        l2 = self._reflect(l2, self.lower_bounds[1], self.upper_bounds[1])
        phi = self._reflect(phi, self.lower_bounds[2], self.upper_bounds[2])

        return torch.stack([l1, l2, phi], dim=-1)
