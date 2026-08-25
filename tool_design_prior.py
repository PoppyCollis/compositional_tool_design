import torch
from config import DEVICE, DesignPriorConfig


class ToolPrior:

    def __init__(self, device=DEVICE):

        self.device = device
        self.lower_bounds = torch.tensor(
            [DesignPriorConfig.L_MIN, DesignPriorConfig.L_MIN, -DesignPriorConfig.PHI_MAX],
            device=device,
        )
        self.upper_bounds = torch.tensor(
            [DesignPriorConfig.L_MAX, DesignPriorConfig.L_MAX, DesignPriorConfig.PHI_MAX],
            device=device,
        )

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
