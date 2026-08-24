import torch
from config import DesignPriorConfig


class ToolPrior:
    
    def __init__(self,):
        
        self.lower_bounds = torch.Tensor([DesignPriorConfig.L_MIN, DesignPriorConfig.L_MIN, 0.0])
        self.upper_bounds = torch.Tensor([DesignPriorConfig.L_MAX, DesignPriorConfig.L_MAX, 2* torch.pi])
        

    def sample(self, n):
        """Sample from uniform prior: l1,l2 from uniform box and theta from unform circle (in radians)"""
        
        u = torch.rand(n, self.lower_bounds.shape[0]) # batch size n
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
        """Enforce measure-preserving geometric constraints to tau: reflect l1,l2; wrap θ"""
        l1, l2, theta = tau[..., 0], tau[..., 1], tau[..., 2]

        l1 = self._reflect(l1, self.lower_bounds[0], self.upper_bounds[0])
        l2 = self._reflect(l2, self.lower_bounds[1], self.upper_bounds[1])
        theta = torch.remainder(theta, 2 * torch.pi)

        return torch.stack([l1, l2, theta], dim=-1)
