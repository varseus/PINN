
import torch
import torch.nn as nn

class ComplexMSELoss(nn.MSELoss):
    """Mean square error of magnitude squared of two complex numbers
    """
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.MSELoss()(torch.abs(input)**2, torch.abs(target)**2)
    

def upper_bound_loss(predictions, targets, upper_bound=0.2):
    """Outler resiliant loss function, using SmoothL1Loss and clamp.

    Args:
        predictions: prediction
        targets: target
        upper_bound: maximum loss (losses above this are clamp)
    """
    return torch.clamp(torch.nn.SmoothL1Loss(beta=0.1)(predictions, targets), max=upper_bound)