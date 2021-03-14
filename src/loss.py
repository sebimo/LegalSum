# The loss is very important for summarization. Especially for extractive summarization, as here we have a really small number of positive samples
# in comparison to the negative samples.
from typing import List, Callable

import torch
import torch.nn.functional as F

# HammingLoss and SubsetLoss are common for multilabel classification, as they emphasis positive examples


class CombinedLoss:

    def __init__(self, losses: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]], weights: List[float]):
        assert len(losses) == len(weights)
        self.losses = losses
        self.weights = weights

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = self.weights[0] * self.losses[0](x,y)
        for l, w in zip(self.losses[1:], self.weights[1:]):
            loss += w * l(x,y)
        return loss

def HammingLossHinge(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Because we cannot directly optimize the HammingLoss, we need to optimize a surrogate loss function
    l = hinge(torch.mul(x,y))
    return torch.sum(torch.mean(l, dim=1))

def HammingLossLogistic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Because we cannot directly optimize the HammingLoss, we need to optimize a surrogate loss function
    l = logistic(torch.mul(x,y))
    return torch.sum(torch.mean(l, dim=1))

def SubsetLossHinge(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    l = hinge(torch.mul(x,y))
    values, _ = torch.max(l, dim=1)
    return torch.sum(values)

def SubsetLossLogistic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    l = logistic(torch.mul(x,y))
    values, _ = torch.max(l, dim=1)
    return torch.sum(values)

def hinge(e: torch.Tensor) -> torch.Tensor:
    return F.relu(1 - e)

def logistic(e: torch.Tensor) -> torch.Tensor:
    return torch.log(1 + torch.exp(-e))