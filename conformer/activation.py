import math
import numpy as np
import torch.nn as nn
from torch import Tensor


class ReLU(nn.Module):
    def __init__(self) -> None:
        super(ReLU).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return np.maximum(0, x)


class Swish(nn.Module):
    """
    Swish is smooth and a non-monotonic function.
    using Swish is better than ReLU.
    """
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x * x.sigmoid()


class GLU(nn.Module):
    """
    GLU: Gated Linear Units
    It helps gradient vanishing problem to relieve.
    """
    def __init__(self, dim) -> None:
        super(GLU, self).__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        x, gate = x.chunk(2, dim=self.dim)
        return x * gate.sigmoid()
