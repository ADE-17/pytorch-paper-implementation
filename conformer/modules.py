
import torch.nn as nn
from torch import Tensor

class ResidualConnection(nn.Module):
    def __init__(self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0):
        super(ResidualConnection, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor 
    
    
    def forward(self, x: Tensor) -> Tensor:
        return (self.module(x) * self.module_factor) + (x * self.input_factor)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)
