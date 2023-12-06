
import torch.nn as nn
from torch import Tensor
from activation import Swish
from modules import Linear

count = 0

class FeedForward(nn.Module):
    """
    Args:
        d_model (int): Dimension of conformer encoder
        expansion_factor (int): Expansion factor of feed forward. you can multiply it by d_model.
        dropout_p (float): Ratio of dropout_p
    """

    def __init__(self, encoder_dim: int = 512, expansion_factor: int = 4, dropout_p: float = 0.1) -> None:
        super(FeedForward, self).__init__()
        self.encoder_dim = encoder_dim
        self.dropout_p = dropout_p
        self.expand_dim = encoder_dim * expansion_factor

        self.sequential = nn.Sequential(
            nn.LayerNorm(self.encoder_dim),
            Linear(self.encoder_dim, self.expand_dim, bias=True),
            Swish(),
            nn.Dropout(p=self.dropout_p),
            Linear(self.expand_dim, self.encoder_dim, bias=True),
            nn.Dropout(p=self.dropout_p),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.sequential(x)
