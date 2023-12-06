
import torch.nn as nn
from torch import Tensor
from encoder import ConformerEncoder
from typing import Tuple

class Conformer(nn.Module):
    def __init__(self, 
            num_class: int,
            input_dim: int = 80, 
            encoder_dim: int = 512,
            num_encoder_layer: int = 17,
            num_attention_head: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            conv_kernel_size: int = 31,
            dropout_p: float = 0.1, 
            half_step_residual: bool = True) -> None:
        super(Conformer, self).__init__()
        
        self.encoder = ConformerEncoder(
            input_dim = input_dim,
            encoder_dim = encoder_dim,
            num_layer = num_encoder_layer,
            num_attention_head = num_attention_head,
            feed_forward_expansion_factor = feed_forward_expansion_factor,
            conv_expansion_factor = conv_expansion_factor,
            dropout_p = dropout_p,
            conv_kernel_size = conv_kernel_size,
            half_step_residual= half_step_residual,
        )
        self.fc = nn.Linear(encoder_dim, num_class, bias=False)
        self.log_softmax = nn.Sequential(nn.LogSoftmax(dim=-1))

    def forward(self, x: Tensor, x_len: Tensor) -> Tuple[Tensor, Tensor]:
        encoder_output, encoder_output_len = self.encoder(x, x_len)
        output = self.fc(encoder_output)
        output = self.log_softmax(output)
        return output, encoder_output_len
