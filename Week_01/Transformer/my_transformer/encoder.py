import torch.nn as nn
from torch import Tensor
from typing import Optional
from .attention import MultiHeadAttention
from .feedforward import FeedForwardLayer, DropoutLayer
from .normalization import LayerNormalization
from .residual import ResidualConnection

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForwardLayer(d_model, d_ff)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.dropout1 = DropoutLayer(dropout)
        self.dropout2 = DropoutLayer(dropout)
        self.residual1 = ResidualConnection()
        self.residual2 = ResidualConnection()
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        mask = None

        # 1. Self-Attention 레이어 적용
        attn_output = self.self_attn(x,x,x, mask)
        
        # 2. 잔차 연결 및 레이어 정규화
        x = self.residual1(x, self.dropout1(attn_output))
        x = self.norm1(x)
        
        # 3. Feed Forward 레이어 적용
        ff_output = self.ff(x)
        
        # 4. 잔차 연결 및 레이어 정규화
        x = self.residual2(x, self.dropout2(ff_output))
        x = self.norm2(x)
        
        return x
