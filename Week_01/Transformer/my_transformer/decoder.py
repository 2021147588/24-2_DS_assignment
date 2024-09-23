import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .feedforward import FeedForwardLayer, DropoutLayer
from .normalization import LayerNormalization
from .residual import ResidualConnection


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.enc_dec_attention = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForwardLayer(d_model, d_ff)

        self.norm1 = LayerNormalization(d_model)
        self.residual1 = ResidualConnection()

        self.norm2 = LayerNormalization(d_model)
        self.residual2 = ResidualConnection()

        self.norm3 = LayerNormalization(d_model)
        self.residual3 = ResidualConnection()

        self.dropout1 = DropoutLayer(dropout)
        self.dropout2 = DropoutLayer(dropout)
        self.dropout3 = DropoutLayer(dropout)

        

    def forward(self, 
                x: torch.Tensor, 
                memory: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
         # 마스크드 셀프 어텐션
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(self.residual1(x, self.dropout1(self_attn_output)))
        
        # 인코더-디코더 어텐션
        enc_dec_attn_output = self.enc_dec_attention(x, memory, memory, src_mask)
        x = self.norm2(self.residual2(x, self.dropout2(enc_dec_attn_output)))
        
        # 포지션 와이즈 피드포워드
        ff_output = self.ff(x)
        x = self.norm3(self.residual3(x, self.dropout3(ff_output)))
        return x