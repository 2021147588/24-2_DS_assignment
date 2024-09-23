import torch
import torch.nn as nn
import math
from torch import Tensor

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)

class PositionEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super(PositionEmbedding, self).__init__()

        # PE값을 저장할 텐서 생성
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  
        
        # [0, 1, 2, ..., max_len-1]의 값을 갖는 1차원 텐서를 생성
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        _2i = torch.arange(0, d_model, step=2).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model))) 


    def forward(self, x: Tensor) -> Tensor:
        return self.encoding[:x.size(1), :]