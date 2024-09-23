import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional, Tuple

class QueryLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(QueryLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class KeyLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(KeyLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ValueLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(ValueLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ScaledDotProductAttention(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        d_tensor = k.size(3)

        k_t = k.transpose(2, 3)  # Key 텐서의 두 번째 차원(시퀀스 길이)과 세 번째 차원(임베딩 차원)을 서로 교환
        score = (q @ k_t) / math.sqrt(d_tensor)  
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)
        score = F.softmax(score,dim=-1)
        v = score @ v
        return v, score


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        
        self.query_layers = QueryLayer(d_model, n_heads)
        self.key_layers = KeyLayer(d_model, n_heads)
        self.value_layers = ValueLayer(d_model, n_heads)
        self.attention = ScaledDotProductAttention()
        self.fc = nn.Linear(n_heads * d_model, d_model)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        batch_size = Q.size(0)

        # 1. 선형 변환 및 차원 분할
        Q = self.query_layers(Q).view(batch_size, -1, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        K = self.key_layers(K).view(batch_size, -1, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        V = self.value_layers(V).view(batch_size, -1, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)

        # 2. Scaled Dot-Product Attention 적용
        if mask is not None:
            mask = mask.unsqueeze(1)  # 마스크 차원 확장
        
        output, attn_weights = self.attention(Q, K, V, mask)

        # 3. 여러 헤드의 결과를 결합
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_model)

        # 4. 최종 선형 레이어 적용
        output = self.fc(output)

        return output
