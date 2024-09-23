import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class EmbeddingLayer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, projection_dim):
        super(EmbeddingLayer, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        
        self.patch_embedding = nn.Conv2d(
            in_channels, projection_dim, kernel_size=patch_size, stride=patch_size
        )
        
        self.position_embedding = nn.Embedding(num_embeddings=self.num_patches, embedding_dim=projection_dim)
        
        self.register_buffer("positions", torch.arange(start=0, end=self.num_patches, step=1).long())

    def forward(self, x):
        # 패치 생성
        # 입력 x의 형식: (batch_size, in_channels, img_size, img_size)
        x = x.permute(0, 3, 1, 2)  # (batch_size, channels, image_size, image_size)
        

        patches = self.patch_embedding(x)  # (batch_size, projection_dim, num_patches**0.5, num_patches**0.5)
        
        patches = patches.flatten(2).transpose(1, 2)  # (batch_size, num_patches, projection_dim)
        
        device = x.device
        positions = self.positions.to(device)
        encoded_positions = self.position_embedding(positions)
        

        encoded_patches = patches + encoded_positions
        return encoded_patches

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,projection_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        
        self.attention = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=num_heads, dropout=dropout)

    def forward(self, query, key, value):

        attn_output, _ = self.attention(query, key, value)

        return attn_output
    
class MLP(nn.Module):
    def __init__(self,in_features, hidden_units, dropout_rate):
        super(MLP, self).__init__()
        layers = []
        input_dim = in_features #?
        for units in hidden_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.GELU())  # GELU 활성화 함수 사용
            layers.append(nn.Dropout(dropout_rate))
            input_dim = units

        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)
    
class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, layer_norm_eps=1e-6):
        super(Block, self).__init__()
        projection_dim = 64
        transformer_units = [projection_dim*2,projection_dim]
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        
        self.mlp = MLP(in_features=embed_dim ,hidden_units=transformer_units, dropout_rate=dropout)

    def forward(self, encoded_patches):
        x1 = self.norm1(encoded_patches)
        x2 = encoded_patches + self.attn(x1,x1,x1)
        x = x2 + self.mlp(self.norm2(x2))
        return x 

#여기까지 Encoder 구현 끝!!


class VisionTransformer(nn.Module):
    def __init__(self, img_size=65, patch_size=8, in_channels=1, num_classes=3, embed_dim=64, depth=8, num_heads=4, dropout=0.1, layer_norm_eps=1e-6):
        super(VisionTransformer, self).__init__()
        mlp_head_units = [2048, 1024]
        self.embedding = EmbeddingLayer(img_size, patch_size, in_channels, embed_dim)
        self.register_buffer("positions", torch.arange(0, embed_dim).long())

        self.encoder = nn.ModuleList([
            Block(embed_dim, num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout =  nn.Dropout(0.5)
        self.mlp = MLP(embed_dim, mlp_head_units, 0.5)
        self.classifier = nn.Linear(mlp_head_units[-1], num_classes)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.embedding(x).to(x.device)  # x출력 = (batch_size, num_patches, num_patches, embed_dim)
        for layer in self.encoder:
            x = layer(x)  # Transformer Block 통과
       
        x = self.norm(x)
        x = x.mean(dim=1)  # (batch_size, embed_dim): 패치 평균
        x = self.dropout(x)
        x = self.mlp(x)
        logits = self.classifier(x)

        return logits

    
