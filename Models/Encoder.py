import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleEncoder1D(nn.Module):
    def __init__(self, in_channels, latent_dim=64, out_dim=64, kernel_size=5, padding=2):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, latent_dim, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # → (B, 32, 1)
        )
        self.proj = nn.Linear(latent_dim, out_dim)

    def forward(self, x):  # x: (B, C, T)
        x = self.net(x).squeeze(-1)  # (B, 32)
        x = self.proj(x)             # (B, out_dim)
        return x

class MLPEncoder(nn.Module):
    def __init__(self, in_channels, out_dim=64):
        super().__init__()
        self.out_dim = out_dim

        self.net = nn.Sequential(
            nn.Flatten(),  # (B, C*T)
            nn.Linear(in_channels * 100, 128),  # assuming fixed T=100 for simplicity
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):  # x: (B, C, T)
        x = self.net(x)    # (B, out_dim)
        return x

class SimpleLSTMEncoder1D(nn.Module):
    def __init__(self, in_channels, latent_dim=64, out_dim=64):
        super().__init__()
        # x: (B, C, T) → LSTM 입력: (B, T, C)
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=latent_dim,
            batch_first=True,
            bidirectional=False,
        )
        self.proj = nn.Linear(latent_dim, out_dim)

    def forward(self, x):          # x: (B, C, T)
        x = x.transpose(1, 2)      # (B, T, C)
        out, (h_n, c_n) = self.lstm(x)
        x = h_n[-1]                # 마지막 layer의 마지막 hidden state: (B, latent_dim)
        x = self.proj(x)           # (B, out_dim)
        return x

class SimpleLSTMAttnEncoder1D(nn.Module):
    """
    입력:  x (B, C, T)
    출력:  (B, out_dim)
    방식:  LSTM으로 time-wise 특징 추출 → (학습되는) attention pooling으로 T축 가중합 → Linear proj
    """
    def __init__(self, in_channels, latent_dim=64, out_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=latent_dim,
            batch_first=True,
            bidirectional=False,
        )
        # additive attention (self-attentive pooling): score_t = v^T tanh(W h_t)
        self.attn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, 1),
        )
        self.proj = nn.Linear(latent_dim, out_dim)

    def forward(self, x):               # (B, C, T)
        x = x.transpose(1, 2)           # (B, T, C)  (LSTM batch_first 입력 형태) 
        h, _ = self.lstm(x)             # (B, T, latent_dim) [web:3]

        scores = self.attn(h)           # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # time dimension에 대해 softmax → (B, T, 1) 

        context = (weights * h).sum(dim=1)      # attention weighted sum → (B, latent_dim) 
        out = self.proj(context)        # (B, out_dim)
        return out

import torch
import torch.nn as nn
import math

class SimpleTransformerEncoder1D(nn.Module):
    def __init__(self, in_channels, latent_dim=64, out_dim=64, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(in_channels, latent_dim)  # (B, T, C) → (B, T, D)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim*4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Transformer 출력 → (B, D, 1) → (B, D)
        self.proj = nn.Linear(latent_dim, out_dim)

    def forward(self, x: torch.Tensor):  # x: (B, C, T)
        x = x.transpose(1, 2)      # (B, T, C)
        x = self.embed(x)          # (B, T, D)
        x = self.transformer(x)    # (B, T, D) self-attention 적용
        x = x.transpose(1, 2)      # (B, D, T) pool 위해
        x = self.pool(x).squeeze(-1)  # (B, D)
        x = self.proj(x)           # (B, out_dim)
        return x
