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