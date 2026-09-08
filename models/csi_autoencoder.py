import torch
import torch.nn as nn
from typing import Tuple


class ResidualBlock(nn.Module):
    """
    Residual Block for deep CSI decoder reconstruction.
    """
    def __init__(self, channels: int = 2, hidden_dim: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act2(out + residual)
        return out


class CSIEncoder(nn.Module):
    """
    CNN Encoder for CSI angular-delay representation.
    Compresses (B, 2, 32, 32) tensor (2048 scalars) down to a compact latent vector of size (B, latent_dim).
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.features = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten()
        )
        self.fc_latent = nn.Linear(2048, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        latent = self.fc_latent(feat)
        return latent


class CSIDecoder(nn.Module):
    """
    Powerful Residual Decoder for CSI reconstruction at gNodeB.
    Reconstructs latent vector (B, latent_dim) back to (B, 2, 32, 32) CSI tensor.
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc_expand = nn.Linear(latent_dim, 2048)
        self.res1 = ResidualBlock(channels=2, hidden_dim=8)
        self.res2 = ResidualBlock(channels=2, hidden_dim=8)
        self.out_conv = nn.Conv2d(2, 2, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.fc_expand(latent)
        x = x.view(-1, 2, 32, 32)
        x = self.res1(x)
        x = self.res2(x)
        x = self.out_conv(x)
        out = self.sigmoid(x)
        return out


class CSIAutoencoder(nn.Module):
    """
    End-to-End DeepCSI Autoencoder.
    Supports Compression Ratios: CR=4 (latent_dim=512), CR=16 (latent_dim=128), CR=32 (latent_dim=64).
    """
    def __init__(self, compression_ratio: int = 16):
        super().__init__()
        self.compression_ratio = compression_ratio
        
        # Calculate scalar budget: 2 * 32 * 32 = 2048 total scalars
        total_scalars = 2048
        if total_scalars % compression_ratio != 0:
            raise ValueError(f"Total scalars (2048) must be divisible by compression_ratio ({compression_ratio}).")
        
        self.latent_dim = total_scalars // compression_ratio
        self.encoder = CSIEncoder(self.latent_dim)
        self.decoder = CSIDecoder(self.latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            reconstruction: Tensor of shape (B, 2, 32, 32)
            latent: Tensor of shape (B, latent_dim)
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent
