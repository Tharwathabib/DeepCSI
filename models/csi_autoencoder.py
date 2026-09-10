import torch
import torch.nn as nn
from typing import Tuple


class ResidualBlock(nn.Module):
    """
    RefineNet-style residual block for deep CSI decoder reconstruction.

    The default widths (8, 16) match the RefineNet block of CsiNet
    (Wen, Shih & Jin, IEEE WCL 2018), which the DeepCSI decoder is modelled on:

        Conv2d(2, 8, 3)  -> BN -> LeakyReLU
        Conv2d(8, 16, 3) -> BN -> LeakyReLU
        Conv2d(16, 2, 3) -> BN
        add(shortcut)    -> LeakyReLU

    Note the final conv is followed by BatchNorm only; the activation comes after
    the residual addition. Pass hidden_dims=(8,) to recover the narrower 2->8->2
    block used earlier in this project, for ablation purposes.
    """
    def __init__(self, channels: int = 2, hidden_dims: Tuple[int, ...] = (8, 16)):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one width.")
        self.channels = channels
        self.hidden_dims = tuple(hidden_dims)

        layers = []
        in_ch = channels
        for width in self.hidden_dims:
            layers.append(nn.Conv2d(in_ch, width, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(width))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = width
        # Project back to the input channel count. BatchNorm but no activation:
        # the LeakyReLU is applied after the residual addition.
        layers.append(nn.Conv2d(in_ch, channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(channels))

        self.body = nn.Sequential(*layers)
        self.act_out = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act_out(self.body(x) + x)


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
    def __init__(self, latent_dim: int, refine_widths: Tuple[int, ...] = (8, 16)):
        super().__init__()
        self.latent_dim = latent_dim
        self.refine_widths = tuple(refine_widths)
        self.fc_expand = nn.Linear(latent_dim, 2048)
        self.res1 = ResidualBlock(channels=2, hidden_dims=self.refine_widths)
        self.res2 = ResidualBlock(channels=2, hidden_dims=self.refine_widths)
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
    def __init__(self, compression_ratio: int = 16, refine_widths: Tuple[int, ...] = (8, 16)):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.refine_widths = tuple(refine_widths)

        # Calculate scalar budget: 2 * 32 * 32 = 2048 total scalars
        total_scalars = 2048
        if total_scalars % compression_ratio != 0:
            raise ValueError(f"Total scalars (2048) must be divisible by compression_ratio ({compression_ratio}).")

        self.latent_dim = total_scalars // compression_ratio
        self.encoder = CSIEncoder(self.latent_dim)
        self.decoder = CSIDecoder(self.latent_dim, refine_widths=self.refine_widths)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            reconstruction: Tensor of shape (B, 2, 32, 32)
            latent: Tensor of shape (B, latent_dim)
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent
