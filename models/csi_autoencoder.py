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

    Two feature extractors are available:

    csinet  Two stacked 3x3 convolutions, as in CsiNet.
    crnet   Parallel branches with 1x9, 9x1 and 3x3 kernels, concatenated.
            Angular-delay CSI is anisotropic -- a single propagation cluster
            appears as a streak extended along one axis and narrow along the
            other -- so isotropic 3x3 kernels are a poor match for the
            structure. The elongated kernels cover a cluster in one hop.
            This is a change of inductive bias, not of capacity, which is what
            the measured 2-4 dB train/val gap calls for: the model is
            overfitting, so adding parameters would make it worse.
    """
    def __init__(self, latent_dim: int, arch: str = "csinet"):
        super().__init__()
        self.latent_dim = latent_dim
        self.arch = arch

        if arch == "csinet":
            self.features = nn.Sequential(
                nn.Conv2d(2, 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(8, 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Flatten()
            )
        elif arch == "crnet":
            self.features = MultiResolutionBlock(in_channels=2, out_channels=2)
        else:
            raise ValueError(f"Unknown encoder arch: {arch!r}")

        # The feature stage deliberately returns to 2 channels before the fully
        # connected layer. Keeping more would multiply the FC weight count by
        # that factor -- at CR=4 the FC is already ~1M of the ~1.05M encoder
        # parameters, so widening here is the fastest way to overfit harder.
        self.fc_latent = nn.Linear(2048, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        latent = self.fc_latent(feat)
        return latent


class MultiResolutionBlock(nn.Module):
    """
    CRNet-style parallel feature extractor.

    Branch A: 3x3                      -- local, isotropic
    Branch B: 1x9 then 9x1             -- wide along delay, then along angle
    Concatenate, then project back to `out_channels`.

    The 1x9/9x1 factorisation gives a 9x9 receptive field for far fewer
    parameters than a dense 9x9 kernel.
    """
    def __init__(self, in_channels: int = 2, out_channels: int = 2, width: int = 8):
        super().__init__()

        def cbr(cin, cout, k, p):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, padding=p),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.branch_local = cbr(in_channels, width, 3, 1)
        self.branch_wide = nn.Sequential(
            cbr(in_channels, width, (1, 9), (0, 4)),
            cbr(width, width, (9, 1), (4, 0)),
        )
        self.fuse = cbr(2 * width, out_channels, 1, 0)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        merged = torch.cat([self.branch_local(x), self.branch_wide(x)], dim=1)
        return self.flatten(self.fuse(merged))


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
    def __init__(self, compression_ratio: int = 16, refine_widths: Tuple[int, ...] = (8, 16),
                 arch: str = "csinet"):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.refine_widths = tuple(refine_widths)
        self.arch = arch

        # Calculate scalar budget: 2 * 32 * 32 = 2048 total scalars
        total_scalars = 2048
        if total_scalars % compression_ratio != 0:
            raise ValueError(f"Total scalars (2048) must be divisible by compression_ratio ({compression_ratio}).")

        self.latent_dim = total_scalars // compression_ratio
        self.encoder = CSIEncoder(self.latent_dim, arch=arch)
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


def infer_refine_widths(state_dict: dict, default=(8, 16)) -> tuple:
    """
    Recover the decoder's RefineNet widths from the weights themselves.

    A checkpoint without a `refine_widths` key is not necessarily the current
    default: the block was widened from (8,) to (8, 16) on this branch, and
    anything trained before that carries the narrow shape. Assuming the default
    makes load_state_dict raise, and the API catches that exception and logs it,
    so the model simply vanishes from /health with no failure anyone notices.

    The widths are the output channels of every conv in a refine block except
    the last, which returns to the 2 real/imag channels:
        decoder.res1.body.0.weight  (8, 2, 3, 3)   -> 8
        decoder.res1.body.3.weight  (16, 8, 3, 3)  -> 16
        decoder.res1.body.6.weight  (2, 16, 3, 3)  -> output, not a width
    """
    convs = sorted(
        (int(k.split(".")[3]), v.shape[0])
        for k, v in state_dict.items()
        if k.startswith("decoder.res1.body.") and k.endswith(".weight") and v.dim() == 4
    )
    if len(convs) < 2:
        return tuple(default)
    return tuple(out for _, out in convs[:-1])


def build_from_checkpoint(checkpoint: dict, compression_ratio: int, device="cpu") -> "CSIAutoencoder":
    """
    Rebuild the exact architecture a checkpoint was trained with, then load it.

    Three separate call sites -- the API, the evaluator and preflight -- each
    read `refine_widths` from the checkpoint but ignored `arch`, so any model
    trained with `--arch crnet` (which the ablation produces) could not be
    loaded by any of them. load_state_dict raises on the shape mismatch, and in
    the backend that exception is caught and logged, which leaves the model
    quietly missing from /health rather than failing startup outright.

    Centralising the reconstruction is the fix: one place now knows how to turn
    a checkpoint back into a model, so the fields cannot drift apart again.
    """
    state = checkpoint["model_state_dict"]
    # Prefer the recorded width; fall back to reading it off the weights rather
    # than to a hardcoded default, which would be wrong for any checkpoint
    # predating the (8,) -> (8, 16) widening.
    widths = checkpoint.get("refine_widths") or infer_refine_widths(state)
    model = CSIAutoencoder(
        compression_ratio=compression_ratio,
        refine_widths=tuple(widths),
        arch=checkpoint.get("arch", "csinet"),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model
