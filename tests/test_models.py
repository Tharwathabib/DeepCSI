import pytest
import torch
import sys
from pathlib import Path

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from models.csi_autoencoder import CSIAutoencoder, CSIEncoder, CSIDecoder


def test_latent_dimensions():
    model_cr4 = CSIAutoencoder(compression_ratio=4)
    model_cr16 = CSIAutoencoder(compression_ratio=16)
    model_cr32 = CSIAutoencoder(compression_ratio=32)

    assert model_cr4.latent_dim == 512
    assert model_cr16.latent_dim == 128
    assert model_cr32.latent_dim == 64


def test_encoder_decoder_forward_shape():
    x = torch.randn(8, 2, 32, 32)

    for cr, expected_latent in [(4, 512), (16, 128), (32, 64)]:
        model = CSIAutoencoder(compression_ratio=cr)
        recon, latent = model(x)

        assert latent.shape == (8, expected_latent)
        assert recon.shape == (8, 2, 32, 32)
        assert recon.min() >= 0.0 and recon.max() <= 1.0  # Sigmoid output range
