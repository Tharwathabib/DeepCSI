"""
Uniform scalar quantisation of the feedback latent.

Without this the project's compression claim counts SCALARS, not BITS. A latent
of 512 float32 values is 16384 bits, not "75% smaller" in any sense a radio
engineer would accept -- the uplink control channel carries bits. Quantising the
latent is what turns a dimension-reduction claim into a bandwidth claim.

The quantiser range is fitted once on the training set and shipped with the
model, exactly like the decoder weights. It is NOT derived per sample: doing
that would require sending the range alongside each payload, and those side-info
bits would have to be counted. A fixed range means the payload is exactly
latent_dim * bits and nothing else.
"""

import numpy as np


def fit_range(latents: np.ndarray, quantile: float = 0.999) -> tuple:
    """
    Symmetric quantiser range covering `quantile` of the observed magnitudes.

    A hard min/max would let a single outlier stretch the range and waste most
    of the levels on values that never occur; clipping the top 0.1% costs far
    less than the resolution it buys back.
    """
    s = float(np.quantile(np.abs(latents), quantile))
    return -s, s


def uniform_quantize(z: np.ndarray, bits: int, lo: float, hi: float) -> np.ndarray:
    """Round to one of 2**bits levels spanning [lo, hi], returning dequantised values."""
    levels = 2 ** bits - 1
    step = (hi - lo) / levels
    idx = np.clip(np.round((z - lo) / step), 0, levels)
    return (idx * step + lo).astype(np.float32)


def payload_bits(latent_dim: int, bits: int) -> int:
    """Feedback payload for one CSI report. No side info: the range is fixed offline."""
    return latent_dim * bits


# A full-resolution CSI report: 2 x 32 x 32 real scalars at float32.
RAW_CSI_BITS = 2 * 32 * 32 * 32
