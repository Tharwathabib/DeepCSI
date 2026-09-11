import sys
from pathlib import Path

import numpy as np
import pytest

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.transforms import (
    spatial_frequency_to_angular_delay,
    angular_delay_to_spatial_frequency,
    truncate_delay,
    split_complex,
    combine_complex,
    compute_energy_retention,
)

NT, NC = 32, 256


def sparse_channel(n_paths=6, max_tau=20, seed=0):
    """A few specular paths with short delays -- the sparsity the pipeline assumes."""
    rng = np.random.default_rng(seed)
    H = np.zeros((NT, NC), dtype=np.complex128)
    for _ in range(n_paths):
        tau = int(rng.integers(0, max_tau))
        aoa = rng.uniform(0, 2 * np.pi)
        gain = (rng.normal() + 1j * rng.normal()) * np.exp(-0.1 * tau)
        steer = np.exp(1j * np.arange(NT) * aoa)
        delay = np.exp(-2j * np.pi * tau * np.arange(NC) / NC)
        H += gain * np.outer(steer, delay)
    return H


def test_round_trip_is_exact():
    rng = np.random.default_rng(0)
    H = rng.normal(size=(NT, NC)) + 1j * rng.normal(size=(NT, NC))
    back = angular_delay_to_spatial_frequency(spatial_frequency_to_angular_delay(H))
    assert np.abs(H - back).max() < 1e-9


def test_delay_axis_uses_inverse_dft():
    """
    A path of delay tau must land at index tau on the delay axis.

    This is the regression guard for a real bug: the transform originally used
    np.fft.fft2, and a forward DFT places delay tau at index Nc-tau. Truncating
    to the first 32 taps then discarded the channel and kept noise, while every
    other test still passed because nothing exercised these functions.
    """
    for tau in (0, 1, 5, 17):
        steer = np.exp(1j * np.arange(NT) * 0.7)
        delay = np.exp(-2j * np.pi * tau * np.arange(NC) / NC)
        H_ad = spatial_frequency_to_angular_delay(np.outer(steer, delay))

        energy_per_tap = np.sum(np.abs(H_ad) ** 2, axis=0)
        peak = int(np.argmax(energy_per_tap))
        assert peak == tau, f"delay {tau} landed at index {peak}, expected {tau}"


def test_sparse_channel_energy_survives_truncation():
    """The 32-tap truncation is only valid if short-delay energy lands early."""
    retention = compute_energy_retention(
        spatial_frequency_to_angular_delay(sparse_channel()), max_delay=32
    )
    assert retention > 95.0, f"only {retention:.1f}% retained"


def test_retention_is_scale_invariant():
    """Retention is a ratio, so scaling the channel must not change it.

    DeepMIMO channels carry real path loss: a typical O1 user has total energy
    near 1e-10, and the weakest are near 1e-16. An absolute epsilon in the
    denominator therefore dominates the signal and drives retention to zero for
    the majority of real users -- which is what made the median RX2 user look
    like 34% retention instead of 99.4%. Synthetic O(1) data never exposes this.
    """
    H_ad = spatial_frequency_to_angular_delay(sparse_channel())
    reference = compute_energy_retention(H_ad, max_delay=32)

    for scale in (1e-5, 1e-8, 1e-11):
        scaled = compute_energy_retention(H_ad * scale, max_delay=32)
        assert abs(scaled - reference) < 1e-6, (
            f"scaling by {scale:g} moved retention {reference:.2f}% -> {scaled:.2f}%"
        )


def test_retention_of_an_all_zero_channel_is_zero():
    """Users with no ray-traced path reach this helper before they are dropped."""
    assert compute_energy_retention(np.zeros((NT, NC), dtype=complex), max_delay=32) == 0.0


def test_white_noise_retention_matches_tap_fraction():
    """Sanity check on the metric itself: white noise spreads energy uniformly."""
    rng = np.random.default_rng(1)
    H = rng.normal(size=(NT, NC)) + 1j * rng.normal(size=(NT, NC))
    retention = compute_energy_retention(spatial_frequency_to_angular_delay(H), max_delay=32)
    assert abs(retention - 100 * 32 / NC) < 4.0


def test_truncate_and_split_shapes():
    H_ad = spatial_frequency_to_angular_delay(sparse_channel())
    assert truncate_delay(H_ad, 32).shape == (NT, 32)
    assert truncate_delay(np.stack([H_ad, H_ad]), 32).shape == (2, NT, 32)

    x = split_complex(truncate_delay(H_ad, 32))
    assert x.shape == (2, NT, 32)
    assert np.abs(combine_complex(x) - truncate_delay(H_ad, 32)).max() < 1e-12

    xb = split_complex(np.stack([H_ad, H_ad]))
    assert xb.shape == (2, 2, NT, NC)


def test_batched_transform_matches_per_sample():
    """The transform operates on the last two axes, so batching must be a no-op."""
    batch = np.stack([sparse_channel(seed=s) for s in range(3)])
    batched = spatial_frequency_to_angular_delay(batch)
    for i in range(len(batch)):
        single = spatial_frequency_to_angular_delay(batch[i])
        assert np.abs(batched[i] - single).max() < 1e-9
