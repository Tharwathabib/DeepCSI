"""
The DeepMIMO angular-delay stage transforms in chunks to keep peak memory down.
Chunking must not change any reported number, so these compare it against the
single-shot computation it replaced.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from data.prepare_deepmimo import to_angular_delay, N_DELAYS
from utils.transforms import spatial_frequency_to_angular_delay, truncate_delay, compute_energy_retention


def channels(n=300, seed=0):
    """Delay-sparse (n, 32, 256) channels, like ray-traced users."""
    rng = np.random.default_rng(seed)
    H = np.zeros((n, 32, 256), dtype=np.complex64)
    for i in range(n):
        for _ in range(rng.integers(3, 9)):
            tau = int(rng.integers(0, 24))
            steer = np.exp(1j * np.arange(32) * rng.uniform(-1.5, 1.5))
            delay = np.exp(-2j * np.pi * tau * np.arange(256) / 256)
            H[i] += (rng.normal() + 1j * rng.normal()) * np.outer(steer, delay)
    # Real path loss: DeepMIMO users span tens of dB, which is what broke the
    # epsilon-guarded retention helper.
    return (H * (10.0 ** (rng.uniform(-6, -3, size=(n, 1, 1))))).astype(np.complex64)


@pytest.mark.parametrize("chunk", [50, 128, 1000])
def test_chunking_does_not_change_the_result(chunk):
    H = channels()
    ad, report = to_angular_delay(H, chunk=chunk)

    expected = truncate_delay(spatial_frequency_to_angular_delay(H), N_DELAYS)
    assert ad.shape == expected.shape
    assert np.abs(ad - expected).max() < 1e-5, "chunked transform differs from single-shot"

    # A chunk larger than the array is the single-shot case; all must agree.
    ref = to_angular_delay(H, chunk=10_000)[1]["energy_retention_percent"]
    assert report["energy_retention_percent"] == pytest.approx(ref, abs=1e-3)


def test_aggregate_retention_matches_the_helper():
    """The accumulated ratio must equal the whole-array computation."""
    H = channels(200, seed=1)
    _, report = to_angular_delay(H, chunk=64)
    full = spatial_frequency_to_angular_delay(H)
    assert report["energy_retention_percent"] == pytest.approx(
        compute_energy_retention(full, max_delay=N_DELAYS), abs=1e-3
    )


def test_output_is_complex64_not_promoted():
    """
    numpy's FFT promotes to complex128. Storing that would double the memory the
    chunking exists to save, so the accumulator stays complex64.
    """
    ad, _ = to_angular_delay(channels(80, seed=2), chunk=32)
    assert ad.dtype == np.complex64
