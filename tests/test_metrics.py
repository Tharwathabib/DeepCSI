import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.metrics import (
    nmse_db,
    nmse_db_numpy,
    cosine_similarity_torch,
    cosine_similarity_numpy,
    beamforming_gain_torch,
    beamforming_gain_numpy,
    beamforming_loss_db,
)


def test_nmse_identical_signal_torch():
    y = torch.rand(4, 2, 32, 32)
    val = nmse_db(y, y).item()
    assert np.isfinite(val)
    assert val <= -90.0  # Identical signals yield extremely negative dB value


def test_nmse_identical_signal_numpy():
    y = np.random.rand(4, 2, 32, 32).astype(np.float32)
    val = nmse_db_numpy(y, y)
    assert np.isfinite(val)
    assert val <= -90.0


def test_nmse_diff_signal_torch():
    y_true = torch.ones(2, 2, 32, 32)
    y_pred = torch.zeros(2, 2, 32, 32)
    val = nmse_db(y_pred, y_true).item()
    assert np.isfinite(val)
    assert abs(val - 0.0) < 1e-3  # (||1-0||^2 / ||1||^2) = 1 => 10 log10(1) = 0 dB


def test_beamforming_gain_identical():
    y = np.random.randn(8, 2, 32, 32).astype(np.float32)
    gain_np = beamforming_gain_numpy(y, y)
    loss_db = beamforming_loss_db(gain_np)
    assert np.isclose(gain_np, 1.0, atol=1e-5)
    assert np.isclose(loss_db, 0.0, atol=1e-4)

    y_torch = torch.from_numpy(y)
    gain_torch = beamforming_gain_torch(y_torch, y_torch).item()
    assert np.isclose(gain_torch, 1.0, atol=1e-5)


def test_beamforming_gain_orthogonal():
    # Construct orthogonal complex channels
    # sample 1 has real/imag energy only on left half, sample 2 only on right half
    y1 = np.zeros((1, 2, 32, 32), dtype=np.float32)
    y2 = np.zeros((1, 2, 32, 32), dtype=np.float32)
    y1[:, :, :16, :] = 1.0
    y2[:, :, 16:, :] = 1.0

    gain = beamforming_gain_numpy(y1, y2)
    assert np.isclose(gain, 0.0, atol=1e-6)


def test_beamforming_gain_phase_shift_invariance():
    # In massive MIMO, global phase shift does not alter beamforming gain
    y = np.random.randn(4, 2, 32, 32).astype(np.float32)
    h_complex = y[:, 0, :, :] + 1j * y[:, 1, :, :]
    # Apply 90-degree phase shift e^(j * pi/2) = j
    h_shifted = h_complex * 1j
    y_shifted = np.stack([h_shifted.real, h_shifted.imag], axis=1).astype(np.float32)

    gain = beamforming_gain_numpy(y_shifted, y)
    assert np.isclose(gain, 1.0, atol=1e-5)


def test_beamforming_gain_torch_numpy_parity():
    np.random.seed(42)
    y1 = np.random.randn(10, 2, 32, 32).astype(np.float32)
    y2 = y1 + 0.1 * np.random.randn(10, 2, 32, 32).astype(np.float32)

    gain_np = beamforming_gain_numpy(y2, y1)
    gain_torch = beamforming_gain_torch(torch.from_numpy(y2), torch.from_numpy(y1)).item()
    assert np.isclose(gain_np, gain_torch, atol=1e-5)


def test_beamforming_gain_single_sample_shape():
    y = np.random.randn(2, 32, 32).astype(np.float32)
    gain = beamforming_gain_numpy(y, y)
    assert isinstance(gain, float)
    assert np.isclose(gain, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Regression tests for the normalisation-aware metrics.
#
# Two bugs motivated these. Both would have survived the switch to real data, so
# both are reproduced here to make sure they cannot silently return.
# ---------------------------------------------------------------------------

CSINET_NORM = {"scheme": "csinet_offset", "offset": 0.5, "min": -0.5, "max": 0.5}


def test_offset_derived_from_min_max_matches_explicit_offset():
    from utils.metrics import offset_from_norm_params

    assert offset_from_norm_params(None) == 0.0
    assert offset_from_norm_params({}) == 0.0
    # CsiNet convention: 0.5 is the complex zero point.
    assert np.isclose(offset_from_norm_params({"min": -0.5, "max": 0.5}), 0.5)
    # An explicit offset key wins over the derived value.
    assert np.isclose(offset_from_norm_params(CSINET_NORM), 0.5)
    # Synthetic min-max data: offset = -min / (max - min).
    assert np.isclose(offset_from_norm_params({"min": -1.5, "max": 0.5}), 0.75)


def test_constant_prediction_scores_zero_rho_when_deoffset():
    """
    A model that emits a constant carries no channel information at all, so rho
    must be 0. Without the de-offset the shared DC component dominates the inner
    product and rho saturates above 99%, which is exactly what made the original
    beamforming numbers meaningless.
    """
    rng = np.random.default_rng(0)
    h = np.zeros((32, 32), dtype=np.complex128)
    for _ in range(5):
        a, d = rng.integers(0, 32), rng.integers(0, 12)
        h[a, d] += (rng.normal() + 1j * rng.normal()) * np.exp(-0.15 * d)

    truth = np.stack([h.real, h.imag])[None, ...] / (2 * np.abs(h).max()) + 0.5
    constant = np.full_like(truth, 0.5)

    rho_fixed = cosine_similarity_numpy(constant, truth, norm_params=CSINET_NORM)
    rho_buggy = cosine_similarity_numpy(constant, truth)

    assert rho_fixed < 1e-6, "constant prediction must score rho = 0"
    assert rho_buggy > 0.99, "sanity check: the un-offset form really is saturated"


def test_rho_degrades_monotonically_with_noise():
    """rho must actually discriminate between good and bad reconstructions."""
    rng = np.random.default_rng(1)
    truth = np.clip(rng.normal(0.5, 0.05, (16, 2, 32, 32)), 0, 1).astype(np.float32)

    rhos = [
        cosine_similarity_numpy(
            np.clip(truth + rng.normal(0, sigma, truth.shape), 0, 1).astype(np.float32),
            truth,
            norm_params=CSINET_NORM,
        )
        for sigma in (0.001, 0.01, 0.05, 0.2)
    ]
    assert all(a > b for a, b in zip(rhos, rhos[1:])), f"rho not monotonic: {rhos}"


def test_aggregate_nmse_is_log_of_mean_not_mean_of_log():
    """
    nmse_db_aggregate must implement 10*log10(mean(ratio)), the CsiNet/CRNet
    convention. On a batch with per-sample variation it is strictly larger
    (less optimistic) than the mean-of-logs form, by Jensen's inequality.
    """
    from utils.metrics import nmse_db_aggregate, nmse_db_aggregate_numpy

    rng = np.random.default_rng(2)
    truth = rng.normal(0, 1, (128, 2, 32, 32)).astype(np.float32)
    # Log-uniform error scale gives the spread that separates the two forms.
    sigmas = 10 ** rng.uniform(-3.0, -1.0, (128, 1, 1, 1))
    recon = (truth + rng.normal(0, 1, truth.shape) * sigmas).astype(np.float32)

    mean_of_logs = nmse_db_numpy(recon, truth)
    log_of_mean = nmse_db_aggregate_numpy(recon, truth)

    assert log_of_mean > mean_of_logs + 1.0, (
        f"expected the aggregate form to be materially less optimistic, "
        f"got {log_of_mean:.2f} vs {mean_of_logs:.2f}"
    )

    # torch and numpy implementations must agree.
    torch_val = nmse_db_aggregate(torch.from_numpy(recon), torch.from_numpy(truth)).item()
    assert np.isclose(torch_val, log_of_mean, atol=1e-3)


def test_aggregate_nmse_identical_signal():
    from utils.metrics import nmse_db_aggregate_numpy

    y = np.random.rand(4, 2, 32, 32).astype(np.float32)
    assert nmse_db_aggregate_numpy(y, y) <= -90.0


def test_metrics_unchanged_without_norm_params():
    """Backward compatibility: omitting norm_params preserves the old behaviour."""
    rng = np.random.default_rng(3)
    y1 = rng.normal(0, 1, (8, 2, 32, 32)).astype(np.float32)
    y2 = (y1 + 0.1 * rng.normal(0, 1, y1.shape)).astype(np.float32)

    h1 = (y1[:, 0] + 1j * y1[:, 1]).reshape(8, -1)
    h2 = (y2[:, 0] + 1j * y2[:, 1]).reshape(8, -1)
    expected = np.mean(
        np.abs(np.sum(np.conj(h2) * h1, axis=-1))
        / (np.linalg.norm(h2, axis=-1) * np.linalg.norm(h1, axis=-1))
    )
    assert np.isclose(cosine_similarity_numpy(y2, y1), expected, atol=1e-6)

