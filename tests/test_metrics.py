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

