import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.metrics import nmse_db, nmse_db_numpy


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
