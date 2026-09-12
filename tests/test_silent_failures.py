"""
Regressions for defects that produced wrong answers without raising anything.

Each of these shipped, ran clean, and printed a plausible number. That is what
makes them worth a test: a crash gets noticed, a quiet 35 dB does not.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from models.csi_autoencoder import CSIAutoencoder, build_from_checkpoint
from utils.metrics import (warn_if_offset_data_measured_raw, nmse_db_aggregate_numpy,
                           beamforming_gain_numpy)

NORM = {"scheme": "csinet_offset", "offset": 0.5, "min": -0.5, "max": 0.5}


def offset_data(n=32, seed=0):
    """Sparse angular-delay tensors in [0,1] with 0.5 as complex zero."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 2, 32, 32), dtype=np.float32)
    for i in range(n):
        for _ in range(8):
            a, d = rng.integers(0, 32), rng.integers(0, 6)
            x[i, :, a, d] += rng.normal(scale=0.2, size=2)
    return np.clip(x + 0.5, 0.0, 1.0)


# --------------------------------------------------------------------------
# 1. Measuring offset data without norm_params
# --------------------------------------------------------------------------

def test_measuring_offset_data_without_norm_params_warns():
    """
    The project's founding bug. Every metric takes norm_params optionally and
    silently measures the raw tensor without it, which inflates NMSE ~35 dB and
    pins rho near 1.0. Passing it everywhere was the first fix and it did not
    hold -- models/dct_baseline.py was missed and printed -37.61 dB. The
    condition is now detected from the data instead of trusted to the caller.
    """
    x = offset_data()
    with pytest.warns(RuntimeWarning, match="offset-normalised"):
        nmse_db_aggregate_numpy(x * 0.9 + 0.05, x, None)


def test_de_offset_data_does_not_warn():
    """The detector must stay silent on genuinely zero-centred data."""
    import warnings
    centred = (offset_data() - 0.5).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nmse_db_aggregate_numpy(centred * 0.9, centred, None)


def test_detector_fires_only_on_the_real_mistake():
    x = offset_data()
    assert warn_if_offset_data_measured_raw(x, 0.0, "t") is True
    # An explicit offset means the caller handled it.
    assert warn_if_offset_data_measured_raw(x, 0.5, "t") is False
    # Zero-centred data is the legitimate raw case.
    assert warn_if_offset_data_measured_raw(x - 0.5, 0.0, "t") is False


def test_the_gap_the_warning_is_about_is_still_enormous():
    """Quantify it, so nobody dismisses the warning as pedantic."""
    x = offset_data(64)
    pred = x + np.float32(0.01) * np.random.default_rng(1).normal(size=x.shape).astype(np.float32)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = nmse_db_aggregate_numpy(pred, x, None)
        raw_gain = beamforming_gain_numpy(pred, x, norm_params=None)
    deoffset = nmse_db_aggregate_numpy(pred, x, NORM)
    deoffset_gain = beamforming_gain_numpy(pred, x, norm_params=NORM)

    assert raw - deoffset < -20, f"raw {raw:.1f} vs de-offset {deoffset:.1f} dB"
    assert raw_gain > 0.99 and deoffset_gain < raw_gain


def test_dct_baseline_entry_point_loads_norm_params():
    """
    models/dct_baseline.py's main() must not fall back to None. It did, and
    running it printed -37.61 dB / 99.98% -- within a dB of the discredited
    figures the project started from.
    """
    src = (root_dir / "models" / "dct_baseline.py").read_text(encoding="utf-8")
    main_src = src.split("def main(")[1]
    assert "norm_params.json" in main_src, "main() does not read the normalisation metadata"
    assert "evaluate_dct_baseline(test_data, norm_params=" in main_src, \
        "main() calls the evaluator without passing norm_params"


# --------------------------------------------------------------------------
# 2. Rebuilding a checkpoint without its architecture
# --------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ["csinet", "crnet"])
def test_checkpoint_round_trips_through_the_shared_builder(tmp_path, arch):
    """
    The API, evaluator and preflight each read refine_widths but ignored arch,
    so a crnet checkpoint could not be loaded by any of them. In the backend
    load_state_dict's exception was caught and logged, leaving the model quietly
    absent from /health instead of failing startup.
    """
    model = CSIAutoencoder(compression_ratio=16, refine_widths=(8, 16), arch=arch)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "refine_widths": (8, 16),
        "arch": arch,
        "data_source": "unit-test",
    }
    path = tmp_path / "ckpt.pt"
    torch.save(ckpt, path)

    loaded = build_from_checkpoint(torch.load(path, map_location="cpu", weights_only=False), 16, "cpu")
    assert loaded.encoder.arch == arch
    x = torch.from_numpy(offset_data(4))
    with torch.no_grad():
        assert torch.allclose(loaded(x)[0], model.eval()(x)[0], atol=1e-6)


def test_no_caller_rebuilds_a_model_without_the_builder():
    """
    Guard against the duplication coming back. Anything that reconstructs a
    model from a checkpoint must go through build_from_checkpoint, which is the
    single place that knows arch and refine_widths travel together.
    """
    offenders = []
    for rel in ("backend/app.py", "models/evaluate.py", "preflight.py", "models/quantize_eval.py"):
        text = (root_dir / rel).read_text(encoding="utf-8")
        if "load_state_dict" in text and "build_from_checkpoint" not in text:
            offenders.append(rel)
    assert not offenders, f"rebuild a checkpoint without the shared builder: {offenders}"
