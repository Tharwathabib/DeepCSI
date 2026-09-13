"""
Regressions for the six defects the second code review found. Each shipped
green: the suite passed, the demo ran, and the numbers were wrong or the crash
was one absent column away.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from models.csi_autoencoder import CSIAutoencoder, build_from_checkpoint, infer_refine_widths
from utils.metrics import beamforming_gain_numpy, beamforming_gain_torch, cosine_similarity_numpy

NORM = {"scheme": "csinet_offset", "offset": 0.5, "min": -0.5, "max": 0.5}


def heterogeneous_pair(n=200, seed=0):
    """Predictions whose quality VARIES across the batch -- where Jensen bites."""
    rng = np.random.default_rng(seed)
    true = np.zeros((n, 2, 32, 32), dtype=np.float32)
    for i in range(n):
        for _ in range(6):
            a, d = rng.integers(0, 32), rng.integers(0, 8)
            true[i, :, a, d] += rng.normal(scale=0.2, size=2)
    true = np.clip(true + 0.5, 0, 1)
    noise = rng.uniform(0.002, 0.08, size=(n, 1, 1, 1)).astype(np.float32)
    pred = np.clip(true + noise * rng.normal(size=true.shape).astype(np.float32), 0, 1)
    return pred, true


def test_beamforming_gain_averages_squares_not_squares_the_average():
    """
    G = rho^2, so the batch figure must be mean(rho^2). Squaring the mean rho
    computes (E rho)^2, which by Jensen is strictly smaller whenever quality
    varies. preflight and the API used that path while metrics.csv averaged per
    sample, so one quantity was published two ways, ~6 points apart.
    """
    pred, true = heterogeneous_pair()
    batch = beamforming_gain_numpy(pred, true, norm_params=NORM)
    per_sample = beamforming_gain_numpy(pred, true, return_per_sample=True, norm_params=NORM)

    assert batch == pytest.approx(float(np.mean(per_sample)), abs=1e-6)

    rho = cosine_similarity_numpy(pred, true, return_per_sample=True, norm_params=NORM)
    wrong = float(np.mean(rho)) ** 2
    assert batch > wrong, "Jensen gap vanished -- the test data is not heterogeneous"


def test_torch_and_numpy_beamforming_agree():
    pred, true = heterogeneous_pair(120, seed=3)
    t = float(beamforming_gain_torch(torch.from_numpy(pred), torch.from_numpy(true),
                                     norm_params=NORM))
    n = beamforming_gain_numpy(pred, true, norm_params=NORM)
    assert t == pytest.approx(n, abs=1e-5)


@pytest.mark.parametrize("widths", [(8,), (8, 16)])
def test_refine_widths_are_recovered_from_weights(tmp_path, widths):
    """
    A checkpoint without a refine_widths key is not necessarily the current
    default -- the block was widened from (8,) to (8, 16) on this branch.
    Assuming the default makes load_state_dict raise, and the API catches that,
    so the model silently disappears from /health.
    """
    model = CSIAutoencoder(compression_ratio=16, refine_widths=widths)
    assert infer_refine_widths(model.state_dict()) == widths

    # A checkpoint from before the key existed.
    path = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": model.state_dict()}, path)
    loaded = build_from_checkpoint(torch.load(path, map_location="cpu", weights_only=False), 16, "cpu")
    # Assert unconditionally. Guarding this with hasattr() would make the whole
    # check evaporate the moment the attribute is renamed, which is the same
    # latent-vacuous assertion this suite exists to stamp out.
    assert loaded.refine_widths == widths

    x = torch.from_numpy(heterogeneous_pair(4, seed=5)[1])
    with torch.no_grad():
        assert torch.allclose(loaded(x)[0], model.eval()(x)[0], atol=1e-6)


def test_per_sample_scheme_is_not_scaled_by_a_global_span():
    """
    DeepMIMO records min/max -0.5/+0.5 to encode the OFFSET convention, but it is
    normalised per sample by peak -- there is no global scale. Applying that 1.0
    span to a raw matrix maps everything to ~0.5, i.e. a constant.
    """
    from backend.app import normalize_external

    raw = (np.random.default_rng(0).normal(scale=1e-5, size=(2, 32, 32))).astype(np.float32)
    per_sample = {"offset": 0.5, "min": -0.5, "max": 0.5, "normalisation": "per_sample_peak"}

    scaled, denorm = normalize_external(raw, per_sample)
    assert scaled.min() >= 0.0 and scaled.max() <= 1.0
    # It must actually use the range, not collapse to the offset.
    assert scaled.std() > 0.05, f"per-sample scaling collapsed to a constant (std {scaled.std():.4g})"
    assert np.allclose(denorm(scaled), raw, atol=1e-9)

    # The global-scale branch still round-trips for the synthetic convention.
    glob = {"offset": 0.5, "min": -1.0, "max": 1.0}
    s2, d2 = normalize_external(np.clip(raw * 1e4, -1, 1), glob)
    assert np.allclose(d2(s2), np.clip(raw * 1e4, -1, 1), atol=1e-6)


def test_gain_axis_floor_survives_a_missing_column():
    """min() over an empty sequence raises; the chart must degrade, not crash."""
    from frontend.app import _gain_axis_floor
    assert _gain_axis_floor([], []) == 0.0
    assert _gain_axis_floor([90.0, 22.0], []) == pytest.approx(14.0)
    assert _gain_axis_floor([float("nan")], []) == 0.0
