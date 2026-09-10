import os
import json
import numpy as np
import pytest
from pathlib import Path


def test_processed_dataset_shapes_and_types():
    data_dir = Path("data/processed")
    if not (data_dir / "train.npy").exists():
        pytest.skip("Dataset files not generated yet.")

    train = np.load(data_dir / "train.npy")
    val = np.load(data_dir / "val.npy")
    test = np.load(data_dir / "test.npy")

    assert train.shape == (7000, 2, 32, 32)
    assert val.shape == (1000, 2, 32, 32)
    assert test.shape == (2000, 2, 32, 32)

    assert train.dtype == np.float32
    assert val.dtype == np.float32
    assert test.dtype == np.float32


def test_normalization_range():
    data_dir = Path("data/processed")
    if not (data_dir / "train.npy").exists():
        pytest.skip("Dataset files not generated yet.")

    train = np.load(data_dir / "train.npy")
    assert train.min() >= 0.0
    assert train.max() <= 1.0


def test_norm_params_json():
    json_path = Path("data/processed/norm_params.json")
    if not json_path.exists():
        pytest.skip("Normalization JSON not found.")

    with open(json_path, "r") as f:
        params = json.load(f)

    assert "min" in params
    assert "max" in params
    assert "shape" in params
    assert params["shape"] == [2, 32, 32]


# ---------------------------------------------------------------------------
# Real-data track: COST2100 tensors produced by data/prepare_cost2100.py.
# Skipped until the dataset has been prepared, in the same style as above.
# ---------------------------------------------------------------------------

COST2100_DIR = Path("data/processed_cost2100")


def _require_cost2100():
    if not (COST2100_DIR / "train.npy").exists():
        pytest.skip("COST2100 dataset not prepared. Run data/prepare_cost2100.py.")


def test_cost2100_shapes_and_types():
    _require_cost2100()

    # Sample counts depend on --subsample, so only the per-sample geometry is
    # fixed. The channel layout is what the model contract depends on.
    for name in ("train", "val", "test"):
        arr = np.load(COST2100_DIR / f"{name}.npy")
        assert arr.ndim == 4, f"{name}: expected 4 dims, got {arr.shape}"
        assert arr.shape[1:] == (2, 32, 32), f"{name}: got {arr.shape}"
        assert arr.dtype == np.float32, f"{name}: got {arr.dtype}"
        assert len(arr) > 0


def test_cost2100_values_are_finite_and_in_range():
    _require_cost2100()

    for name in ("train", "val", "test"):
        arr = np.load(COST2100_DIR / f"{name}.npy")
        assert np.isfinite(arr).all(), f"{name}: contains NaN or Inf"
        # The decoder's Sigmoid head cannot represent anything outside [0, 1].
        assert arr.min() >= 0.0 and arr.max() <= 1.0, f"{name}: outside [0,1]"


def test_cost2100_norm_params_describe_the_offset_convention():
    _require_cost2100()

    with open(COST2100_DIR / "norm_params.json") as f:
        params = json.load(f)

    assert params["shape"] == [2, 32, 32]
    assert params["scheme"] == "csinet_offset"
    # 0.5 is the complex zero point in the released data.
    assert params["offset"] == pytest.approx(0.5)
    # min/max are kept so the backend's affine denormalisation, which computes
    # x * (max - min) + min, reduces to exactly x - 0.5.
    assert params["min"] == pytest.approx(-0.5)
    assert params["max"] == pytest.approx(0.5)
    assert params["split_origin"] == "shipped_with_dataset"


def test_cost2100_offset_round_trips_through_metrics():
    """
    The offset derived from norm_params must reproduce CsiNet's de-offset
    exactly, since every reported metric depends on it.
    """
    _require_cost2100()
    from utils.metrics import offset_from_norm_params, to_complex_numpy

    with open(COST2100_DIR / "norm_params.json") as f:
        params = json.load(f)

    assert offset_from_norm_params(params) == pytest.approx(0.5)

    arr = np.load(COST2100_DIR / "test.npy")[:8]
    expected = ((arr[:, 0] - 0.5) + 1j * (arr[:, 1] - 0.5)).reshape(len(arr), -1)
    np.testing.assert_allclose(to_complex_numpy(arr, params), expected, rtol=1e-6)


def test_cost2100_channel_energy_is_nonzero():
    """
    A sample with no energy would make the NMSE denominator vanish. The metrics
    carry an epsilon, but a genuinely empty sample still signals a broken load.
    """
    _require_cost2100()

    arr = np.load(COST2100_DIR / "test.npy")
    h = (arr[:, 0] - 0.5) + 1j * (arr[:, 1] - 0.5)
    energy = np.sum(np.abs(h) ** 2, axis=(1, 2))
    assert energy.min() > 1e-8, f"{int((energy <= 1e-8).sum())} empty sample(s)"
