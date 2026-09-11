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
# Real-data tracks: tensors produced by data/prepare_cost2100.py and
# data/prepare_deepmimo.py. Both land in the same (N, 2, 32, 32) float32 format
# with the same csinet_offset normalisation contract, so they share one set of
# checks. Each is skipped until that dataset has been prepared.
# ---------------------------------------------------------------------------

REAL_DATASETS = {
    "cost2100": (Path("data/processed_cost2100"), "data/prepare_cost2100.py"),
    "deepmimo": (Path("data/processed_deepmimo"), "data/prepare_deepmimo.py"),
}

# Backwards-compatible alias for the COST2100 path used by older tests.
COST2100_DIR = REAL_DATASETS["cost2100"][0]

real_dataset = pytest.mark.parametrize("dataset", sorted(REAL_DATASETS))


def _require(dataset):
    """Skip unless the named dataset has been prepared; return its directory."""
    directory, script = REAL_DATASETS[dataset]
    if not (directory / "train.npy").exists():
        pytest.skip(f"{dataset} dataset not prepared. Run {script}.")
    return directory


def _require_cost2100():
    return _require("cost2100")


@real_dataset
def test_real_shapes_and_types(dataset):
    directory = _require(dataset)

    # Sample counts depend on --samples / --subsample, so only the per-sample
    # geometry is fixed. The channel layout is the model contract.
    for name in ("train", "val", "test"):
        arr = np.load(directory / f"{name}.npy")
        assert arr.ndim == 4, f"{name}: expected 4 dims, got {arr.shape}"
        assert arr.shape[1:] == (2, 32, 32), f"{name}: got {arr.shape}"
        assert arr.dtype == np.float32, f"{name}: got {arr.dtype}"
        assert len(arr) > 0


@real_dataset
def test_real_values_are_finite_and_in_range(dataset):
    directory = _require(dataset)

    for name in ("train", "val", "test"):
        arr = np.load(directory / f"{name}.npy")
        assert np.isfinite(arr).all(), f"{name}: contains NaN or Inf"
        # The decoder's Sigmoid head cannot represent anything outside [0, 1].
        assert arr.min() >= 0.0 and arr.max() <= 1.0, f"{name}: outside [0,1]"


@real_dataset
def test_real_norm_params_describe_the_offset_convention(dataset):
    directory = _require(dataset)

    with open(directory / "norm_params.json") as f:
        params = json.load(f)

    assert params["shape"] == [2, 32, 32]
    assert params["scheme"] == "csinet_offset"
    # 0.5 is the complex zero point under this convention.
    assert params["offset"] == pytest.approx(0.5)
    # min/max are kept so the backend's affine denormalisation, which computes
    # x * (max - min) + min, reduces to exactly x - 0.5.
    assert params["min"] == pytest.approx(-0.5)
    assert params["max"] == pytest.approx(0.5)
    # How the split was formed must be recorded, because it decides what the
    # val/test scores actually measure.
    assert params["split_origin"] in {"shipped_with_dataset", "random_shuffle_70_10_20"}


@real_dataset
def test_real_offset_round_trips_through_metrics(dataset):
    """
    The offset derived from norm_params must reproduce CsiNet's de-offset
    exactly, since every reported metric depends on it.
    """
    directory = _require(dataset)
    from utils.metrics import offset_from_norm_params, to_complex_numpy

    with open(directory / "norm_params.json") as f:
        params = json.load(f)

    assert offset_from_norm_params(params) == pytest.approx(0.5)

    arr = np.load(directory / "test.npy")[:8]
    expected = ((arr[:, 0] - 0.5) + 1j * (arr[:, 1] - 0.5)).reshape(len(arr), -1)
    np.testing.assert_allclose(to_complex_numpy(arr, params), expected, rtol=1e-6)


@real_dataset
def test_real_channel_energy_is_nonzero(dataset):
    """
    A sample with no energy would make the NMSE denominator vanish. Ray tracing
    genuinely returns zero-power receivers (no path to the transmitter), so the
    preparation step must drop them rather than pass them through as zeros.
    """
    directory = _require(dataset)

    arr = np.load(directory / "test.npy")
    h = (arr[:, 0] - 0.5) + 1j * (arr[:, 1] - 0.5)
    energy = np.sum(np.abs(h) ** 2, axis=(1, 2))
    assert energy.min() > 1e-8, f"{int((energy <= 1e-8).sum())} empty sample(s)"


def test_deepmimo_records_the_full_pipeline():
    """
    The DeepMIMO track is the only one that runs the spatial-frequency -> 2D DFT
    -> delay-truncation pipeline, so its metadata must record what truncation
    cost. This is the evidence behind the 32-tap claim the README makes.
    """
    directory = _require("deepmimo")

    with open(directory / "norm_params.json") as f:
        params = json.load(f)

    assert params["normalisation"] == "per_sample_peak"
    assert "pipeline" in params

    cleaning = params["cleaning"]
    assert cleaning["kept"] == (cleaning["loaded"]
                                - cleaning["dropped_non_finite"]
                                - cleaning["dropped_zero_power"])

    retention = params["delay_truncation"]["energy_retention_percent"]
    assert retention > 90.0, f"delay truncation retains only {retention:.1f}% of energy"
