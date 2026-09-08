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
