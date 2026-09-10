import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.app import LOADED_MODELS, app
from models.csi_autoencoder import CSIAutoencoder

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "device" in data
    assert "available_models" in data
    assert "test_samples_available" in data
    assert "normalization_loaded" in data


def test_invalid_compression_ratio():
    response = client.post("/predict", json={"sample_index": 0, "compression_ratio": 8})
    assert response.status_code == 422  # Unprocessable Entity / Validation Error


def test_invalid_sample_index():
    response = client.post("/predict", json={"sample_index": -5, "compression_ratio": 16})
    assert response.status_code == 422


def test_missing_input_payload():
    # Neither sample_index nor matrices provided
    response = client.post("/predict", json={"compression_ratio": 16})
    assert response.status_code == 422


def test_predict_with_mocked_model(monkeypatch, tmp_path):
    # Set up mock test dataset
    mock_data = np.random.rand(10, 2, 32, 32).astype(np.float32)
    monkeypatch.setattr("backend.app.TEST_DATA", mock_data)

    # Instantiate mock model for CR=16
    mock_model = CSIAutoencoder(compression_ratio=16)
    mock_model.eval()
    LOADED_MODELS[16] = mock_model

    payload = {"sample_index": 2, "compression_ratio": 16}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["sample_index"] == 2
    assert data["input_source"] == "benchmark"
    assert data["compression_ratio"] == 16
    assert data["compressed_dim"] == 128
    assert data["bandwidth_saved_percent"] == 93.75
    assert "beamforming_gain_percent" in data
    assert 0.0 <= data["beamforming_gain_percent"] <= 100.0
    assert "beamforming_loss_db" in data
    assert data["beamforming_loss_db"] <= 0.01
    assert len(data["original_matrix_real"]) == 32
    assert len(data["reconstructed_matrix_real"]) == 32


def test_predict_with_custom_matrix():
    # Instantiate mock model for CR=16 if not loaded
    if 16 not in LOADED_MODELS:
        mock_model = CSIAutoencoder(compression_ratio=16)
        mock_model.eval()
        LOADED_MODELS[16] = mock_model

    # Custom 32x32 random matrices
    real_mat = np.random.uniform(-1.0, 1.0, size=(32, 32)).tolist()
    imag_mat = np.random.uniform(-1.0, 1.0, size=(32, 32)).tolist()

    payload = {
        "matrix_real": real_mat,
        "matrix_imag": imag_mat,
        "compression_ratio": 16,
        "auto_normalize": True
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["sample_index"] is None
    assert data["input_source"] == "custom_json"
    assert data["compression_ratio"] == 16
    assert data["compressed_dim"] == 128
    assert len(data["original_matrix_real"]) == 32
    assert len(data["original_matrix_imag"]) == 32
    assert len(data["reconstructed_matrix_real"]) == 32
    assert len(data["reconstructed_matrix_imag"]) == 32


def test_predict_with_custom_matrix_invalid_shape():
    # 31x32 instead of 32x32
    bad_real = [[0.0] * 32 for _ in range(31)]
    good_imag = [[0.0] * 32 for _ in range(32)]

    payload = {
        "matrix_real": bad_real,
        "matrix_imag": good_imag,
        "compression_ratio": 16
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_upload_npy():
    if 16 not in LOADED_MODELS:
        mock_model = CSIAutoencoder(compression_ratio=16)
        mock_model.eval()
        LOADED_MODELS[16] = mock_model

    # Create in-memory .npy file with shape (2, 32, 32)
    fake_csi = np.random.randn(2, 32, 32).astype(np.float32)
    buf = io.BytesIO()
    np.save(buf, fake_csi)
    buf.seek(0)

    files = {"file": ("test_channel.npy", buf, "application/octet-stream")}
    response = client.post("/predict/upload?compression_ratio=16", files=files)
    assert response.status_code == 200

    data = response.json()
    assert "upload_npy" in data["input_source"]
    assert data["compression_ratio"] == 16
    assert len(data["original_matrix_real"]) == 32
    assert len(data["reconstructed_matrix_real"]) == 32
