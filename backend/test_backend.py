import pytest
from fastapi.testclient import TestClient
import numpy as np
import torch
import sys
from pathlib import Path

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.app import app, LOADED_MODELS
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


def test_invalid_compression_ratio():
    response = client.post("/predict", json={"sample_index": 0, "compression_ratio": 8})
    assert response.status_code == 422  # Unprocessable Entity / Validation Error


def test_invalid_sample_index():
    response = client.post("/predict", json={"sample_index": -5, "compression_ratio": 16})
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
    assert data["compression_ratio"] == 16
    assert data["compressed_dim"] == 128
    assert data["bandwidth_saved_percent"] == 93.75
    assert len(data["original_matrix_real"]) == 32
    assert len(data["reconstructed_matrix_real"]) == 32
