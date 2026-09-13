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

# The metrics are only meaningful on the de-offset channel, and the API now
# refuses to compute them without this. Tests previously left it None, which
# silently measured the raw [0,1] tensor -- and one assertion below was
# passing only because of the saturation that caused.
NORM = {"scheme": "csinet_offset", "offset": 0.5, "min": -0.5, "max": 0.5}


@pytest.fixture(autouse=True)
def _norm_params(monkeypatch):
    monkeypatch.setattr("backend.app.NORM_PARAMS", NORM)

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
    # An UNTRAINED model reconstructing a random tensor must not look good.
    # The previous assertion (<= 0.01 dB) held only because the metric was
    # measured without the offset and saturated near 100% for any output.
    # `<= 0.0` is no better: gain is in [0,1] so 10*log10(gain) is always <= 0,
    # and the check can never fail. Assert something the bug would break.
    assert data["beamforming_loss_db"] < -0.05, (
        "an untrained model showed near-zero beamforming loss -- the offset is "
        "not being removed"
    )
    assert data["beamforming_gain_percent"] < 99.0, (
        "an untrained model scored near-perfect beamforming gain -- the "
        "offset is not being removed"
    )
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


def test_per_sample_scheme_round_trips_through_predict(monkeypatch):
    """
    DeepMIMO is normalised per sample by peak and records min/max of -0.5/+0.5
    only to encode the offset convention. The endpoint must scale by the sample's
    own peak: applying that 1.0 span to a raw matrix maps everything to ~0.5, a
    constant, and denormalises the reply by the wrong factor. Only reachable with
    a per-sample norm_params, so the other tests never touched this branch.
    """
    monkeypatch.setattr("backend.app.NORM_PARAMS", {
        "offset": 0.5, "min": -0.5, "max": 0.5, "normalisation": "per_sample_peak",
    })
    if 16 not in LOADED_MODELS:
        m = CSIAutoencoder(compression_ratio=16)
        m.eval()
        LOADED_MODELS[16] = m

    # Physical scale, like a real ray-traced channel -- far outside [0,1].
    rng = np.random.default_rng(0)
    payload = {
        "matrix_real": rng.normal(scale=2e-5, size=(32, 32)).tolist(),
        "matrix_imag": rng.normal(scale=2e-5, size=(32, 32)).tolist(),
        "compression_ratio": 16,
        "auto_normalize": True,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()

    # The reply comes back at the input's physical scale, not left near 0.5 and
    # not rescaled by the meaningless global span.
    recon = np.array(data["reconstructed_matrix_real"])
    assert np.abs(recon).max() < 1e-3, "output was not returned to the input's scale"
    assert np.isfinite(data["nmse_db"])


def test_predict_rejects_unnormalised_input_when_auto_normalize_is_off():
    """
    The metrics subtract the 0.5 offset unconditionally, so a zero-centred matrix
    would have an offset removed that was never applied -- inflating the NMSE
    denominator and reporting a too-optimistic number instead of failing.
    """
    if 16 not in LOADED_MODELS:
        m = CSIAutoencoder(compression_ratio=16)
        m.eval()
        LOADED_MODELS[16] = m

    payload = {
        "matrix_real": np.random.uniform(-1, 1, (32, 32)).tolist(),
        "matrix_imag": np.random.uniform(-1, 1, (32, 32)).tolist(),
        "compression_ratio": 16,
        "auto_normalize": False,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "0,1" in response.json()["detail"]


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
