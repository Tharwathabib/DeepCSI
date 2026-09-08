import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, validator

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.metrics import nmse_db
from models.csi_autoencoder import CSIAutoencoder

app = FastAPI(
    title="DeepCSI Inference API",
    description="FastAPI Backend for DeepCSI - AI-Native Massive MIMO Channel State Information Compression",
    version="1.0.0"
)

# Global in-memory storage for test dataset and loaded models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DATA: Optional[np.ndarray] = None
LOADED_MODELS: Dict[int, CSIAutoencoder] = {}


class PredictRequest(BaseModel):
    sample_index: int = Field(0, ge=0, description="Test sample index (0 to 999)")
    compression_ratio: int = Field(16, description="Compression Ratio: 4, 16, or 32")

    @validator("compression_ratio")
    def validate_cr(cls, v):
        if v not in [4, 16, 32]:
            raise ValueError("compression_ratio must be 4, 16, or 32")
        return v


class PredictResponse(BaseModel):
    sample_index: int
    compression_ratio: int
    bandwidth_saved_percent: float
    original_shape: List[int]
    original_scalars: int
    compressed_dim: int
    nmse_db: float
    inference_ms: float
    original_matrix_real: List[List[float]]
    reconstructed_matrix_real: List[List[float]]
    original_matrix_imag: List[List[float]]
    reconstructed_matrix_imag: List[List[float]]


@app.on_event("startup")
def load_resources():
    global TEST_DATA, LOADED_MODELS
    test_path = Path("data/processed/test.npy")
    if test_path.exists():
        TEST_DATA = np.load(test_path)
        print(f"[Backend Startup] Loaded test dataset with shape: {TEST_DATA.shape}")
    else:
        print(f"[Backend Startup Warning] Test dataset not found at {test_path}")

    weights_dir = Path("models/weights")
    for cr in [4, 16, 32]:
        weight_path = weights_dir / f"deepcsi_cr{cr}.pt"
        if weight_path.exists():
            try:
                model = CSIAutoencoder(compression_ratio=cr).to(DEVICE)
                checkpoint = torch.load(weight_path, map_location=DEVICE)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                LOADED_MODELS[cr] = model
                print(f"[Backend Startup] Loaded model weight for CR={cr} on {DEVICE}")
            except Exception as e:
                print(f"[Backend Startup Error] Failed loading CR={cr} model: {e}")


@app.get("/health")
def get_health():
    """Health check endpoint providing system status and available models."""
    available_crs = sorted(list(LOADED_MODELS.keys()))
    num_samples = len(TEST_DATA) if TEST_DATA is not None else 0
    return {
        "status": "ok",
        "device": str(DEVICE),
        "available_models": available_crs,
        "test_samples_available": num_samples
    }


@app.get("/sample/{index}")
def get_sample(index: int):
    """Retrieve raw test sample matrix given an index."""
    if TEST_DATA is None:
        raise HTTPException(status_code=500, detail="Test dataset not loaded on server.")
    if index < 0 or index >= len(TEST_DATA):
        raise HTTPException(
            status_code=404,
            detail=f"Sample index {index} out of range (0 to {len(TEST_DATA)-1})."
        )
    sample = TEST_DATA[index]  # (2, 32, 32)
    return {
        "sample_index": index,
        "shape": list(sample.shape),
        "matrix_real": sample[0].tolist(),
        "matrix_imag": sample[1].tolist()
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Perform CSI Compression & Reconstruction for a target test sample and compression ratio.
    """
    if TEST_DATA is None:
        raise HTTPException(status_code=500, detail="Test dataset not loaded on server.")
    if req.sample_index < 0 or req.sample_index >= len(TEST_DATA):
        raise HTTPException(
            status_code=404,
            detail=f"Sample index {req.sample_index} out of bounds (0 to {len(TEST_DATA)-1})."
        )
    if req.compression_ratio not in LOADED_MODELS:
        weights_file = f"models/weights/deepcsi_cr{req.compression_ratio}.pt"
        raise HTTPException(
            status_code= status.HTTP_404_NOT_FOUND,
            detail=f"Model weights for CR={req.compression_ratio} not loaded. "
                   f"Expected file '{weights_file}'. "
                   f"Train model using: python models/train.py --compression-ratio {req.compression_ratio}"
        )

    model = LOADED_MODELS[req.compression_ratio]
    sample_np = TEST_DATA[req.sample_index]  # (2, 32, 32)
    sample_tensor = torch.from_numpy(sample_np).unsqueeze(0).to(DEVICE)  # (1, 2, 32, 32)

    # Synchronize and measure inference latency
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        recon_tensor, latent_tensor = model(sample_tensor)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    inference_ms = (time.time() - start_time) * 1000.0

    # Calculate NMSE in dB
    nmse_val = float(nmse_db(recon_tensor, sample_tensor).item())

    recon_np = recon_tensor.squeeze(0).cpu().numpy()  # (2, 32, 32)
    latent_dim = model.latent_dim
    original_scalars = 2048
    bandwidth_saved = (1.0 - (latent_dim / original_scalars)) * 100.0

    return PredictResponse(
        sample_index=req.sample_index,
        compression_ratio=req.compression_ratio,
        bandwidth_saved_percent=round(bandwidth_saved, 2),
        original_shape=[2, 32, 32],
        original_scalars=original_scalars,
        compressed_dim=latent_dim,
        nmse_db=round(nmse_val, 2),
        inference_ms=round(inference_ms, 3),
        original_matrix_real=sample_np[0].tolist(),
        reconstructed_matrix_real=recon_np[0].tolist(),
        original_matrix_imag=sample_np[1].tolist(),
        reconstructed_matrix_imag=recon_np[1].tolist()
    )
