import io
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from models.csi_autoencoder import CSIAutoencoder
from utils.metrics import beamforming_gain_numpy, beamforming_loss_db, nmse_db

# Global in-memory storage for test dataset, normalization metadata, and loaded models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DATA: Optional[np.ndarray] = None
NORM_PARAMS: Optional[Dict] = None
LOADED_MODELS: Dict[int, CSIAutoencoder] = {}


def load_resources():
    """Load test dataset, normalization parameters, and PyTorch model checkpoints."""
    global TEST_DATA, LOADED_MODELS, NORM_PARAMS

    data_dir = Path(os.getenv("DATA_DIR", "data/processed"))
    test_path = data_dir / "test.npy"
    if test_path.exists():
        TEST_DATA = np.load(test_path)
        print(f"[Backend Startup] Loaded test dataset with shape: {TEST_DATA.shape}")
    else:
        print(f"[Backend Startup Warning] Test dataset not found at {test_path}")

    norm_path = data_dir / "norm_params.json"
    if norm_path.exists():
        with open(norm_path, "r") as f:
            NORM_PARAMS = json.load(f)
        print(
            f"[Backend Startup] Loaded normalization metadata: min={NORM_PARAMS.get('min'):.4f}, "
            f"max={NORM_PARAMS.get('max'):.4f}"
        )
    else:
        print(f"[Backend Startup Warning] Normalization metadata not found at {norm_path}")

    weights_dir = Path(os.getenv("WEIGHTS_DIR", "models/weights"))
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup resource loading and clean shutdown."""
    load_resources()
    yield
    # Cleanup on shutdown if needed
    LOADED_MODELS.clear()


app = FastAPI(
    title="DeepCSI Inference API",
    description="FastAPI Backend for DeepCSI - AI-Native Massive MIMO Channel State Information Compression",
    version="1.1.0",
    lifespan=lifespan
)

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    sample_index: Optional[int] = Field(None, ge=0, description="Test benchmark sample index (0 to 1999)")
    matrix_real: Optional[List[List[float]]] = Field(None, description="32x32 Real component matrix")
    matrix_imag: Optional[List[List[float]]] = Field(None, description="32x32 Imaginary component matrix")
    compression_ratio: int = Field(16, description="Compression Ratio: 4, 16, or 32")
    auto_normalize: bool = Field(True, description="Automatically scale input to [0,1] and denormalize output")

    @field_validator("compression_ratio")
    @classmethod
    def validate_cr(cls, v: int) -> int:
        if v not in [4, 16, 32]:
            raise ValueError("compression_ratio must be 4, 16, or 32")
        return v

    @model_validator(mode="after")
    def validate_inputs(self):
        # Must supply either sample_index OR both matrix_real & matrix_imag
        if self.sample_index is None and (self.matrix_real is None or self.matrix_imag is None):
            raise ValueError("Must provide either 'sample_index' OR both 'matrix_real' and 'matrix_imag'")

        if self.matrix_real is not None and self.matrix_imag is not None:
            if len(self.matrix_real) != 32 or any(len(row) != 32 for row in self.matrix_real):
                raise ValueError("matrix_real must have dimensions (32, 32)")
            if len(self.matrix_imag) != 32 or any(len(row) != 32 for row in self.matrix_imag):
                raise ValueError("matrix_imag must have dimensions (32, 32)")

        return self


class PredictResponse(BaseModel):
    sample_index: Optional[int] = None
    input_source: str = "benchmark"
    compression_ratio: int
    bandwidth_saved_percent: float
    original_shape: List[int]
    original_scalars: int
    compressed_dim: int
    nmse_db: float
    beamforming_gain_percent: float
    beamforming_loss_db: float
    inference_ms: float
    original_matrix_real: List[List[float]]
    reconstructed_matrix_real: List[List[float]]
    original_matrix_imag: List[List[float]]
    reconstructed_matrix_imag: List[List[float]]


def run_model_inference(sample_np: np.ndarray, compression_ratio: int):
    """
    Execute autoencoder forward pass on a normalized (2, 32, 32) float32 tensor.
    Returns:
        recon_np: (2, 32, 32) reconstructed normalized array
        inference_ms: latency in milliseconds
        nmse_val: NMSE in dB
        bf_gain_val: MRT beamforming power gain
        bf_loss_val: beamforming loss in dB
        latent_dim: number of latent scalars
    """
    if compression_ratio not in LOADED_MODELS:
        weights_file = f"models/weights/deepcsi_cr{compression_ratio}.pt"
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model weights for CR={compression_ratio} not loaded. "
                   f"Expected file '{weights_file}'."
        )

    model = LOADED_MODELS[compression_ratio]
    sample_tensor = torch.from_numpy(sample_np).unsqueeze(0).to(DEVICE)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        recon_tensor, latent_tensor = model(sample_tensor)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    inference_ms = (time.time() - start_time) * 1000.0

    nmse_val = float(nmse_db(recon_tensor, sample_tensor).item())
    recon_np = recon_tensor.squeeze(0).cpu().numpy()

    bf_gain_val = float(beamforming_gain_numpy(recon_np, sample_np))
    bf_loss_val = float(beamforming_loss_db(bf_gain_val))

    return recon_np, inference_ms, nmse_val, bf_gain_val, bf_loss_val, model.latent_dim


@app.get("/health")
def get_health():
    """Health check endpoint providing system status, device, and available models."""
    available_crs = sorted(list(LOADED_MODELS.keys()))
    num_samples = len(TEST_DATA) if TEST_DATA is not None else 0
    return {
        "status": "ok",
        "device": str(DEVICE),
        "available_models": available_crs,
        "test_samples_available": num_samples,
        "normalization_loaded": NORM_PARAMS is not None
    }


@app.post("/reload")
def reload_server_resources():
    """Dynamically reload dataset, normalization metadata, and checkpoints."""
    load_resources()
    return {
        "status": "reloaded",
        "models_loaded": list(LOADED_MODELS.keys()),
        "test_samples": len(TEST_DATA) if TEST_DATA is not None else 0,
        "normalization_loaded": NORM_PARAMS is not None
    }


@app.get("/sample/{index}")
def get_sample(index: int):
    """Retrieve raw benchmark test sample matrix given an index."""
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
    Perform CSI Compression & Reconstruction for either a benchmark sample or an external custom matrix.
    """
    if req.sample_index is not None:
        if TEST_DATA is None:
            raise HTTPException(status_code=500, detail="Test dataset not loaded on server.")
        if req.sample_index < 0 or req.sample_index >= len(TEST_DATA):
            raise HTTPException(
                status_code=404,
                detail=f"Sample index {req.sample_index} out of bounds (0 to {len(TEST_DATA)-1})."
            )
        sample_np = TEST_DATA[req.sample_index].astype(np.float32)
        raw_sample = sample_np
        input_source = "benchmark"
    else:
        raw_sample = np.stack([
            np.array(req.matrix_real, dtype=np.float32),
            np.array(req.matrix_imag, dtype=np.float32)
        ], axis=0)
        input_source = "custom_json"

        if req.auto_normalize and NORM_PARAMS is not None:
            norm_min = float(NORM_PARAMS["min"])
            norm_max = float(NORM_PARAMS["max"])
            sample_np = (raw_sample - norm_min) / (norm_max - norm_min + 1e-10)
            sample_np = np.clip(sample_np, 0.0, 1.0).astype(np.float32)
        else:
            sample_np = raw_sample

    recon_np, inference_ms, nmse_val, bf_gain_val, bf_loss_val, latent_dim = run_model_inference(
        sample_np, req.compression_ratio
    )

    if req.sample_index is None and req.auto_normalize and NORM_PARAMS is not None:
        norm_min = float(NORM_PARAMS["min"])
        norm_max = float(NORM_PARAMS["max"])
        denorm_recon = recon_np * (norm_max - norm_min) + norm_min
        orig_out_real = raw_sample[0].tolist()
        orig_out_imag = raw_sample[1].tolist()
        recon_out_real = denorm_recon[0].tolist()
        recon_out_imag = denorm_recon[1].tolist()
    else:
        orig_out_real = sample_np[0].tolist()
        orig_out_imag = sample_np[1].tolist()
        recon_out_real = recon_np[0].tolist()
        recon_out_imag = recon_np[1].tolist()

    original_scalars = 2048
    bandwidth_saved = (1.0 - (latent_dim / original_scalars)) * 100.0

    return PredictResponse(
        sample_index=req.sample_index,
        input_source=input_source,
        compression_ratio=req.compression_ratio,
        bandwidth_saved_percent=round(bandwidth_saved, 2),
        original_shape=[2, 32, 32],
        original_scalars=original_scalars,
        compressed_dim=latent_dim,
        nmse_db=round(nmse_val, 2),
        beamforming_gain_percent=round(bf_gain_val * 100.0, 2),
        beamforming_loss_db=round(bf_loss_val, 2),
        inference_ms=round(inference_ms, 3),
        original_matrix_real=orig_out_real,
        reconstructed_matrix_real=recon_out_real,
        original_matrix_imag=orig_out_imag,
        reconstructed_matrix_imag=recon_out_imag
    )


@app.post("/predict/upload", response_model=PredictResponse)
async def predict_upload(
    file: UploadFile = File(..., description="Upload .npy file containing (2, 32, 32) float32 or (32, 32) complex64 CSI matrix"),
    compression_ratio: int = 16,
    auto_normalize: bool = True
):
    """
    Perform CSI Compression & Reconstruction on an uploaded .npy CSI tensor.
    Supports either shape (2, 32, 32) float32 or shape (32, 32) complex64/complex128.
    """
    if compression_ratio not in [4, 16, 32]:
        raise HTTPException(status_code=422, detail="compression_ratio must be 4, 16, or 32")

    contents = await file.read()
    try:
        arr = np.load(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse .npy file: {e}")

    if arr.ndim == 3 and arr.shape == (2, 32, 32):
        raw_sample = arr.astype(np.float32)
    elif arr.ndim == 2 and arr.shape == (32, 32) and np.iscomplexobj(arr):
        raw_sample = np.stack([np.real(arr), np.imag(arr)], axis=0).astype(np.float32)
    else:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid tensor shape. Expected (2, 32, 32) float or (32, 32) complex, got {arr.shape}."
        )

    # Auto-detect if array is already in [0, 1] range vs raw physical scale
    is_already_normalized = bool(raw_sample.min() >= 0.0 and raw_sample.max() <= 1.0)

    if auto_normalize and NORM_PARAMS is not None and not is_already_normalized:
        norm_min = float(NORM_PARAMS["min"])
        norm_max = float(NORM_PARAMS["max"])
        sample_np = (raw_sample - norm_min) / (norm_max - norm_min + 1e-10)
        sample_np = np.clip(sample_np, 0.0, 1.0).astype(np.float32)
        should_denorm = True
    else:
        sample_np = raw_sample
        should_denorm = False

    recon_np, inference_ms, nmse_val, bf_gain_val, bf_loss_val, latent_dim = run_model_inference(
        sample_np, compression_ratio
    )

    if should_denorm and NORM_PARAMS is not None:
        norm_min = float(NORM_PARAMS["min"])
        norm_max = float(NORM_PARAMS["max"])
        denorm_recon = recon_np * (norm_max - norm_min) + norm_min
        orig_out_real = raw_sample[0].tolist()
        orig_out_imag = raw_sample[1].tolist()
        recon_out_real = denorm_recon[0].tolist()
        recon_out_imag = denorm_recon[1].tolist()
    else:
        orig_out_real = sample_np[0].tolist()
        orig_out_imag = sample_np[1].tolist()
        recon_out_real = recon_np[0].tolist()
        recon_out_imag = recon_np[1].tolist()

    original_scalars = 2048
    bandwidth_saved = (1.0 - (latent_dim / original_scalars)) * 100.0

    return PredictResponse(
        sample_index=None,
        input_source=f"upload_npy ({file.filename})",
        compression_ratio=compression_ratio,
        bandwidth_saved_percent=round(bandwidth_saved, 2),
        original_shape=[2, 32, 32],
        original_scalars=original_scalars,
        compressed_dim=latent_dim,
        nmse_db=round(nmse_val, 2),
        beamforming_gain_percent=round(bf_gain_val * 100.0, 2),
        beamforming_loss_db=round(bf_loss_val, 2),
        inference_ms=round(inference_ms, 3),
        original_matrix_real=orig_out_real,
        reconstructed_matrix_real=recon_out_real,
        original_matrix_imag=orig_out_imag,
        reconstructed_matrix_imag=recon_out_imag
    )
