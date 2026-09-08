import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.metrics import nmse_db
from models.csi_autoencoder import CSIAutoencoder


def check(description: str, condition: bool, err_msg: str = ""):
    status_str = "PASS" if condition else "FAIL"
    symbol = "[OK]" if condition else "[FAIL]"
    print(f"[{status_str}] {symbol} {description}")
    if not condition:
        print(f"       ERROR: {err_msg}")
        sys.exit(1)


def main():
    print("===========================================================")
    print("              DEEPCSI SYSTEM PREFLIGHT CHECK              ")
    print("===========================================================\n")

    # 1. Environment & Dependencies
    check("Python version >= 3.8", sys.version_info >= (3, 8))
    
    try:
        import torch
        import numpy
        import scipy
        import fastapi
        import uvicorn
        import streamlit
        import plotly
        import requests
        check("Required Python packages installed", True)
    except ImportError as e:
        check("Required Python packages installed", False, str(e))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"       Info: PyTorch {torch.__version__} | Device: {device}")

    # 2. Dataset Files & Shapes
    data_dir = Path("data/processed")
    train_p = data_dir / "train.npy"
    val_p = data_dir / "val.npy"
    test_p = data_dir / "test.npy"
    norm_p = data_dir / "norm_params.json"

    check("Dataset files exist", train_p.exists() and val_p.exists() and test_p.exists() and norm_p.exists(),
          "Missing dataset. Run: python data/generate_data.py")

    train_data = np.load(train_p)
    val_data = np.load(val_p)
    test_data = np.load(test_p)

    check("Train shape is (7000, 2, 32, 32)", train_data.shape == (7000, 2, 32, 32))
    check("Val shape is (1000, 2, 32, 32)", val_data.shape == (1000, 2, 32, 32))
    check("Test shape is (2000, 2, 32, 32)", test_data.shape == (2000, 2, 32, 32))
    check("Data dtype is float32", train_data.dtype == np.float32)

    with open(norm_p, "r") as f:
        norm_params = json.load(f)
    check("Normalization metadata present", "min" in norm_params and "max" in norm_params)

    # 3. Model Weights & Inference
    weights_dir = Path("models/weights")
    test_tensor = torch.from_numpy(test_data[:10]).to(device)

    for cr in [4, 16, 32]:
        w_file = weights_dir / f"deepcsi_cr{cr}.pt"
        check(f"CR={cr} weight file exists ({w_file.name})", w_file.exists(),
              f"Missing model weights. Run: python models/train.py --compression-ratio {cr}")

        try:
            model = CSIAutoencoder(compression_ratio=cr).to(device)
            checkpoint = torch.load(w_file, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            with torch.no_grad():
                recon, latent = model(test_tensor)
                nmse_val = nmse_db(recon, test_tensor).item()

            check(f"CR={cr} model forward pass & finite NMSE ({nmse_val:.2f} dB)", np.isfinite(nmse_val))
            check(f"CR={cr} latent shape matches expected ({2048//cr})", latent.shape == (10, 2048 // cr))
        except Exception as e:
            check(f"CR={cr} model verification", False, str(e))

    # 4. Optional Backend API Reachability
    try:
        resp = requests.get("http://localhost:8000/health", timeout=1)
        if resp.status_code == 200:
            print("       Info: FastAPI backend is running and reachable on http://localhost:8000")
        else:
            print("       Info: FastAPI backend returned non-200 status (Server may be starting).")
    except Exception:
        print("       Info: FastAPI backend not running locally (Will be started for demo).")

    print("\n===========================================================")
    print("               DEEPCSI PREFLIGHT: PASS                     ")
    print("===========================================================")


if __name__ == "__main__":
    main()
