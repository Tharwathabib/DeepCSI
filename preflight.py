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

from utils.metrics import nmse_db_aggregate, beamforming_gain_torch
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
    # DATA_DIR / WEIGHTS_DIR match the environment variables the backend reads,
    # so the same settings drive preflight and the running API.
    data_dir = Path(os.getenv("DATA_DIR", "data/processed"))
    print(f"       Info: data dir = {data_dir}")
    train_p = data_dir / "train.npy"
    val_p = data_dir / "val.npy"
    test_p = data_dir / "test.npy"
    norm_p = data_dir / "norm_params.json"

    check("Dataset files exist", train_p.exists() and val_p.exists() and test_p.exists() and norm_p.exists(),
          "Missing dataset. Run one of:\n"
          "         python data/generate_data.py                  (synthetic)\n"
          "         python data/prepare_cost2100.py --mat-dir ...  (real COST2100)")

    train_data = np.load(train_p)
    val_data = np.load(val_p)
    test_data = np.load(test_p)

    # Sample counts differ between the synthetic set and the COST2100 splits, so
    # only the per-sample geometry is fixed.
    for name, arr in (("Train", train_data), ("Val", val_data), ("Test", test_data)):
        check(f"{name} shape is (N, 2, 32, 32) [N={len(arr)}]",
              arr.ndim == 4 and arr.shape[1:] == (2, 32, 32))
    check("Data dtype is float32", train_data.dtype == np.float32)

    with open(norm_p, "r") as f:
        norm_params = json.load(f)
    check("Normalization metadata present", "min" in norm_params and "max" in norm_params)
    print(f"       Info: source = {norm_params.get('source', 'synthetic')} | "
          f"scheme = {norm_params.get('scheme', 'minmax')}")

    # 3. Model Weights & Inference
    weights_dir = Path(os.getenv("WEIGHTS_DIR", "models/weights"))
    test_tensor = torch.from_numpy(test_data[:10]).to(device)

    for cr in [4, 16, 32]:
        w_file = weights_dir / f"deepcsi_cr{cr}.pt"
        check(f"CR={cr} weight file exists ({w_file.name})", w_file.exists(),
              f"Missing model weights. Run: python models/train.py --compression-ratio {cr}")

        try:
            checkpoint = torch.load(w_file, map_location=device, weights_only=False)
            refine_widths = tuple(checkpoint.get("refine_widths", (8, 16)))
            model = CSIAutoencoder(compression_ratio=cr, refine_widths=refine_widths).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            # Weights trained on one dataset and tested against another produce
            # nonsense; catch it here rather than during the demo.
            ckpt_source = checkpoint.get("data_source", "unknown")
            data_source = norm_params.get("source", "unknown")
            if ckpt_source != "unknown" and data_source != "unknown":
                check(f"CR={cr} weights match dataset ({ckpt_source})",
                      ckpt_source == data_source,
                      f"Weights trained on '{ckpt_source}' but data is '{data_source}'.")

            with torch.no_grad():
                recon, latent = model(test_tensor)
                # Measured on the de-offset channel; without norm_params these
                # are inflated by ~35 dB and rho saturates near 1.0.
                nmse_val = nmse_db_aggregate(recon, test_tensor, norm_params).item()
                bf_val = beamforming_gain_torch(
                    recon, test_tensor, norm_params=norm_params
                ).item()

            check(f"CR={cr} model forward pass & finite NMSE ({nmse_val:.2f} dB)", np.isfinite(nmse_val))
            check(f"CR={cr} downstream MRT beamforming gain valid ({bf_val*100:.2f}%)", 0.0 <= bf_val <= 1.0)
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
