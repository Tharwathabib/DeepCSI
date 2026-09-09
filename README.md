# DeepCSI — AI-Native Massive MIMO CSI Compression

> **Target:** FDD Massive MIMO / 5G-Advanced & 6G Research Prototype  
> **Primary Stack:** Python, PyTorch, FastAPI, Streamlit, Plotly, NumPy, SciPy  
> **Sprint Execution:** September 8–12, 2026

---

## Executive Summary

**DeepCSI** is an AI-native Channel State Information (CSI) compression system for Frequency Division Duplex (FDD) massive MIMO wireless networks. 

In FDD systems, the gNodeB requires explicit downlink CSI feedback from the User Equipment (UE) to construct precoding matrices. For a 32-antenna, 256-subcarrier massive MIMO array, raw spatial-frequency CSI consists of $32 \times 256 = 8192$ complex coefficients ($16,384$ float32 values $\approx 64 \text{ KiB}$).

By exploiting angular-delay domain sparsity, truncating delay taps to $32 \times 32$, and deploying CNN-based autoencoders, **DeepCSI** achieves up to **$96.875\%$ scalar dimension reduction** while maintaining high reconstruction fidelity ($\text{NMSE} \le -15 \text{ dB}$).

```text
UE / Channel Estimator
        │
        ▼
Spatial-Frequency CSI H  (32 x 256 complex)
        │
        ▼
2D FFT / DFT  ──>  Angular-Delay Representation
        │
        ▼
Truncation  ──>  32 x 32 Delay-Truncated Matrix (2048 scalars)
        │
        ▼
Real / Imag Split  (2, 32, 32)
        │
        ▼
CNN Encoder  ──>  Latent Vector (512 / 128 / 64 scalars)
        │
        ▼  [ FDD Uplink Feedback ]
        │
Residual Decoder (gNodeB)
        │
        ▼
Reconstructed CSI  ──>  Inverse 2D FFT  ──>  Downlink Precoding
```

---

## Key Performance Indicators

| Compression Ratio (CR) | Retained Scalars | Scalar Reduction | DeepCSI NMSE | DeepCSI MRT Beamforming Gain | DCT Baseline NMSE | DCT MRT Beamforming Gain |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CR = 4** | 512 | **75.00%** | -38.45 dB | **99.98%** (-0.00 dB) | -49.52 dB | 100.00% (-0.00 dB) |
| **CR = 16** | 128 | **93.75%** | -38.17 dB | **99.98%** (-0.00 dB) | -42.52 dB | 99.99% (-0.00 dB) |
| **CR = 32** | 64 | **96.88%** | -37.82 dB | **99.98%** (-0.00 dB) | -40.92 dB | 99.99% (-0.00 dB) |

*Downstream validation note: DeepCSI evaluates both reconstruction error (NMSE) and downstream communication utility via Normalized Maximum Ratio Transmission (MRT) Beamforming Gain $G = \frac{|\hat{\mathbf{h}}^H \mathbf{h}|^2}{\|\hat{\mathbf{h}}\|^2 \|\mathbf{h}\|^2}$. At $\text{CR}=16$ (93.75% scalar reduction), the compressed feedback retains **99.98% of maximum beamforming power**, losing less than $0.01\text{ dB}$ of effective received SNR.*

*Scalar accounting note: DeepCSI compresses the $2 \times 32 \times 32 = 2048$ scalar angular-delay representation down to $128$ float32 latent values at $\text{CR}=16$, representing a $93.75\%$ reduction in uplink feedback scalar dimension.*

---

## Repository Structure

```text
deepcsi-core/
├── README.md
├── requirements.txt
├── .gitignore
├── preflight.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── train.npy         (7000, 2, 32, 32)
│   │   ├── val.npy           (1000, 2, 32, 32)
│   │   ├── test.npy          (2000, 2, 32, 32)
│   │   └── norm_params.json
│   └── generate_data.py
│
├── models/
│   ├── __init__.py
│   ├── csi_autoencoder.py   (Encoder, Decoder, Residual Blocks)
│   ├── train.py             (PyTorch training pipeline)
│   ├── evaluate.py          (Evaluation & metric logging)
│   ├── dct_baseline.py      (2D DCT comparison)
│   └── weights/
│       ├── deepcsi_cr4.pt
│       ├── deepcsi_cr16.pt
│       └── deepcsi_cr32.pt
│
├── backend/
│   ├── __init__.py
│   ├── app.py               (FastAPI inference server)
│   └── test_backend.py      (Endpoint tests)
│
├── frontend/
│   └── app.py               (Streamlit + Plotly interactive UI)
│
├── utils/
│   ├── __init__.py
│   ├── transforms.py        (2D FFT / IFFT, delay truncation)
│   ├── metrics.py           (NMSE in dB calculation)
│   └── seed.py              (Deterministic RNG seed)
│
├── results/
│   ├── metrics.csv
│   ├── nmse_comparison.csv
│   └── figures/
│
└── tests/
    ├── test_data.py
    ├── test_models.py
    └── test_metrics.py
```

---

## Quickstart & Reproducibility Guide

### 1. Environment Setup

```bash
# Create and activate environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset

```bash
python data/generate_data.py
```
*Generates 10,000 samples (7k train / 1k val / 2k test) of $2 \times 32 \times 32$ angular-delay CSI matrices with global min-max normalization.*

### 3. Train DeepCSI Autoencoders

```bash
# Train CR=4, CR=16, and CR=32 models
python models/train.py --compression-ratio 4 --epochs 25
python models/train.py --compression-ratio 16 --epochs 25
python models/train.py --compression-ratio 32 --epochs 25
```

### 4. Run Model Evaluation & DCT Baseline

```bash
python models/evaluate.py
```
*Evaluates trained autoencoders against 2D DCT baseline across all compression ratios and exports metrics to `results/metrics.csv`.*

### 5. Launch FastAPI Backend

```bash
uvicorn backend.app:app --reload --port 8000
```
*API available at `http://localhost:8000`. Test `/health` and `/predict` endpoints.*

### 6. Launch Interactive Streamlit Dashboard

```bash
streamlit run frontend/app.py
```
*Opens interactive web UI for real-time model inference, heatmaps, error visualization, and metric benchmarking.*

### 7. Execute System Preflight Verification

```bash
python preflight.py
```
*Runs comprehensive end-to-end sanity check. Expect output: `DEEPCSI PREFLIGHT: PASS`.*

---

## Team & Attribution

- **Mohammed Elfeky** — SPOC / Technical Coordination & Evaluation Pipeline
- **Adel Hazem** — Channel Data Generation & Domain Sanity Checks
- **Karim ElBatran** — Preprocessing, Normalization & Dataset Splitting
- **Ziaad** — FDD/TDD Domain Research & Presentation Support
- **Mano** — CNN Encoder Architecture & Compression Optimization
- **Mohamed Salama** — Residual Decoder Architecture & Model Tuning
- **Hanin Elsherif** — Training Pipeline & Hyperparameter Tuning
- **Maria Ramy** — FastAPI Backend API Implementation & Endpoint Contracts
- **Yehia** — Backend Integration & Startup Validation
- **Tharwat** — Streamlit Dashboard & Plotly Heatmap Visualizations
- **Hazem Mohamed** — Frontend/Backend Integration & UI Testing
