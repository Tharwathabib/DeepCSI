# DeepCSI — AI-Native Massive MIMO CSI Compression

> **Target:** FDD Massive MIMO / 5G-Advanced & 6G Research Prototype  
> **Primary Stack:** Python, PyTorch, FastAPI, Streamlit, Plotly, NumPy, SciPy  
> **Sprint Execution:** September 8–12, 2026

---

## Executive Summary

**DeepCSI** is an AI-native Channel State Information (CSI) compression system for Frequency Division Duplex (FDD) massive MIMO wireless networks. 

In FDD systems, the gNodeB requires explicit downlink CSI feedback from the User Equipment (UE) to construct precoding matrices. For a 32-antenna, 256-subcarrier massive MIMO array, raw spatial-frequency CSI consists of $32 \times 256 = 8192$ complex coefficients ($16,384$ float32 values $\approx 64 \text{ KiB}$).

By exploiting angular-delay domain sparsity, truncating delay taps to $32 \times 32$, and deploying CNN-based autoencoders, **DeepCSI** achieves up to **$96.875\%$ scalar dimension reduction**, outperforming a classical 2D-DCT baseline at every compression ratio tested by an increasing margin as the feedback budget shrinks.

The original $-15\text{ dB}$ NMSE target is **met on both tracks**: $-15.14\text{ dB}$ at CR=4 on real ray-traced DeepMIMO data, and $-17.11\text{ dB}$ on the synthetic generator. DeepCSI also beats **PCA/KLT — the optimal linear compressor — at every compression ratio on synthetic**, and at CR=16 and CR=32 on DeepMIMO. See Key Performance Indicators below.

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

Measured on a 10,000-sample synthetic test set, 35,000 training samples,
60 epochs per model. Reproduce with:
```bash
python data/generate_data.py --samples 50000 --output-dir data/processed_big
python models/train.py --data-dir data/processed_big -cr {4,16,32} \
    --epochs 60 --weight-decay 2e-6 --output-dir models/weights_big --results-dir results_big
python models/evaluate.py --data-dir data/processed_big \
    --weights-dir models/weights_big --results-dir results_big
```

| Compression Ratio (CR) | Retained | Scalar Reduction | **DeepCSI NMSE** | DeepCSI MRT Gain | PCA | DCT |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CR = 4** | 512 | **75.00%** | **-17.11 dB** | **98.27%** | -14.03 dB | -10.97 dB |
| **CR = 16** | 128 | **93.75%** | **-10.97 dB** | **92.69%** | -4.37 dB | -4.08 dB |
| **CR = 32** | 64 | **96.88%** | **-7.53 dB** | **83.71%** | -2.50 dB | -2.47 dB |

**DeepCSI beats both baselines at every compression ratio** — +3.1 / +6.6 /
+5.0 dB over PCA, the optimal *linear* compressor, and +6.1 / +6.9 / +5.1 dB
over DCT. In beamforming terms that is 93% of maximum MRT power at CR=16 where
PCA retains 63%.

#### Getting the weight decay right was worth 5.9 dB

The earlier figures on this track were **-11.22 / -8.48 / -5.31 dB**. Two
changes account for the difference, and the second is not obvious:

1. **5× more training data** (7,000 → 35,000 samples). The old run had a 3.63 dB
   train/val gap — it was memorising.
2. **Scaling weight decay with dataset size.** `torch.optim.Adam` applies decay
   *per optimizer step*, so 5× the data means 5× the steps per epoch and 5× the
   effective regularisation. Keeping `1e-5` on the larger set made results far
   *worse* (-3 dB, unstable); the scale-corrected `2e-6` gave -17.11 dB.

Measured, CR=4, isolating one variable at a time:

| train samples | weight decay | NMSE |
| ---: | ---: | ---: |
| 7,000 | 1e-5 | -11.22 dB |
| 7,000 | 0 | -9.74 dB |
| 35,000 | 0 | ~-10.6 dB |
| 35,000 | 1e-5 | unstable, ~-3 dB |
| **35,000** | **2e-6** | **-17.11 dB** |

Neither change works without the other. More data alone gains 0.9 dB; the right
decay for that data size turns it into 5.9 dB. `--optimizer adamw` sidesteps the
coupling entirely and is the cleaner long-term fix — see `models/train.py`.

### Results on real ray-traced data (DeepMIMO)

The headline numbers. Measured on a 12,000-sample held-out test set from
DeepMIMO scenario O1 at 3.5 GHz, pooled across four base-station / user-grid
pairs, 42,000 training samples, 60 epochs per model.

Reproduce with:
```bash
python data/prepare_deepmimo.py --output-dir data/processed_deepmimo
python models/train.py --data-dir data/processed_deepmimo -cr {4,16,32} \
    --epochs 60 --loss nmse --weight-decay 0 --output-dir models/weights_deepmimo
python models/evaluate.py --data-dir data/processed_deepmimo \
    --weights-dir models/weights_deepmimo --results-dir results_deepmimo
```

To run the live demo against these artifacts instead of the synthetic ones,
point the two environment variables at them — no code change needed:
```bash
DATA_DIR=data/processed_deepmimo WEIGHTS_DIR=models/weights_deepmimo python preflight.py
DATA_DIR=data/processed_deepmimo WEIGHTS_DIR=models/weights_deepmimo \
    python -m uvicorn backend.app:app --port 8001
```

| CR | Retained | **DeepCSI** | PCA / KLT | 2D DCT | ρ | MRT gain |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4** | 512 | **-15.14 dB** | **-17.00 dB** | -6.37 dB | 0.9859 | 97.20% |
| **16** | 128 | **-12.35 dB** | -4.43 dB | -1.80 dB | 0.9719 | 94.48% |
| **32** | 64 | **-9.87 dB** | -2.36 dB | -0.95 dB | 0.9497 | 90.25% |

**PCA (KLT) is the baseline that matters**, not DCT. A linear autoencoder with
$k$ components is the *optimal* linear compressor under MSE, and unlike DCT it
learns its basis from training data and sends a fixed $k$ numbers — exactly the
payload shape of the autoencoder's latent vector. DCT is credited with $K$
coefficients without paying the ~11 bits each to say *which* $K$ it kept.

Against that stronger bar, DeepCSI wins by **7.9 dB at CR=16** and **7.5 dB at
CR=32** — and **loses by 1.9 dB at CR=4**. The loss is structural rather than a
tuning failure: this dataset's angular-delay tensor has an intrinsic rank of
291 (90% of variance) out of 2048, so a latent of 512 is *not* a bottleneck and
PCA is performing near-lossless linear reconstruction. Where the latent
genuinely binds, the nonlinear model is far ahead. That crossover is the honest
version of the "learned compression wins when the budget is tight" claim.

> **Not comparable to published CsiNet/CRNet numbers.** Those are measured on
> COST2100 indoor; this is DeepMIMO O1 outdoor. The CR=16 and CR=32 figures
> here exceed the published ones, but on a different and evidently easier
> benchmark at high compression — it is a different channel model, not a better
> model. COST2100 is a paid dataset and is out of scope for this project, so a
> like-for-like comparison to the literature is a limitation we accept and
> state, not outstanding work.

### Quantized feedback — the actual bandwidth claim

Scalar reduction is not compression. The uplink carries **bits**, and a latent
of 512 float32 values is 16,384 of them. `models/quantize_eval.py` quantizes the
latent uniformly and measures what each bit budget actually buys. The quantizer
range is fitted on training latents and ships with the model, so the payload is
exactly `latent_dim × bits` with no side information.

A full-resolution CSI report is **65,536 bits** (2×32×32 float32).

| Config | Feedback | vs raw | NMSE | MRT gain |
| :--- | ---: | ---: | ---: | ---: |
| CR=32 @ 4-bit | 256 bits | **256×** | -7.97 dB | 84.81% |
| CR=16 @ 4-bit | 512 bits | **128×** | -11.01 dB | 92.38% |
| **CR=16 @ 6-bit** | **768 bits** | **85×** | **-12.26 dB** | **94.37%** |
| CR=16 @ 8-bit | 1024 bits | 64× | -12.34 dB | 94.48% |
| CR=4 @ 4-bit | 2048 bits | 32× | -10.95 dB | 92.21% |

**The scalar compression ratio is the wrong axis.** At any fixed bit budget a
smaller latent at higher precision beats a larger latent at lower precision:
CR=4 spends 2048 bits to score -10.95 dB, while CR=16 reaches -12.34 dB on
*half* that. The sweet spot is **CR=16 at 6 bits — 85× compression for 0.10 dB**
of loss against unquantized float32.

Reproduce: `python models/quantize_eval.py --data-dir data/processed_deepmimo --weights-dir models/weights_deepmimo`

#### The DCT baseline, counted fairly

DCT picks its $K$ coefficients per sample by magnitude, so the receiver has to
be told *which* ones — a cost the original comparison ignored. A fixed-size
latent pays nothing for this, because its meaning is positional. Charging DCT
the information-theoretic floor $\log_2\binom{2048}{K}$ (generous; a real
encoder does worse):

| Method | Values @6-bit | Index | **Total** | NMSE |
| :--- | ---: | ---: | ---: | ---: |
| DCT K=128 | 768 | 686 | **1454 bits** | -1.80 dB |
| DCT K=64 | 384 | 407 | **791 bits** | -0.95 dB |
| **DeepCSI CR=16 @6-bit** | 768 | 0 | **768 bits** | **-12.26 dB** |

At a matched budget — 791 bits for DCT against 768 for DeepCSI — the learned
codec is **11.3 dB ahead**. The index overhead is why: at $K=64$ it is more than
half of DCT's payload.

*Metric definitions: NMSE is $10\log_{10}\left(\mathbb{E}\left[\|\mathbf{h}-\hat{\mathbf{h}}\|^2 / \|\mathbf{h}\|^2\right]\right)$ and MRT beamforming gain is $G = \frac{|\hat{\mathbf{h}}^H \mathbf{h}|^2}{\|\hat{\mathbf{h}}\|^2 \|\mathbf{h}\|^2}$. Both are computed on the **de-offset complex channel**, i.e. after removing the constant that maps zero to 0.5 in the normalized tensor. Measuring them on the raw $[0,1]$ tensor instead inflates NMSE by roughly 35 dB and saturates $G$ near 100% for any model — see `REAL_DATA.md`.*

> **Scope:** these numbers come from the synthetic 3GPP-*inspired* generator, not
> a validated channel model, so they are **not comparable to published results**
> and should not be quoted as such. The COST2100 track in `REAL_DATA.md` exists
> to produce numbers that are; it is blocked only on the dataset download.

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
