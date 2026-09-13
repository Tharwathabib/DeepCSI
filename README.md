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
60 epochs per model. **This is the default track** — `preflight.py`, the API and
the dashboard all read it with no environment variables set. Reproduce with:
```bash
python data/generate_data.py --samples 50000
python models/train.py -cr {4,16,32} --epochs 60 --weight-decay 2e-6
python models/evaluate.py
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
tuning failure, and the eigenspectrum says exactly why. The variance a
$k$-component linear basis must discard sets a floor that *no* linear
compressor can beat, $10\log_{10}\left(\sum_{i>k}\lambda_i / \sum_i \lambda_i\right)$,
measured on the test set's own covariance:

| Latent | Variance in top $k$ | Linear floor | PCA measured | DeepCSI |
| :--- | ---: | ---: | ---: | ---: |
| 512 (CR=4) | 98.38% | **-17.90 dB** | -17.00 dB | -15.14 dB |
| 128 (CR=16) | 64.09% | -4.45 dB | -4.43 dB | **-12.35 dB** |
| 64 (CR=32) | 42.26% | -2.39 dB | -2.36 dB | **-9.87 dB** |

At CR=4 a linear basis still has 98.38% of the energy to work with, so PCA is
doing near-lossless reconstruction and there is almost nothing for a nonlinear
model to add — it is *not a bottleneck*. At CR=16 and CR=32 the floor collapses
to -4.45 and -2.39 dB, and PCA lands within **0.03 dB** of it — it is essentially
optimal, exactly as the theory says. DeepCSI goes **7.9 dB and 7.5 dB below** a
bound that binds every linear method, reconstructing detail no linear basis of
that width can represent at all. That crossover is the honest version of the
"learned compression wins when the budget is tight" claim.

Reproduce: `python models/eigenspectrum.py --data-dir data/processed_deepmimo --results-dir results_deepmimo`
(`results_deepmimo/eigenspectrum.csv`).

> **Not comparable to published CsiNet/CRNet numbers.** Those are measured on a
> different indoor benchmark; this is DeepMIMO O1 outdoor. The CR=16 and CR=32
> figures here exceed the published ones, but on a different and evidently
> easier benchmark at high compression — a different channel model, not a better
> model. Running on the literature's own benchmark is out of scope, so this is a
> limitation we accept and state, not outstanding work.

### The two tracks side by side

Both tracks run the **same architecture** (`csinet`, `refine_widths 8 16`,
60 epochs) through the same evaluation code. They do *not* share a training
recipe — synthetic uses `--weight-decay 2e-6` with the default MSE loss, DeepMIMO
uses `--loss nmse --weight-decay 0` — so read the columns as *two measured
configurations*, not as a controlled experiment isolating the data:

| | | Synthetic (3GPP-inspired) | DeepMIMO O1 (ray-traced) | Δ |
| :--- | :--- | ---: | ---: | ---: |
| **NMSE** | CR=4 | **-17.11 dB** | -15.14 dB | +1.97 |
| | CR=16 | -10.97 dB | **-12.35 dB** | -1.38 |
| | CR=32 | -7.53 dB | **-9.87 dB** | -2.34 |
| **MRT gain** | CR=4 | **98.27%** | 97.20% | |
| | CR=16 | 92.69% | **94.48%** | |
| | CR=32 | 83.71% | **90.25%** | |
| **vs PCA** | CR=4 | **+3.08 dB** | -1.86 dB | |
| | CR=16 | +6.60 dB | **+7.92 dB** | |
| | CR=32 | +5.03 dB | **+7.51 dB** | |
| **Spread** | CR=4 → CR=32 | 9.58 dB | **5.27 dB** | |
| **Linear floor** | 512 (CR=4) | -16.61 dB | **-17.90 dB** | |
| | 64 (CR=32) | **-3.44 dB** | -2.39 dB | |
| **Intrinsic rank** | 90% of variance | 266 / 2048 | 298 / 2048 | |
| **Test samples** | | 10,000 | 12,000 | |

Three things worth saying out loud — each of which survives the recipe
difference, because none of them turns on a sub-dB gap:

1. **The real-data track is harder at CR=4 and easier at CR=32.** Synthetic wins
   the headline number by 2 dB, but loses by 2.3 dB where compression actually
   bites. Quoting only CR=4 would flatter the synthetic generator.
2. **The real-data curve is flatter** — 5.27 dB across the CR range against
   9.58 dB. On the configurations shipped here, the ray-traced channels degrade
   more gracefully under aggressive compression than the synthetic generator
   suggests: the more favourable result for the system claim, and the less
   favourable one for the generator's realism.
3. **DeepCSI's margin over PCA is *larger* on real data** where the latent
   binds (+7.9 / +7.5 dB against +6.6 / +5.0 dB). The one place it loses is
   CR=4 on DeepMIMO, for the eigenspectrum reason above. The synthetic
   generator understates the case for learned compression at high CR.

> **Quote the DeepMIMO numbers.** The synthetic track is a generator, not a
> validated channel model; it exists to exercise the pipeline and to isolate
> training effects (see the weight-decay study above). The real-data figures are
> the ones that belong in a claim.

`results/eigenspectrum.csv` and `results_deepmimo/eigenspectrum.csv` back the
rank and floor rows; `results/metrics.csv` and `results_deepmimo/metrics.csv`
back everything else.

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
> and should not be quoted as such. The DeepMIMO figures above are the ones
> measured on real ray-traced propagation — quote those.

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
│   ├── processed/                    (synthetic, the default track)
│   │   ├── train.npy         (35000, 2, 32, 32)
│   │   ├── val.npy           (5000, 2, 32, 32)
│   │   ├── test.npy          (10000, 2, 32, 32)
│   │   └── norm_params.json
│   ├── processed_deepmimo/           (real ray-traced, same layout)
│   ├── generate_data.py
│   └── prepare_deepmimo.py  (DeepMIMO O1 -> angular-delay tensors)
│
├── models/
│   ├── __init__.py
│   ├── csi_autoencoder.py   (Encoder, Decoder, Residual Blocks)
│   ├── train.py             (PyTorch training pipeline)
│   ├── evaluate.py          (Evaluation & metric logging)
│   ├── dct_baseline.py      (2D DCT comparison)
│   ├── pca_baseline.py      (PCA/KLT — the optimal linear compressor)
│   ├── eigenspectrum.py     (Linear NMSE floor per latent size)
│   ├── quantize_eval.py     (Feedback bits vs NMSE sweep)
│   ├── ablation.py          (Generalisation-option sweep)
│   ├── weights/
│   │   ├── deepcsi_cr4.pt
│   │   ├── deepcsi_cr16.pt
│   │   └── deepcsi_cr32.pt
│   └── weights_deepmimo/    (same three, real-data track)
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
├── results/                 (synthetic track — CSVs are tracked in git)
│   ├── metrics.csv
│   ├── nmse_comparison.csv
│   ├── beamforming_comparison.csv
│   ├── eigenspectrum.csv
│   ├── pca_baseline_metrics.csv
│   ├── dct_baseline_metrics.csv
│   ├── ablation.csv
│   ├── training_log_cr{4,16,32}.csv
│   └── figures/
│
├── results_deepmimo/        (real-data track; adds quantization.csv, no ablation)
│
├── REAL_DATA.md             (methodology, corrections, withdrawn claims)
│
└── tests/
    ├── test_data.py
    ├── test_models.py
    ├── test_metrics.py
    ├── test_transforms.py
    ├── test_eigenspectrum.py
    ├── test_pca_baseline.py
    ├── test_quantize.py
    ├── test_prepare_deepmimo.py
    ├── test_split_proportions.py
    ├── test_review_regressions.py
    └── test_silent_failures.py
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
python data/generate_data.py --samples 50000
```
*Generates 50,000 samples (35k train / 5k val / 10k test) of $2 \times 32 \times 32$
angular-delay CSI matrices, scaled to $[0,1]$ about a 0.5 complex zero point with
the robust quantile scale (`--norm-mode robust`, the default). Takes ~45 s and is
seeded: the same command reproduces the arrays byte-for-byte.*

> **Pass `--samples 50000`.** The bare command builds a 10,000-sample set, which
> is a *different dataset* from the one every number in this README was measured
> on — and it trains to about -9.7 dB at CR=4 rather than -17.11 dB. The sample
> count is not a free parameter here; see *Getting the weight decay right* above.

### 3. Train DeepCSI Autoencoders — *optional, the weights are in the repo*

The three checkpoints per track are tracked in git (11 MB each track), so after
step 2 you can skip straight to step 4 and `preflight.py` will pass. Retrain only
if you are reproducing the numbers or changing the model.

```bash
# Train CR=4, CR=16, and CR=32 models
python models/train.py --compression-ratio 4  --epochs 60 --weight-decay 2e-6
python models/train.py --compression-ratio 16 --epochs 60 --weight-decay 2e-6
python models/train.py --compression-ratio 32 --epochs 60 --weight-decay 2e-6
```
*`--weight-decay 2e-6` is required to reproduce the published figures: the
default of 0 costs ~6.5 dB at CR=4, and the older `1e-5` is unstable on 35k
samples. Roughly 35 min per model on a modern laptop CPU (no GPU needed);
the per-epoch loss and NMSE are written to `results/training_log_cr{4,16,32}.csv`.*

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
