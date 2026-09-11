# DeepCSI — real-data track (`karim/real-data`)

Replaces the synthetic CSI surrogate with the **COST2100** benchmark released
with CsiNet, and fixes three measurement bugs that made the previous results
unciteable. Owned by Karim (preprocessing / splitting / normalisation / data
validation).

The synthetic path is untouched — `data/generate_data.py` still works exactly as
before, and `main` is unaffected.

---

## Why the old numbers were wrong

The pipeline reported **−38 dB NMSE** and **99.98% beamforming gain at every
compression ratio**. Two things should have raised alarms: NMSE barely moved
across an 8× change in latent size, and a plain DCT baseline appeared to beat
published neural methods by 30 dB.

Neither figure was a result. All three causes are measurement bugs, and all
three would have survived the switch to real data.

### 1. NMSE was measured on the raw `[0,1]` tensor — worth ~38 dB

A constant offset cancels in the squared error but **not** in the signal power.
On sparse normalised CSI the 0.5 DC term dominates `sum(x²)`:

```
mean power on raw [0,1] tensor : 512.2205
mean power on de-offset channel:   0.1579
denominator inflation          :   35.11 dB

NMSE on raw [0,1] tensor       :  -47.96 dB   <- what this repo reported
NMSE on de-offset channel      :   -9.30 dB   <- what CsiNet reports
```

This is arithmetic, not data. The old code would have reported ≈−38 dB on real
COST2100 data too. CsiNet computes NMSE on `(x_real − 0.5) + 1j(x_imag − 0.5)`,
which is what `nmse_db_aggregate(..., norm_params)` now does.

### 2. ρ / beamforming gain had the same offset problem

The shared DC component dominated the inner product, pinning ρ near 1.0
regardless of quality. A model emitting a **constant** — carrying zero channel
information — scored **99.94%**:

| reconstruction | ρ (old) | ρ (fixed) |
|---|---:|---:|
| perfect | 100.00% | 100.00% |
| light noise (σ=0.01) | 99.98% | 86.70% |
| heavy noise (σ=0.10) | 98.07% | 19.06% |
| **constant 0.5 (predicts nothing)** | **99.94%** | **0.00%** |
| pure noise, no signal | 98.05% | 1.81% |

### 3. NMSE used the wrong averaging convention — worth ~10 dB

The code computed `mean(10·log10(ratio))` (mean-of-logs). CsiNet, CRNet and the
rest of the literature use `10·log10(mean(ratio))` (log-of-mean). By Jensen's
inequality the old form is optimistically biased; measured here: **10.67 dB**.

### Knock-on effect

Correcting the denominator also fixes the DCT baseline, which was inflated the
same way — from an implausible −46.20 dB to **−10.80 dB** at CR=4. It no longer
beats published neural methods, which is the expected relationship.

---

## 4. The synthetic generator's normalisation prevented learning

Once the metrics were honest, the synthetic path scored **+0.72 dB at CR=4** —
worse than transmitting nothing. Not a metric artifact this time; a real defect
in `data/generate_data.py`.

Global min-max took its range from a handful of rare cluster peaks
(raw min/max `−3.43 / +4.15`), squashing everything else:

```
98.1% of pixels within +/-0.01 of the zero point
std = 0.0066  ->  the data used ~4% of the [0,1] range
```

The autoencoder minimised MSE by collapsing to the mean — its reconstruction std
was 27% of the target's. Meanwhile a DCT baseline needing no training at all
scored −10.73 dB on the same data, proving the data was compressible and the
representation was the problem.

`normalize_robust()` now sets the scale from a high quantile of `|h|` and clips
the outliers beyond it, which is how the released COST2100 data is prepared:

```
x = clip(h / (2S) + 0.5, 0, 1),   S = quantile(|h|, 0.9995)
```

std rose 0.0066 → 0.0239 (3.6×) while clipping only 0.05% of values, and CR=4
went **+0.72 dB → −11.36 dB**. The recorded `min`/`max` of `−S`/`+S` make the
existing affine denormalisation reduce to exactly `(x − 0.5) · 2S`, so the
metrics and backend treat synthetic and COST2100 data identically.

`--norm-mode minmax` still reproduces the old behaviour if anyone needs it.

---

---

## The DeepMIMO track (what actually shipped)

The COST2100 download is blocked (Google Drive folder-mode `gdown` returns
nothing for files over ~100 MB). **DeepMIMO** replaced it and is arguably better:
ray-traced from real building geometry rather than a statistical model, and it
downloads from Python with no quota to hit.

```python
import deepmimo as dm
dm.download('o1_3p5')          # O1 urban canyon, 3.5 GHz, 2.27 GB
```

Unlike COST2100, DeepMIMO ships raw ray-tracing rather than pre-made
angular-delay tensors, so it forces the **full pipeline** into the execution
path for the first time:

```
ray tracing -> H(32 antennas x 256 subcarriers)
            -> 2D DFT (FFT over antennas, IFFT over subcarriers)
            -> truncate to 32 delay taps
            -> per-sample [0,1] with 0.5 as complex zero
            -> shuffle -> 70/10/20 split
```

That is exactly the flow the README has always advertised and nothing
implemented. Running it immediately surfaced a bug — see "FFT direction" below.

### Measured, not assumed

**Delay truncation is justified.** Truncating 256 subcarriers to 32 delay taps
retains **96.49%** of channel energy (per-user p5 91.04%, median 97.69%). The
README asserted this was safe from day one with no measurement behind it.

**FFT direction was wrong.** `spatial_frequency_to_angular_delay` used `fft2`.
A path of delay tau has response `exp(-2j*pi*tau*k/Nc)`, so a *forward* DFT
along subcarriers places it at index `Nc-tau` — the far end. Truncating to the
first 32 taps would have kept noise and discarded the channel:

| | energy retained by 32-of-256 truncation |
|---|---|
| before (`fft2`) | 75.8% — split between index 0 and index 255 |
| after (`ifft` on subcarriers) | 100.0% |

Invisible because the functions were dead code with no test. Now covered by
`tests/test_transforms.py`.

### Data-engineering decisions

| Decision | Why |
|---|---|
| Sample users **uniformly**, not a contiguous slice | DeepMIMO orders receivers by grid position. The first 2000 span **5.8 dB** of path loss; a uniform sample spans **48.5 dB**. |
| **Shuffle before splitting** | Otherwise val/test measure generalisation across geography, not compression. |
| **Drop** zero-power users | Ray tracing returns exactly zero where no path exists. These divide by zero in per-sample normalisation and vanish the NMSE denominator. They are users the BS cannot serve, not outliers to clip. |
| **Per-sample peak** normalisation | See below. |

**Why peak scaling despite its low spread.** Peak normalisation leaves std ≈
0.018, and quantile/RMS scales spread the data much further — but they clip, and
the clipped values are the peaks that carry the energy. Measured round-trip NMSE
floors, before any model is involved:

| scheme | clipped | NMSE floor |
|---|---:|---:|
| **peak** | 0.000% | **lossless** |
| q=0.999 | 0.146% | −6.39 dB |
| 4·rms | 0.335% | −3.43 dB |
| q=0.99 | 1.025% | −1.62 dB |

A −6.39 dB floor caps the result below what the model already reaches. Peak is
the only lossless option, and low spread is the price.

This is **not** the synthetic normalisation bug repeating. There a single
*global* scale set by dataset-wide outliers squashed every sample into ~4% of
the range. Here each sample sets its own scale and reaches 0 and 1 at its own
peak; the low std reflects genuine channel sparsity.

```bash
python data/prepare_deepmimo.py --samples 30000 --output-dir data/processed_deepmimo
python models/train.py --data-dir data/processed_deepmimo --compression-ratio 4
```

---

## Model selection: what the ablation actually said

The training logs show the models **overfit** at every compression ratio, and
the gap widens with capacity (−3.63 dB at CR=4, −2.16 at CR=32). So a bigger or
deeper model is the wrong move; four changes aimed at generalisation were tried
instead, each behind a flag defaulting to existing behaviour.

`python models/ablation.py --compression-ratio 16 --epochs 50`

| variant | val NMSE | train/val gap | vs baseline |
|---|---:|---:|---:|
| **augment** | **−7.85 dB** | **−0.06** | **−0.22** ✅ |
| baseline | −7.63 dB | −0.62 | — |
| crnet_encoder | −7.05 dB | −0.30 | +0.58 ❌ |
| cosine_lr | −6.05 dB | −1.05 | +1.58 ❌ |
| nmse_loss | −5.20 dB | −0.23 | +2.43 ❌ |
| all_combined | −2.06 dB | +0.02 | **+5.57** ❌❌ |

**Only augmentation helped**, and it worked for the predicted reason: it
collapsed the generalisation gap from −0.62 to −0.06 dB. Three plausible,
literature-backed ideas made things worse, and **shipping all four together
would have cost 5.57 dB** while giving no way to identify the culprit.

Caveat on the sweep: at 50 epochs the baseline gap is only −0.62 dB, whereas the
overfitting that motivated these changes appeared at 90 epochs (−2.28 dB). The
generalisation-oriented options are therefore being judged in a regime that is
not yet strongly overfitting, and `augment` in particular may look better still
at full length.

The recipe in use is **baseline + `--augment`**.

---

## Choosing the DeepMIMO base station and user grid

Which transmitter and receiver grid you pick decides whether the data is
learnable at all, and this is not obvious up front.

Median angular-delay energy concentration over 1500 users:

| config | top-1 bin | top-10 bins | bins for 90% | users with a path |
|---|---:|---:|---:|---:|
| TX5 / RX0 | 54.9% | 93.4% | **7** | 100% |
| TX15 / RX0 | 39.0% | 87.0% | 14 | 42% |
| **TX5 / RX2** | **35.9%** | **84.8%** | **18** | 61% |
| *synthetic reference* | *24.5%* | *84.8%* | *13* | — |

RX0 is a line-of-sight-dominated street grid: **half the channel energy sits in
a single bin out of 1024**. Training on it collapses to the mean at ~0 dB —
the model cannot locate one needle in a haystack from an MSE gradient, which is
the same symptom the synthetic min-max bug produced for a different reason.

RX2 (grid RX_3) has the richer scattering the architecture assumes, and its
top-10 concentration matches the synthetic reference exactly. The cost is
discarding the ~39% of receivers with no ray-traced path to the transmitter.

Loading more paths does **not** help: `max_paths` 10 and 25 give identical
statistics, because the scenario contains no more than 10 paths per receiver.

---

## What changed

| File | Change |
|---|---|
| `data/prepare_cost2100.py` | **New.** Loads COST2100 `.mat` → `(N,2,32,32)` float32 + `norm_params.json`. Splits ship with the dataset, so no normalisation leakage. |
| `utils/metrics.py` | `offset_from_norm_params`, `to_complex_*`, `nmse_db_aggregate*`, `nmse_per_sample_db`. ρ and gain take `norm_params`. |
| `models/csi_autoencoder.py` | Decoder RefineNet blocks widened `2→8→2` → `2→8→16→2`, matching CsiNet. Width is a parameter, so the old block stays available for ablation. |
| `models/train.py` | `--train-samples`, `--patience`, `--amp`, per-epoch CSV log; LR schedule now steps on the same metric used for checkpoint selection. |
| `models/evaluate.py` | De-offset metrics, published CsiNet/CRNet reference columns, rebuilds models with the checkpoint's architecture. |
| `models/dct_baseline.py` | Same de-offset treatment, so the comparison is like-for-like. |
| `backend/app.py`, `preflight.py` | De-offset metrics; honour `DATA_DIR`/`WEIGHTS_DIR`; warn on dataset/weights mismatch. |
| `frontend/app.py` | Removed hardcoded `−38.3 dB` / `99.98%` offline placeholders and the unconditional PASS badges. |
| `notebooks/deepcsi_colab.ipynb` | **New.** Runs the whole pipeline on a free Colab GPU. |

---

## Running it

### On Colab (training)

There is no NVIDIA GPU on the dev laptop (AMD iGPU, `torch+cpu`), so training
runs on a free Colab T4. The 2–3 GB dataset download happens **inside Colab**;
nothing lands on your disk.

Open `notebooks/deepcsi_colab.ipynb` → `Runtime → Change runtime type → T4 GPU`
→ run all. It downloads COST2100, prepares tensors, trains CR = 4/16/32,
evaluates, runs sanity checks, and zips ~25 MB of artifacts to download.

### Locally (demo)

```bash
unzip -o deepcsi_artifacts.zip -d .

# PowerShell
$env:DATA_DIR="data/processed_cost2100"
python preflight.py
uvicorn backend.app:app --reload --port 8000
streamlit run frontend/app.py
```

---

## What to expect, and what to tell the team

**The NMSE will get much worse. That is the win.** Published references on this
benchmark (indoor):

| CR | CsiNet | CRNet | DeepCSI (ours) |
|---:|-------:|------:|---------------:|
| 4  | −17.36 dB | −26.99 dB | _measure it_ |
| 16 |  −8.65 dB | −11.35 dB | _measure it_ |
| 32 |  −6.24 dB |  −8.93 dB | _measure it_ |

Landing near −14 to −17 dB at CR=4 is a **good** result: it means the numbers are
real and comparable to a published paper on the same benchmark. Anything near
−38 dB means a metric is being computed on offset data again.

Two things the corrected pipeline gives the presentation that the old one could
not: an NMSE-vs-CR curve that actually **slopes**, and a beamforming metric that
**discriminates** between compression ratios instead of reading 99.98% everywhere.

The notebook's section 8 checks all three of these automatically before you
present anything.

---

## Known limitations (say these out loud)

- **The DCT baseline is still credited too generously.** It keeps the top K
  coefficients but never pays for transmitting *which* K (~11 bits each at
  K=128), so its real feedback cost is understated versus a fixed-size latent.
- **The FFT pipeline is not implemented.** `utils/transforms.py` has working
  `spatial_frequency_to_angular_delay` / `truncate_delay` helpers that nothing
  calls. COST2100 ships pre-truncated to 32 delay taps, so the README's
  32×256 → 2D FFT → 32×32 flow remains unbuilt.
- **Latent dims are float counts, not a bitstream.** "93.75% reduction" is a
  scalar-dimension claim, not a bandwidth claim. Quantisation and entropy coding
  are future work.
- **COST2100 is a channel model, not measurements.** It is the standard
  benchmark for this task, but it is simulated — same category as the synthetic
  generator, just validated and comparable to published work.
