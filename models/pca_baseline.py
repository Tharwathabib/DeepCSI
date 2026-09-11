"""
PCA (KLT) baseline for DeepCSI.

This is the baseline the project was missing, and it is the strongest fair
comparison available: a linear autoencoder with k components is the OPTIMAL
linear compressor under MSE, so its score is the bar any learned nonlinear
model has to clear to justify itself.

It is a fairer comparison than the DCT baseline in two ways. DCT uses a fixed
basis and is credited with K coefficients without paying to transmit WHICH K it
kept (~11 bits each). PCA, like DeepCSI, learns its basis from the training set
and sends a FIXED k numbers, so the feedback payload is exactly the same shape
as the autoencoder's latent vector. The basis itself is shared offline once,
which is the same deal DeepCSI's decoder weights get.

The basis is fit on TRAIN ONLY and applied to TEST -- fitting on test would
leak, and would be the same mistake the original global min-max normalisation
made.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.metrics import (nmse_per_sample_db, nmse_distribution, beamforming_gain_numpy,
                           beamforming_loss_db, spectral_efficiency)


def fit_pca(train_data: np.ndarray, max_components: int = 512):
    """
    Return (mean, components) of the training set, components shaped (k, 2048).

    Works on the raw [0,1] tensor: mean-centring absorbs the 0.5 offset exactly,
    so no separate de-offset step is needed here. Uses the 2048x2048 covariance
    rather than an SVD of the full data matrix, which is both cheaper and exact
    for this shape.
    """
    x = train_data.reshape(len(train_data), -1).astype(np.float64)
    mean = x.mean(axis=0)
    xc = x - mean
    cov = (xc.T @ xc) / len(xc)
    # eigh returns ascending eigenvalues; take the largest max_components.
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1][:max_components]
    return mean, vecs[:, order].T.copy()


def evaluate_pca_baseline(train_data: np.ndarray, test_data: np.ndarray,
                          norm_params: dict = None, snr_db: float = 10.0):
    """Evaluate PCA at the same latent budgets DeepCSI uses."""
    crs = [4, 16, 32]
    total_scalars = 2048
    shape = test_data.shape[1:]
    results = []

    print("=== Evaluating PCA (KLT) Baseline ===")
    mean, components = fit_pca(train_data, max_components=total_scalars // min(crs))
    flat_test = test_data.reshape(len(test_data), -1).astype(np.float64)

    for cr in crs:
        k = total_scalars // cr
        basis = components[:k]

        start = time.time()
        latent = (flat_test - mean) @ basis.T
        encode_ms = ((time.time() - start) / len(test_data)) * 1000.0

        start = time.time()
        recon_flat = latent @ basis + mean
        decode_ms = ((time.time() - start) / len(test_data)) * 1000.0

        # The autoencoder's sigmoid cannot leave [0,1]; hold PCA to the same
        # output range so the comparison is not handing it an extra degree of
        # freedom the learned model does not have.
        recon = np.clip(recon_flat, 0.0, 1.0).reshape(len(test_data), *shape).astype(np.float32)

        nmse_list = nmse_per_sample_db(recon, test_data, norm_params)
        mean_nmse = float(np.mean(nmse_list))
        std_nmse = float(np.std(nmse_list))
        aggregate_nmse = float(10.0 * np.log10(np.mean(10.0 ** (nmse_list / 10.0)) + 1e-12))

        gains = np.asarray([
            beamforming_gain_numpy(recon[i], test_data[i], norm_params=norm_params)
            for i in range(len(test_data))
        ])
        mean_gain = float(np.mean(gains))
        loss_db = float(beamforming_loss_db(mean_gain))

        print(f"PCA CR={cr:2d} | k={k:4d} | NMSE: {aggregate_nmse:7.2f} dB (agg) "
              f"/ {mean_nmse:7.2f} +/- {std_nmse:4.2f} dB | "
              f"BF Gain: {mean_gain*100:5.2f}% ({loss_db:5.2f} dB)")

        results.append({
            "method": "PCA Baseline",
            "compression_ratio": cr,
            "latent_dim": k,
            "scalar_reduction_percent": round((1.0 - k / total_scalars) * 100.0, 2),
            "nmse_db_mean": round(mean_nmse, 2),
            "nmse_db_std": round(std_nmse, 2),
            "nmse_db_aggregate": round(aggregate_nmse, 2),
            "cosine_similarity_rho": round(float(np.mean(np.sqrt(gains))), 4),
            "beamforming_gain_mean": round(mean_gain, 4),
            "beamforming_gain_std": round(float(np.std(gains)), 4),
            "beamforming_loss_db": round(loss_db, 2),
            "spectral_efficiency_bps_hz": round(float(np.mean(spectral_efficiency(gains, snr_db))), 4),
            **{key: round(v, 4) for key, v in nmse_distribution(nmse_list).items()},
            "encoder_ms_ue": round(encode_ms, 4),
            "decoder_ms_gnb": round(decode_ms, 4),
            "inference_ms": round(encode_ms + decode_ms, 4),
            # A projection matrix per side, the direct analogue of the
            # autoencoder's encoder/decoder parameter counts.
            "encoder_params": k * total_scalars,
            "decoder_params": k * total_scalars,
        })

    return pd.DataFrame(results)


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="PCA baseline at DeepCSI's latent budgets.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--train-samples", type=int, default=20000,
                        help="Cap on training rows used to fit the basis.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train = np.load(data_dir / "train.npy")
    test = np.load(data_dir / "test.npy")
    if len(train) > args.train_samples:
        idx = np.random.default_rng(0).choice(len(train), args.train_samples, replace=False)
        train = train[idx]

    norm_path = data_dir / "norm_params.json"
    norm_params = json.loads(norm_path.read_text()) if norm_path.exists() else None

    print(f"Fitting PCA on {len(train)} training samples from {data_dir}, "
          f"evaluating on {len(test)} test samples.")
    df = evaluate_pca_baseline(train, test, norm_params=norm_params)

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pca_baseline_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"\nPCA baseline metrics saved to '{out_path}'")


if __name__ == "__main__":
    main()
