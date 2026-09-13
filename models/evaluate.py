import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.metrics import (
    nmse_db_aggregate,
    nmse_per_sample_db,
    nmse_distribution,
    beamforming_gain_torch,
    beamforming_loss_db,
    spectral_efficiency,
)

from models.csi_autoencoder import CSIAutoencoder, build_from_checkpoint
from models.dct_baseline import evaluate_dct_baseline
from models.pca_baseline import evaluate_pca_baseline

# SNR at which spectral efficiency is reported. Stated explicitly because the
# figure is meaningless without it.
SNR_DB = 10.0

# Published CsiNet/CRNet reference values used to sit here and were appended to
# the results table. They are measured on a different indoor benchmark that is
# out of scope for this project, so the columns could never be a like-for-like
# comparison against DeepMIMO O1 outdoor. Printing them beside our own would
# invite exactly the comparison the README says cannot be made, so they are
# gone rather than dormant.


def measure_latency(model, test_tensor, device, n_runs: int = 100):
    """
    Measure encoder and decoder latency separately, at batch size 1.

    The two halves run on different hardware in a real deployment -- the encoder
    on the UE, the decoder at the gNodeB -- so a combined figure hides the only
    number that decides feasibility. Batch size 1 is what a UE actually does; the
    full-batch figure is throughput, not latency, and is reported separately
    under its own name rather than being called latency.

    Reports the median over n_runs, which is robust to scheduler jitter in a way
    the mean is not.
    """
    single = test_tensor[:1]
    enc, dec = [], []

    with torch.no_grad():
        for _ in range(10):                      # warmup
            model.decoder(model.encoder(single))
        if device.type == "cuda":
            torch.cuda.synchronize()

        for _ in range(n_runs):
            t0 = time.perf_counter()
            latent = model.encoder(single)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            model.decoder(latent)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            enc.append((t1 - t0) * 1000.0)
            dec.append((t2 - t1) * 1000.0)

        # Amortised full-batch cost: throughput, for comparison with the above.
        t0 = time.perf_counter()
        model(test_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "encoder_ms_ue": float(np.median(enc)),
        "decoder_ms_gnb": float(np.median(dec)),
        # Amortised full-batch cost. Kept under the name inference_ms because
        # the dashboard reads that column, but it is throughput, not latency:
        # a UE encodes one channel at a time and pays encoder_ms_ue.
        "inference_ms": float(batch_ms / len(test_tensor)),
    }


def evaluate_deepcsi_model(model, test_tensor, device, norm_params=None):
    """
    Evaluate trained DeepCSI autoencoder model on test dataset.
    Measures NMSE, beamforming gain, and inference latency.
    """
    model.eval()
    test_tensor = test_tensor.to(device)
    num_samples = len(test_tensor)

    # Warmup pass
    with torch.no_grad():
        _, _ = model(test_tensor[:10])
        if device.type == "cuda":
            torch.cuda.synchronize()

    timing = measure_latency(model, test_tensor, device)

    with torch.no_grad():
        reconstructions, latents = model(test_tensor)

    # Compute per-sample NMSE and beamforming gain
    with torch.no_grad():
        # Per-sample NMSE on the de-offset channel, vectorised. Measuring this on
        # the raw [0,1] tensor divides by DC-dominated power and reports ~35 dB
        # better than reality.
        nmse_per_sample = nmse_per_sample_db(
            reconstructions.cpu().numpy(), test_tensor.cpu().numpy(), norm_params
        )

        # Dataset-level NMSE in the CsiNet/CRNet convention, 10*log10(mean(ratio)).
        # This is the figure that is comparable to published results.
        aggregate_nmse = float(nmse_db_aggregate(reconstructions, test_tensor, norm_params).item())

        # Batch beamforming gain computation. norm_params removes the DC offset
        # baked into [0,1]-normalised data; without it rho saturates near 1.0 for
        # every model and the metric carries no information.
        gains_tensor = beamforming_gain_torch(
            reconstructions, test_tensor, return_per_sample=True, norm_params=norm_params
        )
        gains_list = gains_tensor.cpu().numpy()

    mean_nmse = float(np.mean(nmse_per_sample))
    std_nmse = float(np.std(nmse_per_sample))
    mean_gain = float(np.mean(gains_list))
    std_gain = float(np.std(gains_list))
    loss_db = float(beamforming_loss_db(mean_gain))

    stats = {
        "nmse_db_mean": mean_nmse,
        "nmse_db_std": std_nmse,
        "nmse_db_aggregate": aggregate_nmse,
        # rho itself, not just rho^2 -- CsiNet and CRNet report rho, so without
        # this column the results cannot be placed beside published tables.
        "cosine_similarity_rho": float(np.mean(np.sqrt(gains_list))),
        "beamforming_gain_mean": mean_gain,
        "beamforming_gain_std": std_gain,
        "beamforming_loss_db": loss_db,
        "spectral_efficiency_bps_hz": float(np.mean(spectral_efficiency(gains_list, SNR_DB))),
        **nmse_distribution(nmse_per_sample),
        **timing,
        "encoder_params": sum(p.numel() for p in model.encoder.parameters()),
        "decoder_params": sum(p.numel() for p in model.decoder.parameters()),
    }

    return stats, reconstructions.cpu().numpy(), latents.cpu().numpy()


def generate_plots(df_combined, sample_orig, sample_recon_cr16, sample_dct_cr16, figures_dir):
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: NMSE vs Compression Ratio
    plt.figure(figsize=(8, 5))
    df_deepcsi = df_combined[df_combined["method"] == "DeepCSI"]
    df_dct = df_combined[df_combined["method"] == "DCT Baseline"]

    # Plot the same NMSE the results table and README quote -- the log-of-mean
    # aggregate. Plotting nmse_db_mean here instead would put a different number
    # on the figure than in the text, differing by more than a dB.
    nmse_col = "nmse_db_aggregate" if "nmse_db_aggregate" in df_combined else "nmse_db_mean"

    plt.plot(df_deepcsi["compression_ratio"], df_deepcsi[nmse_col], "o-", color="#1f77b4", linewidth=2.5, label="DeepCSI Autoencoder")
    plt.plot(df_dct["compression_ratio"], df_dct[nmse_col], "s--", color="#ff7f0e", linewidth=2.5, label="2D DCT Baseline")

    df_pca = df_combined[df_combined["method"] == "PCA Baseline"]
    if not df_pca.empty:
        plt.plot(df_pca["compression_ratio"], df_pca[nmse_col], "^-.", color="#9467bd",
                 linewidth=2.5, label="PCA / KLT Baseline (optimal linear)")

    plt.axhline(-15, color="red", linestyle=":", label="Target Requirement (-15 dB)")
    plt.title("NMSE vs Compression Ratio (CR)", fontsize=14, fontweight="bold")
    plt.xlabel("Compression Ratio (CR)", fontsize=12)
    plt.ylabel("Test NMSE (dB)", fontsize=12)
    plt.xticks([4, 16, 32], ["CR=4", "CR=16", "CR=32"])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(figures_dir / "nmse_vs_cr.png", dpi=300)
    plt.close()

    # Plot 2: Downstream Beamforming Power Gain vs Compression Ratio
    plt.figure(figsize=(8, 5))
    deepcsi_gain_pct = df_deepcsi["beamforming_gain_mean"] * 100.0
    dct_gain_pct = df_dct["beamforming_gain_mean"] * 100.0

    plt.plot(df_deepcsi["compression_ratio"], deepcsi_gain_pct, "o-", color="#2ca02c", linewidth=2.5, label="DeepCSI MRT Gain")
    plt.plot(df_dct["compression_ratio"], dct_gain_pct, "s--", color="#d62728", linewidth=2.5, label="2D DCT MRT Gain")

    df_pca_g = df_combined[df_combined["method"] == "PCA Baseline"]
    if not df_pca_g.empty:
        plt.plot(df_pca_g["compression_ratio"], df_pca_g["beamforming_gain_mean"] * 100.0,
                 "^-.", color="#9467bd", linewidth=2.5, label="PCA / KLT MRT Gain")

    plt.axhline(90.0, color="#7f7f7f", linestyle=":", label="90% Power Retention Target")
    plt.title("Downstream MRT Beamforming Gain vs Compression Ratio", fontsize=14, fontweight="bold")
    plt.xlabel("Compression Ratio (CR)", fontsize=12)
    plt.ylabel("Normalized Beamforming Power Gain (%)", fontsize=12)
    plt.xticks([4, 16, 32], ["CR=4", "CR=16", "CR=32"])
    # Fit the axis to the data instead of the hardcoded (50, 102), which was set
    # when every gain read ~99.98% because of the offset bug. Real baselines now
    # reach far lower -- DeepMIMO DCT hits 22.0% at CR=32 -- and a fixed floor of
    # 50 silently pushes those bars off the figure, which is the same class of
    # defect as a chart that lies by axis choice.
    all_gains = pd.concat([deepcsi_gain_pct, dct_gain_pct] +
                          ([df_pca_g["beamforming_gain_mean"] * 100.0] if not df_pca_g.empty else []))
    plt.ylim(max(0.0, float(all_gains.min()) - 8.0), 102)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(figures_dir / "beamforming_gain_vs_cr.png", dpi=300)
    plt.close()

    # Plot 3: Heatmap Visualizations for CR=16 (Real Channel)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    orig_real = sample_orig[0]
    recon_deepcsi_real = sample_recon_cr16[0]
    recon_dct_real = sample_dct_cr16[0]
    error_deepcsi = np.abs(orig_real - recon_deepcsi_real)

    im0 = axes[0].imshow(orig_real, cmap="viridis", aspect="auto")
    axes[0].set_title("Original CSI (Real)", fontsize=11, fontweight="bold")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(recon_deepcsi_real, cmap="viridis", aspect="auto")
    axes[1].set_title("DeepCSI CR=16 Recon", fontsize=11, fontweight="bold")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(recon_dct_real, cmap="viridis", aspect="auto")
    axes[2].set_title("DCT CR=16 Recon", fontsize=11, fontweight="bold")
    plt.colorbar(im2, ax=axes[2])

    im3 = axes[3].imshow(error_deepcsi, cmap="hot", aspect="auto")
    axes[3].set_title("DeepCSI Abs Error", fontsize=11, fontweight="bold")
    plt.colorbar(im3, ax=axes[3])

    for ax in axes:
        ax.set_xlabel("Delay Tap")
        ax.set_ylabel("Antenna")

    plt.tight_layout()
    plt.savefig(figures_dir / "reconstruction_heatmaps.png", dpi=300)
    plt.close()
    print(f"Generated comparison figures in '{figures_dir}'")


def main():
    parser = argparse.ArgumentParser(description="DeepCSI Model Evaluation Pipeline")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Directory containing test dataset")
    parser.add_argument("--weights-dir", type=str, default="models/weights", help="Directory containing trained model weights")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save evaluation metrics")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== DeepCSI Evaluation Pipeline | Device: {device} ===")

    test_path = Path(args.data_dir) / "test.npy"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {test_path}.\n"
            "Run one of:\n"
            "  python data/generate_data.py                 (synthetic)\n"
            "  python data/prepare_deepmimo.py               (real ray-traced)"
        )

    # The normalisation metadata drives the de-offset used by rho and by the
    # aggregate NMSE. Evaluating without it silently reproduces the saturated
    # beamforming numbers this pipeline used to report.
    norm_params = None
    norm_path = Path(args.data_dir) / "norm_params.json"
    if norm_path.exists():
        with open(norm_path) as f:
            norm_params = json.load(f)
        print(f"Dataset source: {norm_params.get('source', 'unknown')} "
              f"(scheme={norm_params.get('scheme', 'minmax')})")
    else:
        print(f"WARNING: no norm_params.json in {args.data_dir}; "
              "rho will be measured without removing the DC offset and will be "
              "saturated near 1.0.")

    test_data_np = np.load(test_path)
    test_tensor = torch.from_numpy(test_data_np)
    total_scalars = 2048

    deepcsi_results = []
    reconstructions_dict = {}

    crs = [4, 16, 32]
    for cr in crs:
        weight_path = Path(args.weights_dir) / f"deepcsi_cr{cr}.pt"
        if not weight_path.exists():
            print(f"Warning: Weights for CR={cr} not found at '{weight_path}'. Run training first.")
            continue

        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)

        # Rebuild through the shared helper: both refine_widths and arch have to
        # come from the checkpoint, or a crnet model cannot be evaluated at all.
        model = build_from_checkpoint(checkpoint, cr, device)

        # Warn loudly if the weights were trained on a different dataset than the
        # test set currently loaded -- silently mixing them produces nonsense.
        ckpt_source = checkpoint.get("data_source", "unknown")
        current_source = (norm_params or {}).get("source", "unknown")
        if ckpt_source != "unknown" and current_source != "unknown" and ckpt_source != current_source:
            print(f"  WARNING: CR={cr} weights were trained on '{ckpt_source}' "
                  f"but the test set is '{current_source}'.")

        stats, recons, latents = evaluate_deepcsi_model(
            model, test_tensor, device, norm_params=norm_params
        )
        reconstructions_dict[cr] = recons

        latent_dim = total_scalars // cr
        scalar_reduction = (1.0 - (latent_dim / total_scalars)) * 100.0

        print(f"DeepCSI CR={cr:2d} | Latent: {latent_dim:4d} | "
              f"NMSE {stats['nmse_db_aggregate']:6.2f} dB "
              f"(p50 {stats['nmse_db_median']:6.2f}, p95 {stats['nmse_db_p95']:6.2f}) | "
              f"rho {stats['cosine_similarity_rho']:.4f} | "
              f"G {stats['beamforming_gain_mean']*100:5.2f}% | "
              f"enc {stats['encoder_ms_ue']:.3f} ms / dec {stats['decoder_ms_gnb']:.3f} ms")

        deepcsi_results.append({
            "method": "DeepCSI",
            "compression_ratio": cr,
            "latent_dim": latent_dim,
            "scalar_reduction_percent": round(scalar_reduction, 2),
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()},
        })

    df_deepcsi = pd.DataFrame(deepcsi_results)

    # Run DCT Baseline evaluation on the same test samples, same scalar budget
    # and the same normalisation metadata, so the comparison is like-for-like.
    df_dct = evaluate_dct_baseline(test_data_np, norm_params=norm_params, snr_db=SNR_DB)

    # PCA is the stronger baseline: a linear autoencoder with k components is the
    # optimal linear compressor, so it is the bar the learned model has to clear.
    # Its basis is fit on TRAIN, never on the test samples scored here.
    train_path = Path(args.data_dir) / "train.npy"
    if train_path.exists():
        pca_train = np.load(train_path)
        if len(pca_train) > 20000:
            sel = np.random.default_rng(0).choice(len(pca_train), 20000, replace=False)
            pca_train = pca_train[sel]
        df_pca = evaluate_pca_baseline(pca_train, test_data_np,
                                       norm_params=norm_params, snr_db=SNR_DB)
    else:
        print(f"WARNING: {train_path} not found; skipping the PCA baseline.")
        df_pca = pd.DataFrame()

    # Combine results
    df_combined = pd.concat([df_deepcsi, df_dct, df_pca], ignore_index=True)

    # Stamp provenance onto every row. These CSVs are tracked in git as the
    # evidence behind the README, and several tracks (7k synthetic, 35k
    # synthetic, pooled DeepMIMO) write the same filename into different
    # directories. Without this a reader comparing a CSV to the README finds a
    # ~6 dB disagreement and no way to tell which run they are holding.
    df_combined.insert(0, "dataset_source", (norm_params or {}).get("source", "unknown"))
    df_combined.insert(1, "data_dir", str(args.data_dir))
    df_combined.insert(2, "n_test", len(test_data_np))

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df_combined.to_csv(results_dir / "metrics.csv", index=False)

    # Pivot tables for direct comparison. Use the aggregate NMSE -- the
    # log-of-mean convention -- since that is what published results use.
    nmse_col = "nmse_db_aggregate" if "nmse_db_aggregate" in df_combined else "nmse_db_mean"
    pivot_nmse = df_combined.pivot(index="compression_ratio", columns="method", values=nmse_col)

    pivot_nmse.to_csv(results_dir / "nmse_comparison.csv")

    pivot_gain = df_combined.pivot(index="compression_ratio", columns="method", values="beamforming_gain_mean")
    pivot_gain.to_csv(results_dir / "beamforming_comparison.csv")

    print(f"\nSaved combined metrics to '{results_dir / 'metrics.csv'}'")
    print("\n=== NMSE Comparison Table (dB, log-of-mean convention) ===")
    print(pivot_nmse.round(2).to_string())
    print("\n=== Downstream Beamforming Power Gain Comparison (G = rho^2) ===")
    print((pivot_gain * 100.0).round(2).to_string() + " %")

    # Generate before/after split comparison if baseline 80/10/10 metrics exist
    baseline_csv = results_dir / "metrics_80_10_10.csv"
    if baseline_csv.exists():
        df_base = pd.read_csv(baseline_csv)
        comparison_rows = []
        for _, row_curr in df_combined.iterrows():
            m = row_curr["method"]
            cr = row_curr["compression_ratio"]
            base_match = df_base[(df_base["method"] == m) & (df_base["compression_ratio"] == cr)]
            if not base_match.empty:
                b_row = base_match.iloc[0]
                nmse_before = b_row["nmse_db_mean"]
                std_before = b_row["nmse_db_std"]
                lat_before = b_row["inference_ms"]
                nmse_after = row_curr["nmse_db_mean"]
                std_after = row_curr["nmse_db_std"]
                lat_after = row_curr["inference_ms"]
                delta_nmse = round(nmse_after - nmse_before, 2)
                comparison_rows.append({
                    "method": m,
                    "compression_ratio": cr,
                    "nmse_before_80_10_10": nmse_before,
                    "std_before_80_10_10": std_before,
                    "nmse_after_70_10_20": nmse_after,
                    "std_after_70_10_20": std_after,
                    "delta_nmse_db": delta_nmse,
                    "latency_before_ms": lat_before,
                    "latency_after_ms": lat_after
                })
        df_comparison = pd.DataFrame(comparison_rows)
        df_comparison.to_csv(results_dir / "split_comparison.csv", index=False)
        print("\n=== Dataset Split Accuracy Comparison: 80/10/10 vs 70/10/20 ===")
        print(df_comparison.to_string(index=False))

    # Generate visual figures if CR=16 model was evaluated
    if 16 in reconstructions_dict:
        from models.dct_baseline import compress_reconstruct_dct_sample
        dct_sample_cr16 = compress_reconstruct_dct_sample(test_data_np[0], retained_scalars=128)
        generate_plots(
            df_combined=df_combined,
            sample_orig=test_data_np[0],
            sample_recon_cr16=reconstructions_dict[16][0],
            sample_dct_cr16=dct_sample_cr16,
            figures_dir=results_dir / "figures"
        )


if __name__ == "__main__":
    main()
