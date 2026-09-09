import argparse
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

from utils.metrics import nmse_db, beamforming_gain_torch, beamforming_loss_db
from models.csi_autoencoder import CSIAutoencoder
from models.dct_baseline import evaluate_dct_baseline


def evaluate_deepcsi_model(model, test_tensor, device):
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

    # Latency timing
    start_time = time.time()
    with torch.no_grad():
        reconstructions, latents = model(test_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed_ms = ((time.time() - start_time) / num_samples) * 1000.0

    # Compute per-sample NMSE and beamforming gain
    nmse_per_sample = []
    with torch.no_grad():
        for i in range(num_samples):
            pred = reconstructions[i:i+1]
            true = test_tensor[i:i+1]
            val = nmse_db(pred, true).item()
            nmse_per_sample.append(val)

        # Batch beamforming gain computation
        gains_tensor = beamforming_gain_torch(reconstructions, test_tensor, return_per_sample=True)
        gains_list = gains_tensor.cpu().numpy().tolist()

    mean_nmse = float(np.mean(nmse_per_sample))
    std_nmse = float(np.std(nmse_per_sample))
    mean_gain = float(np.mean(gains_list))
    std_gain = float(np.std(gains_list))
    loss_db = float(beamforming_loss_db(mean_gain))

    return mean_nmse, std_nmse, mean_gain, std_gain, loss_db, elapsed_ms, reconstructions.cpu().numpy(), latents.cpu().numpy()


def generate_plots(df_combined, sample_orig, sample_recon_cr16, sample_dct_cr16, figures_dir):
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: NMSE vs Compression Ratio
    plt.figure(figsize=(8, 5))
    df_deepcsi = df_combined[df_combined["method"] == "DeepCSI"]
    df_dct = df_combined[df_combined["method"] == "DCT Baseline"]

    plt.plot(df_deepcsi["compression_ratio"], df_deepcsi["nmse_db_mean"], "o-", color="#1f77b4", linewidth=2.5, label="DeepCSI Autoencoder")
    plt.plot(df_dct["compression_ratio"], df_dct["nmse_db_mean"], "s--", color="#ff7f0e", linewidth=2.5, label="2D DCT Baseline")

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

    plt.axhline(90.0, color="#7f7f7f", linestyle=":", label="90% Power Retention Target")
    plt.title("Downstream MRT Beamforming Gain vs Compression Ratio", fontsize=14, fontweight="bold")
    plt.xlabel("Compression Ratio (CR)", fontsize=12)
    plt.ylabel("Normalized Beamforming Power Gain (%)", fontsize=12)
    plt.xticks([4, 16, 32], ["CR=4", "CR=16", "CR=32"])
    plt.ylim(50, 102)
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
        raise FileNotFoundError(f"Test data not found at {test_path}. Run data/generate_data.py first.")

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

        checkpoint = torch.load(weight_path, map_location=device)
        model = CSIAutoencoder(compression_ratio=cr).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        mean_nmse, std_nmse, mean_gain, std_gain, loss_db, latency, recons, latents = evaluate_deepcsi_model(model, test_tensor, device)
        reconstructions_dict[cr] = recons

        latent_dim = total_scalars // cr
        scalar_reduction = (1.0 - (latent_dim / total_scalars)) * 100.0

        print(f"DeepCSI CR={cr:2d} | Latent: {latent_dim:4d} | NMSE: {mean_nmse:6.2f} +/- {std_nmse:4.2f} dB | BF Gain: {mean_gain*100:5.2f}% ({loss_db:5.2f} dB) | Latency: {latency:.3f} ms/sample")

        deepcsi_results.append({
            "method": "DeepCSI",
            "compression_ratio": cr,
            "latent_dim": latent_dim,
            "nmse_db_mean": round(mean_nmse, 2),
            "nmse_db_std": round(std_nmse, 2),
            "beamforming_gain_mean": round(mean_gain, 4),
            "beamforming_gain_std": round(std_gain, 4),
            "beamforming_loss_db": round(loss_db, 2),
            "inference_ms": round(latency, 3),
            "scalar_reduction_percent": round(scalar_reduction, 2)
        })

    df_deepcsi = pd.DataFrame(deepcsi_results)
    
    # Run DCT Baseline evaluation
    df_dct = evaluate_dct_baseline(test_data_np)

    # Combine results
    df_combined = pd.concat([df_deepcsi, df_dct], ignore_index=True)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df_combined.to_csv(results_dir / "metrics.csv", index=False)
    
    # Pivot tables for direct comparison
    pivot_nmse = df_combined.pivot(index="compression_ratio", columns="method", values="nmse_db_mean")
    pivot_nmse.to_csv(results_dir / "nmse_comparison.csv")

    pivot_gain = df_combined.pivot(index="compression_ratio", columns="method", values="beamforming_gain_mean")
    pivot_gain.to_csv(results_dir / "beamforming_comparison.csv")

    print(f"\nSaved combined metrics to '{results_dir / 'metrics.csv'}'")
    print("\n=== NMSE Comparison Table (dB) ===")
    print(pivot_nmse.to_string())
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

    print(f"\nSaved combined metrics to '{results_dir / 'metrics.csv'}'")
    print("\n=== NMSE Comparison Table (dB) ===")
    print(pivot_nmse.to_string())

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
