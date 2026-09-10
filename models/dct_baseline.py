import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.fft import dctn, idctn

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.metrics import nmse_per_sample_db, beamforming_gain_numpy, beamforming_loss_db


def compress_reconstruct_dct_sample(sample_2ch: np.ndarray, retained_scalars: int) -> np.ndarray:
    """
    Apply 2D DCT on 2-channel tensor (2, 32, 32), retain top absolute scalar coefficients across channels,
    zero the rest, and perform inverse 2D DCT.

    Parameters:
        sample_2ch: np.ndarray of shape (2, 32, 32)
        retained_scalars: Total number of float coefficients to retain across both channels.

    Returns:
        reconstructed: np.ndarray of shape (2, 32, 32)
    """
    # 2D DCT per channel
    dct_real = dctn(sample_2ch[0], norm="ortho")
    dct_imag = dctn(sample_2ch[1], norm="ortho")

    # Flatten and combine coefficients
    combined_coeff = np.stack([dct_real, dct_imag], axis=0)  # (2, 32, 32)
    abs_coeff = np.abs(combined_coeff).flatten()

    # Find threshold for top K coefficients
    if retained_scalars < len(abs_coeff):
        threshold = np.partition(abs_coeff, -retained_scalars)[-retained_scalars]
        mask = np.abs(combined_coeff) >= threshold
        # If ties exceed K, truncate exact count
        if np.sum(mask) > retained_scalars:
            flat_mask = mask.flatten()
            indices = np.where(flat_mask)[0]
            drop_indices = indices[retained_scalars:]
            flat_mask[drop_indices] = False
            mask = flat_mask.reshape(combined_coeff.shape)
    else:
        mask = np.ones_like(combined_coeff, dtype=bool)

    truncated_coeff = combined_coeff * mask

    # Inverse 2D DCT per channel
    recon_real = idctn(truncated_coeff[0], norm="ortho")
    recon_imag = idctn(truncated_coeff[1], norm="ortho")

    return np.stack([recon_real, recon_imag], axis=0)


def evaluate_dct_baseline(test_data: np.ndarray, norm_params: dict = None):
    """
    Evaluate DCT baseline performance across CR=4, CR=16, CR=32 on test set.

    norm_params is threaded through to the beamforming metric so the baseline is
    measured exactly the same way as DeepCSI. Note the fairness caveat: this
    baseline is credited with K retained coefficients but does not pay for
    transmitting WHICH K coefficients were kept (~11 bits each at K=128), so its
    real feedback cost is understated relative to a fixed-size latent vector.
    """
    crs = [4, 16, 32]
    total_scalars = 2048
    results = []

    print("=== Evaluating 2D DCT Baseline ===")
    for cr in crs:
        retained = total_scalars // cr
        gain_list = []
        recon_list = []

        start_time = time.time()
        for i in range(len(test_data)):
            sample = test_data[i]
            recon = compress_reconstruct_dct_sample(sample, retained_scalars=retained)
            gain = beamforming_gain_numpy(recon, sample, norm_params=norm_params)
            gain_list.append(gain)
            recon_list.append(recon)
        elapsed_ms = ((time.time() - start_time) / len(test_data)) * 1000.0

        # NMSE on the de-offset complex channel, matching how DeepCSI is scored.
        recon_arr = np.stack(recon_list)
        nmse_list = nmse_per_sample_db(recon_arr, test_data, norm_params)
        mean_nmse = float(np.mean(nmse_list))
        std_nmse = float(np.std(nmse_list))
        aggregate_nmse = float(
            10.0 * np.log10(np.mean(10.0 ** (nmse_list / 10.0)) + 1e-12)
        )
        mean_gain = float(np.mean(gain_list))
        std_gain = float(np.std(gain_list))
        loss_db = float(beamforming_loss_db(mean_gain))
        scalar_reduction = (1.0 - (retained / total_scalars)) * 100.0

        print(f"DCT CR={cr:2d} | Retained: {retained:4d} scalars | NMSE: {aggregate_nmse:6.2f} dB (agg) "
              f"/ {mean_nmse:6.2f} +/- {std_nmse:4.2f} dB (per-sample) | "
              f"BF Gain: {mean_gain*100:5.2f}% ({loss_db:5.2f} dB) | Latency: {elapsed_ms:.3f} ms/sample")

        results.append({
            "method": "DCT Baseline",
            "compression_ratio": cr,
            "latent_dim": retained,
            "nmse_db_mean": round(mean_nmse, 2),
            "nmse_db_std": round(std_nmse, 2),
            "nmse_db_aggregate": round(aggregate_nmse, 2),
            "beamforming_gain_mean": round(mean_gain, 4),
            "beamforming_gain_std": round(std_gain, 4),
            "beamforming_loss_db": round(loss_db, 2),
            "inference_ms": round(elapsed_ms, 3),
            "scalar_reduction_percent": round(scalar_reduction, 2)
        })

    return pd.DataFrame(results)


def main():
    test_path = Path("data/processed/test.npy")
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found at {test_path}. Run data/generate_data.py first.")

    test_data = np.load(test_path)
    df_dct = evaluate_dct_baseline(test_data)

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    dct_csv_path = output_dir / "dct_baseline_metrics.csv"
    df_dct.to_csv(dct_csv_path, index=False)
    print(f"\nDCT baseline metrics saved to '{dct_csv_path}'")


if __name__ == "__main__":
    main()
