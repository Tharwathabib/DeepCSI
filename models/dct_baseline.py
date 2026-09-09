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

from utils.metrics import nmse_db_numpy, beamforming_gain_numpy, beamforming_loss_db


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


def evaluate_dct_baseline(test_data: np.ndarray):
    """
    Evaluate DCT baseline performance across CR=4, CR=16, CR=32 on test set.
    """
    crs = [4, 16, 32]
    total_scalars = 2048
    results = []

    print("=== Evaluating 2D DCT Baseline ===")
    for cr in crs:
        retained = total_scalars // cr
        nmse_list = []
        gain_list = []

        start_time = time.time()
        for i in range(len(test_data)):
            sample = test_data[i]
            recon = compress_reconstruct_dct_sample(sample, retained_scalars=retained)
            nmse = nmse_db_numpy(recon, sample)
            gain = beamforming_gain_numpy(recon, sample)
            nmse_list.append(nmse)
            gain_list.append(gain)
        elapsed_ms = ((time.time() - start_time) / len(test_data)) * 1000.0

        mean_nmse = float(np.mean(nmse_list))
        std_nmse = float(np.std(nmse_list))
        mean_gain = float(np.mean(gain_list))
        std_gain = float(np.std(gain_list))
        loss_db = float(beamforming_loss_db(mean_gain))
        scalar_reduction = (1.0 - (retained / total_scalars)) * 100.0

        print(f"DCT CR={cr:2d} | Retained: {retained:4d} scalars | NMSE: {mean_nmse:6.2f} +/- {std_nmse:4.2f} dB | BF Gain: {mean_gain*100:5.2f}% ({loss_db:5.2f} dB) | Latency: {elapsed_ms:.3f} ms/sample")

        results.append({
            "method": "DCT Baseline",
            "compression_ratio": cr,
            "latent_dim": retained,
            "nmse_db_mean": round(mean_nmse, 2),
            "nmse_db_std": round(std_nmse, 2),
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
