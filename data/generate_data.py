import argparse
import os
import json
import numpy as np
from pathlib import Path
import sys

# Ensure root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.seed import set_seed
from utils.transforms import split_complex, compute_energy_retention


def generate_synthetic_csi(
    num_samples: int = 10000,
    n_antennas: int = 32,
    n_delays: int = 32,
    seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic 3GPP-inspired sparse complex angular-delay CSI tensors.
    
    Structure:
    - Sparse multipath cluster components in angle-delay domain.
    - Exponential delay power profile decay along the delay tap dimension.
    
    Output shape: (num_samples, 2, n_antennas, n_delays) float32
    """
    set_seed(seed)
    
    # Delay tap index profile
    delay_indices = np.arange(n_delays)
    delay_decay = np.exp(-0.15 * delay_indices)  # (32,)
    
    csi_complex_list = []
    
    for _ in range(num_samples):
        # Generate random multipath cluster locations
        n_clusters = np.random.randint(3, 9)
        H_ad = np.zeros((n_antennas, n_delays), dtype=np.complex64)
        
        for _ in range(n_clusters):
            angle_idx = np.random.randint(0, n_antennas)
            delay_idx = np.random.randint(0, min(16, n_delays))  # Most energy in early delay taps
            
            # Complex gain with Rayeligh fading amplitude & random phase
            amplitude = np.random.rayleigh(scale=1.0) * delay_decay[delay_idx]
            phase = np.random.uniform(0, 2 * np.pi)
            gain = amplitude * np.exp(1j * phase)
            
            # Angular leakage / spread (Gaussian envelope across adjacent antennas & delays)
            for a_offset in range(-2, 3):
                a_pos = (angle_idx + a_offset) % n_antennas
                for d_offset in range(0, 3):
                    d_pos = delay_idx + d_offset
                    if d_pos < n_delays:
                        spatial_weight = np.exp(-0.8 * (a_offset ** 2))
                        delay_weight = np.exp(-0.5 * (d_offset ** 2))
                        H_ad[a_pos, d_pos] += gain * spatial_weight * delay_weight
                        
        csi_complex_list.append(H_ad)
        
    csi_complex = np.array(csi_complex_list, dtype=np.complex64)  # (N, 32, 32)
    
    # Check energy retention
    avg_energy_retention = np.mean([compute_energy_retention(h, max_delay=32) for h in csi_complex])
    print(f"Dataset generated. Average Energy Retention (32 delay taps): {avg_energy_retention:.2f}%")
    
    # Split real and imaginary channels -> (N, 2, 32, 32)
    csi_tensor = split_complex(csi_complex).astype(np.float32)
    return csi_tensor


def normalize_robust(raw_tensor: np.ndarray, quantile: float = 0.9995):
    """
    Scale to [0, 1] with 0.5 as the complex zero point, using a robust scale.

    Global min-max is a poor fit for sparse CSI. The extremes come from a handful
    of cluster peaks, so dividing by (max - min) squashes the bulk of the data
    into a sliver of the range: on this generator 98% of pixels land within
    +/-0.01 of the zero point (std 0.0066, about 4% of [0,1]). The autoencoder
    then minimises MSE by collapsing to the mean, scoring ~0 dB NMSE -- worse
    than a DCT baseline that needs no training at all.

    Instead we set the scale from a high quantile of |h| and clip the rare
    outliers beyond it, which is how the released COST2100/CsiNet data is
    prepared. The mapping is

        x = clip(h / (2 * S) + 0.5, 0, 1),   S = quantile(|h|, q)

    so the recorded min/max of -S/+S make the existing affine denormalisation,
    x * (max - min) + min, reduce to exactly (x - 0.5) * 2S.

    Returns (normalized_tensor, scale, clipped_fraction).
    """
    scale = float(np.quantile(np.abs(raw_tensor), quantile))
    if scale <= 0:
        raise ValueError("Robust scale collapsed to zero; the dataset is empty or constant.")

    scaled = raw_tensor / (2.0 * scale) + 0.5
    clipped_fraction = float(np.mean((scaled < 0.0) | (scaled > 1.0)))
    return np.clip(scaled, 0.0, 1.0).astype(np.float32), scale, clipped_fraction


def main():
    parser = argparse.ArgumentParser(description="DeepCSI Synthetic Dataset Generator")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--norm-mode", choices=["robust", "minmax"], default="robust",
        help="robust (default): quantile scale with 0.5 as the zero point, matching "
             "the COST2100 convention. minmax: the original global min-max, kept "
             "for reproducing earlier runs -- it squashes sparse CSI and the model "
             "will not learn from it."
    )
    parser.add_argument("--norm-quantile", type=float, default=0.9995,
                        help="Quantile of |h| defining the robust scale.")
    args = parser.parse_args()

    print("=== DeepCSI Synthetic Dataset Generator ===")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    total_samples = args.samples
    seed = args.seed

    print(f"Generating {total_samples} angular-delay CSI samples...")
    raw_tensor = generate_synthetic_csi(num_samples=total_samples, seed=seed)

    print(f"Raw Tensor Range: min={raw_tensor.min():.6f}, max={raw_tensor.max():.6f}")

    if args.norm_mode == "robust":
        normalized_tensor, scale, clipped = normalize_robust(raw_tensor, args.norm_quantile)
        norm_min, norm_max = -scale, scale
        print(f"Normalization: robust (q={args.norm_quantile}), scale S={scale:.6f}, "
              f"clipped {clipped*100:.3f}% of values")
    else:
        norm_min = float(raw_tensor.min())
        norm_max = float(raw_tensor.max())
        normalized_tensor = ((raw_tensor - norm_min) / (norm_max - norm_min + 1e-10)).astype(np.float32)
        clipped = 0.0
        print("Normalization: global min-max (legacy)")

    print(f"Normalized Tensor: min={normalized_tensor.min():.6f}, "
          f"max={normalized_tensor.max():.6f}, std={normalized_tensor.std():.6f}")
    # A tiny std means the signal occupies a sliver of the representable range,
    # which is the failure mode described in normalize_robust.
    if normalized_tensor.std() < 0.02:
        print("  WARNING: normalized std is very small. The model will likely "
              "collapse to the mean and score ~0 dB NMSE.")

    # Train / Val / Test Split: 70% / 10% / 20%.
    # Computed from the actual sample count, not hardcoded. The previous
    # 7000/1000/2000 literals only equalled 70/10/20 at exactly 10000 samples;
    # with --samples 50000 they yielded 7000/1000/42000, silently starving
    # training to 14% of the data while inflating the test set.
    n = len(normalized_tensor)
    n_train, n_val = int(0.7 * n), int(0.1 * n)
    train_data = normalized_tensor[:n_train]
    val_data = normalized_tensor[n_train:n_train + n_val]
    test_data = normalized_tensor[n_train + n_val:]

    print(f"Train set shape: {train_data.shape}, dtype={train_data.dtype}")
    print(f"Val set shape:   {val_data.shape}, dtype={val_data.dtype}")
    print(f"Test set shape:  {test_data.shape}, dtype={test_data.dtype}")

    # Save processed numpy arrays
    np.save(output_dir / "train.npy", train_data)
    np.save(output_dir / "val.npy", val_data)
    np.save(output_dir / "test.npy", test_data)

    # Save normalization parameters. offset/scheme mirror the COST2100 metadata
    # so utils.metrics and the backend treat both datasets identically.
    norm_params = {
        "source": f"synthetic_3gpp_inspired_{args.norm_mode}",
        "scheme": "csinet_offset" if args.norm_mode == "robust" else "minmax",
        "offset": 0.5 if args.norm_mode == "robust" else None,
        "min": norm_min,
        "max": norm_max,
        "clipped_fraction": clipped,
        "shape": list(train_data.shape[1:]),
        "num_train": len(train_data),
        "num_val": len(val_data),
        "num_test": len(test_data),
        "seed": seed
    }
    if norm_params["offset"] is None:
        del norm_params["offset"]
    with open(output_dir / "norm_params.json", "w") as f:
        json.dump(norm_params, f, indent=2)

    print(f"Successfully saved dataset and normalization metadata to {output_dir}")


if __name__ == "__main__":
    main()
