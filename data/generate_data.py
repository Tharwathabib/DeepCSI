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


def main():
    print("=== DeepCSI Synthetic Dataset Generator ===")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    total_samples = 10000
    seed = 42
    
    print(f"Generating {total_samples} angular-delay CSI samples...")
    raw_tensor = generate_synthetic_csi(num_samples=total_samples, seed=seed)
    
    # Calculate global min-max normalization parameters on full dataset (or train set)
    global_min = float(raw_tensor.min())
    global_max = float(raw_tensor.max())
    print(f"Raw Tensor Range: min={global_min:.6f}, max={global_max:.6f}")
    
    # Normalize tensor to [0, 1] range
    normalized_tensor = (raw_tensor - global_min) / (global_max - global_min + 1e-10)
    print(f"Normalized Tensor Range: min={normalized_tensor.min():.6f}, max={normalized_tensor.max():.6f}")
    
    # Train / Val / Test Split: 7000 / 1000 / 2000 (70% / 10% / 20%)
    train_data = normalized_tensor[:7000]
    val_data = normalized_tensor[7000:8000]
    test_data = normalized_tensor[8000:]
    
    print(f"Train set shape: {train_data.shape}, dtype={train_data.dtype}")
    print(f"Val set shape:   {val_data.shape}, dtype={val_data.dtype}")
    print(f"Test set shape:  {test_data.shape}, dtype={test_data.dtype}")
    
    # Save processed numpy arrays
    np.save(output_dir / "train.npy", train_data)
    np.save(output_dir / "val.npy", val_data)
    np.save(output_dir / "test.npy", test_data)
    
    # Save normalization parameters
    norm_params = {
        "min": global_min,
        "max": global_max,
        "shape": list(train_data.shape[1:]),
        "num_train": len(train_data),
        "num_val": len(val_data),
        "num_test": len(test_data),
        "seed": seed
    }
    with open(output_dir / "norm_params.json", "w") as f:
        json.dump(norm_params, f, indent=2)
        
    print(f"Successfully saved dataset and normalization metadata to {output_dir}")


if __name__ == "__main__":
    main()
