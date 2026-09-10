"""
COST2100 dataset preparation for DeepCSI.

Converts the pre-processed COST2100 CSI feedback dataset released with CsiNet
(Wen, Shih & Jin, "Deep Learning for Massive MIMO CSI Feedback", IEEE WCL 2018)
into the (N, 2, 32, 32) float32 tensors the DeepCSI pipeline already consumes.

Unlike data/generate_data.py, which synthesises angular-delay tensors directly,
this reads real channel realisations from the COST2100 channel model. The train /
val / test partition ships with the dataset as three separate .mat files, so
there is no re-splitting here and no possibility of normalisation leakage across
the split boundary.

Normalisation convention
------------------------
The released data is already scaled to [0, 1] with 0.5 as the complex zero point.
CsiNet recovers the complex channel as

    H = (x[..., 0, :, :] - 0.5) + 1j * (x[..., 1, :, :] - 0.5)

We record this as scheme="csinet_offset" in norm_params.json so that metrics and
the backend de-offset consistently instead of hardcoding the constant. See
utils/metrics.py::to_complex.

Usage
-----
    python data/prepare_cost2100.py --mat-dir data/raw/COST2100
    python data/prepare_cost2100.py --mat-dir data/raw/COST2100 --subsample 20000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Ensure root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.seed import set_seed

# Channel geometry: 32 transmit antennas x 32 retained delay taps, split into
# real/imaginary channels -> 2048 scalars per sample.
N_ANTENNAS = 32
N_DELAYS = 32
N_CHANNELS = 2
SCALARS_PER_SAMPLE = N_CHANNELS * N_ANTENNAS * N_DELAYS

# The environment suffix ("in" / "out") selects the indoor 5.3 GHz pico-cell or
# the outdoor 300 MHz rural scenario.
ENVIRONMENTS = {
    "indoor": {
        "suffix": "in",
        "label": "cost2100_indoor_5.3GHz",
        "description": "Indoor pico-cell, 5.3 GHz",
    },
    "outdoor": {
        "suffix": "out",
        "label": "cost2100_outdoor_300MHz",
        "description": "Outdoor rural, 300 MHz",
    },
}

# Split name -> the stem of the .mat file holding it. CsiNet's naming is
# DATA_Htrain{in,out}.mat / DATA_Hval{in,out}.mat / DATA_Htest{in,out}.mat.
SPLIT_FILES = {
    "train": "DATA_Htrain{suffix}.mat",
    "val": "DATA_Hval{suffix}.mat",
    "test": "DATA_Htest{suffix}.mat",
}

# Every split file stores its payload under this MATLAB variable name.
MAT_KEY = "HT"


def load_mat_array(path: Path, key: str = MAT_KEY) -> np.ndarray:
    """
    Read one COST2100 .mat file and return its raw (N, 2048) array.

    scipy handles MATLAB v7 and below. The v7.3 format is HDF5 underneath and
    raises NotImplementedError, so fall back to h5py and undo its transposed
    axis order.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing COST2100 file: {path}\n"
            f"Download the dataset and place the .mat files in {path.parent}.\n"
            f"See notebooks/deepcsi_colab.ipynb for the download cells."
        )

    from scipy.io import loadmat

    try:
        mat = loadmat(str(path))
        if key not in mat:
            available = [k for k in mat.keys() if not k.startswith("__")]
            raise KeyError(f"Key '{key}' not found in {path.name}. Available: {available}")
        return np.asarray(mat[key])
    except NotImplementedError:
        # MATLAB v7.3 / HDF5. h5py returns datasets transposed relative to MATLAB.
        import h5py

        with h5py.File(str(path), "r") as f:
            if key not in f:
                raise KeyError(f"Key '{key}' not found in {path.name}. Available: {list(f.keys())}")
            return np.asarray(f[key]).T


def to_csi_tensor(raw: np.ndarray) -> np.ndarray:
    """
    Reshape a raw (N, 2048) COST2100 array into (N, 2, 32, 32) float32.

    C-order reshape puts the first 1024 scalars in channel 0 (real) and the
    next 1024 in channel 1 (imaginary), matching CsiNet's channels_first layout.
    """
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] != SCALARS_PER_SAMPLE:
        raise ValueError(
            f"Expected raw array of shape (N, {SCALARS_PER_SAMPLE}), got {raw.shape}."
        )
    return raw.reshape(len(raw), N_CHANNELS, N_ANTENNAS, N_DELAYS).astype(np.float32)


def validate_split(name: str, data: np.ndarray) -> dict:
    """
    Sanity-check one split and return a summary dict.

    Raises on anything that would silently poison training: wrong shape or
    dtype, non-finite values, or samples that fall outside the [0, 1] range the
    Sigmoid output head can represent.
    """
    if data.ndim != 4 or data.shape[1:] != (N_CHANNELS, N_ANTENNAS, N_DELAYS):
        raise ValueError(f"{name}: expected (N, 2, 32, 32), got {data.shape}")
    if data.dtype != np.float32:
        raise ValueError(f"{name}: expected float32, got {data.dtype}")

    n_nan = int(np.isnan(data).sum())
    n_inf = int(np.isinf(data).sum())
    if n_nan or n_inf:
        raise ValueError(f"{name}: found {n_nan} NaN and {n_inf} Inf values")

    d_min, d_max = float(data.min()), float(data.max())
    if d_min < 0.0 or d_max > 1.0:
        raise ValueError(
            f"{name}: values outside [0, 1] (min={d_min:.6f}, max={d_max:.6f}). "
            "The decoder's Sigmoid head cannot represent these."
        )

    # Per-sample energy of the de-offset complex channel. A near-zero-energy
    # sample would make its NMSE denominator vanish and blow up the metric.
    complex_data = (data[:, 0] - 0.5) + 1j * (data[:, 1] - 0.5)
    energy = np.sum(np.abs(complex_data) ** 2, axis=(1, 2))

    summary = {
        "name": name,
        "num_samples": int(len(data)),
        "shape": list(data.shape[1:]),
        "dtype": str(data.dtype),
        "min": d_min,
        "max": d_max,
        "mean": float(data.mean()),
        "energy_mean": float(energy.mean()),
        "energy_min": float(energy.min()),
        "energy_max": float(energy.max()),
    }

    print(
        f"  {name:<6} n={summary['num_samples']:<7} shape={tuple(data.shape[1:])} "
        f"{data.dtype}  range=[{d_min:.4f}, {d_max:.4f}]  mean={summary['mean']:.4f}"
    )
    print(
        f"         complex energy: mean={energy.mean():.4f} "
        f"min={energy.min():.6f} max={energy.max():.4f}"
    )
    if energy.min() < 1e-6:
        print(f"         WARNING: {int((energy < 1e-6).sum())} near-zero-energy sample(s)")

    return summary


def report_delay_profile(data: np.ndarray, top_n: int = 8) -> dict:
    """
    Report how channel energy is distributed across delay taps.

    The project README asserts that truncating to 32 delay taps is safe because
    later taps are negligible, without ever measuring it. COST2100 data arrives
    pre-truncated to 32 taps, so we cannot verify what was discarded upstream --
    but we can at least show how concentrated the retained taps are, which is
    the evidence behind the sparsity claim the model relies on.
    """
    complex_data = (data[:, 0] - 0.5) + 1j * (data[:, 1] - 0.5)
    # Energy per delay tap, summed over antennas and averaged over samples.
    per_tap = np.mean(np.sum(np.abs(complex_data) ** 2, axis=1), axis=0)
    total = float(per_tap.sum())
    fractions = per_tap / (total + 1e-12)

    cumulative = np.cumsum(fractions)
    print(f"  Energy concentration across {N_DELAYS} retained delay taps:")
    print(f"    taps 0-{top_n - 1}: {cumulative[top_n - 1] * 100:.2f}% of total energy")
    print(f"    taps 0-15: {cumulative[15] * 100:.2f}%")
    top_str = ", ".join(f"{i}:{f * 100:.1f}%" for i, f in enumerate(fractions[:top_n]))
    print(f"    per-tap: {top_str}")

    return {
        "per_tap_energy_fraction": [float(f) for f in fractions],
        "cumulative_fraction_first_8": float(cumulative[7]),
        "cumulative_fraction_first_16": float(cumulative[15]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert the COST2100 CSI feedback dataset into DeepCSI tensors."
    )
    parser.add_argument(
        "--mat-dir",
        type=str,
        default="data/raw/COST2100",
        help="Directory holding the DATA_H*.mat files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed_cost2100",
        help="Where to write train/val/test.npy and norm_params.json.",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="indoor",
        choices=sorted(ENVIRONMENTS.keys()),
        help="COST2100 scenario to prepare.",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Cap the TRAINING split at this many samples (val/test kept whole).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for subsampling.")
    args = parser.parse_args()

    set_seed(args.seed)

    env = ENVIRONMENTS[args.environment]
    mat_dir = Path(args.mat_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== DeepCSI COST2100 Dataset Preparation ===")
    print(f"Scenario:   {env['description']}")
    print(f"Source dir: {mat_dir.resolve()}")
    print(f"Output dir: {output_dir.resolve()}")
    print()

    splits = {}
    for split_name, template in SPLIT_FILES.items():
        filename = template.format(suffix=env["suffix"])
        path = mat_dir / filename
        print(f"Loading {filename} ...")
        raw = load_mat_array(path)
        data = to_csi_tensor(raw)

        if split_name == "train" and args.subsample is not None:
            if args.subsample < len(data):
                idx = np.random.default_rng(args.seed).choice(
                    len(data), size=args.subsample, replace=False
                )
                idx.sort()
                data = data[idx]
                print(f"  Subsampled training split to {args.subsample} samples.")
            else:
                print(
                    f"  --subsample {args.subsample} >= available {len(data)}; "
                    "keeping the full split."
                )

        splits[split_name] = data

    print()
    print("Validation:")
    summaries = {name: validate_split(name, data) for name, data in splits.items()}

    print()
    report = report_delay_profile(splits["train"])

    print()
    for name, data in splits.items():
        out_path = output_dir / f"{name}.npy"
        np.save(out_path, data)
        print(f"Saved {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    # The min/max fields exist for backward compatibility with the synthetic
    # pipeline's denormalisation code, which computes x * (max - min) + min.
    # With min=-0.5 and max=0.5 that reduces to exactly x - 0.5.
    norm_params = {
        "source": env["label"],
        "scheme": "csinet_offset",
        "offset": 0.5,
        "min": -0.5,
        "max": 0.5,
        "shape": [N_CHANNELS, N_ANTENNAS, N_DELAYS],
        "num_train": summaries["train"]["num_samples"],
        "num_val": summaries["val"]["num_samples"],
        "num_test": summaries["test"]["num_samples"],
        "seed": args.seed,
        "split_origin": "shipped_with_dataset",
        "delay_energy": report,
    }
    with open(output_dir / "norm_params.json", "w") as f:
        json.dump(norm_params, f, indent=2)
    print(f"Saved {output_dir / 'norm_params.json'}")

    print()
    print("Done. Train with:")
    print(f"  python models/train.py --data-dir {output_dir} --compression-ratio 4")


if __name__ == "__main__":
    main()
