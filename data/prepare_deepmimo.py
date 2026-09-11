"""
DeepMIMO dataset preparation for DeepCSI.

Builds (N, 2, 32, 32) angular-delay tensors from ray-traced DeepMIMO channels,
running the full pipeline the README advertises and the synthetic generator
skips:

    ray tracing -> H (32 antennas x 256 subcarriers, spatial-frequency)
                -> 2D DFT (FFT over antennas, IFFT over subcarriers)
                -> truncate to 32 delay taps
                -> per-sample normalisation to [0,1] with 0.5 as complex zero
                -> train / val / test split

Unlike COST2100, DeepMIMO is ray-traced from real building geometry, so the
channels carry genuine spatial structure -- and unlike the synthetic generator,
nothing here is invented.

Usage:
    python data/prepare_deepmimo.py --scenario o1_3p5_downloaded --samples 40000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.seed import set_seed
from utils.transforms import (
    spatial_frequency_to_angular_delay,
    truncate_delay,
    split_complex,
    compute_energy_retention,
)

N_ANTENNAS = 32
N_SUBCARRIERS = 256
N_DELAYS = 32


def load_channels(scenario: str, tx_set: int, rx_set: int, n_users: int,
                  seed: int, chunk: int = 5000) -> np.ndarray:
    """
    Compute spatial-frequency channels for a uniform sample of users.

    Users are sampled uniformly across the whole grid rather than taken from the
    front: DeepMIMO orders receivers by grid position, so a contiguous slice is
    one small patch of street with almost no path-loss diversity (measured: 5.8 dB
    across the first 2000 users, against tens of dB across the full grid).

    Channels are computed in chunks and reduced to 32x32 by the caller, because
    the intermediate 32x256 representation is 32x larger than what we keep.
    """
    import deepmimo as dm

    sets = {s.id: s for s in dm.get_txrx_sets(scenario)}
    total_rx = sets[rx_set].num_points
    n_users = min(n_users, total_rx)

    rng = np.random.default_rng(seed)
    idxs = np.sort(rng.choice(total_rx, size=n_users, replace=False))
    print(f"Sampling {n_users} of {total_rx} users from RX set {rx_set} "
          f"({sets[rx_set].name}), TX set {tx_set} ({sets[tx_set].name})")

    params = dm.ChannelParameters()
    params.bs_antenna.shape = np.array([N_ANTENNAS, 1])     # 32-element ULA
    params.ue_antenna.shape = np.array([1, 1])              # single RX antenna
    params.ofdm.subcarriers = N_SUBCARRIERS
    params.ofdm.selected_subcarriers = np.arange(N_SUBCARRIERS)
    params.freq_domain = 1

    out = []
    for start in range(0, n_users, chunk):
        part = idxs[start:start + chunk]
        d = dm.load(scenario, tx_sets={tx_set: [0]}, rx_sets={rx_set: part.tolist()})
        ds = d[0] if hasattr(d, "__len__") and not hasattr(d, "compute_channels") else d
        ch = np.asarray(ds.compute_channels(params))        # (n, 1, 32, 256)
        out.append(ch[:, 0, :, :].astype(np.complex64))     # drop the RX-antenna axis
        print(f"  users {start:>6}-{start + len(part):>6}  ->  {out[-1].shape}")

    return np.concatenate(out, axis=0)


def clean(H: np.ndarray) -> tuple:
    """
    Drop users whose channel cannot be used, and report why.

    Ray tracing produces genuinely empty channels for receivers with no path to
    the transmitter (inside buildings, deep shadow). Their energy is exactly
    zero, which would make the NMSE denominator vanish and the per-sample
    normalisation divide by zero. These are not outliers to be clipped -- they
    are users the base station could not serve at all, and they belong out of
    the dataset rather than in it as zeros.
    """
    n0 = len(H)
    finite = np.isfinite(H).all(axis=(1, 2))
    energy = np.sum(np.abs(H) ** 2, axis=(1, 2))
    keep = finite & (energy > 0)

    report = {
        "loaded": int(n0),
        "dropped_non_finite": int((~finite).sum()),
        "dropped_zero_power": int((finite & (energy <= 0)).sum()),
        "kept": int(keep.sum()),
    }
    print(f"Cleaning: {report['loaded']} loaded, "
          f"{report['dropped_non_finite']} non-finite, "
          f"{report['dropped_zero_power']} zero-power (no ray-traced path), "
          f"{report['kept']} kept")

    H = H[keep]
    e = np.sum(np.abs(H) ** 2, axis=(1, 2))
    dr = 10 * np.log10(e.max() / e.min())
    print(f"  path-loss dynamic range across kept users: {dr:.1f} dB")
    report["path_loss_dynamic_range_db"] = round(float(dr), 2)
    return H, report


def to_angular_delay(H_sf: np.ndarray) -> tuple:
    """2D DFT, then truncate to the first 32 delay taps. Reports what truncation costs."""
    H_ad_full = spatial_frequency_to_angular_delay(H_sf)
    retention = compute_energy_retention(H_ad_full, max_delay=N_DELAYS)
    per_sample = np.array([
        compute_energy_retention(h, max_delay=N_DELAYS) for h in H_ad_full[:2000]
    ])
    print(f"Delay truncation {N_SUBCARRIERS} -> {N_DELAYS} taps: "
          f"{retention:.2f}% of total energy retained "
          f"(per-user p5 {np.percentile(per_sample, 5):.2f}%, "
          f"median {np.median(per_sample):.2f}%)")
    return truncate_delay(H_ad_full, N_DELAYS), {
        "energy_retention_percent": round(float(retention), 3),
        "energy_retention_p5_percent": round(float(np.percentile(per_sample, 5)), 3),
    }


def normalize_per_sample(H_ad: np.ndarray) -> np.ndarray:
    """
    Map each sample to [0,1] with 0.5 as the complex zero point, scaled by its
    own peak so the full range is used.

    Per-sample rather than dataset-wide because path loss spans tens of dB here:
    a global scale would compress every distant user into a sliver around 0.5 and
    reproduce exactly the collapse-to-the-mean failure the synthetic generator
    hit. This discards absolute amplitude, which is standard for CSI feedback --
    the gNodeB needs the channel's direction to beamform, and amplitude is
    signalled separately.

    Dividing by max(|real|, |imag|) rather than max|H| makes the mapping fill
    [0,1] exactly instead of only the middle ~71% of it.

    Scaling by the peak rather than a high quantile looks wasteful -- it leaves
    the resulting tensor with a small standard deviation (~0.018), since sparse
    channels put most of their mass near the zero point. Quantile and RMS scales
    spread the data out considerably more, but they clip, and the clipped values
    are precisely the peaks that carry the energy. Measured round-trip NMSE
    floors, before any model is involved:

        peak      0.000% clipped   no floor (exact)
        q=0.999   0.146% clipped   -6.39 dB
        4*rms     0.335% clipped   -3.43 dB
        q=0.99    1.025% clipped   -1.62 dB

    A -6.39 dB floor would cap the result below what the model already reaches,
    so the peak scale is the only lossless option and the low spread is the
    price. Note this is a different situation from the synthetic generator's
    failure: there a single GLOBAL scale set by dataset-wide outliers squashed
    every sample into ~4% of the range, whereas here each sample sets its own
    scale and reaches 0 and 1 at its own peak.
    """
    peak = np.maximum(
        np.abs(H_ad.real).max(axis=(1, 2)),
        np.abs(H_ad.imag).max(axis=(1, 2)),
    ).astype(np.float32)
    scaled = H_ad / (2.0 * peak[:, None, None]) + (0.5 + 0.5j)
    x = split_complex(scaled.astype(np.complex64)).astype(np.float32)
    # Guard against float32 rounding pushing a peak a few ULP outside [0,1];
    # this clips nothing of substance (measured 0.000% above).
    return np.clip(x, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Prepare a DeepMIMO scenario for DeepCSI.")
    parser.add_argument("--scenario", default="o1_3p5_downloaded")
    parser.add_argument("--output-dir", default="data/processed_deepmimo")
    parser.add_argument("--samples", type=int, default=40000, help="Users to sample.")
    parser.add_argument("--tx-set", type=int, default=5, help="Base station TX set id.")
    parser.add_argument("--rx-set", type=int, default=0, help="User grid RX set id.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== DeepCSI DeepMIMO Dataset Preparation ===")
    H_sf = load_channels(args.scenario, args.tx_set, args.rx_set, args.samples, args.seed)
    print(f"Spatial-frequency channels: {H_sf.shape}")

    H_sf, clean_report = clean(H_sf)
    H_ad, energy_report = to_angular_delay(H_sf)
    print(f"Angular-delay (truncated): {H_ad.shape}")

    data = normalize_per_sample(H_ad)
    print(f"Normalised tensor: {data.shape} {data.dtype} "
          f"range [{data.min():.4f}, {data.max():.4f}] std {data.std():.4f}")

    # Shuffle before splitting. DeepMIMO orders users by grid position, so an
    # unsplit-shuffled dataset would put one end of the street in train and the
    # other in test, turning the val/test score into a generalisation-across-
    # geography measurement rather than a compression measurement.
    perm = np.random.default_rng(args.seed).permutation(len(data))
    data = data[perm]

    n = len(data)
    n_train, n_val = int(0.7 * n), int(0.1 * n)
    splits = {
        "train": data[:n_train],
        "val": data[n_train:n_train + n_val],
        "test": data[n_train + n_val:],
    }

    for name, arr in splits.items():
        assert arr.ndim == 4 and arr.shape[1:] == (2, N_ANTENNAS, N_DELAYS), arr.shape
        assert np.isfinite(arr).all()
        assert arr.min() >= 0.0 and arr.max() <= 1.0
        np.save(out_dir / f"{name}.npy", arr)
        print(f"Saved {name}.npy {arr.shape} ({arr.nbytes / 1e6:.0f} MB)")

    norm_params = {
        "source": f"deepmimo_{args.scenario}_tx{args.tx_set}_rx{args.rx_set}",
        "scheme": "csinet_offset",
        "offset": 0.5,
        "min": -0.5,
        "max": 0.5,
        "normalisation": "per_sample_peak",
        "shape": [2, N_ANTENNAS, N_DELAYS],
        "num_train": len(splits["train"]),
        "num_val": len(splits["val"]),
        "num_test": len(splits["test"]),
        "seed": args.seed,
        "split_origin": "random_shuffle_70_10_20",
        "pipeline": f"raytracing -> H({N_ANTENNAS}x{N_SUBCARRIERS}) -> 2D DFT -> "
                    f"truncate {N_DELAYS} taps -> per-sample [0,1]",
        "cleaning": clean_report,
        "delay_truncation": energy_report,
    }
    with open(out_dir / "norm_params.json", "w") as f:
        json.dump(norm_params, f, indent=2)
    print(f"Saved {out_dir / 'norm_params.json'}")
    print(f"\nTrain with:\n  python models/train.py --data-dir {out_dir} --compression-ratio 4")


if __name__ == "__main__":
    main()
