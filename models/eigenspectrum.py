"""
Eigenspectrum analysis: why PCA is competitive at one compression ratio and
hopeless at the next.

`models/pca_baseline.py` measures what PCA actually scores. This script explains
it. The PCA reconstruction error at k components is exactly the variance left in
the discarded tail of the covariance eigenspectrum, so

    NMSE_floor(k) = 10 * log10( sum(ev[k:]) / sum(ev) )

is a lower bound on what ANY linear compressor with a k-dimensional latent can
achieve on the split whose covariance produced those eigenvalues.

**The split matters, and getting it wrong stops it being a bound.** The reported
PCA and DeepCSI numbers are measured on TEST, so the floor has to come from the
test covariance; a floor computed on train is a bound on train reconstruction
only, and the test set can beat it by chance. It does here -- on the DeepMIMO
track the train floor at k=128 is -4.38 dB while PCA scores -4.43 dB on test,
which looks like a linear method beating its own limit and is really just two
different samples. This script therefore defaults to `--split test`.

PCA fit on train and applied to test still lands at or above the test floor,
because the floor is the optimum over all rank-k linear maps on test and PCA's
basis was not fit there.

The bound is what makes the cross-track comparison interpretable. DeepCSI beats
PCA by ~8 dB at CR=16 on DeepMIMO but LOSES to it by 1.9 dB at CR=4, and the
reason is not model capacity: it is that this data's spectrum puts 98.6% of its
energy in the first 512 components, so a 512-dim linear latent is barely a
bottleneck at all. Reporting the floor alongside the measured score turns that
from a surprising result into an arithmetic one.

Run on both tracks to compare:
    python models/eigenspectrum.py --data-dir data/processed \
        --results-dir results
    python models/eigenspectrum.py --data-dir data/processed_deepmimo \
        --results-dir results_deepmimo
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.metrics import offset_from_norm_params


def eigenspectrum(train_data: np.ndarray, norm_params: dict, subsample: int = 0,
                  seed: int = 0, chunk: int = 4096):
    """
    Covariance eigenvalues of the flattened training tensor, descending.

    The offset is removed first so the spectrum describes the complex channel
    rather than a DC term -- the same de-offset convention the metrics use. Note
    that mean-centring would absorb a constant offset anyway; removing it
    explicitly keeps this function honest if the data is ever stored unoffset.

    Computed from the exact 2048x2048 scatter matrix, accumulated in chunks, not
    from an SVD of a sample. **Subsampling biases this measurement and the bias
    is not small.** With n samples the empirical covariance has rank <= n-1, so a
    subsample makes the leading components look like they explain more of the
    variance than they do, and the floor at large k comes out too optimistic.
    Measured on the DeepMIMO track, latent 512:

        8,000-sample SVD  ->  -18.41 dB   (and -18.16 to -18.41 across seeds)
        full 42,000       ->  -17.54 dB

    0.87 dB, all of it flattering the linear baseline at exactly the compression
    ratio where PCA is competitive. `subsample` is kept only to demonstrate that
    effect; leave it at 0 for the real number.

    Returns (eigenvalues, n_used, n_features).
    """
    X = train_data.reshape(len(train_data), -1)
    offset = offset_from_norm_params(norm_params)

    n_used = len(X) if subsample <= 0 else min(len(X), subsample)
    if n_used < len(X):
        idx = np.random.default_rng(seed).choice(len(X), n_used, replace=False)
        X = X[idx]

    n_features = X.shape[1]
    # Two streaming passes: the mean, then the centred scatter matrix. Chunked so
    # peak memory is chunk x 2048 float64 rather than the whole 42k x 2048.
    total = np.zeros(n_features, dtype=np.float64)
    for start in range(0, n_used, chunk):
        total += X[start:start + chunk].astype(np.float64).sum(axis=0)
    mean = total / n_used
    if offset:
        # The offset shifts the mean by a constant and cancels on centring; carry
        # it explicitly so the arithmetic is the same either way.
        mean = mean - offset

    scatter = np.zeros((n_features, n_features), dtype=np.float64)
    for start in range(0, n_used, chunk):
        block = X[start:start + chunk].astype(np.float64)
        if offset:
            block = block - offset
        block = block - mean
        scatter += block.T @ block

    # Symmetric and positive semi-definite, so eigvalsh is exact and cheaper than
    # an SVD. Clamp the tiny negative eigenvalues rounding produces at the tail.
    eigenvalues = np.linalg.eigvalsh(scatter)[::-1]
    return np.clip(eigenvalues, 0.0, None), n_used, n_features


def floor_table(eigenvalues: np.ndarray, n_features: int = 2048,
                latent_dims=(64, 128, 512)) -> pd.DataFrame:
    """
    Variance retained and the implied linear NMSE floor at each latent size.

    `n_features` is the ambient dimension (2048 here), which sets the compression
    ratio. It is NOT len(eigenvalues): an SVD over n samples returns only
    min(n, n_features) values, so reading the ratio off the array length reports
    CR=1 for a small batch. The eigenvalues beyond that count are exactly zero --
    the data cannot have rank above its sample count -- so the total variance and
    the residual at any k are still correct.
    """
    total = eigenvalues.sum()
    if total <= 0:
        raise ValueError("Eigenspectrum has zero total variance; the data is constant.")

    cumulative = np.cumsum(eigenvalues) / total
    rows = []
    for k in latent_dims:
        if k > n_features:
            continue
        # Fewer eigenvalues than k means the tail is all zeros: nothing is lost.
        retained = float(cumulative[k - 1]) if k <= len(eigenvalues) else 1.0
        residual = max(1.0 - retained, 1e-15)
        rows.append({
            "latent_dim": k,
            "compression_ratio": n_features // k,
            "variance_retained_percent": round(retained * 100, 4),
            "residual_variance": residual,
            "linear_nmse_floor_db": round(10 * np.log10(residual), 4),
        })
    return pd.DataFrame(rows)


def rank_for_variance(eigenvalues: np.ndarray, fractions=(0.9, 0.99)) -> dict:
    """Smallest component count reaching each cumulative-variance fraction."""
    cumulative = np.cumsum(eigenvalues) / eigenvalues.sum()
    return {f: int(np.searchsorted(cumulative, f) + 1) for f in fractions}


def main():
    parser = argparse.ArgumentParser(description="DeepCSI eigenspectrum / linear NMSE floor")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--subsample", type=int, default=0,
                        help="Estimate from a random subsample instead of all "
                             "training data. Biases the floor optimistically at "
                             "large latent sizes (0.87 dB at k=512 on DeepMIMO); "
                             "for demonstrating that effect, not for reporting.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["test", "train"], default="test",
                        help="Split whose covariance defines the floor. Defaults "
                             "to test, the split the reported metrics use -- a "
                             "train floor does not bound a test score.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    split_path = data_dir / f"{args.split}.npy"
    norm_path = data_dir / "norm_params.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"{args.split.capitalize()} data not found at {split_path}.\n"
            "Run one of:\n"
            "  python data/generate_data.py --samples 50000  (synthetic, default track)\n"
            "  python data/prepare_deepmimo.py               (real ray-traced)"
        )
    if not norm_path.exists():
        raise FileNotFoundError(
            f"{norm_path} not found. It carries the offset that separates the "
            "complex channel from the 0.5 DC term."
        )

    norm_params = json.loads(norm_path.read_text())
    split_data = np.load(split_path)
    source = norm_params.get("source", "unknown")
    print(f"=== DeepCSI Eigenspectrum | source: {source} | split: {args.split} ===")

    eigenvalues, n_used, n_features = eigenspectrum(
        split_data, norm_params, subsample=args.subsample, seed=args.seed,
    )
    ranks = rank_for_variance(eigenvalues)
    print(f"Samples used: {n_used} of {len(split_data)} | features: {n_features}")
    print(f"Intrinsic rank: {ranks[0.9]} components for 90% of variance, "
          f"{ranks[0.99]} for 99%")

    df = floor_table(eigenvalues, n_features=n_features)
    df.insert(0, "dataset_source", source)
    df.insert(1, "split", args.split)
    df.insert(2, "n_samples_used", n_used)
    df.insert(3, "rank_90pct_variance", ranks[0.9])
    df.insert(4, "rank_99pct_variance", ranks[0.99])

    print()
    print(df[["latent_dim", "compression_ratio", "variance_retained_percent",
              "linear_nmse_floor_db"]].to_string(index=False))

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "eigenspectrum.csv"
    df.to_csv(out, index=False)
    print(f"\nEigenspectrum floors saved to '{out}'")


if __name__ == "__main__":
    main()
