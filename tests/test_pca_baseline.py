import sys
from pathlib import Path

import numpy as np
import pytest

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from models.pca_baseline import fit_pca, evaluate_pca_baseline

NORM = {"scheme": "csinet_offset", "offset": 0.5, "min": -0.5, "max": 0.5}


def sparse_set(n=400, seed=0):
    """Delay-sparse tensors in [0,1] with 0.5 as zero, like the real datasets."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 2, 32, 32), dtype=np.float32)
    for i in range(n):
        for _ in range(6):
            a, d = rng.integers(0, 32), rng.integers(0, 8)
            x[i, :, a, d] += rng.normal(scale=0.15, size=2)
    return np.clip(x + 0.5, 0.0, 1.0)


def test_full_rank_basis_reconstructs_exactly():
    """A basis with as many components as dimensions is a change of coordinates."""
    data = sparse_set(300)
    mean, comps = fit_pca(data, max_components=2048)
    flat = data.reshape(len(data), -1).astype(np.float64)
    recon = (flat - mean) @ comps.T @ comps + mean
    assert np.abs(recon - flat).max() < 1e-8


def test_nmse_improves_monotonically_with_components():
    """More retained components can never reconstruct worse."""
    train, test = sparse_set(500, seed=1), sparse_set(200, seed=2)
    df = evaluate_pca_baseline(train, test, norm_params=NORM)
    by_cr = df.set_index("compression_ratio")["nmse_db_aggregate"]
    assert by_cr[4] < by_cr[16] < by_cr[32], f"not monotonic in latent size:\n{by_cr}"


def test_basis_is_fit_on_train_not_test():
    """
    Fitting on test would leak. A basis fit on an UNRELATED training set must
    score strictly worse than one fit on data matching the test distribution --
    if it does not, the function is looking at the test set.
    """
    test = sparse_set(200, seed=2)
    matched = evaluate_pca_baseline(sparse_set(500, seed=3), test, norm_params=NORM)
    # A mismatched train set: energy at long delays instead of short ones.
    rng = np.random.default_rng(9)
    other = np.clip(np.float32(rng.normal(scale=0.15, size=(500, 2, 32, 32))) + 0.5, 0, 1)
    mismatched = evaluate_pca_baseline(other, test, norm_params=NORM)

    m = matched.set_index("compression_ratio")["nmse_db_aggregate"]
    x = mismatched.set_index("compression_ratio")["nmse_db_aggregate"]
    assert m[16] < x[16], (
        f"basis fit on unrelated data scored {x[16]:.2f} dB vs {m[16]:.2f} dB on "
        "matched data -- the basis is not coming from the training set"
    )


def test_output_schema_matches_dct_baseline():
    """Both baselines feed the same metrics.csv, so the columns must line up."""
    pytest.importorskip("scipy")
    from models.dct_baseline import evaluate_dct_baseline

    test = sparse_set(120, seed=4)
    pca = evaluate_pca_baseline(sparse_set(300, seed=5), test, norm_params=NORM)
    dct = evaluate_dct_baseline(test, norm_params=NORM)
    missing = set(dct.columns) - set(pca.columns)
    assert not missing, f"PCA baseline is missing columns the DCT baseline emits: {missing}"
