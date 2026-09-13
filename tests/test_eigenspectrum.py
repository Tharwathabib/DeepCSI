import sys
from pathlib import Path

import numpy as np
import pytest

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from models.eigenspectrum import eigenspectrum, floor_table, rank_for_variance
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


def test_floor_bounds_measured_pca_on_the_same_split():
    """
    The whole point of the floor: no linear compressor can beat it.

    The floor must come from the covariance of the split the score is measured
    on. PCA's basis is fit on train and applied to test, so it cannot beat the
    TEST floor -- that floor is the optimum over all rank-k linear maps on test.
    """
    train, test = sparse_set(1200, seed=1), sparse_set(600, seed=2)
    ev, _, n_features = eigenspectrum(test, NORM)
    floors = floor_table(ev, n_features).set_index("compression_ratio")["linear_nmse_floor_db"]
    measured = evaluate_pca_baseline(train, test, norm_params=NORM)
    measured = measured.set_index("compression_ratio")["nmse_db_aggregate"]

    for cr in (4, 16, 32):
        assert measured[cr] >= floors[cr] - 1e-6, (
            f"CR={cr}: measured PCA {measured[cr]:.2f} dB beats the linear floor "
            f"{floors[cr]:.2f} dB for its own split, which is impossible."
        )


def test_a_floor_from_the_wrong_split_is_not_a_bound():
    """
    Regression guard for the mistake this script shipped with once: a floor
    computed on train says nothing rigorous about a test score, and on the real
    DeepMIMO data the test score beat the train floor by 0.05 dB, which reads as
    a linear method beating its own limit.

    Two independent draws give different covariances, so the train floor and the
    test floor differ. The test asserts they are genuinely not interchangeable.
    """
    def floors_of(data):
        eigenvalues, _, n_features = eigenspectrum(data, NORM)
        table = floor_table(eigenvalues, n_features)
        return table.set_index("compression_ratio")["linear_nmse_floor_db"]

    fa = floors_of(sparse_set(800, seed=11))
    fb = floors_of(sparse_set(800, seed=12))
    assert any(abs(fa[cr] - fb[cr]) > 1e-9 for cr in (4, 16, 32)), (
        "two different samples produced identical floors; the test cannot "
        "distinguish the splits"
    )


def test_floor_improves_with_latent_size():
    """Keeping more components can only shrink the discarded tail."""
    ev, _, n_features = eigenspectrum(sparse_set(600, seed=3), NORM, subsample=600)
    df = floor_table(ev, n_features).set_index("compression_ratio")["linear_nmse_floor_db"]
    assert df[4] < df[16] < df[32], f"floor not monotonic in latent size:\n{df}"


def test_compression_ratio_comes_from_features_not_spectrum_length():
    """
    The compression ratio is a property of the representation (2048 scalars), not
    of how long the eigenvalue array happens to be. A rank-deficient spectrum --
    what any SVD over fewer samples than features returns -- must still be
    labelled CR=32/16/4, not 1/4/9.
    """
    truncated = np.linspace(10.0, 0.1, 300)
    assert len(truncated) < 2048, "precondition: spectrum shorter than the feature count"
    table = floor_table(truncated, n_features=2048).set_index("latent_dim")
    assert table["compression_ratio"].tolist() == [32, 16, 4]
    # k=512 runs past the supplied spectrum, so the discarded tail is all zeros.
    assert table.loc[512, "variance_retained_percent"] == 100.0
    # k=64 and k=128 sit inside it and must still lose something.
    assert table.loc[128, "variance_retained_percent"] < 100.0


def test_offset_is_removed_before_the_svd():
    """
    A 0.5 DC term is a rank-one direction of enormous variance. If it survived
    into the covariance it would dominate the first eigenvalue and flatter every
    floor -- the same offset mistake that cost this project 38 dB on NMSE.

    Mean-centring absorbs a constant, so the spectrum must be identical whether
    or not the offset is declared.
    """
    data = sparse_set(400, seed=4)
    with_offset, _, _ = eigenspectrum(data, NORM, subsample=400)
    without, _, _ = eigenspectrum(data, None, subsample=400)
    np.testing.assert_allclose(with_offset, without, rtol=1e-9, atol=1e-9)


def test_rank_is_the_smallest_count_reaching_the_threshold():
    """rank_for_variance must return a count that clears the bar, and no slack."""
    ev, _, _ = eigenspectrum(sparse_set(600, seed=5), NORM, subsample=600)
    cumulative = np.cumsum(ev) / ev.sum()
    for frac, k in rank_for_variance(ev, fractions=(0.5, 0.9, 0.99)).items():
        assert cumulative[k - 1] >= frac, f"{k} components fall short of {frac}"
        assert k == 1 or cumulative[k - 2] < frac, f"{k} components is one too many for {frac}"


def test_constant_data_is_rejected_not_silently_scored():
    """
    A constant dataset has no variance to retain. Dividing by a zero total would
    yield nan floors that read as "no information lost"; the failure has to be
    loud instead.
    """
    with pytest.raises(ValueError, match="zero total variance"):
        floor_table(np.zeros(2048), 2048)
