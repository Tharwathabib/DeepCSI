import sys
from pathlib import Path

import numpy as np

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.quantize import fit_range, uniform_quantize, payload_bits, RAW_CSI_BITS


def latents(n=2000, seed=0):
    return np.random.default_rng(seed).normal(scale=2.0, size=(n, 128)).astype(np.float32)


def test_more_bits_never_increases_error():
    z = latents()
    lo, hi = fit_range(z)
    errs = [np.abs(uniform_quantize(z, b, lo, hi) - z).mean() for b in (2, 3, 4, 6, 8, 10)]
    assert all(a > b for a, b in zip(errs, errs[1:])), f"error not monotonic in bits: {errs}"


def test_output_stays_inside_the_fitted_range():
    z = latents()
    lo, hi = fit_range(z)
    q = uniform_quantize(z, 4, lo, hi)
    assert q.min() >= lo - 1e-5 and q.max() <= hi + 1e-5


def test_high_precision_is_near_lossless_inside_the_range():
    z = latents()
    lo, hi = fit_range(z, quantile=1.0)          # full range, nothing clipped
    q = uniform_quantize(z, 16, lo, hi)
    assert np.abs(q - z).max() < (hi - lo) / (2 ** 16 - 1)


def test_range_is_fitted_not_taken_from_the_data_being_quantised():
    """
    The codebook ships with the model, so a shifted test distribution must
    actually suffer. If it did not, the range would be adapting per call and
    the payload would owe side-information bits that nothing is counting.
    """
    train, test = latents(seed=1), latents(seed=2) + 10.0
    lo, hi = fit_range(train)
    err_matched = np.abs(uniform_quantize(latents(seed=3), 4, lo, hi) - latents(seed=3)).mean()
    err_shifted = np.abs(uniform_quantize(test, 4, lo, hi) - test).mean()
    assert err_shifted > err_matched * 5


def test_payload_accounting():
    # A CR=16 report at 6 bits: 128 latent scalars, 768 bits, 85.3x vs raw.
    assert payload_bits(128, 6) == 768
    assert RAW_CSI_BITS == 65536
    assert round(RAW_CSI_BITS / payload_bits(128, 6), 1) == 85.3
