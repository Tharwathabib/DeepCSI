import warnings

import torch
import numpy as np
from typing import Union, Optional


# ---------------------------------------------------------------------------
# Normalisation handling
# ---------------------------------------------------------------------------

def warn_if_offset_data_measured_raw(y_true, offset: float, caller: str) -> bool:
    """
    Catch the defect this project was founded on, at the point it happens.

    Every metric here takes norm_params as an OPTIONAL argument and falls back
    to measuring the raw tensor when it is absent. That fallback is correct for
    genuinely zero-centred data and catastrophically wrong for the [0,1] data
    this project actually stores, where a constant 0.5+0.5j DC term inflates
    NMSE by ~35 dB and pins rho near 1.0 for any output at all. Those were the
    original -38.45 dB and 99.98% figures.

    Passing norm_params everywhere was the first fix, and it did not hold: one
    caller was missed, and running it printed -37.61 dB and 99.98% again. So the
    condition is detected from the data instead of trusted to the caller.

    Offset-normalised data sits inside [0,1] with a mean near 0.5. A de-offset
    channel is centred near 0. The two are unmistakable, so this warns only on
    the real mistake. Returns True when it fired, which makes it testable.
    """
    if offset != 0.0:
        return False
    arr = y_true.detach().cpu().numpy() if hasattr(y_true, "detach") else np.asarray(y_true)
    if arr.size == 0:
        return False
    lo, hi, mean = float(arr.min()), float(arr.max()), float(arr.mean())
    if lo < -1e-6 or hi > 1.0 + 1e-6 or abs(mean - 0.5) > 0.05:
        return False
    warnings.warn(
        f"{caller}: measuring data that looks offset-normalised "
        f"(range [{lo:.3f}, {hi:.3f}], mean {mean:.3f}) without norm_params. "
        "The 0.5 DC term inflates NMSE by roughly 35 dB and saturates rho near "
        "1.0 for any prediction. Pass norm_params=json.load(<data-dir>/"
        "norm_params.json).",
        RuntimeWarning,
        stacklevel=3,
    )
    return True


def offset_from_norm_params(norm_params: Optional[dict]) -> float:
    """
    Return the constant that must be subtracted from a normalised tensor before
    it can be read as a complex channel.

    Both normalisation schemes used in this project are affine,
    x = (h - min) / (max - min). Undoing that gives h = x * (max - min) + min.
    Cosine similarity is invariant to the positive real scale (max - min) because
    it cancels in the ratio, but NOT to the additive term, so the shift is the
    only part that matters here:

        h  proportional to  x - (-min / (max - min))

    For CsiNet-convention data (min=-0.5, max=0.5) this evaluates to 0.5,
    matching the reference implementation's `x_real - 0.5 + 1j * (x_imag - 0.5)`.
    For the synthetic min-max data it recovers the correct scheme-specific shift.

    Passing None returns 0.0, i.e. treat the tensor as already zero-centred.
    """
    if not norm_params:
        return 0.0
    if "offset" in norm_params:
        return float(norm_params["offset"])
    lo = float(norm_params.get("min", 0.0))
    hi = float(norm_params.get("max", 1.0))
    span = hi - lo
    if abs(span) < 1e-12:
        return 0.0
    return -lo / span


def to_complex_torch(x: torch.Tensor, norm_params: Optional[dict] = None) -> torch.Tensor:
    """
    Convert a (B, 2, H, W) real/imag tensor into a (B, H*W) complex tensor.

    Removing the normalisation offset first is essential: on [0, 1] data every
    element carries a common ~0.5+0.5j DC component, which dominates any inner
    product and pins cosine similarity near 1.0 regardless of reconstruction
    quality.
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    offset = offset_from_norm_params(norm_params)
    warn_if_offset_data_measured_raw(x, offset, "to_complex_torch")
    real = x[:, 0, :, :] - offset
    imag = x[:, 1, :, :] - offset
    return torch.complex(real, imag).reshape(x.size(0), -1)


def to_complex_numpy(x: np.ndarray, norm_params: Optional[dict] = None) -> np.ndarray:
    """NumPy counterpart of to_complex_torch. Returns a (B, H*W) complex array."""
    if x.ndim == 3:
        x = x[np.newaxis, ...]
    offset = offset_from_norm_params(norm_params)
    warn_if_offset_data_measured_raw(x, offset, "to_complex_numpy")
    real = x[:, 0, :, :] - offset
    imag = x[:, 1, :, :] - offset
    return (real + 1j * imag).reshape(len(x), -1)


# ---------------------------------------------------------------------------
# NMSE
# ---------------------------------------------------------------------------

def nmse_db(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Per-sample NMSE in dB, averaged over the batch: mean(10 * log10(mse / power)).

    This is the mean of the logs. It is a fine training/monitoring signal but is
    NOT the convention used by the CSI feedback literature, so do not report it
    alongside published numbers -- use nmse_db_aggregate for that.

    Parameters:
        y_pred: Predicted/Reconstructed tensor of shape (B, C, H, W)
        y_true: Ground truth tensor of shape (B, C, H, W)

    Returns:
        Mean NMSE in dB across the batch (scalar tensor)
    """
    numerator = torch.sum((y_true - y_pred) ** 2, dim=(1, 2, 3))
    denominator = torch.sum(y_true ** 2, dim=(1, 2, 3)) + 1e-10
    nmse_linear = numerator / denominator
    return torch.mean(10.0 * torch.log10(nmse_linear + 1e-10))


def nmse_db_numpy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    NumPy counterpart of nmse_db (mean of the per-sample logs).

    Parameters:
        y_pred: Predicted array of shape (B, C, H, W) or (C, H, W)
        y_true: Ground truth array of shape (B, C, H, W) or (C, H, W)

    Returns:
        Mean NMSE in dB (float)
    """
    if y_pred.ndim == 3:
        y_pred = y_pred[np.newaxis, ...]
        y_true = y_true[np.newaxis, ...]

    numerator = np.sum((y_true - y_pred) ** 2, axis=(1, 2, 3))
    denominator = np.sum(y_true ** 2, axis=(1, 2, 3)) + 1e-10
    nmse_linear = numerator / denominator
    return float(np.mean(10.0 * np.log10(nmse_linear + 1e-10)))


def nmse_per_sample_db(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    norm_params: Optional[dict] = None
) -> np.ndarray:
    """
    Per-sample NMSE in dB as a (B,) array, measured on the de-offset complex
    channel when norm_params is supplied.

    Use this rather than looping nmse_db_numpy over samples: on normalised data
    the un-offset form divides by the DC-dominated signal power and reports a
    figure roughly 35 dB too optimistic (see nmse_db_aggregate).
    """
    if y_pred.ndim == 3:
        y_pred = y_pred[np.newaxis, ...]
        y_true = y_true[np.newaxis, ...]

    if norm_params:
        h_pred = to_complex_numpy(y_pred, norm_params)
        h_true = to_complex_numpy(y_true, norm_params)
        numerator = np.sum(np.abs(h_true - h_pred) ** 2, axis=-1)
        denominator = np.sum(np.abs(h_true) ** 2, axis=-1) + 1e-10
    else:
        warn_if_offset_data_measured_raw(y_true, 0.0, "nmse_per_sample_db")
        numerator = np.sum((y_true - y_pred) ** 2, axis=(1, 2, 3))
        denominator = np.sum(y_true ** 2, axis=(1, 2, 3)) + 1e-10

    return 10.0 * np.log10(numerator / denominator + 1e-10)


def nmse_db_aggregate(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    norm_params: Optional[dict] = None
) -> torch.Tensor:
    """
    Dataset-level NMSE in dB: 10 * log10(mean(mse / power)) -- the log of the mean.

    This is the convention used by CsiNet, CRNet and the rest of the CSI feedback
    literature, and is the number to quote when comparing against published
    results. It differs from nmse_db(): by Jensen's inequality the mean-of-logs
    form is optimistically biased relative to this one.

    When norm_params is supplied the error and signal power are measured on the
    de-offset complex channel, which is what the reference implementations do.
    Supplying it matters enormously and is not a cosmetic detail: the squared
    error is unchanged by a constant offset because it cancels in the
    subtraction, but the signal power is not. On sparse [0,1]-normalised CSI the
    0.5 DC term dominates sum(x^2), inflating the denominator by ~35 dB and
    making the reported NMSE roughly that much too optimistic.

    Parameters:
        y_pred: Predicted tensor of shape (B, C, H, W)
        y_true: Ground truth tensor of shape (B, C, H, W)
        norm_params: Optional normalisation metadata (see offset_from_norm_params)

    Returns:
        NMSE in dB (scalar tensor)
    """
    if norm_params:
        h_pred = to_complex_torch(y_pred, norm_params)
        h_true = to_complex_torch(y_true, norm_params)
        numerator = torch.sum(torch.abs(h_true - h_pred) ** 2, dim=-1)
        denominator = torch.sum(torch.abs(h_true) ** 2, dim=-1) + 1e-10
    else:
        warn_if_offset_data_measured_raw(y_true, 0.0, "nmse_db_aggregate")
        numerator = torch.sum((y_true - y_pred) ** 2, dim=(1, 2, 3))
        denominator = torch.sum(y_true ** 2, dim=(1, 2, 3)) + 1e-10

    return 10.0 * torch.log10(torch.mean(numerator / denominator) + 1e-10)


def nmse_db_aggregate_numpy(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    norm_params: Optional[dict] = None
) -> float:
    """NumPy counterpart of nmse_db_aggregate. Returns NMSE in dB (float)."""
    if y_pred.ndim == 3:
        y_pred = y_pred[np.newaxis, ...]
        y_true = y_true[np.newaxis, ...]

    if norm_params:
        h_pred = to_complex_numpy(y_pred, norm_params)
        h_true = to_complex_numpy(y_true, norm_params)
        numerator = np.sum(np.abs(h_true - h_pred) ** 2, axis=-1)
        denominator = np.sum(np.abs(h_true) ** 2, axis=-1) + 1e-10
    else:
        warn_if_offset_data_measured_raw(y_true, 0.0, "nmse_db_aggregate_numpy")
        numerator = np.sum((y_true - y_pred) ** 2, axis=(1, 2, 3))
        denominator = np.sum(y_true ** 2, axis=(1, 2, 3)) + 1e-10

    return float(10.0 * np.log10(np.mean(numerator / denominator) + 1e-10))


# ---------------------------------------------------------------------------
# Cosine similarity / beamforming gain
# ---------------------------------------------------------------------------

def cosine_similarity_torch(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    return_per_sample: bool = False,
    norm_params: Optional[dict] = None
) -> torch.Tensor:
    """
    Compute complex Generalized Cosine Similarity (rho) for PyTorch tensors.
    rho = |h_pred^H * h_true| / (||h_pred|| * ||h_true||) in [0.0, 1.0]

    Pass norm_params whenever the tensors are normalised. Without it the DC
    offset inherent to [0, 1] data saturates rho near 1.0 for every model, which
    makes the metric useless for comparing compression ratios.

    Parameters:
        y_pred: Predicted tensor of shape (B, 2, H, W) or (2, H, W)
        y_true: Ground truth tensor of shape (B, 2, H, W) or (2, H, W)
        return_per_sample: If True, returns (B,) tensor; otherwise returns scalar mean.
        norm_params: Optional normalisation metadata (see offset_from_norm_params)

    Returns:
        Cosine similarity rho (scalar tensor or 1D tensor)
    """
    h_pred = to_complex_torch(y_pred, norm_params)
    h_true = to_complex_torch(y_true, norm_params)

    inner_prod = torch.abs(torch.sum(torch.conj(h_pred) * h_true, dim=-1))
    norm_pred = torch.linalg.norm(h_pred, dim=-1)
    norm_true = torch.linalg.norm(h_true, dim=-1)

    rho = inner_prod / (norm_pred * norm_true + 1e-10)
    rho = torch.clamp(rho, 0.0, 1.0)

    return rho if return_per_sample else torch.mean(rho)


def beamforming_gain_torch(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    return_per_sample: bool = False,
    norm_params: Optional[dict] = None
) -> torch.Tensor:
    """
    Compute normalized beamforming power gain G = rho^2 in [0.0, 1.0] for PyTorch tensors.
    G = 1.0 indicates perfect alignment of the MRT beamformer w = h_pred / ||h_pred||.

    Parameters:
        y_pred: Predicted tensor of shape (B, 2, H, W) or (2, H, W)
        y_true: Ground truth tensor of shape (B, 2, H, W) or (2, H, W)
        return_per_sample: If True, returns (B,) tensor; otherwise returns scalar mean.
        norm_params: Optional normalisation metadata (see offset_from_norm_params)

    Returns:
        Beamforming power gain G in [0.0, 1.0]
    """
    rho = cosine_similarity_torch(
        y_pred, y_true, return_per_sample=return_per_sample, norm_params=norm_params
    )
    return rho ** 2


def cosine_similarity_numpy(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    return_per_sample: bool = False,
    norm_params: Optional[dict] = None
) -> Union[float, np.ndarray]:
    """
    Compute complex Generalized Cosine Similarity (rho) for NumPy arrays.
    rho = |h_pred^H * h_true| / (||h_pred|| * ||h_true||) in [0.0, 1.0]

    Parameters:
        y_pred: Predicted array of shape (B, 2, H, W) or (2, H, W)
        y_true: Ground truth array of shape (B, 2, H, W) or (2, H, W)
        return_per_sample: If True, returns array of shape (B,); otherwise float mean.
        norm_params: Optional normalisation metadata (see offset_from_norm_params)

    Returns:
        Cosine similarity rho (float or 1D array)
    """
    h_pred = to_complex_numpy(y_pred, norm_params)
    h_true = to_complex_numpy(y_true, norm_params)

    inner_prod = np.abs(np.sum(np.conj(h_pred) * h_true, axis=-1))
    norm_pred = np.linalg.norm(h_pred, axis=-1)
    norm_true = np.linalg.norm(h_true, axis=-1)

    rho = inner_prod / (norm_pred * norm_true + 1e-10)
    rho = np.clip(rho, 0.0, 1.0)

    if return_per_sample:
        return rho
    return float(np.mean(rho))


def beamforming_gain_numpy(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    return_per_sample: bool = False,
    norm_params: Optional[dict] = None
) -> Union[float, np.ndarray]:
    """
    Compute normalized beamforming power gain G = rho^2 in [0.0, 1.0] for NumPy arrays.
    G = 1.0 indicates perfect alignment with ideal MRT precoding.

    Parameters:
        y_pred: Predicted array of shape (B, 2, H, W) or (2, H, W)
        y_true: Ground truth array of shape (B, 2, H, W) or (2, H, W)
        return_per_sample: If True, returns array of shape (B,); otherwise float mean.
        norm_params: Optional normalisation metadata (see offset_from_norm_params)

    Returns:
        Beamforming power gain G in [0.0, 1.0] (float or 1D array)
    """
    rho = cosine_similarity_numpy(
        y_pred, y_true, return_per_sample=return_per_sample, norm_params=norm_params
    )
    gain = rho ** 2
    if return_per_sample:
        return gain
    return float(gain)


def beamforming_loss_db(gain: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute beamforming power loss in dB: 10 * log10(gain).
    Loss is <= 0.0 dB, where 0.0 dB represents zero beamforming power loss.
    """
    if isinstance(gain, np.ndarray):
        return 10.0 * np.log10(np.clip(gain, 1e-10, 1.0))
    return float(10.0 * np.log10(max(gain, 1e-10)))


def spectral_efficiency(gain: Union[float, np.ndarray], snr_db: float = 10.0) -> Union[float, np.ndarray]:
    """
    Single-user MRT spectral efficiency in bit/s/Hz: log2(1 + snr * G).

    Translates the beamforming gain into the quantity a link budget is actually
    written in. With perfect CSI G = 1 and this reduces to the Shannon rate at
    the given SNR, so comparing against that upper bound gives the throughput
    cost of imperfect feedback.

    Note this is an upper bound on what the compression costs: it assumes the
    only impairment is beam misalignment, ignoring interference, scheduling and
    feedback delay.
    """
    snr_linear = 10.0 ** (snr_db / 10.0)
    return np.log2(1.0 + snr_linear * np.asarray(gain))


def nmse_distribution(nmse_per_sample_db_values: np.ndarray) -> dict:
    """
    Summarise a per-sample NMSE distribution by percentiles rather than mean/std.

    NMSE in dB is heavy-tailed and asymmetric, so "mean +/- std" implies a
    symmetric spread that does not exist and understates the bad tail. p95 is
    the number that answers "how poorly does this do on the users it handles
    worst", which matters more than the average for a scheduler.
    """
    v = np.asarray(nmse_per_sample_db_values, dtype=np.float64)
    p5, p50, p95 = np.percentile(v, [5, 50, 95])
    return {
        "nmse_db_p5": float(p5),
        "nmse_db_median": float(p50),
        "nmse_db_p95": float(p95),
        "nmse_db_worst": float(v.max()),
    }
