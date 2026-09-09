import torch
import numpy as np
from typing import Union, Tuple


def nmse_db(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Compute Normalized Mean Squared Error (NMSE) in dB for PyTorch tensors.

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
    Compute Normalized Mean Squared Error (NMSE) in dB for NumPy arrays.

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


def cosine_similarity_torch(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    return_per_sample: bool = False
) -> torch.Tensor:
    """
    Compute complex Generalized Cosine Similarity (rho) for PyTorch tensors.
    rho = |h_pred^H * h_true| / (||h_pred|| * ||h_true||) in [0.0, 1.0]

    Parameters:
        y_pred: Predicted tensor of shape (B, 2, H, W) or (2, H, W)
        y_true: Ground truth tensor of shape (B, 2, H, W) or (2, H, W)
        return_per_sample: If True, returns (B,) tensor; otherwise returns scalar mean.

    Returns:
        Cosine similarity rho (scalar tensor or 1D tensor)
    """
    if y_pred.dim() == 3:
        y_pred = y_pred.unsqueeze(0)
        y_true = y_true.unsqueeze(0)

    # Convert 2-channel real/imag to complex vectors
    h_pred = torch.complex(y_pred[:, 0, :, :], y_pred[:, 1, :, :]).reshape(y_pred.size(0), -1)
    h_true = torch.complex(y_true[:, 0, :, :], y_true[:, 1, :, :]).reshape(y_true.size(0), -1)

    inner_prod = torch.abs(torch.sum(torch.conj(h_pred) * h_true, dim=-1))
    norm_pred = torch.linalg.norm(h_pred, dim=-1)
    norm_true = torch.linalg.norm(h_true, dim=-1)

    rho = inner_prod / (norm_pred * norm_true + 1e-10)
    rho = torch.clamp(rho, 0.0, 1.0)

    return rho if return_per_sample else torch.mean(rho)


def beamforming_gain_torch(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    return_per_sample: bool = False
) -> torch.Tensor:
    """
    Compute normalized beamforming power gain G = rho^2 in [0.0, 1.0] for PyTorch tensors.
    G = 1.0 indicates perfect alignment of the MRT beamformer w = h_pred / ||h_pred||.

    Parameters:
        y_pred: Predicted tensor of shape (B, 2, H, W) or (2, H, W)
        y_true: Ground truth tensor of shape (B, 2, H, W) or (2, H, W)
        return_per_sample: If True, returns (B,) tensor; otherwise returns scalar mean.

    Returns:
        Beamforming power gain G in [0.0, 1.0]
    """
    rho = cosine_similarity_torch(y_pred, y_true, return_per_sample=return_per_sample)
    return rho ** 2


def cosine_similarity_numpy(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    return_per_sample: bool = False
) -> Union[float, np.ndarray]:
    """
    Compute complex Generalized Cosine Similarity (rho) for NumPy arrays.
    rho = |h_pred^H * h_true| / (||h_pred|| * ||h_true||) in [0.0, 1.0]

    Parameters:
        y_pred: Predicted array of shape (B, 2, H, W) or (2, H, W)
        y_true: Ground truth array of shape (B, 2, H, W) or (2, H, W)
        return_per_sample: If True, returns array of shape (B,); otherwise float mean.

    Returns:
        Cosine similarity rho (float or 1D array)
    """
    if y_pred.ndim == 3:
        y_pred = y_pred[np.newaxis, ...]
        y_true = y_true[np.newaxis, ...]

    # Convert 2-channel real/imag to complex vectors
    h_pred = (y_pred[:, 0, :, :] + 1j * y_pred[:, 1, :, :]).reshape(len(y_pred), -1)
    h_true = (y_true[:, 0, :, :] + 1j * y_true[:, 1, :, :]).reshape(len(y_true), -1)

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
    return_per_sample: bool = False
) -> Union[float, np.ndarray]:
    """
    Compute normalized beamforming power gain G = rho^2 in [0.0, 1.0] for NumPy arrays.
    G = 1.0 indicates perfect alignment with ideal MRT precoding.

    Parameters:
        y_pred: Predicted array of shape (B, 2, H, W) or (2, H, W)
        y_true: Ground truth array of shape (B, 2, H, W) or (2, H, W)
        return_per_sample: If True, returns array of shape (B,); otherwise float mean.

    Returns:
        Beamforming power gain G in [0.0, 1.0] (float or 1D array)
    """
    rho = cosine_similarity_numpy(y_pred, y_true, return_per_sample=return_per_sample)
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

