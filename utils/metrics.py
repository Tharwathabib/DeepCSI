import torch
import numpy as np
from typing import Union


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
