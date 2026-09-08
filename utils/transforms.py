import numpy as np


def spatial_frequency_to_angular_delay(H_sf: np.ndarray) -> np.ndarray:
    """
    Transform Spatial-Frequency CSI matrix (Nt x Nc complex)
    to Angular-Delay domain via 2D DFT.

    H_AD = F_ant * H * F_sub^H  <=> 2D FFT along antenna & subcarrier axes.
    """
    return np.fft.fft2(H_sf)


def angular_delay_to_spatial_frequency(H_ad: np.ndarray) -> np.ndarray:
    """
    Transform Angular-Delay CSI matrix (Nt x Nc complex)
    back to Spatial-Frequency domain via 2D inverse DFT.
    """
    return np.fft.ifft2(H_ad)


def truncate_delay(H_ad: np.ndarray, max_delay: int = 32) -> np.ndarray:
    """
    Truncate angular-delay representation along the delay tap axis (last dimension) to max_delay.
    Input shape: (Nt, Nc) or (B, Nt, Nc)
    Output shape: (Nt, max_delay) or (B, Nt, max_delay)
    """
    if H_ad.ndim == 2:
        return H_ad[:, :max_delay]
    elif H_ad.ndim == 3:
        return H_ad[:, :, :max_delay]
    else:
        raise ValueError(f"Unsupported array dimension: {H_ad.ndim}")


def split_complex(H: np.ndarray) -> np.ndarray:
    """
    Split complex tensor (..., Nt, Nc) into real and imaginary channels.
    Output shape: (..., 2, Nt, Nc) where index 0 is Real, index 1 is Imaginary.
    """
    real_part = np.real(H)
    imag_part = np.imag(H)
    if H.ndim == 2:
        return np.stack([real_part, imag_part], axis=0)
    elif H.ndim == 3:
        return np.stack([real_part, imag_part], axis=1)
    else:
        raise ValueError(f"Unsupported array dimension: {H.ndim}")


def combine_complex(x: np.ndarray) -> np.ndarray:
    """
    Combine (..., 2, Nt, Nc) real/imag stack back into complex tensor (..., Nt, Nc).
    """
    if x.ndim == 3:
        return x[0] + 1j * x[1]
    elif x.ndim == 4:
        return x[:, 0] + 1j * x[:, 1]
    else:
        raise ValueError(f"Unsupported array dimension: {x.ndim}")


def compute_energy_retention(H_ad: np.ndarray, max_delay: int = 32) -> float:
    """
    Compute percentage of total channel energy retained after delay tap truncation.
    """
    total_energy = np.sum(np.abs(H_ad) ** 2)
    if H_ad.ndim == 2:
        truncated_energy = np.sum(np.abs(H_ad[:, :max_delay]) ** 2)
    elif H_ad.ndim == 3:
        truncated_energy = np.sum(np.abs(H_ad[:, :, :max_delay]) ** 2)
    else:
        raise ValueError(f"Unsupported array dimension: {H_ad.ndim}")

    return float((truncated_energy / (total_energy + 1e-10)) * 100.0)
