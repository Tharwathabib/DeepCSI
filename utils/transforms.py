import numpy as np


def spatial_frequency_to_angular_delay(H_sf: np.ndarray) -> np.ndarray:
    """
    Transform Spatial-Frequency CSI (..., Nt, Nc complex) to the Angular-Delay domain.

    Antenna axis (-2): forward DFT, mapping a ULA steering vector to an angle bin.
    Subcarrier axis (-1): INVERSE DFT, mapping a path of delay tau to index tau.

    The direction of the subcarrier transform matters and is not a convention
    choice. A path of delay tau has frequency response exp(-2j*pi*tau*k/Nc), so a
    forward DFT along k places it at index Nc-tau -- the far END of the axis.
    Truncating to the first 32 taps after a forward DFT therefore discards the
    channel and keeps noise. The inverse DFT places it at index tau, which is
    what makes delay truncation valid. This also matches CsiNet, which returns
    to the frequency domain with a forward FFT along the delay axis.
    """
    return np.fft.fft(np.fft.ifft(H_sf, axis=-1), axis=-2)


def angular_delay_to_spatial_frequency(H_ad: np.ndarray) -> np.ndarray:
    """
    Transform Angular-Delay CSI (..., Nt, Nd complex) back to Spatial-Frequency.

    Exact inverse of spatial_frequency_to_angular_delay.
    """
    return np.fft.fft(np.fft.ifft(H_ad, axis=-2), axis=-1)


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

    The denominator is guarded by an exact zero test, not by adding an epsilon.
    Retention is a ratio, so an absolute floor silently rescales it whenever the
    channel carries real path loss: a DeepMIMO user at 8.6e-11 total energy sits
    BELOW a 1e-10 epsilon, and every such user then reports near-zero retention
    no matter how delay-concentrated it actually is. That understated the median
    RX2 user as 34% when the true figure is 99.4%.
    """
    total_energy = float(np.sum(np.abs(H_ad) ** 2))
    if H_ad.ndim == 2:
        truncated_energy = float(np.sum(np.abs(H_ad[:, :max_delay]) ** 2))
    elif H_ad.ndim == 3:
        truncated_energy = float(np.sum(np.abs(H_ad[:, :, :max_delay]) ** 2))
    else:
        raise ValueError(f"Unsupported array dimension: {H_ad.ndim}")

    if total_energy == 0.0:
        return 0.0
    return float(truncated_energy / total_energy * 100.0)
