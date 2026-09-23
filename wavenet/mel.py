import torch
import torchaudio.transforms as AT

SR, N_FFT, HOP, N_MELS = 8000, 512, 128, 80

_mel_fn = AT.MelSpectrogram(
    sample_rate=SR,
    n_fft=N_FFT,
    hop_length=HOP,
    n_mels=N_MELS,
    f_min=0,
    f_max=SR // 2,
    power=1.0,
    center=True,
)


def log_mel(x):
    """Compute the log-magnitude mel spectrogram of a waveform.

    Uses an 80-band mel filterbank over 0 to SR/2 Hz, with n_fft=512 and a
    hop of 128 samples. Magnitudes are clamped to at least 1e-5 before the
    log, so silent frames bottom out at about -11.5 instead of -inf.

    Args:
        x: Waveform of shape (..., T), sampled at SR (8 kHz).

    Returns:
        Log-mel spectrogram of shape (..., N_MELS, F), where
        F = T // HOP + 1 because frames are centred.
    """
    return torch.log(_mel_fn(x).clamp(min=1e-5))


def compute_stats(mels):
    """Compute per-band normalisation statistics over a set of spectrograms.

    All spectrograms are concatenated along the time axis, so each band's
    statistics are taken over every frame in the dataset. Longer clips
    therefore carry more weight.

    Args:
        mels: Sequence of log-mel spectrograms, each of shape (N_MELS, F_i).
            The frame counts F_i may differ.

    Returns:
        A (mean, std) tuple of tensors, each of shape (N_MELS,). std is the
        unbiased (Bessel-corrected) standard deviation.
    """
    mels = torch.cat(mels, dim=1)

    mean = torch.mean(mels, dim=1)
    std = torch.std(mels, 1)
    return mean, std


def standardise(m, mean, std):
    """Normalise a spectrogram to zero mean and unit variance per mel band.

    Args:
        m: Log-mel spectrogram of shape (N_MELS, F).
        mean: Per-band means of shape (N_MELS,), from compute_stats.
        std: Per-band standard deviations of shape (N_MELS,), from
            compute_stats.

    Returns:
        Standardised spectrogram of shape (N_MELS, F).
    """

    # (N_MELS, F) - (N_MELS, 1): each band's stat broadcasts across all frames
    m_z = (m - mean[:, None]) / std[:, None]

    return m_z
