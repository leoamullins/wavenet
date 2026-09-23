import math
import random
from pathlib import Path

import torch
import soundfile as sf
import torchaudio.functional as AF

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WAV_DIR = DATA_DIR / "LJSpeech-1.1" / "wavs"


# ---------- 1. mu-law ----------
def mu_law_encode(x, mu=255):
    """Quantise a waveform to mu-law codes.

    Applies mu-law companding, then rounds onto mu + 1 evenly spaced levels.
    Quiet samples get finer resolution than loud ones.

    Args:
        x: Waveform with values in [-1, 1], any shape.
        mu: Largest code. The output has mu + 1 levels.

    Returns:
        Long tensor with the same shape as x and values in 0..mu.
    """
    F = torch.sign(x) * torch.log1p(mu * x.abs()) / math.log1p(mu)
    return torch.floor((F + 1) / 2 * mu + 0.5).long()


def mu_law_decode(q, mu=255):
    """Convert mu-law codes back to a waveform.

    Inverse of mu_law_encode, up to the rounding error from quantisation.

    Args:
        q: Integer codes in 0..mu, any shape.
        mu: Largest code. Must match the value used to encode.

    Returns:
        Float tensor with the same shape as q and values in [-1, 1].
    """
    y = 2 * q.float() / mu - 1
    return torch.sign(y) * torch.expm1(y.abs() * math.log1p(mu)) / mu


# ---------- 2. one clip ----------
def trim_silence(x, top_db=30):
    """Drop leading and trailing samples quieter than top_db below the peak.

    Only the ends are trimmed. Quiet stretches in the middle of the clip
    are kept.

    Args:
        x: 1-D waveform.
        top_db: Threshold in dB below the clip's peak absolute amplitude.

    Returns:
        The slice of x from the first to the last sample above the
        threshold. If no sample is above it (e.g. an all-zero clip), x is
        returned unchanged.
    """
    thresh = x.abs().max() * 10 ** (-top_db / 20)
    idx = (x.abs() > thresh).nonzero().squeeze(1)
    if idx.numel() == 0:
        return x
    return x[idx[0].item() : idx[-1].item() + 1]


def load_clip(path, sr=8000, top_db=30):
    """Load an audio file as a mono, resampled, silence-trimmed waveform.

    Args:
        path: Path to an audio file readable by soundfile.
        sr: Target sample rate in Hz.
        top_db: Silence threshold passed to trim_silence.

    Returns:
        1-D float32 tensor at sr Hz with values clamped to [-1, 1].
    """
    w, orig_sr = sf.read(path, dtype="float32")
    if w.ndim == 2:  # stereo -> mono (not needed for LJ, but safe)
        w = w.mean(axis=1)
    x = torch.from_numpy(w)
    if orig_sr != sr:
        x = AF.resample(x, orig_sr, sr)
    x = x.clamp(-1, 1)  # resampling can overshoot slightly
    return trim_silence(x, top_db)


# ---------- 3. preprocess (run once) ----------
def preprocess(sr=8000, val_frac=0.05, n_clips=None, seed=0, out_dir=DATA_DIR):
    """Encode LJSpeech into one flat stream of mu-law codes per split.

    Clips are shuffled with seed and split into train and val by clip, so
    no clip appears in both. Each clip is loaded with load_clip and
    mu-law encoded. All clips in a split are then concatenated with no
    separator and saved as a uint8 tensor to out_dir/{split}_{sr}.pt.

    Args:
        sr: Target sample rate in Hz.
        val_frac: Fraction of clips held out for validation (at least one).
        n_clips: If set, use only this many clips after shuffling. Useful
            for quick experiments.
        seed: Seed for the shuffle that decides the split.
        out_dir: Directory to write the .pt files to.
    """
    files = sorted(WAV_DIR.glob("*.wav"))
    random.Random(seed).shuffle(files)
    if n_clips is not None:
        files = files[:n_clips]

    n_val = max(1, int(len(files) * val_frac))
    splits = {"val": files[:n_val], "train": files[n_val:]}  # split BY CLIP

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, paths in splits.items():
        encoded = []
        for i, p in enumerate(paths):
            encoded.append(mu_law_encode(load_clip(p, sr)).to(torch.uint8))
            if (i + 1) % 1000 == 0:
                print(f"  {name}: {i + 1}/{len(paths)}")
        data = torch.cat(encoded)
        torch.save(data, out_dir / f"{name}_{sr}.pt")
        print(
            f"{name}: {len(paths)} clips, {len(data):,} samples, {len(data) / sr / 3600:.2f} h"
        )


class AudioChunks(torch.utils.data.Dataset):
    """Fixed-length (input, target) windows cut from a flat stream of codes.

    Each item is a window of T codes, with the target shifted one step
    ahead of the input. The first rf - 1 targets are set to -1 so they can
    be skipped with ignore_index=-1 in the loss. Those positions don't
    have a full receptive field of real context.

    Args:
        data: 1-D tensor of mu-law codes, e.g. from preprocess.
        T: Window length in samples.
        rf: Receptive field of the model in samples. Must be <= T.
        n_items: Number of items per epoch when random is True.
        random: If True, each item starts at a random offset, so an epoch
            is n_items random windows. If False, the stream is cut into
            consecutive non-overlapping windows. Use this for a
            deterministic validation pass.
    """

    def __init__(self, data, T, rf, n_items=10_000, random=True):
        assert len(data) > T + 1
        assert rf <= T
        self.data = data
        self.T, self.rf, self.random = T, rf, random
        self.n_items = n_items if random else (len(data) - 1) // T

    def __len__(self):
        return self.n_items

    def __getitem__(self, i):
        """Return the (x, y) pair for item i.

        Returns:
            x: Long tensor of shape (T,), the input codes.
            y: Long tensor of shape (T,), x shifted left by one, with the
                first rf - 1 entries set to -1.
        """
        if self.random:
            off = torch.randint(0, len(self.data) - self.T, (1,)).item()
        else:
            off = i * self.T

        chunk = self.data[off : off + self.T + 1].long()
        x, y = chunk[:-1], chunk[1:].clone()
        y[: self.rf - 1] = -1
        return x, y

