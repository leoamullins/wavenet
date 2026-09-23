import math
import random
from pathlib import Path

import torch
import soundfile as sf
import torchaudio.functional as AF

ROOT = Path(__file__).parent / "data"
WAV_DIR = ROOT / "LJSpeech-1.1" / "wavs"


# ---------- 1. mu-law ----------
def mu_law_encode(x, mu=255):
    """float in [-1, 1] -> long in 0..mu"""
    F = torch.sign(x) * torch.log1p(mu * x.abs()) / math.log1p(mu)
    return torch.floor((F + 1) / 2 * mu + 0.5).long()


def mu_law_decode(q, mu=255):
    """long in 0..mu -> float in [-1, 1]"""
    y = 2 * q.float() / mu - 1
    return torch.sign(y) * torch.expm1(y.abs() * math.log1p(mu)) / mu


# ---------- 2. one clip ----------
def trim_silence(x, top_db=30):
    """Drop leading/trailing samples quieter than top_db below the peak."""
    thresh = x.abs().max() * 10 ** (-top_db / 20)
    idx = (x.abs() > thresh).nonzero().squeeze(1)
    if idx.numel() == 0:
        return x
    return x[idx[0].item() : idx[-1].item() + 1]


def load_clip(path, sr=8000, top_db=30):
    w, orig_sr = sf.read(path, dtype="float32")
    if w.ndim == 2:  # stereo -> mono (not needed for LJ, but safe)
        w = w.mean(axis=1)
    x = torch.from_numpy(w)
    if orig_sr != sr:
        x = AF.resample(x, orig_sr, sr)
    x = x.clamp(-1, 1)  # resampling can overshoot slightly
    return trim_silence(x, top_db)


# ---------- 3. preprocess (run once) ----------
def preprocess(sr=8000, val_frac=0.05, n_clips=None, seed=0, out_dir=ROOT):
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

    def __init__(self, data, T, rf, n_items=10_000, random=True):
        assert len(data) > T + 1
        assert rf <= T
        self.data = data
        self.T, self.rf, self.random = T, rf, random
        self.n_items = n_items if random else (len(data) - 1) // T

    def __len__(self):
        return self.n_items

    def __getitem__(self, i):
        if self.random:
            off = torch.randint(0, len(self.data) - self.T, (1,)).item()
        else:
            off = i * self.T

        chunk = self.data[off : off + self.T + 1].long()
        x, y = chunk[:-1], chunk[1:].clone()
        y[: self.rf - 1] = -1
        return x, y
