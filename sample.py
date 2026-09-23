"""Generate audio from a trained WaveNet checkpoint and plot spectrograms."""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
import torch

from wavenet.data import DATA_DIR, mu_law_decode
from wavenet.model import WaveNet


@torch.no_grad()
def generate(model, n_samples, batch_size=1, temperature=1.0, device="cpu"):
    """Sample one step at a time, feeding each output back as input."""
    rf = model.rf
    # start from silence (mu-law 128 ~ 0.0)
    x = torch.full((batch_size, rf), 128, dtype=torch.long, device=device)
    out = []
    for i in range(n_samples):
        logits = model(x[:, -rf:])[:, :, -1] / temperature
        nxt = torch.multinomial(torch.softmax(logits, dim=1), 1)
        x = torch.cat([x[:, 1:], nxt], dim=1)
        out.append(nxt)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n_samples}")
    return torch.cat(out, dim=1).cpu()


def plot_spectrograms(clips, sr, path):
    fig, axes = plt.subplots(len(clips), 1, figsize=(10, 2.2 * len(clips)), sharex=True)
    for ax, (name, wav) in zip(axes, clips):
        ax.specgram(wav.numpy(), Fs=sr, NFFT=256, noverlap=192, cmap="magma")
        ax.set_title(name, fontsize=10, loc="left")
        ax.set_ylabel("Hz")
    axes[-1].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/colab.pt"))
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--n", type=int, default=3, help="number of clips to generate")
    p.add_argument("--temperature", type=float, nargs="+", default=[1.0])
    p.add_argument("--sr", type=int, default=8000)
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    ck = torch.load(args.ckpt, map_location=device)
    model = WaveNet(**ck["config"]).to(device).eval()
    model.load_state_dict(ck["model"])
    print(
        f"step {ck['step']}, val {ck['val']:.3f} nats "
        f"({ck['val'] / math.log(2):.2f} bits/sample), "
        f"receptive field {model.rf} samples ({1000 * model.rf / args.sr:.0f} ms)"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    n = int(args.seconds * args.sr)

    # a real validation clip for reference
    val = torch.load(DATA_DIR / f"val_{args.sr}.pt")
    real = mu_law_decode(val[:n].long())
    sf.write(args.out / "real.wav", real.numpy(), args.sr)
    clips = [("real (LJSpeech val)", real)]

    for t in args.temperature:
        print(f"generating {args.n} x {args.seconds}s at temperature {t}")
        q = generate(model, n, args.n, t, device)
        for i, row in enumerate(q):
            wav = mu_law_decode(row)
            sf.write(args.out / f"sample_t{t}_{i}.wav", wav.numpy(), args.sr)
            if i == 0:
                clips.append((f"generated, temperature {t}", wav))

    plot_spectrograms(clips, args.sr, args.out / "spectrograms.png")
    print(f"wrote {args.out}/")


if __name__ == "__main__":
    main()
