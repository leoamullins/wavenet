"""Train WaveNet on preprocessed mu-law LJSpeech."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wavenet.data import DATA_DIR, AudioChunks
from wavenet.model import WaveNet


@torch.no_grad()
def evaluate(model, dl, device, max_batches=50):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(dl):
        if i == max_batches:
            break
        x, y = x.to(device), y.to(device)
        losses.append(F.cross_entropy(model(x), y, ignore_index=-1).item())
    model.train()
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sr", type=int, default=8000)
    p.add_argument("--R", type=int, default=64, help="residual channels")
    p.add_argument("--S", type=int, default=128, help="skip channels")
    p.add_argument("--n-layers", type=int, default=9)
    p.add_argument("--n-stacks", type=int, default=2)
    p.add_argument("--T", type=int, default=4096, help="chunk length in samples")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/ckpt.pt"))
    args = p.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("device:", device)

    train_data = torch.load(DATA_DIR / f"train_{args.sr}.pt")
    val_data = torch.load(DATA_DIR / f"val_{args.sr}.pt")

    config = {
        "R": args.R,
        "S": args.S,
        "n_layers": args.n_layers,
        "n_stacks": args.n_stacks,
    }
    model = WaveNet(**config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    T, B = args.T, args.batch_size
    train_dl = DataLoader(
        AudioChunks(train_data, T, model.rf, n_items=10_000), batch_size=B
    )
    val_dl = DataLoader(AudioChunks(val_data, T, model.rf, random=False), batch_size=B)

    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
    step, best_val = 0, float("inf")

    while step < args.max_steps:
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)

            # forward
            logits = model(x)
            loss = F.cross_entropy(logits, y, ignore_index=-1)

            # backward
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % args.eval_every == 0 or step == args.max_steps - 1:
                val = evaluate(model, val_dl, device)
                print(f"step {step}: train {loss.item():.3f}  val {val:.3f}")

                if val < best_val:
                    best_val = val
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "config": config,
                            "step": step,
                            "val": val,
                        },
                        args.ckpt,
                    )

            step += 1
            if step >= args.max_steps:
                break


if __name__ == "__main__":
    main()
