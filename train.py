"""Train WaveNet on preprocessed mu-law LJSpeech.

Default: unconditional model on data/{train,val}_{sr}.pt (from preprocess.py).
--mel:   mel-conditioned vocoder on data/{train,val}_{SR}_mel.pt (from
         preprocess.py --mel).
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wavenet.data import DATA_DIR, AudioChunks, MelChunks
from wavenet.mel import SR, HOP, N_MELS
from wavenet.model import WaveNet


def to_device(batch, device):
    """Move an (x, y) or (x, c, y) batch to device. c is None if absent."""
    x, *c, y = (t.to(device, non_blocking=True) for t in batch)
    return x, (c[0] if c else None), y


@torch.no_grad()
def evaluate(model, dl, device, amp, max_batches=50):
    model.eval()
    losses = []
    for i, batch in enumerate(dl):
        if i == max_batches:
            break
        x, c, y = to_device(batch, device)
        with amp:
            logits = model(x, c)
        losses.append(F.cross_entropy(logits.float(), y, ignore_index=-1).item())
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
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-compile", action="store_true", help="skip torch.compile")
    p.add_argument(
        "--mel", action="store_true", help="train a mel-conditioned vocoder"
    )
    args = p.parse_args()

    if args.mel:
        if args.sr != SR:
            p.error(f"--mel uses the mel.py settings, which require --sr {SR}")
        if args.T % HOP:
            p.error(f"--mel needs --T to be a multiple of the hop ({HOP})")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("device:", device)

    # mixed precision + TF32 on CUDA: bf16 on Ampere+ (A100, L4), fp16 on older (T4)
    cuda = device == "cuda"
    use_bf16 = cuda and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    amp = torch.autocast("cuda", dtype=amp_dtype, enabled=cuda)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda and not use_bf16)
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("amp:", amp_dtype)

    suffix = "_mel" if args.mel else ""
    train_data = torch.load(args.data_dir / f"train_{args.sr}{suffix}.pt")
    val_data = torch.load(args.data_dir / f"val_{args.sr}{suffix}.pt")

    config = {
        "R": args.R,
        "S": args.S,
        "n_layers": args.n_layers,
        "n_stacks": args.n_stacks,
    }
    if args.mel:
        config |= {"n_mels": N_MELS, "hop": HOP}
    model = WaveNet(**config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # compiled wrapper for forward passes; save checkpoints from `model` so the
    # state_dict keys don't get the "_orig_mod." prefix
    fmodel = model if args.no_compile or not cuda else torch.compile(model)

    T, B = args.T, args.batch_size
    loader_kw = dict(
        batch_size=B,
        num_workers=args.num_workers if cuda else 0,
        pin_memory=cuda,
        persistent_workers=cuda and args.num_workers > 0,
    )
    if args.mel:
        train_ds = MelChunks(
            train_data["codes"], train_data["mels"], T, model.rf, n_items=10_000
        )
        val_ds = MelChunks(
            val_data["codes"], val_data["mels"], T, model.rf, random=False
        )
        # train-split stats, needed to standardise new mels at inference time
        mel_stats = {"mean": train_data["mean"], "std": train_data["std"]}
    else:
        train_ds = AudioChunks(train_data, T, model.rf, n_items=10_000)
        val_ds = AudioChunks(val_data, T, model.rf, random=False)
        mel_stats = {}
    train_dl = DataLoader(train_ds, **loader_kw)
    val_dl = DataLoader(val_ds, drop_last=True, **loader_kw)

    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
    step, best_val = 0, float("inf")

    while step < args.max_steps:
        for batch in train_dl:
            x, c, y = to_device(batch, device)

            # forward
            with amp:
                logits = fmodel(x, c)
            loss = F.cross_entropy(logits.float(), y, ignore_index=-1)

            # backward
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if step % args.eval_every == 0 or step == args.max_steps - 1:
                val = evaluate(fmodel, val_dl, device, amp)
                print(f"step {step}: train {loss.item():.3f}  val {val:.3f}")

                if val < best_val:
                    best_val = val
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "config": config,
                            "step": step,
                            "val": val,
                            **mel_stats,
                        },
                        args.ckpt,
                    )

            step += 1
            if step >= args.max_steps:
                break


if __name__ == "__main__":
    main()
