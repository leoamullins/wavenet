"""Train WaveNet on preprocessed mu-law LJSpeech."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wavenet.data import DATA_DIR, AudioChunks
from wavenet.model import WaveNet


@torch.no_grad()
def evaluate(model, dl, device, amp, max_batches=50):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(dl):
        if i == max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with amp:
            logits = model(x)
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
    args = p.parse_args()

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

    train_data = torch.load(args.data_dir / f"train_{args.sr}.pt")
    val_data = torch.load(args.data_dir / f"val_{args.sr}.pt")

    config = {
        "R": args.R,
        "S": args.S,
        "n_layers": args.n_layers,
        "n_stacks": args.n_stacks,
    }
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
    train_dl = DataLoader(
        AudioChunks(train_data, T, model.rf, n_items=10_000), **loader_kw
    )
    val_dl = DataLoader(
        AudioChunks(val_data, T, model.rf, random=False), drop_last=True, **loader_kw
    )

    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
    step, best_val = 0, float("inf")

    while step < args.max_steps:
        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # forward
            with amp:
                logits = fmodel(x)
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
                        },
                        args.ckpt,
                    )

            step += 1
            if step >= args.max_steps:
                break


if __name__ == "__main__":
    main()
