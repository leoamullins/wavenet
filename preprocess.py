"""Preprocess LJSpeech.

Default: resample, trim and mu-law encode into data/{train,val}_{sr}.pt.
--mel:   also extract standardised log-mels, per clip, into data/{train,val}_{SR}_mel.pt.
"""

import argparse

from wavenet.data import preprocess
from wavenet.data import preprocess_mel, SR


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sr", type=int, default=8000)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--n-clips", type=int, default=None, help="subset for quick tests")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--mel", action="store_true", help="per-clip codes + log-mels for the vocoder"
    )
    args = p.parse_args()
    common = {"val_frac": args.val_frac, "n_clips": args.n_clips, "seed": args.seed}

    if args.mel:
        if args.sr is not None and args.sr != SR:
            p.error(f"--mel uses the mel.py settings, which require --sr {SR}")
        preprocess_mel(**common)
    else:
        preprocess(sr=args.sr or 8000, **common)


if __name__ == "__main__":
    main()
