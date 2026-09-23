"""Resample, trim and mu-law encode LJSpeech into data/{train,val}_{sr}.pt."""

import argparse

from wavenet.data import preprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sr", type=int, default=8000)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--n-clips", type=int, default=None, help="subset for quick tests")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    preprocess(sr=args.sr, val_frac=args.val_frac, n_clips=args.n_clips, seed=args.seed)


if __name__ == "__main__":
    main()
