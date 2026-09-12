"""
Ablation runner for the DeepCSI training recipe.

Runs the committed baseline plus one change at a time, then all changes
together, at a fixed compression ratio and epoch budget. The point is to
attribute any improvement to a specific cause rather than asserting that a
bundle of changes helped.

Each variant trains from the same seed on the same data, so differences are
attributable to the flag under test (up to run-to-run variance, which the
`baseline` row re-run at a second seed would bound -- not done here to keep the
sweep short).

Usage:
    python models/ablation.py --compression-ratio 16 --epochs 50
"""

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# name -> extra CLI flags on top of the shared base command
VARIANTS = {
    "baseline":      [],
    "nmse_loss":     ["--loss", "nmse"],
    "cosine_lr":     ["--scheduler", "cosine"],
    "augment":       ["--augment"],
    "crnet_encoder": ["--arch", "crnet"],
    "all_combined":  ["--loss", "nmse", "--scheduler", "cosine", "--augment", "--arch", "crnet"],
}


def main():
    parser = argparse.ArgumentParser(description="DeepCSI training-recipe ablation")
    parser.add_argument("--compression-ratio", "-cr", type=int, default=16, choices=[4, 16, 32])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    # Pinned explicitly, not inherited. The first version of this sweep ran at
    # the then-default 1e-5 and every verdict it produced had to be withdrawn:
    # Adam couples weight decay to the gradient, and the nmse_loss arm's values
    # are ~2700x larger than the MSE arms', so that one variant was effectively
    # immune to a decay the others were fighting. 0 puts every arm on the same
    # footing, which is the only way the comparison means anything.
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--out", type=str, default="results/ablation.csv")
    parser.add_argument("--work-dir", type=str, default="results/_ablation")
    args = parser.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, extra in VARIANTS.items():
        cmd = [
            sys.executable, "models/train.py",
            "--compression-ratio", str(args.compression_ratio),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--data-dir", args.data_dir,
            "--seed", str(args.seed),
            "--weight-decay", str(args.weight_decay),
            "--output-dir", str(work / name),
            "--results-dir", str(work / name),
        ] + extra

        print(f"\n=== {name} ===\n{' '.join(extra) or '(committed defaults)'}", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
            raise SystemExit(f"variant {name} failed")

        log = pd.read_csv(work / name / f"training_log_cr{args.compression_ratio}.csv")
        best = log.loc[log.val_nmse_db.idxmin()]
        rows.append({
            "variant": name,
            "flags": " ".join(extra) or "(defaults)",
            "best_epoch": int(best.epoch),
            "train_nmse_db": round(float(best.train_nmse_db), 2),
            "val_nmse_db": round(float(best.val_nmse_db), 2),
            # The gap is the quantity of interest: the models overfit, so a
            # change that closes the gap is doing the job we picked it for.
            "generalisation_gap_db": round(float(best.train_nmse_db - best.val_nmse_db), 2),
            "minutes": round(elapsed / 60.0, 1),
        })
        print(f"  best val {best.val_nmse_db:.2f} dB "
              f"(gap {best.train_nmse_db - best.val_nmse_db:.2f} dB) in {elapsed/60:.1f} min",
              flush=True)

    df = pd.DataFrame(rows)
    base = df.loc[df.variant == "baseline", "val_nmse_db"].iloc[0]
    df["delta_vs_baseline_db"] = (df.val_nmse_db - base).round(2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\n=== Ablation: CR={args.compression_ratio}, {args.epochs} epochs ===")
    print(df.to_string(index=False))
    print(f"\nNegative delta_vs_baseline_db = better. Saved to {args.out}")


if __name__ == "__main__":
    main()
