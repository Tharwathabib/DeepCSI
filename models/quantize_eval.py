"""
Rate-distortion of the DeepCSI feedback latent.

Answers the question the scalar-reduction figures cannot: for a given number of
FEEDBACK BITS, how much reconstruction accuracy do you get? Produces the
bits-per-report figure that belongs on the slide in place of "93.75% smaller".

The quantiser range is fitted on the training latents and reused for test, so
nothing about the test set leaks into the codebook.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from models.csi_autoencoder import CSIAutoencoder, build_from_checkpoint
from utils.metrics import nmse_db_aggregate_numpy, beamforming_gain_numpy
from utils.quantize import fit_range, uniform_quantize, payload_bits, RAW_CSI_BITS

BIT_WIDTHS = [2, 3, 4, 5, 6, 8]


def main():
    parser = argparse.ArgumentParser(description="Quantised feedback rate-distortion.")
    parser.add_argument("--data-dir", default="data/processed_deepmimo")
    parser.add_argument("--weights-dir", default="models/weights_deepmimo")
    parser.add_argument("--results-dir", default="results_deepmimo")
    parser.add_argument("--test-samples", type=int, default=4000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    norm_path = data_dir / "norm_params.json"
    norm_params = json.loads(norm_path.read_text()) if norm_path.exists() else None

    train = np.load(data_dir / "train.npy")
    test = np.load(data_dir / "test.npy")
    if len(test) > args.test_samples:
        test = test[np.random.default_rng(0).choice(len(test), args.test_samples, replace=False)]
    # Fitting the range needs only enough latents to see the distribution.
    fit_src = train[np.random.default_rng(1).choice(len(train), min(4000, len(train)), replace=False)]

    rows = []
    for cr in (4, 16, 32):
        ckpt_path = Path(args.weights_dir) / f"deepcsi_cr{cr}.pt"
        if not ckpt_path.exists():
            print(f"skipping CR={cr}: {ckpt_path} not found")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_from_checkpoint(ckpt, cr, "cpu")

        with torch.no_grad():
            z_fit = model.encoder(torch.from_numpy(fit_src)).numpy()
            z_test = model.encoder(torch.from_numpy(test)).numpy()
        lo, hi = fit_range(z_fit)

        for bits in [None] + BIT_WIDTHS:
            zq = z_test if bits is None else uniform_quantize(z_test, bits, lo, hi)
            with torch.no_grad():
                recon = model.decoder(torch.from_numpy(zq)).numpy()

            nmse = nmse_db_aggregate_numpy(recon, test, norm_params)
            gain = float(np.mean([
                beamforming_gain_numpy(recon[i], test[i], norm_params=norm_params)
                for i in range(len(test))
            ]))
            n_bits = (model.latent_dim * 32 if bits is None
                      else payload_bits(model.latent_dim, bits))

            rows.append({
                "compression_ratio": cr,
                "latent_dim": model.latent_dim,
                "bits_per_scalar": "float32" if bits is None else bits,
                "feedback_bits": n_bits,
                "compression_vs_raw": round(RAW_CSI_BITS / n_bits, 1),
                "nmse_db": round(float(nmse), 2),
                "beamforming_gain_pct": round(gain * 100, 2),
            })
            label = "float32" if bits is None else f"{bits}-bit "
            print(f"CR={cr:2d} | {label:>7s} | {n_bits:6d} bits/report "
                  f"({RAW_CSI_BITS / n_bits:6.1f}x vs raw) | NMSE {float(nmse):7.2f} dB "
                  f"| G {gain*100:5.2f}%")

    df = pd.DataFrame(rows)
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "quantization.csv"
    df.to_csv(out_path, index=False)
    print(f"\nRaw CSI report for reference: {RAW_CSI_BITS} bits "
          f"(2x32x32 float32).\nSaved to '{out_path}'")


if __name__ == "__main__":
    main()
