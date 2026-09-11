import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.seed import set_seed
from utils.metrics import offset_from_norm_params
from models.csi_autoencoder import CSIAutoencoder


def nmse_ratio_sum(recon: torch.Tensor, target: torch.Tensor, offset: float = 0.0) -> torch.Tensor:
    """
    Sum of per-sample linear NMSE ratios (mse / power) over a batch.

    Accumulating the ratios lets us finish an epoch with
    10*log10(sum_ratios / n_samples), i.e. the log-of-mean convention used by
    CsiNet, CRNet and the rest of the CSI feedback literature.

    `offset` must be the normalisation offset of the data (0.5 for the CsiNet
    [0,1] convention). The squared error is unaffected by it -- a constant
    cancels in the subtraction -- but the signal power is not: on sparse
    normalised CSI the DC term dominates sum(x^2) and inflates the denominator
    by roughly 35 dB, which makes the reported NMSE that much too optimistic.
    """
    target_c = target - offset
    numerator = torch.sum((target_c - (recon - offset)) ** 2, dim=(1, 2, 3))
    denominator = torch.sum(target_c ** 2, dim=(1, 2, 3)) + 1e-10
    return torch.sum(numerator / denominator)


class NMSELoss(nn.Module):
    """
    Mean per-sample linear NMSE: mean( ||h - h_hat||^2 / ||h||^2 ).

    MSE weights every sample by its absolute energy, so high-energy users
    dominate the gradient and low-energy ones are effectively ignored. That is
    exactly the failure the per-sample NMSE percentiles exposed: the worst users
    score around 0 dB, i.e. no usable reconstruction, while the mean looks fine.

    Normalising per sample makes the training objective the same quantity the
    model is scored on, and gives low-energy users equal weight.
    """
    def __init__(self, offset: float = 0.0):
        super().__init__()
        self.offset = offset

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_c = target - self.offset
        num = torch.sum((target_c - (recon - self.offset)) ** 2, dim=(1, 2, 3))
        den = torch.sum(target_c ** 2, dim=(1, 2, 3)) + 1e-10
        return torch.mean(num / den)


def augment_angle_shift(batch: torch.Tensor) -> torch.Tensor:
    """
    Circularly shift each sample along the antenna/angle axis by a random amount.

    The angular axis is a DFT output over a uniform linear array, so it is
    periodic: rolling it yields a physically valid channel for a user at a
    different angle. The delay axis is NOT periodic in this sense -- tap 0 is
    privileged and energy decays with delay -- so it is left alone.

    Gives up to 32 distinct views of every training sample at zero cost, which
    is aimed squarely at the 2-4 dB train/val gap.

    IMPORTANT: this is only valid when every angle is equally likely. That holds
    for the synthetic generator, whose path angles are drawn uniformly. It does
    NOT hold for a ray-traced scenario with one fixed base station, where users
    occupy a narrow angular sector -- see angular_concentration() below.
    """
    shifts = torch.randint(0, batch.shape[2], (batch.shape[0],), device=batch.device)
    # torch.roll cannot take per-sample shifts, so gather with rolled indices.
    idx = (torch.arange(batch.shape[2], device=batch.device).unsqueeze(0) - shifts.unsqueeze(1)) % batch.shape[2]
    idx = idx.view(batch.shape[0], 1, batch.shape[2], 1).expand_as(batch)
    return torch.gather(batch, 2, idx)


def angular_concentration(data: np.ndarray, offset: float) -> float:
    """
    Peak-to-mean ratio of the dataset's average energy profile over angle bins.

    1.0 means energy is spread evenly over all angles, so angle-shift
    augmentation invents nothing. Large values mean users sit in a narrow
    sector and rolling them fabricates channels that never occur.

    Measured: synthetic 1.1 (uniform path angles), DeepMIMO O1 TX5/RX2 5.2
    (one fixed BS, users along a street, dominant angle in 6 bins of 32).
    Augmenting the latter pinned CR=4 at 0.00 dB for 25 epochs -- the model
    emitting a constant -- while the same run without it reached -9.11 dB in 12.
    """
    sample = np.asarray(data[:4000], dtype=np.float64)
    h = np.abs((sample[:, 0] - offset) + 1j * (sample[:, 1] - offset))
    per_angle = h.sum(axis=2)                       # (n, angle)
    profile = per_angle.mean(axis=0)
    return float(profile.max() / profile.mean())


def build_scheduler(optimizer, name: str, epochs: int, warmup_frac: float = 0.05):
    """
    ReduceLROnPlateau (default) or cosine annealing with linear warmup.

    CRNet attributes a substantial part of its margin over CsiNet to the cosine
    schedule rather than to architecture, and the plateau schedule here decayed
    the LR to ~1e-05 and stalled well before the epoch budget ran out.
    """
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

    warmup = max(1, int(epochs * warmup_frac))

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, dataloader, criterion, device, optimizer=None, scaler=None, offset=0.0,
              augment=False):
    """
    Run one epoch. Trains when an optimizer is supplied, otherwise evaluates.

    Returns (avg_loss, nmse_db) where nmse_db uses the de-offset channel and the
    log-of-mean convention -- the only NMSE this script reports, so that no
    inflated figure is ever written where someone might quote it.
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_ratio = 0.0
    n_samples = len(dataloader.dataset)

    for (batch_x,) in dataloader:
        batch_x = batch_x.to(device, non_blocking=True)
        if augment and training:
            batch_x = augment_angle_shift(batch_x)

        with torch.set_grad_enabled(training):
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    recon, _ = model(batch_x)
                    loss = criterion(recon, batch_x)
            else:
                recon, _ = model(batch_x)
                loss = criterion(recon, batch_x)

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            total_loss += loss.item() * len(batch_x)
            total_ratio += nmse_ratio_sum(recon.float(), batch_x, offset).item()

    avg_loss = total_loss / n_samples
    aggregate_nmse = 10.0 * np.log10(total_ratio / n_samples + 1e-12)
    return avg_loss, float(aggregate_nmse)


def main():
    parser = argparse.ArgumentParser(description="DeepCSI Autoencoder Training Script")
    parser.add_argument("--compression-ratio", "-cr", type=int, default=16, choices=[4, 16, 32],
                        help="Compression Ratio (4, 16, or 32)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    # Default 0, not 1e-5. torch.optim.Adam adds weight_decay*w straight to the
    # gradient, so it competes with the data term on absolute scale. An MSE loss
    # on this data is ~3.7e-4, small enough that 1e-5 decay wins: the weights are
    # pulled to zero, the sigmoid outputs a constant 0.5, and NMSE sits at
    # exactly 0.00 dB. Measured on DeepMIMO RX0, CR=4, MSE loss:
    #   weight_decay 1e-5 -> 0.01 dB after 12 epochs (collapsed)
    #   weight_decay 0    -> -1.99 dB after 10 epochs, still improving
    # It is silent -- training "converges", just to the mean.
    #
    # 0 is not universally better, so set this per dataset. On the synthetic set
    # -- 7000 samples with a 3.63 dB train/val gap -- the regularisation is
    # worth 1.48 dB at CR=4 (-11.22 with 1e-5 against -9.74 without). DeepMIMO
    # has 42000 samples and barely overfits, so it gains nothing and loses
    # everything.
    #
    # The principled fix is torch.optim.AdamW, whose decoupled decay is applied
    # to the weights directly rather than added to the gradient, so it cannot
    # out-shout a small loss. Not adopted here only because switching optimiser
    # invalidates every measured number in the README and needs a re-run on both
    # datasets to verify.
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Path to processed data directory")
    parser.add_argument("--output-dir", type=str, default="models/weights",
                        help="Directory to save model weights")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory for the per-epoch training log")
    parser.add_argument("--train-samples", type=int, default=None,
                        help="Cap the training split at this many samples (for quick runs)")
    parser.add_argument("--patience", type=int, default=None,
                        help="Stop early after this many epochs without validation improvement")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed-precision training (CUDA only)")
    parser.add_argument("--refine-widths", type=int, nargs="+", default=[8, 16],
                        help="Decoder RefineNet block widths. Default 8 16 matches CsiNet.")
    # The four flags below default to the existing behaviour so each can be
    # ablated independently against the committed baseline.
    parser.add_argument("--loss", choices=["mse", "nmse"], default="mse",
                        help="nmse weights every sample equally and matches the reported metric.")
    parser.add_argument("--scheduler", choices=["plateau", "cosine"], default="plateau",
                        help="cosine adds linear warmup then cosine annealing.")
    parser.add_argument("--augment", action="store_true",
                        help="Random circular shift along the angle axis.")
    parser.add_argument("--arch", choices=["csinet", "crnet"], default="csinet",
                        help="crnet uses a multi-resolution encoder (1x9/9x1/3x3 branches).")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== DeepCSI Training | CR={args.compression_ratio} | Device: {device} ===")

    data_path = Path(args.data_dir)
    train_path = data_path / "train.npy"
    val_path = data_path / "val.npy"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Training or validation data not found in {data_path}.\n"
            "Run one of:\n"
            "  python data/generate_data.py                 (synthetic)\n"
            "  python data/prepare_cost2100.py --mat-dir ... (real COST2100)"
        )

    # Carry the dataset's normalisation metadata into the checkpoint so a weight
    # file can never be silently paired with data it was not trained on.
    norm_params = None
    norm_path = data_path / "norm_params.json"
    if norm_path.exists():
        with open(norm_path) as f:
            norm_params = json.load(f)
        print(f"Dataset source: {norm_params.get('source', 'unknown')} "
              f"(scheme={norm_params.get('scheme', 'minmax')})")
    else:
        print(f"WARNING: no norm_params.json in {data_path}.")

    offset = offset_from_norm_params(norm_params)
    print(f"NMSE measured on the de-offset channel (offset={offset:.4f}).")

    train_data = torch.from_numpy(np.load(train_path))
    val_data = torch.from_numpy(np.load(val_path))

    if args.train_samples is not None and args.train_samples < len(train_data):
        idx = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(args.seed))
        train_data = train_data[idx[:args.train_samples]]
        print(f"Training on a {args.train_samples}-sample subset.")

    print(f"Train: {tuple(train_data.shape)} | Val: {tuple(val_data.shape)}")

    pin = device.type == "cuda"
    train_loader = DataLoader(TensorDataset(train_data), batch_size=args.batch_size,
                              shuffle=True, pin_memory=pin)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=args.batch_size,
                            shuffle=False, pin_memory=pin)

    model = CSIAutoencoder(
        compression_ratio=args.compression_ratio,
        refine_widths=tuple(args.refine_widths),
        arch=args.arch,
    ).to(device)
    print(f"Model initialized: arch={args.arch}, latent_dim={model.latent_dim}, "
          f"refine_widths={tuple(args.refine_widths)}, "
          f"total_params={sum(p.numel() for p in model.parameters()):,}")

    criterion = NMSELoss(offset) if args.loss == "nmse" else nn.MSELoss()
    print(f"Loss: {args.loss} | scheduler: {args.scheduler} | augment: {args.augment}")
    if args.augment:
        conc = angular_concentration(train_data, offset)
        print(f"  angular concentration (peak/mean over angle bins): {conc:.1f}")
        if conc > 2.0:
            print("  WARNING: this dataset is not angle-shift invariant. Rolling the "
                  "angle axis\n           fabricates channels from angles that never "
                  "occur here, and has been\n           measured to cost ~9 dB on "
                  "DeepMIMO. Consider dropping --augment.")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.scheduler, args.epochs)

    scaler = None
    if args.amp:
        if device.type == "cuda":
            scaler = torch.amp.GradScaler("cuda")
            print("Mixed precision enabled.")
        else:
            print("WARNING: --amp requested but CUDA is unavailable; running in fp32.")

    weights_dir = Path(args.output_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = weights_dir / f"deepcsi_cr{args.compression_ratio}.pt"

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"training_log_cr{args.compression_ratio}.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "epoch", "train_loss", "train_nmse_db",
        "val_loss", "val_nmse_db", "lr", "elapsed_s",
    ])

    best_val_nmse = float("inf")
    best_epoch = 0
    epochs_since_improvement = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_nmse = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, scaler=scaler, offset=offset, augment=args.augment
        )
        val_loss, val_nmse = run_epoch(
            model, val_loader, criterion, device, offset=offset
        )

        # Plateau needs the monitored metric; cosine steps on epoch count.
        scheduler.step(val_nmse) if args.scheduler == "plateau" else scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time

        marker = ""
        if val_nmse < best_val_nmse:
            best_val_nmse = val_nmse
            best_epoch = epoch
            epochs_since_improvement = 0
            marker = "  *"
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "compression_ratio": args.compression_ratio,
                "latent_dim": model.latent_dim,
                "refine_widths": tuple(args.refine_widths),
                "arch": args.arch,
                "epoch": epoch,
                "val_nmse_db": val_nmse,
                "data_source": (norm_params or {}).get("source", "unknown"),
                "norm_params": norm_params,
                "config": vars(args),
            }
            torch.save(checkpoint, best_weights_path)
        else:
            epochs_since_improvement += 1

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"Train NMSE {train_nmse:7.2f} dB | "
              f"Val NMSE {val_nmse:7.2f} dB | "
              f"LR {current_lr:.2e}{marker}")

        log_writer.writerow([
            epoch, f"{train_loss:.8f}", f"{train_nmse:.4f}",
            f"{val_loss:.8f}", f"{val_nmse:.4f}",
            f"{current_lr:.8f}", f"{elapsed:.1f}",
        ])
        log_file.flush()

        if args.patience is not None and epochs_since_improvement >= args.patience:
            print(f"\nNo validation improvement for {args.patience} epochs; stopping early.")
            break

    log_file.close()
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.1f}s.")
    print(f"Best model saved to '{best_weights_path}' "
          f"(Epoch {best_epoch:02d}, Val NMSE: {best_val_nmse:.2f} dB)")
    print(f"Per-epoch log written to '{log_path}'")

    # A model that emits a constant scores an NMSE of exactly 1.0, i.e. 0 dB,
    # because its error equals the signal. Training looks like it converged --
    # the loss curve is smooth and flat -- so this needs saying out loud.
    if best_val_nmse > -0.5:
        print(f"\nWARNING: best val NMSE is {best_val_nmse:.2f} dB, at or above 0 dB.\n"
              "         The model has almost certainly collapsed to predicting the\n"
              "         mean, which scores exactly 0 dB. Known causes, in order:\n"
              "           - weight decay competing with a small MSE loss (use 0)\n"
              "           - --augment on a dataset that is not angle-shift invariant\n"
              "           - a dataset whose intrinsic rank is far below the latent size")


if __name__ == "__main__":
    main()
