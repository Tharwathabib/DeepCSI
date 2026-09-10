import argparse
import csv
import json
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


def run_epoch(model, dataloader, criterion, device, optimizer=None, scaler=None, offset=0.0):
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
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
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
    ).to(device)
    print(f"Model initialized: latent_dim={model.latent_dim}, "
          f"refine_widths={tuple(args.refine_widths)}, "
          f"total_params={sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Step on the same quantity used to select the best checkpoint, so the LR
    # schedule and the checkpointing agree on what "improvement" means.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

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
            optimizer=optimizer, scaler=scaler, offset=offset
        )
        val_loss, val_nmse = run_epoch(
            model, val_loader, criterion, device, offset=offset
        )

        # Select and schedule on the same quantity we report.
        scheduler.step(val_nmse)
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


if __name__ == "__main__":
    main()
