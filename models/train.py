import argparse
import json
import os
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
from utils.metrics import nmse_db
from models.csi_autoencoder import CSIAutoencoder


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_nmse = 0.0
    
    for batch_x in dataloader:
        batch_x = batch_x[0].to(device)
        optimizer.zero_grad()
        
        recon, _ = model(batch_x)
        loss = criterion(recon, batch_x)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(batch_x)
        with torch.no_grad():
            total_nmse += nmse_db(recon, batch_x).item() * len(batch_x)
            
    avg_loss = total_loss / len(dataloader.dataset)
    avg_nmse = total_nmse / len(dataloader.dataset)
    return avg_loss, avg_nmse


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_nmse = 0.0
    
    with torch.no_grad():
        for batch_x in dataloader:
            batch_x = batch_x[0].to(device)
            recon, _ = model(batch_x)
            loss = criterion(recon, batch_x)
            
            total_loss += loss.item() * len(batch_x)
            total_nmse += nmse_db(recon, batch_x).item() * len(batch_x)
            
    avg_loss = total_loss / len(dataloader.dataset)
    avg_nmse = total_nmse / len(dataloader.dataset)
    return avg_loss, avg_nmse


def main():
    parser = argparse.ArgumentParser(description="DeepCSI Autoencoder Training Script")
    parser.add_argument("--compression-ratio", "-cr", type=int, default=16, choices=[4, 16, 32],
                        help="Compression Ratio (4, 16, or 32)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed data directory")
    parser.add_argument("--output-dir", type=str, default="models/weights", help="Directory to save model weights")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== DeepCSI Training | CR={args.compression_ratio} | Device: {device} ===")

    data_path = Path(args.data_dir)
    train_path = data_path / "train.npy"
    val_path = data_path / "val.npy"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Training or validation data not found in {data_path}. Run data/generate_data.py first.")

    train_data = torch.from_numpy(np.load(train_path))
    val_data = torch.from_numpy(np.load(val_path))

    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = CSIAutoencoder(compression_ratio=args.compression_ratio).to(device)
    print(f"Model initialized: latent_dim={model.latent_dim}, total_params={sum(p.numel() for p in model.parameters())}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    weights_dir = Path(args.output_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = weights_dir / f"deepcsi_cr{args.compression_ratio}.pt"

    best_val_nmse = float("inf")
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_nmse = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_nmse = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train NMSE: {train_nmse:6.2f} dB | Val NMSE: {val_nmse:6.2f} dB | LR: {current_lr:.6f}")

        if val_nmse < best_val_nmse:
            best_val_nmse = val_nmse
            best_epoch = epoch
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "compression_ratio": args.compression_ratio,
                "latent_dim": model.latent_dim,
                "epoch": epoch,
                "val_nmse_db": val_nmse,
                "config": vars(args)
            }
            torch.save(checkpoint, best_weights_path)

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.1f}s.")
    print(f"Best Model Saved to '{best_weights_path}' (Epoch {best_epoch:02d}, Val NMSE: {best_val_nmse:.2f} dB)")


if __name__ == "__main__":
    main()
