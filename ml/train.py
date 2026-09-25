"""Training loop for the residual U-Net, with per-epoch CSV experiment logging and frequent checkpointing -
per the project guide's own advice: "Save checkpoints often because free sessions can disconnect" (Colab/Kaggle).

Usage: python ml/train.py [--epochs N] [--lr LR] [--batch-size B] [--run-name NAME]
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent / "model"))
from dataset import load_datasets
from unet import ResidualUNet
from loss import HeavyRainWeightedLoss

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
LOG_DIR = Path(__file__).resolve().parent / "logs"


def run_epoch(model, loader, loss_fn, mask, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        x = batch["x"].to(device)
        bicubic_raw = batch["bicubic_raw"].to(device)
        coarse_nn_raw = batch["coarse_nn_raw"].to(device)
        truth = batch["truth"].to(device)

        with torch.set_grad_enabled(is_train):
            pred = model.predict_rainfall(x, bicubic_raw, coarse_nn_raw)
            loss = loss_fn(pred, truth, mask)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{run_name}.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds, test_ds, stats, valid_mask = load_datasets()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    mask = torch.from_numpy(valid_mask).to(device)
    model = ResidualUNet(in_channels=train_ds.input_tensor.shape[1], base_channels=args.base_channels).to(device)
    loss_fn = HeavyRainWeightedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "batch_size", "base_channels", "seed", "timestamp", "elapsed_s"])

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = run_epoch(model, train_loader, loss_fn, mask, device, optimizer)
            val_loss = run_epoch(model, val_loader, loss_fn, mask, device, optimizer=None)
            elapsed = time.time() - t0

            print(f"[{run_name}] epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} ({elapsed:.1f}s)")
            writer.writerow([epoch, train_loss, val_loss, args.lr, args.batch_size, args.base_channels, args.seed, time.strftime("%Y-%m-%d %H:%M:%S"), f"{elapsed:.1f}"])
            f.flush()

            # checkpoint every epoch (not just best) - free-tier sessions can disconnect without warning
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                 "val_loss": val_loss, "stats": stats, "args": vars(args)},
                CHECKPOINT_DIR / f"{run_name}_last.pt",
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"epoch": epoch, "model_state": model.state_dict(), "val_loss": val_loss, "stats": stats, "args": vars(args)},
                    CHECKPOINT_DIR / f"{run_name}_best.pt",
                )

    print(f"Done. Best val_loss: {best_val_loss:.4f}. Log: {log_path}")


if __name__ == "__main__":
    main()
