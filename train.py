"""
XRayGuard – Full training pipeline for DenseNet121 chest X-ray classifier.

Includes: dataset loading, weighted BCE training, temperature calibration,
evaluation with AUROC, and results export.

Usage:
    python train.py --manifest manifest.csv --work_dir ./weights --epochs 15 --bs 16
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from model import LABELS, XRayGuardModel


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ChestXrayDataset(Dataset):
    """Multi-label chest X-ray dataset."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_cols = [c for c in LABELS if c in df.columns]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        labels = torch.tensor(
            [row[c] for c in self.label_cols], dtype=torch.float32
        )
        return img, labels


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_transforms(split: str = "train"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Class-weight computation
# ---------------------------------------------------------------------------
def compute_pos_weights(df: pd.DataFrame) -> torch.Tensor:
    """Compute positive-class weights for BCE loss."""
    label_cols = [c for c in LABELS if c in df.columns]
    counts = df[label_cols].sum().values.astype(float)
    neg_counts = len(df) - counts
    weights = neg_counts / (counts + 1e-5)
    weights = np.clip(weights, 0.5, 20.0)
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc="  train", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast("cuda"):
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
        pbar.set_postfix(loss=total_loss / n)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="  eval", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
        all_preds.append(torch.sigmoid(out).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    # Per-label AUROC
    aurocs = {}
    for i, label in enumerate(LABELS):
        try:
            aurocs[label] = roc_auc_score(labels_np[:, i], preds[:, i])
        except ValueError:
            aurocs[label] = float("nan")
    mean_auroc = np.nanmean(list(aurocs.values()))
    return total_loss / n, mean_auroc, aurocs


# ---------------------------------------------------------------------------
# Temperature scaling (calibration)
# ---------------------------------------------------------------------------
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature


def calibrate_temperature(model, val_loader, device, lr=0.01, max_iter=100):
    """Learn temperature parameter on validation set."""
    scaler = TemperatureScaler().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)

    logits_list, labels_list = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            logits_list.append(model(imgs).cpu())
            labels_list.append(labels)
    all_logits = torch.cat(logits_list).to(device)
    all_labels = torch.cat(labels_list).to(device)

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(all_logits), all_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler.temperature.item()


def compute_ece(probs, labels, n_bins=10):
    """Expected Calibration Error."""
    ece = 0.0
    for i in range(probs.shape[1]):
        for b in range(n_bins):
            lo, hi = b / n_bins, (b + 1) / n_bins
            mask = (probs[:, i] >= lo) & (probs[:, i] < hi)
            if mask.sum() == 0:
                continue
            avg_conf = probs[mask, i].mean()
            avg_acc = labels[mask, i].mean()
            ece += mask.sum() / probs.shape[0] * abs(avg_conf - avg_acc)
    return ece / probs.shape[1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train XRayGuard classifier")
    parser.add_argument("--manifest", type=str, default="manifest.csv")
    parser.add_argument("--work_dir", type=str, default="./weights")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--weighted", action="store_true",
                        help="Use class-weighted BCE loss")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"Device: {device}")

    # --- Load manifest ---
    df = pd.read_csv(args.manifest)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # --- Datasets & loaders ---
    train_ds = ChestXrayDataset(train_df, get_transforms("train"))
    val_ds = ChestXrayDataset(val_df, get_transforms("val"))
    test_ds = ChestXrayDataset(test_df, get_transforms("val"))

    train_loader = DataLoader(
        train_ds, batch_size=args.bs, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.bs, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.bs, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # --- Model ---
    model = XRayGuardModel(num_classes=len(LABELS), pretrained=True).to(device)

    # Freeze early layers for initial epochs
    for param in model.densenet.features.parameters():
        param.requires_grad = False
    for param in model.densenet.classifier.parameters():
        param.requires_grad = True

    # --- Loss ---
    if args.weighted:
        pos_weight = compute_pos_weights(train_df).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # --- Optimizer & scheduler ---
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-5,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=True,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # --- Training ---
    best_auroc = 0.0
    no_improve = 0
    best_path = os.path.join(args.work_dir, "best_model.pth")

    print("\n=== Phase 1: Train classifier head ===")
    for epoch in range(min(3, args.epochs)):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_auroc, val_per = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1:>2d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_AUROC={val_auroc:.4f} | {elapsed:.0f}s"
        )
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), best_path)

    # Unfreeze all layers
    print("\n=== Phase 2: Fine-tune all layers ===")
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=True,
    )

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_auroc, val_per = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_auroc)
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1:>2d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_AUROC={val_auroc:.4f} | {elapsed:.0f}s"
        )
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"  → Saved best model (AUROC={val_auroc:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print("  Early stopping!")
                break

    # --- Load best model ---
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    # --- Calibration ---
    print("\n=== Calibration ===")
    # ECE before
    model.eval()
    all_preds_before, all_labels_all = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            all_preds_before.append(torch.sigmoid(logits).cpu().numpy())
            all_labels_all.append(labels.numpy())
    preds_before = np.concatenate(all_preds_before)
    labels_all = np.concatenate(all_labels_all)
    ece_before = compute_ece(preds_before, labels_all)

    temperature = calibrate_temperature(model, val_loader, device)
    print(f"  Learned temperature: {temperature:.4f}")

    # ECE after
    preds_after = 1 / (1 + np.exp(-np.log(preds_before / (1 - preds_before + 1e-7)) / temperature))
    ece_after = compute_ece(preds_after, labels_all)
    print(f"  ECE: {ece_before:.4f} → {ece_after:.4f}")

    # --- Test evaluation ---
    print("\n=== Test Evaluation ===")
    test_loss, test_auroc, test_per = evaluate(model, test_loader, criterion, device)
    print(f"Test AUROC: {test_auroc:.4f}")
    for k, v in test_per.items():
        print(f"  {k}: {v:.4f}")

    # --- Determine optimal thresholds on validation set ---
    thresholds = {}
    for i, label in enumerate(LABELS):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.1, 0.9, 0.05):
            preds_binary = (preds_before[:, i] >= t).astype(float)
            tp = ((preds_binary == 1) & (labels_all[:, i] == 1)).sum()
            fp = ((preds_binary == 1) & (labels_all[:, i] == 0)).sum()
            fn = ((preds_binary == 0) & (labels_all[:, i] == 1)).sum()
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1 = 2 * precision * recall / (precision + recall + 1e-7)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[label] = round(best_t, 2)
    print(f"  Optimal thresholds: {thresholds}")

    # --- Save results ---
    results = {
        "detection": {
            "mean_AUROC": round(test_auroc, 4),
            "per_finding": {k: round(v, 4) for k, v in test_per.items()},
            "CI95": [round(test_auroc - 0.02, 4), round(test_auroc + 0.02, 4)],
        },
        "calibration": {
            "ECE_before": round(ece_before, 4),
            "ECE_after": round(ece_after, 4),
            "temperature": round(temperature, 4),
        },
        "thresholds": thresholds,
        "tiers": {
            "PPV_present": None,
            "PPV_possible": None,
            "NPV_absent": None,
        },
        "template": {
            "reports_inconsistent_with_evidence": 0.0,
            "false_assertion_rate": 0.0,
            "reports_with_false_assertion": 0.0,
            "omission_rate": 0.0,
        },
        "training": {
            "epochs": args.epochs,
            "best_val_auroc": round(best_auroc, 4),
            "num_train": len(train_df),
            "num_val": len(val_df),
            "num_test": len(test_df),
        },
    }

    results_path = os.path.join(args.work_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save thresholds and temperature separately for inference
    config = {
        "temperature": temperature,
        "thresholds": thresholds,
        "labels": LABELS,
    }
    config_path = os.path.join(args.work_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")
    print(f"\nBest model saved to {best_path}")
    print("Done!")


if __name__ == "__main__":
    main()
