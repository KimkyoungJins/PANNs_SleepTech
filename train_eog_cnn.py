#!/usr/bin/env python3
"""EOG_CNN 학습 스크립트 - REM vs NREM 분류.

사용법:
    python3 train_eog_cnn.py --version v2
    → checkpoints/hierarchical/v2/stage2_eog_cnn.pth
    → checkpoints/hierarchical/v2/training/stage2_history.json
"""

import os
import sys
import time
import logging
import json
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix,
)

sys.path.insert(0, os.path.dirname(__file__))
from models.eog_cnn import EOG_CNN
from datasets.eog_dataset import EOGDataset

LABEL_NAMES = ['REM', 'NREM']


# ──────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=0.75, gamma=3.0, num_classes=2):
        super().__init__()
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1.0 - alpha], dtype=torch.float32)
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_onehot = torch.zeros_like(probs)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1.0)

        pt = (probs * targets_onehot).sum(dim=1)
        pt = pt.clamp(min=1e-8)

        alpha_t = self.alpha.to(logits.device)[targets]
        focal_weight = alpha_t * (1.0 - pt) ** self.gamma
        loss = -focal_weight * torch.log(pt)

        return loss.mean()


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────
def compute_metrics(all_targets, all_preds):
    acc = np.mean(np.array(all_targets) == np.array(all_preds))
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, labels=[0, 1], zero_division=0)
    macro_f1 = f1.mean()
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])

    return {
        'acc': acc,
        'bal_acc': bal_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_f1': macro_f1,
        'support': support,
        'cm': cm,
    }


def log_metrics(logger, prefix, loss, metrics):
    logger.info(f"  {prefix}: loss={loss:.4f} bal_acc={metrics['bal_acc']:.4f}")
    for i, name in enumerate(LABEL_NAMES):
        logger.info(f"         {name:>4s}: P={metrics['precision'][i]:.4f} "
                     f"R={metrics['recall'][i]:.4f} F1={metrics['f1'][i]:.4f}")
    logger.info(f"         Macro F1: {metrics['macro_f1']:.4f}")


# ──────────────────────────────────────────────
# Setup logging
# ──────────────────────────────────────────────
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'train.log')

    logger = logging.getLogger('eog_cnn')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# ──────────────────────────────────────────────
# Train / Evaluate
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, max_norm=1.0):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for eog, labels in loader:
        eog = eog.to(device)
        labels = labels.to(device, dtype=torch.long)

        logits = model(eog)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        total_loss += loss.item() * eog.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_targets)
    metrics = compute_metrics(all_targets, all_preds)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for eog, labels in loader:
        eog = eog.to(device)
        labels = labels.to(device, dtype=torch.long)

        logits = model(eog)
        loss = criterion(logits, labels)

        total_loss += loss.item() * eog.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_targets)
    metrics = compute_metrics(all_targets, all_preds)
    return avg_loss, metrics


# ──────────────────────────────────────────────
# Version directory helpers
# ──────────────────────────────────────────────
def get_version_dir(base_dir, version):
    """버전별 디렉토리 경로 반환 및 생성."""
    ver_dir = os.path.join(base_dir, 'checkpoints', 'hierarchical', version)
    os.makedirs(os.path.join(ver_dir, 'training'), exist_ok=True)
    os.makedirs(os.path.join(ver_dir, 'results'), exist_ok=True)
    return ver_dir


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, required=True,
                        help='실험 버전 (예: v1, v2)')
    parser.add_argument('--data_dir', type=str,
                        default='../data/data_for_ai/full_ver_3class')
    parser.add_argument('--csv_dir', type=str,
                        default='../data/data_for_ai/full_ver_rem_nrem')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--focal_alpha', type=float, default=0.75)
    parser.add_argument('--focal_gamma', type=float, default=3.0)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, args.data_dir)
    csv_dir = os.path.join(base_dir, args.csv_dir)

    # 버전별 디렉토리
    ver_dir = get_version_dir(base_dir, args.version)
    training_dir = os.path.join(ver_dir, 'training')

    logger = setup_logging(training_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(f"Version: {args.version}")
    logger.info(f"Output: {ver_dir}")
    logger.info(f"Args: {vars(args)}")

    # ===== Datasets =====
    train_dataset = EOGDataset(
        os.path.join(csv_dir, 'train.csv'), data_dir, augment=True)
    val_dataset = EOGDataset(
        os.path.join(csv_dir, 'val.csv'), data_dir, augment=False)

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ===== WeightedRandomSampler =====
    train_labels = train_dataset.get_labels()
    counts = Counter(train_labels)
    n_rem, n_nrem = counts[0], counts[1]
    total = len(train_labels)
    class_weights = [total / (2 * n_rem), total / (2 * n_nrem)]
    sample_weights = [class_weights[l] for l in train_labels]

    logger.info(f"Class counts - REM: {n_rem}, NREM: {n_nrem}")
    logger.info(f"Sampler weights - REM: {class_weights[0]:.2f}, NREM: {class_weights[1]:.2f}")

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    # ===== Model =====
    model = EOG_CNN(num_classes=2).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # ===== Loss / Optimizer / Scheduler =====
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ===== Training loop =====
    best_macro_f1 = 0.0
    patience_counter = 0
    history = []
    low_rem_recall_count = 0

    logger.info("=" * 60)
    logger.info("Training started")
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        # Log
        logger.info(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.1f}s, "
                     f"lr={scheduler.get_last_lr()[0]:.6f})")
        log_metrics(logger, "Train", train_loss, train_metrics)
        log_metrics(logger, "Val  ", val_loss, val_metrics)
        logger.info(f"  Val CM: {val_metrics['cm'].tolist()}")

        # History
        epoch_record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_bal_acc': train_metrics['bal_acc'],
            'val_bal_acc': val_metrics['bal_acc'],
            'val_macro_f1': val_metrics['macro_f1'],
            'val_rem_f1': float(val_metrics['f1'][0]),
            'val_nrem_f1': float(val_metrics['f1'][1]),
            'val_rem_recall': float(val_metrics['recall'][0]),
            'val_nrem_recall': float(val_metrics['recall'][1]),
        }
        history.append(epoch_record)

        # Save last checkpoint
        last_ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'metrics': epoch_record,
        }
        torch.save(last_ckpt, os.path.join(ver_dir, 'stage2_eog_cnn_last.pth'))

        # Best model check (val_macro_f1)
        if val_metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = val_metrics['macro_f1']
            patience_counter = 0
            best_ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'metrics': epoch_record,
            }
            torch.save(best_ckpt, os.path.join(ver_dir, 'stage2_eog_cnn.pth'))
            logger.info(f"  >> New best! Macro F1={best_macro_f1:.4f}")
        else:
            patience_counter += 1
            logger.info(f"  Patience: {patience_counter}/{args.patience}")

        # REM recall safety check
        if val_metrics['recall'][0] < 0.5:
            low_rem_recall_count += 1
            if low_rem_recall_count >= 3:
                logger.info("=" * 60)
                logger.info("[ALERT] REM Recall < 0.5 for 3+ epochs. Stopping.")
                logger.info("Consider: adjust focal alpha/gamma, increase REM oversampling")
                logger.info("=" * 60)
                break
        else:
            low_rem_recall_count = 0

        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    # ===== Summary =====
    logger.info("\n" + "=" * 60)
    logger.info("Training complete")
    logger.info(f"Best val Macro F1: {best_macro_f1:.4f}")
    logger.info("=" * 60)

    # Save history
    history_path = os.path.join(training_dir, 'stage2_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"History saved to {history_path}")

    # Save training args
    args_path = os.path.join(training_dir, 'stage2_args.json')
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Args saved to {args_path}")


if __name__ == '__main__':
    main()
