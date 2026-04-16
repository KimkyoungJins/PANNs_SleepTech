#!/usr/bin/env python3
"""EOG_CNN 평가 스크립트 - test set에서 REM/NREM 성능 평가.

사용법:
    python3 eval_eog_cnn.py --version v2
    → checkpoints/hierarchical/v2/results/eog_cnn_test.json
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix,
)

sys.path.insert(0, os.path.dirname(__file__))
from models.eog_cnn import EOG_CNN
from datasets.eog_dataset import EOGDataset

LABEL_NAMES = ['REM', 'NREM']


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    for eog, labels in loader:
        eog = eog.to(device)
        logits = model(eog)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(labels.numpy())
        all_probs.extend(probs)

    return np.array(all_targets), np.array(all_preds), np.array(all_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, required=True,
                        help='실험 버전 (예: v1, v2)')
    parser.add_argument('--data_dir', type=str,
                        default='../data/data_for_ai/full_ver_3class')
    parser.add_argument('--csv_dir', type=str,
                        default='../data/data_for_ai/full_ver_rem_nrem')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, args.data_dir)
    csv_dir = os.path.join(base_dir, args.csv_dir)

    # 버전별 디렉토리
    ver_dir = os.path.join(base_dir, 'checkpoints', 'hierarchical', args.version)
    results_dir = os.path.join(ver_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    ckpt_path = os.path.join(ver_dir, 'stage2_eog_cnn.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Version: {args.version}")

    # Load model
    model = EOG_CNN(num_classes=2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    print(f"Loaded: {ckpt_path} (epoch {ckpt['epoch']})")

    # Load test dataset
    test_dataset = EOGDataset(
        os.path.join(csv_dir, 'test.csv'), data_dir, augment=False)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    print(f"Test samples: {len(test_dataset)}")

    # Evaluate
    targets, preds, probs = evaluate(model, test_loader, device)

    # Metrics
    acc = np.mean(targets == preds)
    bal_acc = balanced_accuracy_score(targets, preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=[0, 1], zero_division=0)
    macro_f1 = f1.mean()
    cm = confusion_matrix(targets, preds, labels=[0, 1])

    # Print
    print(f"\n{'=' * 60}")
    print(f"EOG_CNN TEST RESULTS (REM vs NREM) — {args.version}")
    print(f"{'=' * 60}")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print()
    print(f"{'':>8s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    print("-" * 50)
    for i, name in enumerate(LABEL_NAMES):
        print(f"{name:>8s} {precision[i]:>10.4f} {recall[i]:>10.4f} "
              f"{f1[i]:>10.4f} {support[i]:>10d}")
    print()
    print("Confusion Matrix:")
    print(f"{'':>8s} {'pred_REM':>10s} {'pred_NREM':>10s}")
    for i, name in enumerate(LABEL_NAMES):
        print(f"{name:>8s} {cm[i][0]:>10d} {cm[i][1]:>10d}")

    # Save
    result_dict = {
        'accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'macro_f1': float(macro_f1),
        'per_class': {
            LABEL_NAMES[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i]),
            } for i in range(2)
        },
        'confusion_matrix': cm.tolist(),
        'checkpoint_epoch': ckpt['epoch'],
    }
    result_path = os.path.join(results_dir, 'eog_cnn_test.json')
    with open(result_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"\nSaved to {result_path}")


if __name__ == '__main__':
    main()
