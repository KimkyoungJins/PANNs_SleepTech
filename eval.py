#!/usr/bin/env python3
"""
BiLSTM REM/NREM 단독 평가 (Stage 2 only).

Ground truth Wake mask 적용 (= Stage 1이 완벽하다고 가정한 best case).
두 가지 평가 방식:
  1. per-window: 가운데 center_window epoch만 (학습과 동일 조건)
  2. per-epoch:  sliding window logits aggregation (실사용 시나리오)

사용:
    cd project1/bilstm
    python3 eval.py --workspace=./workspaces/v1 --split=test --cuda
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from models.bilstm_classifier import BiLSTMClassifier  # noqa: E402
from datasets.seq_dataset import SoundSequenceDataset  # noqa: E402
from utils.losses import MaskedFocalLoss  # noqa: E402


@torch.no_grad()
def evaluate_per_window(model, loader, loss_fn, device):
    """학습 시 검증과 동일 조건: center mask 적용."""
    model.eval()
    all_preds, all_targets = [], []
    total_loss, n_batches = 0.0, 0

    for emb, lbl, msk in loader:
        emb = emb.to(device); lbl = lbl.to(device); msk = msk.to(device)
        logits = model(emb)
        loss = loss_fn(logits, lbl, msk)
        total_loss += loss.item()
        n_batches += 1

        m = (msk > 0).flatten()
        preds = logits.argmax(dim=-1).flatten()[m]
        targets = lbl.flatten()[m]
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return all_preds, all_targets, total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_per_epoch(model, dataset, device):
    """
    환자별 sliding window aggregation:
      각 epoch가 등장하는 모든 window의 logits을 평균 → argmax → epoch별 prediction.
    REM/NREM 라벨 epoch만 metric 산정 (Wake epoch 제외).
    """
    model.eval()

    # 환자별 logit 누적기
    accum = {}
    for p in dataset.patients:
        n = p['num_epochs']
        accum[p['pid']] = {
            'logits': np.zeros((n, 2), dtype=np.float64),
            'counts': np.zeros(n, dtype=np.int64),
            'labels': p['labels'].copy(),   # 0=REM, 1=NREM (Wake도 0 dummy, mask로 구분)
            'mask': p['mask'].copy(),       # 1=REM/NREM, 0=Wake
        }

    seq_len = dataset.seq_len
    for pi, start in dataset.windows:
        p = dataset.patients[pi]
        x = torch.from_numpy(p['emb'][start:start+seq_len]).float().unsqueeze(0).to(device)
        logits = model(x).squeeze(0).cpu().numpy()  # [seq_len, 2]
        accum[p['pid']]['logits'][start:start+seq_len] += logits
        accum[p['pid']]['counts'][start:start+seq_len] += 1

    all_preds, all_targets = [], []
    per_patient = {}

    for pid, info in accum.items():
        valid = info['counts'] > 0
        avg_logits = np.zeros_like(info['logits'])
        avg_logits[valid] = info['logits'][valid] / info['counts'][valid][:, None]
        preds = avg_logits.argmax(axis=1)

        keep = (info['mask'] == 1) & valid   # Sleep epoch && covered by some window
        n_uncovered = int(((info['mask'] == 1) & ~valid).sum())
        if keep.sum() > 0:
            all_preds.append(preds[keep])
            all_targets.append(info['labels'][keep])

        per_patient[pid] = {
            'covered_sleep': int(keep.sum()),
            'uncovered_sleep': n_uncovered,
            'total_epochs': int(len(info['labels'])),
        }

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return all_preds, all_targets, per_patient


def report_metrics(preds, targets, prefix=''):
    bal_acc = balanced_accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
    p, r, f1, _ = precision_recall_fscore_support(
        targets, preds, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(targets, preds, labels=[0, 1])

    print()
    print('=' * 60)
    print(f'  {prefix}  (N={len(targets):,})')
    print('=' * 60)
    print(f'  Balanced Acc: {bal_acc:.4f}')
    print(f'  Macro F1:     {macro_f1:.4f}')
    print(f'  REM:  P={p[0]:.4f}  R={r[0]:.4f}  F1={f1[0]:.4f}')
    print(f'  NREM: P={p[1]:.4f}  R={r[1]:.4f}  F1={f1[1]:.4f}')
    print(f'  Confusion Matrix (rows=true, cols=pred):')
    print(f'         pred_REM  pred_NREM')
    print(f'   REM   {cm[0,0]:>8d}  {cm[0,1]:>9d}')
    print(f'   NREM  {cm[1,0]:>8d}  {cm[1,1]:>9d}')

    return {
        'balanced_acc': float(bal_acc),
        'macro_f1': float(macro_f1),
        'rem': {'precision': float(p[0]), 'recall': float(r[0]), 'f1': float(f1[0])},
        'nrem': {'precision': float(p[1]), 'recall': float(r[1]), 'f1': float(f1[1])},
        'confusion_matrix': cm.tolist(),
        'n_samples': int(len(targets)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_dir', default='./embeddings')
    parser.add_argument('--workspace', default='./workspaces/v1')
    parser.add_argument('--checkpoint', default=None,
                        help='기본: workspace/checkpoints/best_model.pth')
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    ckpt_path = args.checkpoint or os.path.join(
        args.workspace, 'checkpoints/best_model.pth'
    )
    config_path = os.path.join(args.workspace, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    print(f'Checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = BiLSTMClassifier(
        embed_dim=2048, proj_dim=config['proj_dim'],
        hidden_dim=config['hidden_dim'], num_layers=config['num_layers'],
        dropout=config['dropout'], num_classes=2,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Loaded epoch={ckpt["epoch"]} val_macro_f1={ckpt["val_macro_f1"]:.4f}')

    ds = SoundSequenceDataset(
        args.embeddings_dir, args.split,
        seq_len=config['seq_len'],
        center_window=config['center_window'],
        stride=config['stride'],
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    loss_fn = MaskedFocalLoss(
        alpha=config['focal_alpha'], gamma=config['focal_gamma']
    ).to(device)

    # ── 1. per-window ──
    preds_w, targets_w, val_loss = evaluate_per_window(model, loader, loss_fn, device)
    metrics_window = report_metrics(
        preds_w, targets_w, prefix=f'{args.split} — per-window (center {config["center_window"]} only)'
    )
    metrics_window['loss'] = float(val_loss)

    # ── 2. per-epoch (sliding aggregation) ──
    preds_e, targets_e, per_patient = evaluate_per_epoch(model, ds, device)
    metrics_epoch = report_metrics(
        preds_e, targets_e, prefix=f'{args.split} — per-epoch (sliding aggregated)'
    )
    metrics_epoch['per_patient'] = per_patient

    out = {
        'split': args.split,
        'checkpoint': ckpt_path,
        'epoch': int(ckpt['epoch']),
        'per_window': metrics_window,
        'per_epoch': metrics_epoch,
    }
    results_dir = os.path.join(args.workspace, 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f'eval_{args.split}.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
