#!/usr/bin/env python3
"""
BiLSTM REM/NREM 분류기 학습.

입력: bilstm/embeddings/  (extract_embeddings.py 결과물)
출력: bilstm/workspaces/{name}/
        ├── config.json
        ├── history.json
        ├── checkpoints/{best,last}_model.pth
        └── logs/train_*.log

사용:
    cd project1/bilstm
    python3 train.py --workspace=./workspaces/v1 --cuda
"""

import os
import sys
import json
import time
import argparse
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, confusion_matrix,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from models.bilstm_classifier import BiLSTMClassifier  # noqa: E402
from datasets.seq_dataset import SoundSequenceDataset, worker_init_fn  # noqa: E402
from utils.losses import MaskedFocalLoss  # noqa: E402


def setup_logging(workspace: str):
    log_dir = os.path.join(workspace, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_{time.strftime("%Y%m%d_%H%M%S")}.log')

    fmt = '%(asctime)s | %(levelname)s | %(message)s'
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(fmt))
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter(fmt))
    root.addHandler(sh)
    root.addHandler(fh)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_preds, all_targets = [], []

    for emb, lbl, msk in loader:
        emb = emb.to(device, non_blocking=True)
        lbl = lbl.to(device, non_blocking=True)
        msk = msk.to(device, non_blocking=True)

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

    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])

    return {
        'loss': total_loss / max(1, n_batches),
        'macro_f1': float(macro_f1),
        'balanced_acc': float(bal_acc),
        'confusion_matrix': cm.tolist(),
        'num_samples': int(len(all_targets)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_dir', default='./embeddings')
    parser.add_argument('--workspace', default='./workspaces/v1')

    # sequence
    parser.add_argument('--seq_len', type=int, default=40)
    parser.add_argument('--center_window', type=int, default=20)
    parser.add_argument('--stride', type=int, default=20)

    # training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15,
                        help='early stopping patience (epoch)')
    parser.add_argument('--lr_patience', type=int, default=5,
                        help='ReduceLROnPlateau patience (epoch)')
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # loss
    parser.add_argument('--focal_alpha', type=float, default=0.75)
    parser.add_argument('--focal_gamma', type=float, default=3.0)

    # model
    parser.add_argument('--proj_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)

    # misc
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # ── setup ──
    os.makedirs(args.workspace, exist_ok=True)
    setup_logging(args.workspace)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    logging.info(f'Device: {device}')
    if args.cuda and not torch.cuda.is_available():
        logging.warning('--cuda 지정했으나 GPU 미가용 → CPU')

    # config 저장
    with open(os.path.join(args.workspace, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    logging.info(f'Workspace: {os.path.abspath(args.workspace)}')

    # ── Datasets ──
    logging.info('Loading datasets...')
    train_ds = SoundSequenceDataset(
        args.embeddings_dir, 'train',
        seq_len=args.seq_len, center_window=args.center_window,
        stride=args.stride, seed=args.seed,
    )
    val_ds = SoundSequenceDataset(
        args.embeddings_dir, 'val',
        seq_len=args.seq_len, center_window=args.center_window, stride=args.stride,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, worker_init_fn=worker_init_fn,
        pin_memory=(device.type == 'cuda'), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    # ── Model ──
    model = BiLSTMClassifier(
        embed_dim=2048, proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        dropout=args.dropout, num_classes=2,
    ).to(device)
    n_params = model.num_parameters()
    logging.info(f'Model params: {n_params:,} ({n_params/1e6:.2f}M)')

    # ── Loss / Optimizer / Scheduler ──
    loss_fn = MaskedFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=args.lr_patience,
    )

    # ── Train loop ──
    history = []
    best_f1 = -1.0
    patience_counter = 0
    ckpt_dir = os.path.join(args.workspace, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, n_batches = 0.0, 0

        for emb, lbl, msk in train_loader:
            emb = emb.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            msk = msk.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(emb)
            loss = loss_fn(logits, lbl, msk)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(1, n_batches)

        val_metrics = evaluate(model, val_loader, loss_fn, device)
        scheduler.step(val_metrics['macro_f1'])

        elapsed = time.time() - t0
        cm = np.array(val_metrics['confusion_matrix'])
        rem_recall = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
        nrem_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0
        cur_lr = optimizer.param_groups[0]['lr']

        log_line = (
            f'Epoch {epoch:>3d}/{args.num_epochs}  '
            f'tr_loss={train_loss:.4f}  '
            f'val_loss={val_metrics["loss"]:.4f}  '
            f'val_F1={val_metrics["macro_f1"]:.4f}  '
            f'val_BalAcc={val_metrics["balanced_acc"]:.4f}  '
            f'REM_R={rem_recall:.3f} NREM_R={nrem_recall:.3f}  '
            f'lr={cur_lr:.1e}  [{elapsed:.0f}s]'
        )

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_macro_f1': val_metrics['macro_f1'],
            'val_balanced_acc': val_metrics['balanced_acc'],
            'val_confusion_matrix': val_metrics['confusion_matrix'],
            'lr': cur_lr,
            'elapsed_sec': elapsed,
        })

        improved = val_metrics['macro_f1'] > best_f1
        if improved:
            best_f1 = val_metrics['macro_f1']
            patience_counter = 0
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_macro_f1': best_f1,
                'val_balanced_acc': val_metrics['balanced_acc'],
                'config': vars(args),
            }, os.path.join(ckpt_dir, 'best_model.pth'))
            log_line += '  ★ BEST'
        else:
            patience_counter += 1

        # last 항상 저장 (resume용)
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'val_macro_f1': val_metrics['macro_f1'],
        }, os.path.join(ckpt_dir, 'last_model.pth'))

        # history 매 epoch 저장
        with open(os.path.join(args.workspace, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        logging.info(log_line)

        if patience_counter >= args.patience:
            logging.info(
                f'Early stopping at epoch {epoch} '
                f'(patience={args.patience}, best_f1={best_f1:.4f})'
            )
            break

    logging.info(f'학습 완료. best val_macro_f1 = {best_f1:.4f}')


if __name__ == '__main__':
    main()
