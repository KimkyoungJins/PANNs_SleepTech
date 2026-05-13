#!/usr/bin/env python3
"""
Sleep Apnea detector — ResNet22 audio binary classification.

Task: 30초 audio epoch → apnea event 발생 여부 (binary).
시작 가중치: full_ver_2class_finetune (W/S, sleep 도메인 적응됨)
fc_audioset head는 재초기화 (W/S → apnea 다른 task).

사용:
  cd resnet
  python3 train_apnea_resnet22.py --workspace=./workspaces/apnea_v1 --cuda
"""

import os
import sys
import csv
import json
import time
import random
import argparse
import logging

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
)

# ResNet22
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pytorch'))
from models import ResNet22  # noqa: E402

torch.backends.cudnn.benchmark = False

SAMPLE_RATE = 16000
CLIP_SAMPLES = SAMPLE_RATE * 30
WINDOW_SIZE = 512
HOP_SIZE = 160
MEL_BINS = 64
FMIN = 50
FMAX = 8000


def focal_per_sample(logits, targets, alpha, gamma):
    """Per-sample focal loss (binary). class 0 weight=alpha, class 1 weight=1-alpha."""
    alpha_vec = torch.tensor([alpha, 1.0 - alpha], device=logits.device, dtype=torch.float32)
    probs = F.softmax(logits, dim=1)
    pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
    alpha_t = alpha_vec[targets]
    focal_weight = alpha_t * (1.0 - pt).pow(gamma)
    return -focal_weight * torch.log(pt)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        return focal_per_sample(logits, targets, self.alpha, self.gamma).mean()


def load_wav(path):
    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(wav) < CLIP_SAMPLES:
        wav = np.concatenate([wav, np.zeros(CLIP_SAMPLES - len(wav), dtype=np.float32)])
    else:
        wav = wav[:CLIP_SAMPLES]
    return wav.astype(np.float32)


class ApneaDataset(Dataset):
    """Apnea binary classification (0=normal, 1=apnea)."""

    def __init__(self, csv_path, data_dir, split='train', mixup_prob=0.0):
        self.data_dir = data_dir
        self.split = split
        self.is_train = (split == 'train')
        self.mixup_prob = mixup_prob if self.is_train else 0.0

        items = []
        with open(csv_path) as f:
            reader = csv.reader(f); next(reader)
            for row in reader:
                fname, label = row[0], int(row[1])
                pid = fname.split('_')[0]
                items.append((pid, fname, label))
        self.items = items
        labels = np.array([it[2] for it in items])
        self.num_normal = int((labels == 0).sum())
        self.num_apnea = int((labels == 1).sum())

    def __len__(self):
        return len(self.items)

    def get_class_weights(self):
        w = np.empty(len(self.items), dtype=np.float64)
        w_normal = 1.0 / max(1, self.num_normal)
        w_apnea = 1.0 / max(1, self.num_apnea)
        for i, (_, _, lbl) in enumerate(self.items):
            w[i] = w_normal if lbl == 0 else w_apnea
        return w

    def _load_one(self, idx):
        pid, fname, label = self.items[idx]
        path = os.path.join(self.data_dir, pid, fname)
        wav = load_wav(path)
        return wav, label

    def __getitem__(self, idx):
        wav1, y1 = self._load_one(idx)
        if self.is_train and random.random() < self.mixup_prob:
            idx2 = random.randrange(len(self))
            wav2, y2 = self._load_one(idx2)
            lam = float(np.random.beta(0.4, 0.4))
            lam = max(0.1, min(0.9, lam))
            wav = (lam * wav1 + (1.0 - lam) * wav2).astype(np.float32)
            return wav, y1, y2, lam
        return wav1, y1, y1, 1.0


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_preds, all_targets, all_probs = [], [], []
    for wav, y1, y2, lam in loader:
        wav = wav.to(device, non_blocking=True)
        y1 = y1.to(device, non_blocking=True)
        logits = model(wav)['clipwise_output']
        loss = loss_fn(logits, y1)
        total_loss += loss.item(); n_batches += 1
        probs = F.softmax(logits, dim=1)[:, 1]  # P(apnea)
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y1.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)
    p, r, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, labels=[0, 1], zero_division=0)
    try:
        auc = float(roc_auc_score(all_targets, all_probs))
    except Exception:
        auc = 0.0
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    return {
        'loss': total_loss / max(1, n_batches),
        'accuracy': float((all_preds == all_targets).mean()),
        'balanced_acc': float(balanced_accuracy_score(all_targets, all_preds)),
        'macro_f1': float(f1.mean()),
        'auc': auc,
        'normal': {'precision': float(p[0]), 'recall': float(r[0]),
                   'f1': float(f1[0]), 'support': int(support[0])},
        'apnea': {'precision': float(p[1]), 'recall': float(r[1]),
                  'f1': float(f1[1]), 'support': int(support[1])},
        'confusion_matrix': cm.tolist(),
    }


def setup_logging(workspace):
    log_dir = os.path.join(workspace, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_{time.strftime("%Y%m%d_%H%M%S")}.log')
    fmt = '%(asctime)s | %(levelname)s | %(message)s'
    root = logging.getLogger(); root.handlers.clear(); root.setLevel(logging.INFO)
    sh = logging.StreamHandler(); sh.setFormatter(logging.Formatter(fmt))
    fh = logging.FileHandler(log_file); fh.setFormatter(logging.Formatter(fmt))
    root.addHandler(sh); root.addHandler(fh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/data_for_ai/full_ver_3class')
    parser.add_argument('--csv_dir', default='../data/data_for_ai/apnea_2class')
    parser.add_argument('--workspace', default='./workspaces/apnea_v1')
    parser.add_argument('--init_weights',
                        default='./workspaces/full_ver_2class_finetune/checkpoints/best_model.pth',
                        help='sleep 도메인 적응된 W/S 가중치에서 시작 (head는 재초기화)')

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8,
                        help='val_batch_size 줄여서 OOM 방지')
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--lr_patience', type=int, default=3)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    parser.add_argument('--focal_alpha', type=float, default=0.5,
                        help='클래스 비율 양호(55:45)라 0.5 사용')
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--mixup_prob', type=float, default=0.0)
    parser.add_argument('--num_samples_per_epoch', type=int, default=8000,
                        help='WeightedRandomSampler num_samples (~50:50)')

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.workspace, exist_ok=True)
    setup_logging(args.workspace)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    logging.info(f'Device: {device}')

    base = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(base, args.csv_dir)
    data_dir = os.path.join(base, args.data_dir)

    with open(os.path.join(args.workspace, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    logging.info(f'Workspace: {os.path.abspath(args.workspace)}')

    train_ds = ApneaDataset(os.path.join(csv_dir, 'train.csv'), data_dir, 'train',
                             mixup_prob=args.mixup_prob)
    val_ds = ApneaDataset(os.path.join(csv_dir, 'val.csv'), data_dir, 'val')
    logging.info(f'Train: {len(train_ds)} (apnea={train_ds.num_apnea}, normal={train_ds.num_normal})')
    logging.info(f'Val:   {len(val_ds)} (apnea={val_ds.num_apnea}, normal={val_ds.num_normal})')

    weights = train_ds.get_class_weights()
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=args.num_samples_per_epoch, replacement=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
    )

    # Model
    model = ResNet22(
        sample_rate=SAMPLE_RATE, window_size=WINDOW_SIZE, hop_size=HOP_SIZE,
        mel_bins=MEL_BINS, fmin=FMIN, fmax=FMAX, classes_num=2,
    )
    init_path = os.path.join(base, args.init_weights) if not os.path.isabs(args.init_weights) else args.init_weights
    if os.path.exists(init_path):
        ckpt = torch.load(init_path, map_location='cpu', weights_only=False)
        sd = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        sd_no_head = {k: v for k, v in sd.items() if not k.startswith('fc_audioset')}
        md = model.state_dict()
        filtered = {k: v for k, v in sd_no_head.items() if k in md and v.shape == md[k].shape}
        md.update(filtered); model.load_state_dict(md)
        logging.info(f'Loaded {len(filtered)}/{len(sd)} layers from {init_path}')
    else:
        logging.warning(f'init_weights not found: {init_path}, training from scratch')
    model.to(device)

    loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=args.lr_patience,
    )

    history = []
    best_metric = -1.0
    patience_counter = 0
    ckpt_dir = os.path.join(args.workspace, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, n_batches = 0.0, 0

        for wav, y1, y2, lam in train_loader:
            wav = wav.to(device, non_blocking=True)
            y1 = y1.to(device, non_blocking=True)
            y2 = y2.to(device, non_blocking=True)
            lam_t = lam.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            logits = model(wav)['clipwise_output']
            logits = logits.float()  # FP32 loss (FP16 NaN 방지)
            l1 = focal_per_sample(logits, y1, args.focal_alpha, args.focal_gamma)
            l2 = focal_per_sample(logits, y2, args.focal_alpha, args.focal_gamma)
            loss = (lam_t * l1 + (1.0 - lam_t) * l2).mean()

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss += loss.item(); n_batches += 1

        train_loss /= max(1, n_batches)
        torch.cuda.empty_cache()  # OOM 방지: train과 val 사이 캐시 정리
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        torch.cuda.empty_cache()
        scheduler.step(val_metrics['balanced_acc'])

        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]['lr']
        log_line = (
            f'Epoch {epoch:>3d}/{args.num_epochs}  '
            f'tr_loss={train_loss:.4f}  '
            f'val_loss={val_metrics["loss"]:.4f}  '
            f'val_Acc={val_metrics["accuracy"]*100:.2f}%  '
            f'val_BalAcc={val_metrics["balanced_acc"]*100:.2f}%  '
            f'val_F1={val_metrics["macro_f1"]*100:.2f}%  '
            f'val_AUC={val_metrics["auc"]:.3f}  '
            f'apnea_R={val_metrics["apnea"]["recall"]*100:.1f}% '
            f'normal_R={val_metrics["normal"]["recall"]*100:.1f}%  '
            f'lr={cur_lr:.1e}  [{elapsed:.0f}s]'
        )

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val': val_metrics,
            'lr': cur_lr,
            'elapsed_sec': elapsed,
        })

        if val_metrics['balanced_acc'] > best_metric:
            best_metric = val_metrics['balanced_acc']
            patience_counter = 0
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_balanced_acc': val_metrics['balanced_acc'],
                'val_accuracy': val_metrics['accuracy'],
                'val_auc': val_metrics['auc'],
                'config': vars(args),
            }, os.path.join(ckpt_dir, 'best_model.pth'))
            log_line += '  ★ BEST'
        else:
            patience_counter += 1

        torch.save({'model': model.state_dict(), 'epoch': epoch},
                   os.path.join(ckpt_dir, 'last_model.pth'))

        with open(os.path.join(args.workspace, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        logging.info(log_line)

        if patience_counter >= args.patience:
            logging.info(f'Early stopping at epoch {epoch} (best val_BalAcc={best_metric:.4f})')
            break

    logging.info(f'학습 완료. best val_BalAcc = {best_metric:.4f}')


if __name__ == '__main__':
    main()
