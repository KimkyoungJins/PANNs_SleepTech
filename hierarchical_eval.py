#!/usr/bin/env python3
"""
Hierarchical 3-Class 평가:
  Stage 1 (기존 ResNet22 W/S) + Stage 2 (새 BiLSTM R/N)
  → Wake / REM / NREM 3-class 최종 prediction.

기존 EOG_CNN 기반 v1 (3-class 73.7%) 와 비교 가능.

사용:
    cd project1/bilstm
    python3 hierarchical_eval.py --bilstm_workspace=./workspaces/v1 --cuda
"""

import os
import sys
import json
import glob
import time
import argparse

import numpy as np
import torch
import librosa
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_recall_fscore_support, confusion_matrix,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# 1) bilstm/ 우선 → 로컬 models 패키지 캐싱
sys.path.insert(0, THIS_DIR)
from models.bilstm_classifier import BiLSTMClassifier  # noqa: E402

# 2) resnet/pytorch 도 path 추가 (pytorch_utils 등 내부 import 위해)
_RESNET_PYTORCH = os.path.join(THIS_DIR, '..', 'resnet', 'pytorch')
sys.path.append(_RESNET_PYTORCH)

# 3) resnet/pytorch/models.py 를 'resnet_models' 라는 별칭으로 로드 (이름 충돌 회피)
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'resnet_models', os.path.join(_RESNET_PYTORCH, 'models.py')
)
_resnet_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_resnet_models)
ResNet22 = _resnet_models.ResNet22


# ── audio params (extract_embeddings.py와 동일) ──
SAMPLE_RATE = 16000
CLIP_SAMPLES = SAMPLE_RATE * 30
WINDOW_SIZE = 512
HOP_SIZE = 160
MEL_BINS = 64
FMIN = 50
FMAX = 8000


def load_wav(path, sr=SAMPLE_RATE, target_len=CLIP_SAMPLES):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    if len(wav) < target_len:
        wav = np.concatenate([wav, np.zeros(target_len - len(wav), dtype=np.float32)])
    else:
        wav = wav[:target_len]
    return wav.astype(np.float32)


def load_stage1_model(ckpt_path, device):
    """기존 ResNet22 Wake/Sleep 분류기."""
    model = ResNet22(
        sample_rate=SAMPLE_RATE, window_size=WINDOW_SIZE, hop_size=HOP_SIZE,
        mel_bins=MEL_BINS, fmin=FMIN, fmax=FMAX, classes_num=2,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_stage2_model(workspace, device):
    config_path = os.path.join(workspace, 'config.json')
    ckpt_path = os.path.join(workspace, 'checkpoints/best_model.pth')
    with open(config_path) as f:
        config = json.load(f)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = BiLSTMClassifier(
        embed_dim=2048, proj_dim=config['proj_dim'],
        hidden_dim=config['hidden_dim'], num_layers=config['num_layers'],
        dropout=config['dropout'], num_classes=2,
    )
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    return model, config, ckpt


@torch.no_grad()
def stage1_predict(model, wav_paths, device, batch_size=16):
    """wav 리스트 → Wake(0)/Sleep(1) 예측."""
    preds = []
    for i in range(0, len(wav_paths), batch_size):
        batch_paths = wav_paths[i:i + batch_size]
        wavs = np.stack([load_wav(p) for p in batch_paths])
        x = torch.from_numpy(wavs).to(device)
        out = model(x)
        logits = out['clipwise_output']
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


@torch.no_grad()
def stage2_predict(model, embeddings, device, seq_len=40, stride=20):
    """embedding 시퀀스 → REM(0)/NREM(1) sliding window aggregation."""
    n_orig = len(embeddings)
    n = max(n_orig, seq_len)
    if n_orig < seq_len:
        pad = np.zeros((seq_len - n_orig, embeddings.shape[1]), dtype=np.float32)
        embeddings = np.concatenate([embeddings, pad], axis=0)

    logits_sum = np.zeros((n, 2), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)

    starts = list(range(0, n - seq_len + 1, stride))
    if not starts or starts[-1] != n - seq_len:
        starts.append(n - seq_len)

    for s in starts:
        x = torch.from_numpy(embeddings[s:s + seq_len]).float().unsqueeze(0).to(device)
        logits = model(x).squeeze(0).cpu().numpy()
        logits_sum[s:s + seq_len] += logits
        counts[s:s + seq_len] += 1

    valid = counts > 0
    avg = np.zeros_like(logits_sum)
    avg[valid] = logits_sum[valid] / counts[valid][:, None]
    return avg.argmax(axis=1)[:n_orig]


def get_test_patients(embeddings_dir):
    out = []
    for mf in sorted(glob.glob(os.path.join(embeddings_dir, '*_meta.json'))):
        with open(mf) as f:
            meta = json.load(f)
        if meta['split'] != 'test':
            continue
        out.append({
            'pid': meta['patient_id'],
            'epoch_ids': meta['epoch_ids'],
        })
    return out


def evaluate_hierarchical(stage1, stage2, stage2_config, embeddings_dir, data_dir, device):
    test_patients = get_test_patients(embeddings_dir)
    print(f'Test patients: {len(test_patients)}')

    all_targets, all_preds = [], []
    per_patient = {}

    for idx, info in enumerate(test_patients, 1):
        pid = info['pid']
        emb = np.load(os.path.join(embeddings_dir, f'{pid}_emb.npy'))
        raw_labels = np.load(os.path.join(embeddings_dir, f'{pid}_labels.npy'))

        wav_paths = [
            os.path.join(data_dir, pid, f'{pid}_epoch{eid:04d}.wav')
            for eid in info['epoch_ids']
        ]

        t0 = time.time()
        s1 = stage1_predict(stage1, wav_paths, device)
        s2 = stage2_predict(stage2, emb, device,
                            seq_len=stage2_config['seq_len'],
                            stride=stage2_config['stride'])

        # Stage1=0(Wake)→0; Stage1=1→ Stage2(0=REM→1, 1=NREM→2)
        final = np.where(s1 == 0, 0, np.where(s2 == 0, 1, 2))

        all_targets.append(raw_labels)
        all_preds.append(final)

        acc = accuracy_score(raw_labels, final)
        per_patient[pid] = {
            'n_epochs': int(len(raw_labels)),
            'accuracy': float(acc),
            's1_wake_pred_ratio': float((s1 == 0).mean()),
            's1_sleep_pred_ratio': float((s1 == 1).mean()),
            'gt_label_counts': {
                'wake': int(np.sum(raw_labels == 0)),
                'rem':  int(np.sum(raw_labels == 1)),
                'nrem': int(np.sum(raw_labels == 2)),
            },
        }
        dt = time.time() - t0
        print(f'[{idx}/{len(test_patients)}] {pid}: n={len(raw_labels):>4d} '
              f'acc={acc:.4f}  [{dt:.1f}s]')

    return np.concatenate(all_targets), np.concatenate(all_preds), per_patient


def report_3class(targets, preds):
    label_names = ['Wake', 'REM', 'NREM']
    acc = accuracy_score(targets, preds)
    bal_acc = balanced_accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
    p, r, f1, _ = precision_recall_fscore_support(
        targets, preds, labels=[0, 1, 2], zero_division=0
    )
    cm = confusion_matrix(targets, preds, labels=[0, 1, 2])

    print()
    print('=' * 70)
    print(f'  Hierarchical 3-Class Result  (N={len(targets):,})')
    print('=' * 70)
    print(f'  Accuracy:       {acc*100:.2f}%')
    print(f'  Balanced Acc:   {bal_acc*100:.2f}%')
    print(f'  Macro F1:       {macro_f1:.4f}')
    print()
    print(f'  {"":>8s} {"Precision":>10s} {"Recall":>10s} {"F1":>10s}')
    for i, name in enumerate(label_names):
        print(f'  {name:>8s} {p[i]:>10.4f} {r[i]:>10.4f} {f1[i]:>10.4f}')
    print()
    print(f'  Confusion Matrix (rows=true, cols=pred):')
    print(f'  {"":>8s} {"Wake":>8s} {"REM":>8s} {"NREM":>8s}')
    for i, name in enumerate(label_names):
        print(f'  {name:>8s} {cm[i,0]:>8d} {cm[i,1]:>8d} {cm[i,2]:>8d}')
    print()
    print('  Compare with v1 (EOG_CNN hierarchical, 116 patients):')
    print('    3-class acc 73.7% balanced, macro_f1 0.7392')

    return {
        'accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'macro_f1': float(macro_f1),
        'per_class': {
            label_names[i].lower(): {
                'precision': float(p[i]),
                'recall': float(r[i]),
                'f1': float(f1[i]),
            } for i in range(3)
        },
        'confusion_matrix': cm.tolist(),
        'n_samples': int(len(targets)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_dir', default='./embeddings')
    parser.add_argument('--data_dir', default='../data/data_for_ai/full_ver_3class')
    parser.add_argument('--bilstm_workspace', default='./workspaces/v1')
    parser.add_argument('--stage1_ckpt',
                        default='../resnet/workspaces/full_ver_2class/checkpoints/best_model.pth')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print(f'Stage 1 (Wake/Sleep): {args.stage1_ckpt}')
    stage1 = load_stage1_model(args.stage1_ckpt, device)

    print(f'Stage 2 (REM/NREM):   {args.bilstm_workspace}')
    stage2, stage2_config, stage2_ckpt = load_stage2_model(args.bilstm_workspace, device)
    print(f'  loaded epoch={stage2_ckpt["epoch"]} '
          f'val_macro_f1={stage2_ckpt["val_macro_f1"]:.4f}')

    targets, preds, per_patient = evaluate_hierarchical(
        stage1, stage2, stage2_config,
        args.embeddings_dir, args.data_dir, device,
    )

    metrics = report_3class(targets, preds)
    metrics['per_patient'] = per_patient
    metrics['stage1_ckpt'] = args.stage1_ckpt
    metrics['stage2_workspace'] = args.bilstm_workspace
    metrics['stage2_epoch'] = int(stage2_ckpt['epoch'])

    results_dir = os.path.join(args.bilstm_workspace, 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'hierarchical_3class.json')
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
