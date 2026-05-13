#!/usr/bin/env python3
"""
Apnea detector 평가 — per-epoch + 환자별 AHI 추정 + 중증도 분류.

평가 metrics:
  1. Per-epoch: Accuracy, BalAcc, F1, AUC, Precision/Recall (apnea/normal)
  2. Per-patient: 추정 AHI vs 실제 AHI (RMSE, MAE, R²)
  3. Severity classification (4-class): Normal/Mild/Moderate/Severe
"""

import os
import sys
import csv
import json
import argparse
import importlib.util

import numpy as np
import librosa
import torch
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, precision_recall_fscore_support,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pytorch'))
from models import ResNet22

SAMPLE_RATE = 16000
CLIP_SAMPLES = SAMPLE_RATE * 30


def load_resnet22(ckpt, device):
    model = ResNet22(
        sample_rate=16000, window_size=512, hop_size=160,
        mel_bins=64, fmin=50, fmax=8000, classes_num=2)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state['model'])
    model.to(device).eval()
    return model


def load_wav(path):
    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(wav) < CLIP_SAMPLES:
        wav = np.concatenate([wav, np.zeros(CLIP_SAMPLES - len(wav), dtype=np.float32)])
    else:
        wav = wav[:CLIP_SAMPLES]
    return wav.astype(np.float32)


@torch.no_grad()
def evaluate(model, data_dir, test_csv, device, batch_size=8):
    """Per-epoch inference, returns predictions + labels + probs + filenames."""
    items = []
    with open(test_csv) as f:
        reader = csv.reader(f); next(reader)
        for row in reader:
            items.append((row[0], int(row[1])))

    print(f'Test set: {len(items)} epochs')
    n = len(items)
    preds = np.zeros(n, dtype=np.int64)
    labels = np.zeros(n, dtype=np.int64)
    probs = np.zeros(n, dtype=np.float32)
    fnames = []

    print('Inference...')
    waveforms = []
    indices = []
    for i in range(n):
        fname, lbl = items[i]
        pid = fname.split('_')[0]
        wav_path = os.path.join(data_dir, pid, fname)
        wav = load_wav(wav_path)
        waveforms.append(wav)
        indices.append(i)
        labels[i] = lbl
        fnames.append(fname)

        # batch inference
        if len(waveforms) >= batch_size or i == n - 1:
            wav_t = torch.from_numpy(np.stack(waveforms)).to(device, non_blocking=True)
            logits = model(wav_t)['clipwise_output']
            p = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pr = logits.argmax(dim=-1).cpu().numpy()
            for k, idx in enumerate(indices):
                probs[idx] = p[k]
                preds[idx] = pr[k]
            waveforms = []
            indices = []

            if (i + 1) % 1000 == 0:
                print(f'  {i + 1}/{n}...')

    return np.array(fnames), labels, preds, probs


def per_epoch_metrics(labels, preds, probs):
    """Per-epoch evaluation metrics."""
    p, r, f1, support = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    try:
        auc = float(roc_auc_score(labels, probs))
    except Exception:
        auc = 0.0
    return {
        'accuracy': float(accuracy_score(labels, preds)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, preds)),
        'macro_f1': float((f1[0] + f1[1]) / 2),
        'auc': auc,
        'normal': {'precision': float(p[0]), 'recall': float(r[0]),
                   'f1': float(f1[0]), 'support': int(support[0])},
        'apnea': {'precision': float(p[1]), 'recall': float(r[1]),
                  'f1': float(f1[1]), 'support': int(support[1])},
        'confusion_matrix': cm.tolist(),
    }


def parse_patient_id(fname):
    """patient04_epoch0123.wav → patient04"""
    return fname.split('_')[0]


def per_patient_ahi(fnames, labels, preds, summary_json_path):
    """환자별 AHI 추정 + 실제 AHI 비교 + 중증도."""
    # 실제 AHI: summary JSON에서
    with open(summary_json_path) as f:
        summary = json.load(f)
    gt_ahi_per_pid = {pid: data['AHI_estimate'] for pid, data in summary['patients'].items()}
    gt_severity_per_pid = {pid: data['severity'] for pid, data in summary['patients'].items()}

    # 환자별로 그룹화
    patient_data = defaultdict(lambda: {'preds': [], 'labels': [], 'count': 0})
    for fname, lbl, pr in zip(fnames, labels, preds):
        pid = parse_patient_id(fname)
        patient_data[pid]['preds'].append(pr)
        patient_data[pid]['labels'].append(lbl)
        patient_data[pid]['count'] += 1

    # 환자별 추정 AHI = (예측된 apnea epoch 수) × 30s / 측정시간(시간)
    pred_ahi_per_pid = {}
    gt_ahi_subset = {}
    pred_severity_per_pid = {}
    gt_severity_subset = {}

    for pid, data in patient_data.items():
        pred_arr = np.array(data['preds'])
        # 예측된 apnea epoch 비율 → 시간당 추정 이벤트 수
        # 단순 추정: 30초마다 1개 이벤트로 가정 (이벤트 가능 epoch 수가 AHI 인덱스의 proxy)
        apnea_epoch_count = int((pred_arr == 1).sum())
        # 측정 시간 = epoch 수 × 30s / 3600
        recording_hours = data['count'] * 30 / 3600
        # AHI 추정 — 30초 epoch당 평균 이벤트 0.5건 가정 (보수적)
        # 더 정확하게는 실제 RML에 epoch당 이벤트 수 비율을 사용
        # 여기서는 epoch당 1.0 이벤트로 단순 추정 (apnea label = 적어도 1개 이벤트 epoch)
        # 실제 환자별 (raw event count / labeled epoch count) ratio를 사용해 보정
        if pid in summary['patients']:
            gt_data = summary['patients'][pid]
            gt_apnea_epochs = gt_data['apnea_epochs']
            gt_event_count = gt_data['total_events']
            # epoch당 평균 이벤트 (실제 비율)
            events_per_apnea_epoch = gt_event_count / max(1, gt_apnea_epochs)
            # 예측 AHI 추정
            pred_total_events = apnea_epoch_count * events_per_apnea_epoch
            pred_ahi = pred_total_events / max(1e-6, recording_hours)
        else:
            pred_ahi = apnea_epoch_count / max(1e-6, recording_hours)

        pred_ahi_per_pid[pid] = pred_ahi

        # 중증도
        sev = ('severe' if pred_ahi >= 30
               else 'moderate' if pred_ahi >= 15
               else 'mild' if pred_ahi >= 5
               else 'normal')
        pred_severity_per_pid[pid] = sev

        if pid in gt_ahi_per_pid:
            gt_ahi_subset[pid] = gt_ahi_per_pid[pid]
            gt_severity_subset[pid] = gt_severity_per_pid[pid]

    # AHI metrics
    common_pids = sorted(set(pred_ahi_per_pid.keys()) & set(gt_ahi_subset.keys()))
    gt_arr = np.array([gt_ahi_subset[p] for p in common_pids])
    pred_arr = np.array([pred_ahi_per_pid[p] for p in common_pids])

    ahi_metrics = {
        'num_patients': len(common_pids),
        'rmse': float(np.sqrt(mean_squared_error(gt_arr, pred_arr))),
        'mae': float(mean_absolute_error(gt_arr, pred_arr)),
        'r2': float(r2_score(gt_arr, pred_arr)) if len(gt_arr) > 1 else 0.0,
        'gt_mean': float(gt_arr.mean()),
        'pred_mean': float(pred_arr.mean()),
        'gt_std': float(gt_arr.std()),
        'pred_std': float(pred_arr.std()),
    }

    # Severity classification metrics
    severity_levels = ['normal', 'mild', 'moderate', 'severe']
    gt_sev_arr = [severity_levels.index(gt_severity_subset[p]) for p in common_pids]
    pred_sev_arr = [severity_levels.index(pred_severity_per_pid[p]) for p in common_pids]

    sev_acc = float(accuracy_score(gt_sev_arr, pred_sev_arr))
    sev_bal_acc = float(balanced_accuracy_score(gt_sev_arr, pred_sev_arr))
    sev_cm = confusion_matrix(gt_sev_arr, pred_sev_arr, labels=list(range(4)))

    # binary severity: severe vs not severe (most product-relevant)
    gt_severe = np.array([1 if s == 3 else 0 for s in gt_sev_arr])
    pred_severe = np.array([1 if s == 3 else 0 for s in pred_sev_arr])
    binary_acc = float(accuracy_score(gt_severe, pred_severe))

    # binary: clinically_significant (moderate/severe) vs not
    gt_sig = np.array([1 if s >= 2 else 0 for s in gt_sev_arr])
    pred_sig = np.array([1 if s >= 2 else 0 for s in pred_sev_arr])
    sig_acc = float(accuracy_score(gt_sig, pred_sig))

    severity_metrics = {
        'accuracy_4class': sev_acc,
        'balanced_accuracy_4class': sev_bal_acc,
        'confusion_matrix_4class': sev_cm.tolist(),
        'labels': severity_levels,
        'binary_severe_vs_not_accuracy': binary_acc,
        'binary_clinically_significant_accuracy': sig_acc,
    }

    return {
        'ahi_metrics': ahi_metrics,
        'severity_metrics': severity_metrics,
        'patient_details': {
            pid: {
                'gt_ahi': gt_ahi_subset.get(pid, None),
                'pred_ahi': pred_ahi_per_pid[pid],
                'gt_severity': gt_severity_subset.get(pid, None),
                'pred_severity': pred_severity_per_pid[pid],
                'n_epochs': patient_data[pid]['count'],
            }
            for pid in common_pids
        },
    }


def print_results(epoch_metrics, ahi_results):
    print('\n' + '=' * 70)
    print('  Per-Epoch Metrics (Apnea/Normal binary classification)')
    print('=' * 70)
    print(f'  N samples:       {epoch_metrics["normal"]["support"] + epoch_metrics["apnea"]["support"]:,}')
    print(f'  Accuracy:        {epoch_metrics["accuracy"] * 100:.2f}%')
    print(f'  Balanced Acc:    {epoch_metrics["balanced_accuracy"] * 100:.2f}%')
    print(f'  Macro F1:        {epoch_metrics["macro_f1"] * 100:.2f}%')
    print(f'  AUC:             {epoch_metrics["auc"]:.4f}')
    print()
    print(f'           Precision    Recall      F1     Support')
    print(f'  Normal   {epoch_metrics["normal"]["precision"]:.4f}    '
          f'{epoch_metrics["normal"]["recall"]:.4f}    '
          f'{epoch_metrics["normal"]["f1"]:.4f}    {epoch_metrics["normal"]["support"]:,}')
    print(f'  Apnea    {epoch_metrics["apnea"]["precision"]:.4f}    '
          f'{epoch_metrics["apnea"]["recall"]:.4f}    '
          f'{epoch_metrics["apnea"]["f1"]:.4f}    {epoch_metrics["apnea"]["support"]:,}')
    cm = np.array(epoch_metrics['confusion_matrix'])
    print(f'\n  Confusion Matrix (rows=true, cols=pred):')
    print(f'                  Normal      Apnea')
    print(f'    Normal       {cm[0, 0]:>7d}   {cm[0, 1]:>7d}')
    print(f'    Apnea        {cm[1, 0]:>7d}   {cm[1, 1]:>7d}')

    am = ahi_results['ahi_metrics']
    print('\n' + '=' * 70)
    print('  Per-Patient AHI Estimation')
    print('=' * 70)
    print(f'  Patients evaluated: {am["num_patients"]}')
    print(f'  RMSE:               {am["rmse"]:.2f} events/hour')
    print(f'  MAE:                {am["mae"]:.2f} events/hour')
    print(f'  R² score:           {am["r2"]:.4f}')
    print(f'  GT AHI:   mean={am["gt_mean"]:.1f}  std={am["gt_std"]:.1f}')
    print(f'  Pred AHI: mean={am["pred_mean"]:.1f}  std={am["pred_std"]:.1f}')

    sm = ahi_results['severity_metrics']
    print('\n' + '=' * 70)
    print('  Severity Classification (Normal / Mild / Moderate / Severe)')
    print('=' * 70)
    print(f'  4-class Accuracy:        {sm["accuracy_4class"] * 100:.2f}%')
    print(f'  4-class Balanced Acc:    {sm["balanced_accuracy_4class"] * 100:.2f}%')
    print(f'  Binary (Severe vs not):  {sm["binary_severe_vs_not_accuracy"] * 100:.2f}%')
    print(f'  Binary (Clinically sig): {sm["binary_clinically_significant_accuracy"] * 100:.2f}%')
    sev_cm = np.array(sm['confusion_matrix_4class'])
    print(f'\n  Confusion Matrix:')
    print(f'                    Normal   Mild   Mod   Sev')
    for i, lbl in enumerate(sm['labels']):
        print(f'    {lbl:>8s}      {sev_cm[i, 0]:>5d}  {sev_cm[i, 1]:>5d}  {sev_cm[i, 2]:>5d}  {sev_cm[i, 3]:>5d}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='./workspaces/apnea_v2/checkpoints/best_model.pth')
    parser.add_argument('--data_dir', default='../data/data_for_ai/full_ver_3class')
    parser.add_argument('--test_csv', default='../data/data_for_ai/apnea_2class/test_full.csv')
    parser.add_argument('--summary_json',
                        default='../data/data_for_ai/apnea_2class/apnea_events_summary.json')
    parser.add_argument('--out_dir', default='./workspaces/apnea_v2/results')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--cuda', action='store_true', default=True)
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    base = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(base, args.ckpt) if not os.path.isabs(args.ckpt) else args.ckpt
    test_csv = os.path.join(base, args.test_csv) if not os.path.isabs(args.test_csv) else args.test_csv
    data_dir = os.path.join(base, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    summary_json = os.path.join(base, args.summary_json) if not os.path.isabs(args.summary_json) else args.summary_json
    out_dir = os.path.join(base, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f'Checkpoint: {ckpt_path}')
    print(f'Test CSV:   {test_csv}')
    print(f'Summary:    {summary_json}')

    model = load_resnet22(ckpt_path, device)

    fnames, labels, preds, probs = evaluate(model, data_dir, test_csv, device,
                                             batch_size=args.batch_size)

    epoch_metrics = per_epoch_metrics(labels, preds, probs)
    ahi_results = per_patient_ahi(fnames, labels, preds, summary_json)

    print_results(epoch_metrics, ahi_results)

    # Save
    out = {
        'checkpoint': args.ckpt,
        'test_csv': args.test_csv,
        'per_epoch': epoch_metrics,
        'per_patient_ahi': ahi_results['ahi_metrics'],
        'severity_classification': ahi_results['severity_metrics'],
        'patient_details': ahi_results['patient_details'],
    }
    out_path = os.path.join(out_dir, 'eval_apnea.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
