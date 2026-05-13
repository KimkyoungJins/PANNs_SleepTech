#!/usr/bin/env python3
"""
W/S 분류기 (full_ver_2class_finetune) 출력에 temporal smoothing 적용해서 정확도 개선 실험.

방법:
  1. 모든 test_full.csv 샘플에 대해 W/S 예측 + 확률 계산
  2. 환자별로 epoch 순서대로 정렬
  3. 환자 내에서 median filter / min-duration rule 적용
  4. before/after 비교 (정확도, P/R/F1, confusion matrix)

핵심 가정:
  - 단일 epoch noise (W W W S W W W) 는 대부분 모델 오류
  - 양옆 같으면 가운데 흡수
  - sleep apnea 환자라 micro-arousal 일부 보존 필요 → kernel 작게

사용:
  python3 ws_smoothing_experiment.py --kernels 3,5,7 --threshold 0.5
"""

import os
import sys
import csv
import json
import argparse
import importlib.util
from collections import defaultdict

import numpy as np
import librosa
import torch
from scipy.signal import medfilt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix,
)

# ResNet22 import via importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pytorch'))
_resnet_models_path = os.path.join(os.path.dirname(__file__), 'pytorch', 'models.py')
_spec = importlib.util.spec_from_file_location('resnet_models', _resnet_models_path)
_resnet_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_resnet_mod)
ResNet22 = _resnet_mod.ResNet22

SAMPLE_RATE = 16000
CLIP_SAMPLES = SAMPLE_RATE * 30


def load_resnet22(ckpt_path, device):
    model = ResNet22(
        sample_rate=16000, window_size=512, hop_size=160,
        mel_bins=64, fmin=50, fmax=8000, classes_num=2)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
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


def parse_epoch_id(filename):
    """patient21_epoch0042.wav → (patient21, 42)"""
    base = os.path.splitext(filename)[0]
    pid = base.split('_epoch')[0]
    eid = int(base.split('_epoch')[1])
    return pid, eid


@torch.no_grad()
def cache_predictions(model, data_dir, test_csv, device):
    """모든 샘플에 대해 W/S 예측 (P(Sleep))과 ground truth label 캐싱.

    label 매핑: wake=0, sleep=1 (REM[1] + NREM[2] 합치기)
    """
    items = []
    with open(test_csv) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            fname, label_3class = row[0], int(row[1])
            # convert 3-class to 2-class W/S
            label_ws = 0 if label_3class == 0 else 1  # 0=wake, else=sleep
            pid, eid = parse_epoch_id(fname)
            items.append((fname, pid, eid, label_ws))

    # sort by patient then epoch
    items.sort(key=lambda x: (x[1], x[2]))

    n = len(items)
    print(f'Total samples: {n}')
    print(f'Patients: {len(set(it[1] for it in items))}')
    print('Caching W/S predictions...')

    p_sleep = np.zeros(n, dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    pids = []
    eids = []
    fnames = []

    for i, (fname, pid, eid, lbl) in enumerate(items):
        wav_path = os.path.join(data_dir, pid, fname)
        wav = load_wav(wav_path)
        wav_t = torch.from_numpy(wav).unsqueeze(0).to(device)
        logits = model(wav_t)['clipwise_output']
        probs = torch.softmax(logits, dim=1)
        p_sleep[i] = probs[0, 1].item()
        labels[i] = lbl
        pids.append(pid)
        eids.append(eid)
        fnames.append(fname)

        if (i + 1) % 1000 == 0:
            print(f'  {i + 1}/{n}...')

    return np.array(fnames), np.array(pids), np.array(eids), p_sleep, labels


def apply_smoothing(preds, pids, kernel_size):
    """환자 단위로 median filter 적용 (환자 경계 안 넘게)."""
    smoothed = preds.copy()
    unique_pids = np.unique(pids)

    for pid in unique_pids:
        mask = pids == pid
        seg = preds[mask]
        if len(seg) >= kernel_size:
            smoothed[mask] = medfilt(seg, kernel_size=kernel_size)
        # else: 너무 짧으면 그대로 유지

    return smoothed


def apply_min_duration_rule(preds, pids, min_run=2):
    """단일 epoch (양옆이 같은) 흡수.
    W W W S W W W → W W W W W W W (양옆이 W로 같음)
    W W W S S W W → 그대로 유지 (S가 2개)
    """
    smoothed = preds.copy()
    unique_pids = np.unique(pids)

    for pid in unique_pids:
        mask = pids == pid
        seg = smoothed[mask].copy()
        n = len(seg)

        # find runs
        runs = []
        i = 0
        while i < n:
            j = i
            while j < n and seg[j] == seg[i]:
                j += 1
            runs.append((i, j, seg[i]))
            i = j

        # short runs sandwiched between same labels → absorbed
        for k in range(1, len(runs) - 1):
            i_start, i_end, label = runs[k]
            run_len = i_end - i_start
            if run_len < min_run:
                left_label = runs[k - 1][2]
                right_label = runs[k + 1][2]
                if left_label == right_label:
                    seg[i_start:i_end] = left_label

        smoothed[mask] = seg

    return smoothed


def compute_metrics(preds, labels):
    acc = float(accuracy_score(labels, preds))
    bal_acc = float(balanced_accuracy_score(labels, preds))
    p, r, f1, support = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'wake': {
            'precision': float(p[0]), 'recall': float(r[0]),
            'f1': float(f1[0]), 'support': int(support[0]),
        },
        'sleep': {
            'precision': float(p[1]), 'recall': float(r[1]),
            'f1': float(f1[1]), 'support': int(support[1]),
        },
        'macro_f1': float((f1[0] + f1[1]) / 2),
        'confusion_matrix': cm.tolist(),
    }


def print_metrics(name, m):
    print(f'\n{name}:')
    print(f'  Accuracy:  {m["accuracy"]*100:.2f}%')
    print(f'  Balanced:  {m["balanced_accuracy"]*100:.2f}%')
    print(f'  Macro F1:  {m["macro_f1"]*100:.2f}%')
    print(f'  Wake:  P={m["wake"]["precision"]*100:.2f}  R={m["wake"]["recall"]*100:.2f}  F1={m["wake"]["f1"]*100:.2f}  N={m["wake"]["support"]}')
    print(f'  Sleep: P={m["sleep"]["precision"]*100:.2f}  R={m["sleep"]["recall"]*100:.2f}  F1={m["sleep"]["f1"]*100:.2f}  N={m["sleep"]["support"]}')
    cm = np.array(m['confusion_matrix'])
    print(f'  Confusion Matrix (행=true, 열=pred): [[{cm[0,0]}, {cm[0,1]}], [{cm[1,0]}, {cm[1,1]}]]')


def count_changes(before, after):
    """smoothing으로 바뀐 epoch 수."""
    return int((before != after).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='workspaces/full_ver_2class_finetune/checkpoints/best_model.pth')
    parser.add_argument('--data_dir', default='../data/data_for_ai/full_ver_3class')
    parser.add_argument('--test_csv', default='../data/data_for_ai/full_ver_3class/test_full.csv')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Sleep threshold on P(Sleep)')
    parser.add_argument('--kernels', type=str, default='3,5,7',
                        help='comma-separated median filter kernel sizes')
    parser.add_argument('--out_dir', default='./workspaces/full_ver_2class_finetune/smoothing')
    parser.add_argument('--cuda', action='store_true', default=True)
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    base = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(base, args.ckpt)
    test_csv = os.path.join(base, args.test_csv)
    data_dir = os.path.join(base, args.data_dir)
    out_dir = os.path.join(base, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f'Checkpoint: {ckpt_path}')
    model = load_resnet22(ckpt_path, device)
    print('Model loaded.')

    cache_path = os.path.join(out_dir, 'cache.npz')
    if os.path.exists(cache_path):
        print(f'Loading cached predictions: {cache_path}')
        cache = np.load(cache_path, allow_pickle=True)
        fnames, pids, eids, p_sleep, labels = (
            cache['fnames'], cache['pids'], cache['eids'],
            cache['p_sleep'], cache['labels'],
        )
    else:
        fnames, pids, eids, p_sleep, labels = cache_predictions(
            model, data_dir, test_csv, device)
        np.savez(cache_path, fnames=fnames, pids=pids, eids=eids,
                 p_sleep=p_sleep, labels=labels)
        print(f'Cache saved: {cache_path}')

    # baseline (no smoothing)
    raw_preds = (p_sleep > args.threshold).astype(np.int64)
    baseline_metrics = compute_metrics(raw_preds, labels)
    print_metrics(f'BASELINE (no smoothing, thr={args.threshold})', baseline_metrics)

    results = {'baseline': baseline_metrics}

    # 다양한 smoothing 시도
    kernels = [int(k) for k in args.kernels.split(',')]
    for k in kernels:
        smoothed = apply_smoothing(raw_preds, pids, kernel_size=k)
        n_changed = count_changes(raw_preds, smoothed)
        m = compute_metrics(smoothed, labels)
        print_metrics(f'MEDIAN FILTER kernel={k} ({n_changed} epochs changed)', m)
        m['kernel'] = k
        m['n_changed'] = n_changed
        results[f'medfilt_k{k}'] = m

    # min-duration rule
    smoothed_md = apply_min_duration_rule(raw_preds, pids, min_run=2)
    n_changed_md = count_changes(raw_preds, smoothed_md)
    m_md = compute_metrics(smoothed_md, labels)
    print_metrics(f'MIN-DURATION RULE (min_run=2, {n_changed_md} epochs changed)', m_md)
    m_md['min_run'] = 2
    m_md['n_changed'] = n_changed_md
    results['min_duration_2'] = m_md

    # combined: medfilt(3) + min_duration_2
    medfilt3 = apply_smoothing(raw_preds, pids, kernel_size=3)
    combined = apply_min_duration_rule(medfilt3, pids, min_run=2)
    n_changed_c = count_changes(raw_preds, combined)
    m_c = compute_metrics(combined, labels)
    print_metrics(f'COMBINED (medfilt3 + min_duration_2, {n_changed_c} epochs changed)', m_c)
    m_c['n_changed'] = n_changed_c
    results['combined_medfilt3_md2'] = m_c

    # save
    out_path = os.path.join(out_dir, 'results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved: {out_path}')

    # summary table
    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f'{"Method":<35s} {"Acc":>6s} {"Bal":>6s} {"MF1":>6s} {"WakeF1":>7s} {"SleepF1":>8s} {"Δacc":>6s}')
    base_acc = baseline_metrics['accuracy']
    print(f'{"baseline (raw)":<35s} {baseline_metrics["accuracy"]*100:>5.2f}% '
          f'{baseline_metrics["balanced_accuracy"]*100:>5.2f}% '
          f'{baseline_metrics["macro_f1"]*100:>5.2f}% '
          f'{baseline_metrics["wake"]["f1"]*100:>6.2f}% '
          f'{baseline_metrics["sleep"]["f1"]*100:>7.2f}% '
          f'{0:>5.2f}p')
    for key, m in results.items():
        if key == 'baseline':
            continue
        delta = (m['accuracy'] - base_acc) * 100
        print(f'{key:<35s} {m["accuracy"]*100:>5.2f}% '
              f'{m["balanced_accuracy"]*100:>5.2f}% '
              f'{m["macro_f1"]*100:>5.2f}% '
              f'{m["wake"]["f1"]*100:>6.2f}% '
              f'{m["sleep"]["f1"]*100:>7.2f}% '
              f'{delta:>+5.2f}p')


if __name__ == '__main__':
    main()
