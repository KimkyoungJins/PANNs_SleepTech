#!/usr/bin/env python3
"""
Hierarchical Stage 1 threshold sweep.

기존 hierarchical_infer.py의 hard cascade 대신:
  1. 모든 샘플에 대해 Stage 1 P(Sleep) 계산
  2. 모든 샘플에 대해 Stage 2 REM/NREM 예측도 항상 계산 (캐싱)
  3. threshold sweep을 post-process로 빠르게 실행

이렇게 하면 한 번의 inference로 여러 threshold를 평가 가능.

사용:
  python3 hierarchical_threshold_sweep.py --version v3_user_proposed --test_type balanced
  python3 hierarchical_threshold_sweep.py --version v3_user_proposed --test_type both
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
from collections import Counter
from sklearn.metrics import (
    balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix,
)

# EOG_CNN
sys.path.insert(0, os.path.dirname(__file__))
from models.eog_cnn import EOG_CNN

# ResNet22 via importlib to avoid 'models' name collision
_resnet_models_path = os.path.join(os.path.dirname(__file__), 'pytorch', 'models.py')
_spec = importlib.util.spec_from_file_location('resnet_models', _resnet_models_path)
_resnet_mod = importlib.util.module_from_spec(_spec)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pytorch'))
_spec.loader.exec_module(_resnet_mod)
ResNet22 = _resnet_mod.ResNet22

LABEL_NAMES_3CLASS = ['wake', 'rem', 'nrem']
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


def load_eog_cnn(ckpt, device):
    model = EOG_CNN(num_classes=2)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state'])
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
def cache_predictions(resnet, eog_cnn, data_dir, test_csv, device):
    """모든 샘플에 대해 Stage 1 P(Sleep)과 Stage 2 REM/NREM 예측 캐싱."""
    filenames = []
    labels = []
    with open(test_csv) as f:
        reader = csv.reader(f); next(reader)
        for row in reader:
            filenames.append(row[0])
            labels.append(int(row[1]))

    print(f"\nTest: {test_csv}")
    print(f"Samples: {len(filenames)}")
    dist = Counter(labels)
    for k in sorted(dist):
        print(f"  {LABEL_NAMES_3CLASS[k]}: {dist[k]}")

    p_sleep_arr = np.zeros(len(filenames), dtype=np.float32)
    s2_pred_arr = np.zeros(len(filenames), dtype=np.int64)

    print("Caching Stage 1 + Stage 2 predictions for all samples...")
    for i, fname in enumerate(filenames):
        patient = fname.split('_epoch')[0]
        wav_path = os.path.join(data_dir, patient, fname)
        eog_path = os.path.join(data_dir, patient, os.path.splitext(fname)[0] + '_eog.npy')

        # Stage 1
        wav = load_wav(wav_path)
        wav_t = torch.from_numpy(wav).unsqueeze(0).to(device)
        s1_logits = resnet(wav_t)['clipwise_output']
        s1_probs = torch.softmax(s1_logits, dim=1)
        p_sleep_arr[i] = s1_probs[0, 1].item()

        # Stage 2 (항상 계산해서 캐싱 — Wake로 분류돼도)
        eog = np.load(eog_path).astype(np.float32)
        eog_t = torch.from_numpy(eog).unsqueeze(0).to(device)
        s2_logits = eog_cnn(eog_t)
        s2_pred_arr[i] = s2_logits.argmax(dim=1).item()

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(filenames)}...")

    return np.array(filenames), np.array(labels), p_sleep_arr, s2_pred_arr


def evaluate_at_threshold(p_sleep, s2_preds, labels, threshold):
    """Threshold 적용해서 3-class 예측 + metric 계산."""
    # P(Sleep) > threshold이면 Sleep → Stage 2 (s2_pred 0=REM=1, 1=NREM=2)
    # else: Wake (0)
    is_sleep = p_sleep > threshold
    pred = np.where(
        is_sleep,
        np.where(s2_preds == 0, 1, 2),  # Sleep → REM(1) or NREM(2)
        0,  # Wake
    )

    acc = float((pred == labels).mean())
    bal_acc = float(balanced_accuracy_score(labels, pred))
    p, r, f1, support = precision_recall_fscore_support(
        labels, pred, labels=[0, 1, 2], zero_division=0)
    macro_f1 = float(f1.mean())
    cm = confusion_matrix(labels, pred, labels=[0, 1, 2])

    return {
        'threshold': float(threshold),
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'macro_f1': macro_f1,
        'per_class': {
            LABEL_NAMES_3CLASS[i]: {
                'precision': float(p[i]), 'recall': float(r[i]),
                'f1': float(f1[i]), 'support': int(support[i]),
            } for i in range(3)
        },
        'confusion_matrix': cm.tolist(),
    }


def print_summary(results, label):
    print(f"\n{'='*72}")
    print(f"  Threshold Sweep — {label}")
    print(f"{'='*72}")
    print(f"  {'thr':>5s}  {'BalAcc':>7s}  {'MacroF1':>7s}  "
          f"{'W_P':>5s} {'W_R':>5s}  {'R_P':>5s} {'R_R':>5s}  {'N_P':>5s} {'N_R':>5s}")
    for r in results:
        pc = r['per_class']
        print(f"  {r['threshold']:>5.2f}  "
              f"{r['balanced_accuracy']*100:>7.2f}  "
              f"{r['macro_f1']*100:>7.2f}  "
              f"{pc['wake']['precision']*100:>5.1f} {pc['wake']['recall']*100:>5.1f}  "
              f"{pc['rem']['precision']*100:>5.1f} {pc['rem']['recall']*100:>5.1f}  "
              f"{pc['nrem']['precision']*100:>5.1f} {pc['nrem']['recall']*100:>5.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='../data/data_for_ai/full_ver_3class')
    parser.add_argument('--test_type', type=str, default='both',
                        choices=['balanced', 'full', 'both'])
    parser.add_argument('--thresholds', type=str,
                        default='0.50,0.55,0.60,0.65,0.70,0.75',
                        help='comma-separated threshold values to sweep')
    parser.add_argument('--cuda', action='store_true', default=True)
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Version: {args.version}")

    # Load checkpoints
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ver_dir = os.path.join(base_dir, 'checkpoints', 'hierarchical', args.version)
    s1_ckpt = os.path.join(ver_dir, 'stage1_resnet22.pth')
    s2_ckpt = os.path.join(ver_dir, 'stage2_eog_cnn.pth')
    print(f"Stage 1: {s1_ckpt}")
    print(f"Stage 2: {s2_ckpt}")

    resnet = load_resnet22(s1_ckpt, device)
    eog_cnn = load_eog_cnn(s2_ckpt, device)

    data_dir = os.path.join(base_dir, args.data_dir)
    thresholds = [float(t) for t in args.thresholds.split(',')]

    # Cached eval per test type
    test_files = []
    if args.test_type in ('balanced', 'both'):
        test_files.append(('balanced', os.path.join(data_dir, 'test.csv')))
    if args.test_type in ('full', 'both'):
        test_files.append(('full', os.path.join(data_dir, 'test_full.csv')))

    sweep_dir = os.path.join(ver_dir, 'sweep')
    os.makedirs(sweep_dir, exist_ok=True)

    for tag, csv_path in test_files:
        filenames, labels, p_sleep, s2_preds = cache_predictions(
            resnet, eog_cnn, data_dir, csv_path, device)

        # Save cache
        cache_path = os.path.join(sweep_dir, f'cache_{tag}.npz')
        np.savez(cache_path, filenames=filenames, labels=labels,
                 p_sleep=p_sleep, s2_preds=s2_preds)
        print(f"Cache saved: {cache_path}")

        results = [evaluate_at_threshold(p_sleep, s2_preds, labels, t)
                   for t in thresholds]
        print_summary(results, label=f"{tag} ({len(labels)} samples)")

        # Save results
        out_path = os.path.join(sweep_dir, f'sweep_{tag}.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Sweep results saved: {out_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
