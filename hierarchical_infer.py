#!/usr/bin/env python3
"""Hierarchical 추론 파이프라인.

Stage 1: ResNet22 (Mic wav) → Wake(0) vs Sleep(1)
Stage 2: EOG_CNN (2ch EOG)  → REM(0) vs NREM(1)  (Sleep인 경우만)
최종: 3-class → Wake(0), REM(1), NREM(2)

사용법:
    # balanced test
    python3 hierarchical_infer.py --version v2 --test_type balanced

    # unbalanced test
    python3 hierarchical_infer.py --version v2 --test_type full

    # 둘 다
    python3 hierarchical_infer.py --version v2 --test_type both
"""

import os
import sys
import csv
import json
import argparse
from datetime import datetime

import numpy as np
import librosa
import torch
from collections import Counter
from sklearn.metrics import (
    balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix,
)

# EOG_CNN import
sys.path.insert(0, os.path.dirname(__file__))
from models.eog_cnn import EOG_CNN

# ResNet22 import via importlib to avoid 'models' package name collision
import importlib.util
_resnet_models_path = os.path.join(os.path.dirname(__file__), 'pytorch', 'models.py')
_spec = importlib.util.spec_from_file_location('resnet_models', _resnet_models_path)
_resnet_mod = importlib.util.module_from_spec(_spec)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pytorch'))
_spec.loader.exec_module(_resnet_mod)
ResNet22 = _resnet_mod.ResNet22

LABEL_NAMES_3CLASS = ['wake', 'rem', 'nrem']
SAMPLE_RATE = 16000
CLIP_SAMPLES = SAMPLE_RATE * 30


def load_resnet22(checkpoint_path, device, classes_num=2):
    model = ResNet22(
        sample_rate=16000, window_size=512, hop_size=160,
        mel_bins=64, fmin=50, fmax=8000, classes_num=classes_num)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    print(f"ResNet22 loaded: {checkpoint_path}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}, Val Acc: {ckpt.get('val_acc', 0):.4f}")
    return model, ckpt


def load_eog_cnn(checkpoint_path, device):
    model = EOG_CNN(num_classes=2)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    print(f"EOG_CNN loaded: {checkpoint_path}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}, "
          f"Macro F1: {ckpt.get('metrics', {}).get('val_macro_f1', 0):.4f}")
    return model, ckpt


def load_wav(wav_path, sample_rate=16000, clip_samples=480000):
    waveform, _ = librosa.load(wav_path, sr=sample_rate, mono=True)
    if len(waveform) < clip_samples:
        waveform = np.concatenate([
            waveform,
            np.zeros(clip_samples - len(waveform), dtype=np.float32)])
    else:
        waveform = waveform[:clip_samples]
    return waveform.astype(np.float32)


@torch.no_grad()
def hierarchical_predict(resnet, eog_cnn, wav_path, eog_path, device):
    # Stage 1: Wake vs Sleep
    waveform = load_wav(wav_path)
    wav_tensor = torch.from_numpy(waveform).unsqueeze(0).to(device)
    output = resnet(wav_tensor)
    logits = output['clipwise_output']
    stage1_pred = logits.argmax(dim=1).item()

    if stage1_pred == 0:
        return 0, stage1_pred, None

    # Stage 2: REM vs NREM
    eog = np.load(eog_path).astype(np.float32)
    eog_tensor = torch.from_numpy(eog).unsqueeze(0).to(device)
    stage2_logits = eog_cnn(eog_tensor)
    stage2_pred = stage2_logits.argmax(dim=1).item()

    pred_3class = 1 if stage2_pred == 0 else 2
    return pred_3class, stage1_pred, stage2_pred


def run_inference(resnet, eog_cnn, data_dir, test_csv, device):
    """Run hierarchical inference on a test CSV. Returns result dict."""
    filenames = []
    labels_3class = []
    with open(test_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            filenames.append(row[0])
            labels_3class.append(int(row[1]))

    print(f"\nTest: {test_csv}")
    print(f"Samples: {len(filenames)}")
    dist = Counter(labels_3class)
    for k in sorted(dist):
        print(f"  {LABEL_NAMES_3CLASS[k]}: {dist[k]}")

    all_preds = []
    stage1_preds = []
    stage2_preds = []

    print("Running inference...")
    for i, (fname, label) in enumerate(zip(filenames, labels_3class)):
        patient_folder = fname.split('_epoch')[0]
        wav_path = os.path.join(data_dir, patient_folder, fname)
        basename = os.path.splitext(fname)[0]
        eog_path = os.path.join(data_dir, patient_folder, basename + '_eog.npy')

        pred_3class, s1, s2 = hierarchical_predict(
            resnet, eog_cnn, wav_path, eog_path, device)

        all_preds.append(pred_3class)
        stage1_preds.append(s1)
        stage2_preds.append(s2)

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(filenames)}...")

    all_preds = np.array(all_preds)
    all_targets = np.array(labels_3class)

    # Metrics
    acc = np.mean(all_preds == all_targets)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, labels=[0, 1, 2], zero_division=0)
    macro_f1 = f1.mean()
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])

    s1_targets = np.array([0 if t == 0 else 1 for t in all_targets])
    s1_preds_arr = np.array(stage1_preds)
    s1_acc = np.mean(s1_targets == s1_preds_arr)

    s2_correct = 0
    s2_total = 0
    for t, s1, s2 in zip(all_targets, stage1_preds, stage2_preds):
        if s1 == 1 and t != 0:
            s2_total += 1
            expected = 0 if t == 1 else 1
            if s2 == expected:
                s2_correct += 1
    s2_acc = s2_correct / s2_total if s2_total > 0 else 0

    # Print
    print(f"\n{'=' * 60}")
    print(f"Stage 1 Acc: {s1_acc:.4f} | Stage 2 Acc: {s2_acc:.4f} ({s2_total} samples)")
    print(f"3-Class Accuracy: {acc:.4f} | Balanced Acc: {bal_acc:.4f} | Macro F1: {macro_f1:.4f}")
    print(f"{'':>8s} {'P':>8s} {'R':>8s} {'F1':>8s} {'N':>8s}")
    for i, name in enumerate(LABEL_NAMES_3CLASS):
        print(f"{name:>8s} {precision[i]:>8.4f} {recall[i]:>8.4f} {f1[i]:>8.4f} {support[i]:>8d}")

    return {
        'test_csv': test_csv,
        'samples': len(filenames),
        'stage1_accuracy': float(s1_acc),
        'stage2_accuracy': float(s2_acc),
        'stage2_samples': s2_total,
        'accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'macro_f1': float(macro_f1),
        'per_class': {
            LABEL_NAMES_3CLASS[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i]),
            } for i in range(3)
        },
        'confusion_matrix': cm.tolist(),
    }


def update_config(ver_dir, args, resnet_ckpt, eog_ckpt, results):
    """config.json 생성 또는 업데이트."""
    config_path = os.path.join(ver_dir, 'config.json')

    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {
            'version': args.version,
            'created': datetime.now().strftime('%Y-%m-%d'),
        }

    config['updated'] = datetime.now().strftime('%Y-%m-%d %H:%M')

    config['stage1'] = {
        'model': 'ResNet22',
        'task': 'Wake(0) vs Sleep(1)',
        'checkpoint_file': 'stage1_resnet22.pth',
        'epoch': resnet_ckpt.get('epoch', '?'),
        'val_acc': resnet_ckpt.get('val_acc', 0),
    }

    config['stage2'] = {
        'model': 'EOG_CNN',
        'task': 'REM(0) vs NREM(1)',
        'checkpoint_file': 'stage2_eog_cnn.pth',
        'epoch': eog_ckpt.get('epoch', '?'),
        'val_macro_f1': eog_ckpt.get('metrics', {}).get('val_macro_f1', 0),
    }

    config['test_results'] = results

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\nConfig saved: {config_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, required=True,
                        help='실험 버전 (예: v1, v2)')
    parser.add_argument('--data_dir', type=str,
                        default='../data/data_for_ai/full_ver_3class')
    parser.add_argument('--test_type', type=str, default='both',
                        choices=['balanced', 'full', 'both'],
                        help='테스트 유형: balanced, full, both')
    parser.add_argument('--test_csv_balanced', type=str,
                        default='../data/data_for_ai/full_ver_3class/test.csv')
    parser.add_argument('--test_csv_full', type=str,
                        default='../data/data_for_ai/full_ver_3class/test_full.csv')
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, args.data_dir)

    # 버전별 디렉토리
    ver_dir = os.path.join(base_dir, 'checkpoints', 'hierarchical', args.version)
    results_dir = os.path.join(ver_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    resnet_path = os.path.join(ver_dir, 'stage1_resnet22.pth')
    eog_path = os.path.join(ver_dir, 'stage2_eog_cnn.pth')

    if not os.path.exists(resnet_path):
        print(f"[ERROR] {resnet_path} not found")
        print(f"  Stage 1 체크포인트를 {ver_dir}/stage1_resnet22.pth에 복사하세요.")
        sys.exit(1)
    if not os.path.exists(eog_path):
        print(f"[ERROR] {eog_path} not found")
        print(f"  Stage 2 학습을 먼저 실행하세요: python3 train_eog_cnn.py --version {args.version}")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Version: {args.version}\n")

    # Load models
    resnet, resnet_ckpt = load_resnet22(resnet_path, device)
    eog_cnn, eog_ckpt = load_eog_cnn(eog_path, device)

    # Run inference
    all_results = {}

    if args.test_type in ('balanced', 'both'):
        csv_path = os.path.join(base_dir, args.test_csv_balanced)
        result = run_inference(resnet, eog_cnn, data_dir, csv_path, device)
        result_path = os.path.join(results_dir, 'test_balanced.json')
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {result_path}")
        all_results['balanced'] = result

    if args.test_type in ('full', 'both'):
        csv_path = os.path.join(base_dir, args.test_csv_full)
        result = run_inference(resnet, eog_cnn, data_dir, csv_path, device)
        result_path = os.path.join(results_dir, 'test_full.json')
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {result_path}")
        all_results['full'] = result

    # Update config.json
    update_config(ver_dir, args, resnet_ckpt, eog_ckpt, all_results)

    # Update best symlink
    hier_dir = os.path.join(base_dir, 'checkpoints', 'hierarchical')
    best_link = os.path.join(hier_dir, 'best')
    if os.path.islink(best_link):
        os.remove(best_link)
    os.symlink(args.version, best_link)
    print(f"best -> {args.version}")


if __name__ == '__main__':
    main()
