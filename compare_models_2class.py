#!/usr/bin/env python3
"""CNN6 vs ResNet22 2-class 공정 비교 — 동일 테스트셋(149명 balanced)으로 평가."""

import os
import sys
import json
import numpy as np
import torch
import librosa
import csv
from collections import Counter
from sklearn.metrics import (
    balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 경로 설정 ──
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'data_for_ai', 'full_ver_2class')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')

# ── 모델 로드 함수 ──
sys.path.insert(0, os.path.join(BASE_DIR, 'pytorch'))

import importlib.util


def load_model_from_file(model_path, model_class_name, checkpoint_path, device, classes_num=2):
    """모델 파일에서 클래스 로드하고 체크포인트 적용."""
    spec = importlib.util.spec_from_file_location('model_module', model_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ModelClass = getattr(mod, model_class_name)

    model = ModelClass(
        sample_rate=16000, window_size=512, hop_size=160,
        mel_bins=64, fmin=50, fmax=8000, classes_num=classes_num)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    return model, ckpt


def load_wav(wav_path, sr=16000, clip_samples=480000):
    waveform, _ = librosa.load(wav_path, sr=sr, mono=True)
    if len(waveform) < clip_samples:
        waveform = np.concatenate([waveform, np.zeros(clip_samples - len(waveform), dtype=np.float32)])
    else:
        waveform = waveform[:clip_samples]
    return waveform.astype(np.float32)


@torch.no_grad()
def evaluate_model(model, filenames, labels, data_dir, device):
    """모델 평가."""
    all_preds = []
    for i, fname in enumerate(filenames):
        patient_folder = fname.split('_epoch')[0]
        # full_ver_2class는 symlink이므로 full_ver_3class 경로도 시도
        wav_path = os.path.join(data_dir, patient_folder, fname)
        if not os.path.exists(wav_path):
            wav_path = os.path.join(BASE_DIR, '..', 'data', 'data_for_ai', 'full_ver_3class', patient_folder, fname)

        waveform = load_wav(wav_path)
        wav_tensor = torch.from_numpy(waveform).unsqueeze(0).to(device)
        output = model(wav_tensor)
        logits = output['clipwise_output']
        pred = logits.argmax(dim=1).item()
        all_preds.append(pred)

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(filenames)}...")

    all_preds = np.array(all_preds)
    all_targets = np.array(labels)

    acc = np.mean(all_preds == all_targets)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, labels=[0, 1], zero_division=0)
    macro_f1 = f1.mean()
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])

    return {
        'accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'macro_f1': float(macro_f1),
        'wake': {'P': float(precision[0]), 'R': float(recall[0]), 'F1': float(f1[0])},
        'sleep': {'P': float(precision[1]), 'R': float(recall[1]), 'F1': float(f1[1])},
        'cm': cm,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 테스트 데이터 로드
    filenames = []
    labels = []
    with open(TEST_CSV) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            filenames.append(row[0])
            labels.append(int(row[1]))

    c = Counter(labels)
    print(f"Test: {len(filenames)} samples (wake={c[0]}, sleep={c[1]})")

    # 후보 모델 정의
    candidates = {
        'CNN6\n(full_ver)': {
            'model_path': os.path.join(BASE_DIR, '..', 'cnn6', 'pytorch', 'models.py'),
            'model_class': 'Cnn6',
            'checkpoint': os.path.join(BASE_DIR, '..', 'cnn6', 'workspaces', 'full_ver_2class', 'checkpoints', 'best_model.pth'),
        },
        'CNN6\n(ratio_ver)': {
            'model_path': os.path.join(BASE_DIR, '..', 'cnn6', 'pytorch', 'models.py'),
            'model_class': 'Cnn6',
            'checkpoint': os.path.join(BASE_DIR, '..', 'cnn6', 'workspaces', 'ratio_ver_2class', 'checkpoints', 'best_model.pth'),
        },
        'ResNet22\n(v1 best)': {
            'model_path': os.path.join(BASE_DIR, 'pytorch', 'models.py'),
            'model_class': 'ResNet22',
            'checkpoint': os.path.join(BASE_DIR, 'checkpoints', 'hierarchical', 'v1', 'stage1_resnet22.pth'),
        },
    }

    results = {}
    for name, info in candidates.items():
        print(f"\n=== {name.replace(chr(10), ' ')} ===")
        print(f"  Checkpoint: {info['checkpoint']}")

        if not os.path.exists(info['checkpoint']):
            print(f"  [SKIP] 체크포인트 없음")
            continue

        model, ckpt = load_model_from_file(
            info['model_path'], info['model_class'], info['checkpoint'], device)
        print(f"  Epoch: {ckpt.get('epoch', '?')}, Val Acc: {ckpt.get('val_acc', 0):.4f}")

        result = evaluate_model(model, filenames, labels, DATA_DIR, device)
        results[name] = result

        print(f"  Accuracy: {result['accuracy']*100:.2f}%")
        print(f"  Wake:  P={result['wake']['P']:.4f} R={result['wake']['R']:.4f} F1={result['wake']['F1']:.4f}")
        print(f"  Sleep: P={result['sleep']['P']:.4f} R={result['sleep']['R']:.4f} F1={result['sleep']['F1']:.4f}")

        del model
        torch.cuda.empty_cache()

    if not results:
        print("평가할 모델이 없습니다.")
        return

    # ── 비교 이미지 생성 ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f'CNN6 vs ResNet22 — 2-Class (Wake/Sleep) 공정 비교\n'
        f'Test: 149명 Balanced ({len(filenames)} samples, wake:sleep = 1:1)',
        fontsize=16, fontweight='bold')

    model_names = list(results.keys())
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24']

    # Panel 1: Accuracy 비교
    ax = axes[0, 0]
    accs = [results[n]['accuracy'] * 100 for n in model_names]
    bars = ax.bar(range(len(model_names)), accs, color=colors[:len(model_names)])
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Overall Accuracy')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', fontsize=12, fontweight='bold')

    # Panel 2: Per-class F1
    ax = axes[0, 1]
    x = np.arange(len(model_names))
    width = 0.35
    wake_f1 = [results[n]['wake']['F1'] for n in model_names]
    sleep_f1 = [results[n]['sleep']['F1'] for n in model_names]
    bars1 = ax.bar(x - width / 2, wake_f1, width, label='Wake F1', color='#ff6b6b')
    bars2 = ax.bar(x + width / 2, sleep_f1, width, label='Sleep F1', color='#4ecdc4')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-class F1 Score')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                        f'{h:.3f}', ha='center', fontsize=9, fontweight='bold')

    # Panel 3: Precision / Recall 비교
    ax = axes[1, 0]
    metrics = ['Wake P', 'Wake R', 'Sleep P', 'Sleep R']
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)
    for i, name in enumerate(model_names):
        vals = [
            results[name]['wake']['P'],
            results[name]['wake']['R'],
            results[name]['sleep']['P'],
            results[name]['sleep']['R'],
        ]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                       label=name.replace('\n', ' '), color=colors[i])
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                        f'{v:.2f}', ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel('Score')
    ax.set_title('Precision & Recall 비교')
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: Confusion Matrices
    ax = axes[1, 1]
    n_models = len(model_names)
    cm_text = ""
    for i, name in enumerate(model_names):
        cm = results[name]['cm']
        short_name = name.replace('\n', ' ')
        cm_text += f"{short_name}\n"
        cm_text += f"  pred_W  pred_S\n"
        cm_text += f"W  {cm[0][0]:>5d}  {cm[0][1]:>5d}\n"
        cm_text += f"S  {cm[1][0]:>5d}  {cm[1][1]:>5d}\n\n"
    ax.text(0.05, 0.95, cm_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title('Confusion Matrices')
    ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(BASE_DIR, '..', 'report', 'cnn6_vs_resnet22_2class.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n비교 이미지 저장: {output_path}")

    # JSON 결과 저장
    json_path = os.path.join(BASE_DIR, '..', 'report', 'cnn6_vs_resnet22_2class.json')
    json_results = {}
    for name, r in results.items():
        key = name.replace('\n', ' ')
        json_results[key] = {k: v for k, v in r.items() if k != 'cm'}
        json_results[key]['confusion_matrix'] = r['cm'].tolist()
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"JSON 결과 저장: {json_path}")


if __name__ == '__main__':
    main()
