#!/usr/bin/env python3
"""
Audio-only 2-stage CNN cascade inference.

Stage 1: ResNet22 (audio Wake/Sleep, full_ver_2class_finetune)
Stage 2: ResNet22 (audio REM/NREM, v7_remnrem) — Sleep으로 분류된 epoch에만 적용

순수 CNN cascade (BiLSTM, EOG 없음). v3_user_proposed (audio+EOG) 의 EOG 대체용 비교.

사용:
  python3 audio_cnn_cascade_infer.py --test_type both
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
    balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix,
)

# ResNet22 via importlib (avoid 'models' name collision)
_resnet_models_path = os.path.join(os.path.dirname(__file__), 'pytorch', 'models.py')
_spec = importlib.util.spec_from_file_location('resnet_models', _resnet_models_path)
_resnet_mod = importlib.util.module_from_spec(_spec)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pytorch'))
_spec.loader.exec_module(_resnet_mod)
ResNet22 = _resnet_mod.ResNet22

LABEL_NAMES_3CLASS = ['wake', 'rem', 'nrem']
SAMPLE_RATE = 16000
CLIP_SAMPLES = SAMPLE_RATE * 30


def load_resnet22(ckpt_path, device, classes_num=2):
    model = ResNet22(
        sample_rate=16000, window_size=512, hop_size=160,
        mel_bins=64, fmin=50, fmax=8000, classes_num=classes_num)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state['model'] if isinstance(state, dict) and 'model' in state else state
    # filter shape mismatches just in case
    md = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in md and v.shape == md[k].shape}
    md.update(filtered)
    model.load_state_dict(md)
    print(f'  Loaded {len(filtered)}/{len(sd)} layers')
    if isinstance(state, dict):
        epoch = state.get('epoch', '?')
        val_metric = state.get('val_acc', state.get('val_balanced_acc', None))
        print(f'  Epoch: {epoch}, Val metric: {val_metric}')
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
def cascade_predict(stage1, stage2, wav_path, device, sleep_threshold=0.5):
    """순수 audio cascade.

    Stage 1: full_ver_2class_finetune → Wake or Sleep
    Stage 2: v7_remnrem → REM or NREM (Sleep 분류된 epoch만)

    Returns:
      pred_3class: 0=Wake, 1=REM, 2=NREM
      stage1_pred: 0=Wake, 1=Sleep
      stage2_pred: 0=REM, 1=NREM (or None if Wake)
      p_sleep: float
    """
    wav = load_wav(wav_path)
    wav_t = torch.from_numpy(wav).unsqueeze(0).to(device)

    # Stage 1
    s1_logits = stage1(wav_t)['clipwise_output']
    s1_probs = torch.softmax(s1_logits, dim=1)
    p_sleep = s1_probs[0, 1].item()
    stage1_pred = 1 if p_sleep > sleep_threshold else 0

    if stage1_pred == 0:  # Wake
        return 0, stage1_pred, None, p_sleep

    # Stage 2 (Sleep)
    # Note: v7_remnrem trained with class 0=REM, class 1=NREM
    s2_logits = stage2(wav_t)['clipwise_output']
    stage2_pred = s2_logits.argmax(dim=1).item()
    pred_3class = 1 if stage2_pred == 0 else 2  # REM=1, NREM=2
    return pred_3class, stage1_pred, stage2_pred, p_sleep


def cache_predictions(stage1, stage2, data_dir, test_csv, device, sleep_threshold=0.5):
    """모든 샘플에 대해 cascade 예측."""
    items = []
    with open(test_csv) as f:
        reader = csv.reader(f); next(reader)
        for row in reader:
            items.append((row[0], int(row[1])))

    print(f'\nTest: {test_csv}')
    print(f'Samples: {len(items)}')
    dist = Counter(it[1] for it in items)
    for k in sorted(dist):
        print(f'  {LABEL_NAMES_3CLASS[k]}: {dist[k]}')

    preds = np.zeros(len(items), dtype=np.int64)
    labels = np.zeros(len(items), dtype=np.int64)
    s1_preds = np.zeros(len(items), dtype=np.int64)
    s2_preds = np.full(len(items), -1, dtype=np.int64)
    p_sleeps = np.zeros(len(items), dtype=np.float32)

    print('Cascade inference...')
    for i, (fname, lbl) in enumerate(items):
        patient = fname.split('_epoch')[0]
        wav_path = os.path.join(data_dir, patient, fname)
        p3, s1, s2, ps = cascade_predict(stage1, stage2, wav_path, device, sleep_threshold)
        preds[i] = p3
        labels[i] = lbl
        s1_preds[i] = s1
        if s2 is not None:
            s2_preds[i] = s2
        p_sleeps[i] = ps

        if (i + 1) % 1000 == 0:
            print(f'  {i + 1}/{len(items)}...')

    return preds, labels, s1_preds, s2_preds, p_sleeps


def compute_metrics(preds, labels):
    acc = float((preds == labels).mean())
    bal_acc = float(balanced_accuracy_score(labels, preds))
    p, r, f1, support = precision_recall_fscore_support(
        labels, preds, labels=[0, 1, 2], zero_division=0)
    macro_f1 = float(f1.mean())
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    return {
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


def print_metrics(name, m):
    print(f'\n{"=" * 72}')
    print(f'  {name}')
    print(f'{"=" * 72}')
    print(f'  Accuracy:       {m["accuracy"] * 100:.2f}%')
    print(f'  Balanced Acc:   {m["balanced_accuracy"] * 100:.2f}%')
    print(f'  Macro F1:       {m["macro_f1"]:.4f}')
    print(f'\n            Precision     Recall         F1')
    for cls in LABEL_NAMES_3CLASS:
        pc = m['per_class'][cls]
        print(f'    {cls:>5s}     {pc["precision"]:.4f}     '
              f'{pc["recall"]:.4f}     {pc["f1"]:.4f}')
    cm = np.array(m['confusion_matrix'])
    print(f'\n  Confusion Matrix (rows=true, cols=pred):')
    print(f'               Wake      REM     NREM')
    for i, cls in enumerate(LABEL_NAMES_3CLASS):
        print(f'    {cls:>5s}    {cm[i, 0]:>5d}    {cm[i, 1]:>5d}    {cm[i, 2]:>5d}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1_ckpt',
                        default='workspaces/full_ver_2class_finetune/checkpoints/best_model.pth')
    parser.add_argument('--stage2_ckpt',
                        default='../bilstm/workspaces/v7_remnrem/checkpoints/best_model.pth')
    parser.add_argument('--data_dir', default='../data/data_for_ai/full_ver_3class')
    parser.add_argument('--test_type', default='both', choices=['balanced', 'full', 'both'])
    parser.add_argument('--sleep_threshold', type=float, default=0.5)
    parser.add_argument('--out_dir', default='./checkpoints/hierarchical/audio_cnn_cascade')
    parser.add_argument('--cuda', action='store_true', default=True)
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    base = os.path.dirname(os.path.abspath(__file__))
    s1_ckpt = os.path.join(base, args.stage1_ckpt)
    s2_ckpt = os.path.join(base, args.stage2_ckpt)
    data_dir = os.path.join(base, args.data_dir)
    out_dir = os.path.join(base, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f'\nStage 1 (W/S):     {s1_ckpt}')
    stage1 = load_resnet22(s1_ckpt, device, classes_num=2)
    print(f'\nStage 2 (REM/NREM): {s2_ckpt}')
    stage2 = load_resnet22(s2_ckpt, device, classes_num=2)

    print(f'\nSleep threshold: {args.sleep_threshold}')

    test_files = []
    if args.test_type in ('balanced', 'both'):
        test_files.append(('balanced', os.path.join(data_dir, 'test.csv')))
    if args.test_type in ('full', 'both'):
        test_files.append(('full', os.path.join(data_dir, 'test_full.csv')))

    all_results = {}
    for tag, csv_path in test_files:
        preds, labels, s1_preds, s2_preds, p_sleeps = cache_predictions(
            stage1, stage2, data_dir, csv_path, device, args.sleep_threshold)

        # 메인 3-class 결과
        m = compute_metrics(preds, labels)
        print_metrics(f'Audio-only CNN Cascade — {tag} ({len(labels)} samples)', m)

        # Stage별 accuracy도 계산
        s1_true = (labels != 0).astype(int)  # Sleep if not Wake
        s1_acc = float((s1_preds == s1_true).mean())
        # Stage 2 accuracy: Sleep으로 분류된 것 중에서 REM/NREM 정답률
        s2_mask = (s1_preds == 1) & (labels != 0)
        s2_correct = 0
        s2_total = 0
        for i in range(len(preds)):
            if s2_mask[i]:
                gt_remnrem = 0 if labels[i] == 1 else 1  # REM(1)→0, NREM(2)→1
                if s2_preds[i] == gt_remnrem:
                    s2_correct += 1
                s2_total += 1
        s2_acc = s2_correct / max(1, s2_total)

        print(f'\n  Stage 1 (W/S) Accuracy:        {s1_acc * 100:.2f}%')
        print(f'  Stage 2 (REM/NREM) Accuracy:   {s2_acc * 100:.2f}%  '
              f'({s2_total} samples)')

        m['stage1_accuracy'] = s1_acc
        m['stage2_accuracy'] = s2_acc
        m['stage2_samples'] = s2_total
        all_results[tag] = m

        out_path = os.path.join(out_dir, f'test_{tag}.json')
        with open(out_path, 'w') as f:
            json.dump(m, f, indent=2)
        print(f'  Saved: {out_path}')

    # 기록용 메타데이터
    config = {
        'stage1_ckpt': args.stage1_ckpt,
        'stage2_ckpt': args.stage2_ckpt,
        'sleep_threshold': args.sleep_threshold,
        'description': 'Audio-only CNN cascade — pure ResNet22 + ResNet22 (no BiLSTM, no EOG)',
    }
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f'\nDone. Results in {out_dir}')


if __name__ == '__main__':
    main()
