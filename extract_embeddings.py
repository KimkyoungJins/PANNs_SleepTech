#!/usr/bin/env python3
"""
PANNs ResNet22 (AudioSet pretrained, mAP=0.430) → 2048-d embedding 추출.

각 환자별로 30s wav 모두를 ResNet22(frozen, eval)에 통과시켜
fc_audioset 직전 2048-d embedding을 얻어 .npy로 저장한다.

출력 (환자당 3개 파일):
  embeddings/{patient_id}_emb.npy      shape=[N, 2048] float32
  embeddings/{patient_id}_labels.npy   shape=[N]       int64  (0=W, 1=R, 2=N)
  embeddings/{patient_id}_meta.json    {split, epoch_ids, label_counts}

사용:
  cd project1/bilstm
  python3 extract_embeddings.py --cuda
  python3 extract_embeddings.py --cuda --limit 5      # 디버그용 5명만
"""

import os
import sys
import csv
import json
import time
import argparse
import numpy as np
import torch
import librosa

# ── ResNet22 import (sibling resnet/ 디렉토리) ──
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESNET_DIR = os.path.join(THIS_DIR, '..', 'resnet')
sys.path.insert(0, os.path.join(RESNET_DIR, 'pytorch'))

from models import ResNet22  # noqa: E402


# ── audio params (resnet/pytorch/main.py와 동일) ──
SAMPLE_RATE = 16000
CLIP_SAMPLES = SAMPLE_RATE * 30   # 480,000
WINDOW_SIZE = 512
HOP_SIZE = 160
MEL_BINS = 64
FMIN = 50
FMAX = 8000

# AudioSet 원본 PANNs 클래스 수
AUDIOSET_CLASSES = 527


def load_wav(path, sr=SAMPLE_RATE, target_len=CLIP_SAMPLES):
    """wav → float32 [target_len]. 부족하면 zero-pad, 길면 truncate."""
    waveform, _ = librosa.load(path, sr=sr, mono=True)
    if len(waveform) < target_len:
        waveform = np.concatenate(
            [waveform, np.zeros(target_len - len(waveform), dtype=np.float32)]
        )
    else:
        waveform = waveform[:target_len]
    return waveform.astype(np.float32)


def parse_epoch_id(filename):
    """patient01_epoch0001.wav → 1."""
    base = os.path.splitext(filename)[0]
    return int(base.rsplit('epoch', 1)[1])


def build_label_map(data_dir):
    """train + val + test_full CSV → {filename: (label, split)}."""
    label_map = {}
    for split, csv_name in [
        ('train', 'train.csv'),
        ('val', 'val.csv'),
        ('test', 'test_full.csv'),
    ]:
        path = os.path.join(data_dir, csv_name)
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)  # header
            for row in reader:
                fname, label = row[0], int(row[1])
                label_map[fname] = (label, split)
    return label_map


def load_pretrained_resnet22(pretrained_path, device):
    """원본 PANNs ResNet22 로드 (AudioSet 527 클래스, eval 모드, frozen)."""
    model = ResNet22(
        sample_rate=SAMPLE_RATE, window_size=WINDOW_SIZE, hop_size=HOP_SIZE,
        mel_bins=MEL_BINS, fmin=FMIN, fmax=FMAX,
        classes_num=AUDIOSET_CLASSES,
    )
    ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_for_patient(model, device, patient_dir, label_map, batch_size):
    """환자 1명의 모든 labeled epoch에 대해 embedding 추출 (epoch 순서 보존)."""
    pid = os.path.basename(patient_dir.rstrip('/'))
    wav_files = sorted(
        [f for f in os.listdir(patient_dir) if f.endswith('.wav') and f in label_map],
        key=parse_epoch_id,
    )
    if not wav_files:
        return None

    splits = {label_map[f][1] for f in wav_files}
    assert len(splits) == 1, f'{pid}: 한 환자가 여러 split에 걸침 {splits}'
    split = splits.pop()

    embeddings = []
    labels = []
    epoch_ids = []

    for i in range(0, len(wav_files), batch_size):
        batch_files = wav_files[i:i + batch_size]
        waveforms = np.stack(
            [load_wav(os.path.join(patient_dir, f)) for f in batch_files]
        )
        wav_tensor = torch.from_numpy(waveforms).to(device, non_blocking=True)
        out = model(wav_tensor)
        emb = out['embedding'].detach().cpu().numpy()  # [B, 2048]
        embeddings.append(emb)
        for f in batch_files:
            labels.append(label_map[f][0])
            epoch_ids.append(parse_epoch_id(f))

    return {
        'patient_id': pid,
        'split': split,
        'embeddings': np.concatenate(embeddings, axis=0).astype(np.float32),
        'labels': np.array(labels, dtype=np.int64),
        'epoch_ids': np.array(epoch_ids, dtype=np.int32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/data_for_ai/full_ver_3class')
    parser.add_argument('--pretrained_path', default='../resnet/ResNet22_mAP=0.430.pth')
    parser.add_argument('--output_dir', default='./embeddings')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--limit', type=int, default=None,
                        help='디버그용: 처리할 환자 수 제한')
    parser.add_argument('--skip_existing', action='store_true',
                        help='이미 .npy가 있는 환자는 건너뜀')
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if args.cuda and not torch.cuda.is_available():
        print('  [경고] --cuda 지정했으나 GPU 미가용 → CPU로 실행 (매우 느림)')

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Pretrained: {args.pretrained_path}')
    model = load_pretrained_resnet22(args.pretrained_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'ResNet22 loaded ({n_params/1e6:.1f}M params, frozen, eval mode)')

    label_map = build_label_map(args.data_dir)
    print(f'Total labeled epochs (train+val+test_full): {len(label_map)}')

    patient_dirs = sorted(
        os.path.join(args.data_dir, d)
        for d in os.listdir(args.data_dir)
        if d.startswith('patient')
        and os.path.isdir(os.path.join(args.data_dir, d))
    )
    if args.limit:
        patient_dirs = patient_dirs[:args.limit]
    print(f'Patients to process: {len(patient_dirs)}')
    print()

    summary = {'train': 0, 'val': 0, 'test': 0}
    skipped, total_epochs, t_start = [], 0, time.time()

    for idx, pdir in enumerate(patient_dirs, 1):
        pid = os.path.basename(pdir)
        emb_path = os.path.join(args.output_dir, f'{pid}_emb.npy')
        if args.skip_existing and os.path.exists(emb_path):
            print(f'[{idx}/{len(patient_dirs)}] {pid} — SKIP (already exists)')
            continue

        t0 = time.time()
        result = extract_for_patient(model, device, pdir, label_map, args.batch_size)
        if result is None:
            skipped.append(pid)
            print(f'[{idx}/{len(patient_dirs)}] {pid} — SKIP (no labeled wav)')
            continue

        np.save(emb_path, result['embeddings'])
        np.save(os.path.join(args.output_dir, f'{pid}_labels.npy'), result['labels'])

        lc = result['labels']
        meta = {
            'patient_id': pid,
            'split': result['split'],
            'num_epochs': int(len(lc)),
            'epoch_ids': result['epoch_ids'].tolist(),
            'label_counts': {
                'wake': int(np.sum(lc == 0)),
                'rem':  int(np.sum(lc == 1)),
                'nrem': int(np.sum(lc == 2)),
            },
            'embedding_dim': int(result['embeddings'].shape[1]),
            'embedding_source': 'panns_resnet22_audioset_mAP=0.430',
        }
        with open(os.path.join(args.output_dir, f'{pid}_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        summary[result['split']] += 1
        total_epochs += len(lc)
        dt = time.time() - t0
        print(
            f'[{idx}/{len(patient_dirs)}] {pid} ({result["split"]:<5s}) '
            f'{len(lc):>4d} epoch  '
            f'(W:{meta["label_counts"]["wake"]:>3d} '
            f'R:{meta["label_counts"]["rem"]:>3d} '
            f'N:{meta["label_counts"]["nrem"]:>3d})  '
            f'[{dt:.1f}s]'
        )

    elapsed = time.time() - t_start
    print()
    print('=' * 70)
    print(f'완료: train={summary["train"]} val={summary["val"]} test={summary["test"]}')
    print(f'총 epoch: {total_epochs:,}, 소요: {elapsed/60:.1f}분')
    if skipped:
        print(f'Skipped (no labeled wav): {len(skipped)} → {skipped[:10]}')
    print(f'Output: {os.path.abspath(args.output_dir)}/')


if __name__ == '__main__':
    main()
