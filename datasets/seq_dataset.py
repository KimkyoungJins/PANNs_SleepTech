"""
SoundSequenceDataset — 환자별 PANNs embedding 시퀀스 Dataset.

extract_embeddings.py가 만든 .npy를 읽어 40-epoch sequence 단위로 제공한다.

Label 변환:
    원본 (0=Wake, 1=REM, 2=NREM)
    → binary (0=REM, 1=NREM)
    → Wake epoch은 mask=0 으로 무시

각 sample = (emb [T, 2048], label [T], mask [T])
    mask=1: REM/NREM 이며 가운데 center_window 안에 있음 → loss 계산
    mask=0: Wake 이거나 가장자리 (head/tail) → loss 무시

학습:
    - 환자 num_epochs 가중치로 random pick → random start
    - __len__: total_epochs / seq_len (1 epoch ≈ 데이터 1바퀴)

검증/테스트:
    - sliding window (stride 지정) 으로 결정론적 enumerate
"""

import os
import glob
import json
import logging
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset


def remap_labels(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    원본 라벨 → binary + mask.

    Args:
        labels: [N] int  (0=Wake, 1=REM, 2=NREM)

    Returns:
        new_labels: [N] int64  (0=REM, 1=NREM, Wake 위치는 0 dummy)
        mask:       [N] float32 (1=REM/NREM, 0=Wake)
    """
    mask = (labels != 0).astype(np.float32)
    new_labels = np.where(
        labels == 1, 0,    # REM → 0
        np.where(labels == 2, 1, 0)  # NREM → 1, Wake → 0 (dummy, mask=0)
    ).astype(np.int64)
    return new_labels, mask


class SoundSequenceDataset(Dataset):
    """
    Args:
        embeddings_dir:  extract_embeddings.py 결과 폴더
        split:           'train' | 'val' | 'test'
        seq_len:         시퀀스 길이 (default 40)
        center_window:   가운데 몇 epoch에만 loss 적용 (default 20)
                         seq_len - center_window 는 짝수여야 함 (앞뒤 균등)
        stride:          val/test 에서 sliding window stride
        epochs_per_iter: train의 __len__. None이면 total_epochs // seq_len
        seed:            train random sampling seed
    """

    def __init__(
        self,
        embeddings_dir: str,
        split: str,
        seq_len: int = 40,
        center_window: int = 20,
        stride: int = 20,
        epochs_per_iter: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__()
        assert split in {'train', 'val', 'test'}, f'invalid split: {split}'
        assert seq_len > 0 and center_window > 0
        assert center_window <= seq_len, \
            f'center_window({center_window}) > seq_len({seq_len})'
        assert (seq_len - center_window) % 2 == 0, \
            f'seq_len-center must be even (current: {seq_len-center_window})'

        self.embeddings_dir = embeddings_dir
        self.split = split
        self.seq_len = seq_len
        self.center_window = center_window
        self.stride = stride
        self.is_train = (split == 'train')

        # ── 환자별 데이터 로드 ──
        self.patients: List[Dict] = []
        skipped = []
        meta_files = sorted(glob.glob(os.path.join(embeddings_dir, '*_meta.json')))

        for mf in meta_files:
            with open(mf) as f:
                meta = json.load(f)
            if meta['split'] != split:
                continue
            pid = meta['patient_id']
            n = meta['num_epochs']
            if n < seq_len:
                skipped.append((pid, n))
                continue

            emb = np.load(os.path.join(embeddings_dir, f'{pid}_emb.npy'))
            raw = np.load(os.path.join(embeddings_dir, f'{pid}_labels.npy'))
            new_labels, mask = remap_labels(raw)

            self.patients.append({
                'pid': pid,
                'emb': emb,           # [N, 2048] float32
                'labels': new_labels,  # [N] int64
                'mask': mask,          # [N] float32
                'num_epochs': n,
            })

        if not self.patients:
            raise RuntimeError(
                f'No patients loaded for split={split} from {embeddings_dir}'
            )

        # ── 시퀀스 윈도우 enumerate (val/test) ──
        self.windows: List[Tuple[int, int]] = []
        if not self.is_train:
            for pi, p in enumerate(self.patients):
                last_start = p['num_epochs'] - seq_len
                starts = list(range(0, last_start + 1, stride))
                # 마지막 epoch까지 커버 (stride 떨어진 경우 추가)
                if starts and starts[-1] != last_start:
                    starts.append(last_start)
                for s in starts:
                    self.windows.append((pi, s))

        # ── train __len__ 산정 ──
        total_epochs = sum(p['num_epochs'] for p in self.patients)
        if self.is_train:
            self.length = epochs_per_iter or (total_epochs // seq_len)
        else:
            self.length = len(self.windows)

        # ── train RNG (DataLoader worker 마다 fork되도록 seed만 보관) ──
        self._seed = seed

        # ── 로깅 ──
        cd = self.class_distribution()
        logging.info(
            f'[{split}] patients={len(self.patients)} '
            f'(skipped short: {len(skipped)}), '
            f'total_epochs={total_epochs:,}, '
            f'__len__={self.length:,}'
        )
        logging.info(
            f'[{split}] REM={cd["rem"]:,} ({cd["rem_ratio"]*100:.1f}%) '
            f'NREM={cd["nrem"]:,}  '
            f'Wake(masked)={cd["wake"]:,}'
        )

    def __len__(self) -> int:
        return self.length

    def _make_final_mask(self, wake_mask: np.ndarray) -> np.ndarray:
        """Wake-mask AND center-mask. (가운데 center_window 안 + REM/NREM)."""
        margin = (self.seq_len - self.center_window) // 2
        center = np.zeros_like(wake_mask)
        center[margin:margin + self.center_window] = 1.0
        return wake_mask * center

    def _sample_window(self, idx: int) -> Tuple[int, int]:
        if not self.is_train:
            return self.windows[idx]
        # train: 환자별 num_epochs 비례 가중치 → random pick
        weights = np.array(
            [p['num_epochs'] for p in self.patients], dtype=np.float64
        )
        weights /= weights.sum()
        pi = int(np.random.choice(len(self.patients), p=weights))
        p = self.patients[pi]
        max_start = p['num_epochs'] - self.seq_len
        start = int(np.random.randint(0, max_start + 1))
        return pi, start

    def __getitem__(self, idx: int):
        pi, start = self._sample_window(idx)
        p = self.patients[pi]
        end = start + self.seq_len

        emb = p['emb'][start:end]              # [T, 2048]
        labels = p['labels'][start:end]         # [T]
        wake_mask = p['mask'][start:end]        # [T]
        final_mask = self._make_final_mask(wake_mask)

        return (
            torch.from_numpy(emb).float(),
            torch.from_numpy(labels).long(),
            torch.from_numpy(final_mask).float(),
        )

    def class_distribution(self) -> Dict[str, int]:
        rem, nrem, wake = 0, 0, 0
        for p in self.patients:
            wake += int(np.sum(p['mask'] == 0))
            rem += int(np.sum((p['labels'] == 0) & (p['mask'] == 1)))
            nrem += int(np.sum((p['labels'] == 1) & (p['mask'] == 1)))
        total = rem + nrem
        return {
            'wake': wake,
            'rem': rem,
            'nrem': nrem,
            'total_sleep': total,
            'rem_ratio': rem / total if total else 0,
        }

    def get_window_metadata(self, idx: int) -> Tuple[str, int]:
        """val/test에서 (patient_id, start_idx) 반환. 결과 분석용."""
        if self.is_train:
            raise RuntimeError('get_window_metadata is val/test only')
        pi, start = self.windows[idx]
        return self.patients[pi]['pid'], start


def worker_init_fn(worker_id: int):
    """DataLoader worker 마다 다른 numpy seed → train sampling 다양성 보장."""
    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)


# ──────────────────────────────────────────────
# Smoke test (가짜 .npy 만들어서 검증)
# ──────────────────────────────────────────────
if __name__ == '__main__':
    import tempfile
    import shutil

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    tmp = tempfile.mkdtemp()
    print(f'tmp dir: {tmp}')

    try:
        # 가짜 환자 5명: train 3, val 1, test 1
        patient_specs = [
            ('patient01', 'train', 800),
            ('patient02', 'train', 950),
            ('patient03', 'train', 1100),
            ('patient04', 'val', 850),
            ('patient05', 'test', 900),
        ]
        rng = np.random.RandomState(0)
        for pid, split, n in patient_specs:
            # 라벨 분포: Wake 20%, REM 5%, NREM 75%
            labels = rng.choice([0, 1, 2], size=n, p=[0.20, 0.05, 0.75]).astype(np.int64)
            emb = rng.randn(n, 2048).astype(np.float32)
            np.save(os.path.join(tmp, f'{pid}_emb.npy'), emb)
            np.save(os.path.join(tmp, f'{pid}_labels.npy'), labels)
            with open(os.path.join(tmp, f'{pid}_meta.json'), 'w') as f:
                json.dump({
                    'patient_id': pid, 'split': split, 'num_epochs': n,
                    'epoch_ids': list(range(1, n+1)),
                    'label_counts': {'wake': int(np.sum(labels==0)),
                                     'rem': int(np.sum(labels==1)),
                                     'nrem': int(np.sum(labels==2))},
                    'embedding_dim': 2048,
                }, f)

        print('\n=== train ===')
        ds_train = SoundSequenceDataset(tmp, 'train', seq_len=40, center_window=20)
        emb, lbl, msk = ds_train[0]
        print(f'  sample: emb={tuple(emb.shape)} lbl={tuple(lbl.shape)} mask={tuple(msk.shape)}')
        print(f'  mask sum={msk.sum().item():.0f} (max=20, center_window)')
        print(f'  __len__={len(ds_train)}')

        print('\n=== val ===')
        ds_val = SoundSequenceDataset(tmp, 'val', seq_len=40, center_window=20, stride=20)
        print(f'  windows={len(ds_val)}')
        for i in [0, 1, len(ds_val) - 1]:
            pid, start = ds_val.get_window_metadata(i)
            print(f'  window[{i}]: {pid} start={start}')

        print('\n=== test ===')
        ds_test = SoundSequenceDataset(tmp, 'test', seq_len=40, center_window=20, stride=20)
        print(f'  windows={len(ds_test)}')

        # DataLoader smoke
        from torch.utils.data import DataLoader
        loader = DataLoader(
            ds_train, batch_size=4, shuffle=False,
            num_workers=0, worker_init_fn=worker_init_fn,
        )
        emb, lbl, msk = next(iter(loader))
        print(f'\n=== DataLoader batch ===')
        print(f'  emb={tuple(emb.shape)} lbl={tuple(lbl.shape)} mask={tuple(msk.shape)}')

        print('\n✓ SoundSequenceDataset tests passed')
    finally:
        shutil.rmtree(tmp)
