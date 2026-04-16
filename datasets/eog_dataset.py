"""EOG Dataset - CSV 기반 EOG .npy 로딩 Dataset with optional augmentation."""

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset


class EOGDataset(Dataset):
    """EOG 신호 기반 REM/NREM 분류용 Dataset.

    CSV 형식: filename,label
        filename: patient{pnum}_epoch{idx}.wav
        label: 0(rem) or 1(nrem)

    wav 파일명에서 _eog.npy 경로를 유추:
        patient01_epoch0001.wav → patient01/patient01_epoch0001_eog.npy
    """

    def __init__(self, csv_path, data_dir, augment=False):
        """
        Args:
            csv_path: CSV 파일 경로 (full_ver_rem_nrem/train.csv 등)
            data_dir: EOG .npy 파일이 있는 base 디렉토리 (full_ver/)
            augment: True이면 학습용 augmentation 적용
        """
        self.data_dir = data_dir
        self.augment = augment
        self.filenames = []
        self.labels = []

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                self.filenames.append(row[0])
                self.labels.append(int(row[1]))

        self.data_num = len(self.filenames)

    def __len__(self):
        return self.data_num

    def _augment(self, eog):
        """Train-time augmentation on numpy array (2, 3000).

        - Time shift: ±10 samples (100ms at 100Hz)
        - Gaussian noise: σ=0.01
        - Channel dropout: 10% prob, zero out one channel
        - Amplitude scaling: 0.9~1.1 random
        """
        # Time shift
        shift = np.random.randint(-10, 11)
        if shift != 0:
            eog = np.roll(eog, shift, axis=1)
            if shift > 0:
                eog[:, :shift] = 0.0
            else:
                eog[:, shift:] = 0.0

        # Gaussian noise
        eog = eog + np.random.normal(0, 0.01, eog.shape).astype(np.float32)

        # Channel dropout (10% prob)
        if np.random.random() < 0.1:
            ch = np.random.randint(0, 2)
            eog[ch, :] = 0.0

        # Amplitude scaling
        scale = np.random.uniform(0.9, 1.1)
        eog = eog * scale

        return eog

    def __getitem__(self, index):
        filename = self.filenames[index]
        label = self.labels[index]

        basename = os.path.splitext(filename)[0]
        eog_filename = basename + '_eog.npy'
        patient_folder = filename.split('_epoch')[0]
        eog_path = os.path.join(self.data_dir, patient_folder, eog_filename)

        eog = np.load(eog_path).astype(np.float32)  # (2, 3000)

        if self.augment:
            eog = self._augment(eog)

        eog_tensor = torch.from_numpy(eog)
        return eog_tensor, label

    def get_labels(self):
        """WeightedRandomSampler용 라벨 리스트 반환."""
        return self.labels
