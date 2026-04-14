"""
LSTM용 데이터 제너레이터
- 같은 환자의 연속 에포크를 seq_len개씩 묶어서 반환
- 라벨은 시퀀스 중간 에포크의 라벨을 사용
"""

import numpy as np
import csv
import os
import logging
import librosa
from collections import defaultdict

import torch.utils.data
import config

LABEL_MAP_2CLASS = {0: 0, 1: 1, 2: 1}


class SleepSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, audio_dir, sample_rate=16000, seq_len=10, stride=1):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.clip_samples = sample_rate * 30
        self.seq_len = seq_len
        self.stride = stride

        # CSV 읽기
        filenames = []
        labels = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                label = int(row[1])
                if config.classes_num == 2:
                    label = LABEL_MAP_2CLASS[label]
                filenames.append(row[0])
                labels.append(label)

        # 환자별로 에포크를 그룹화하고 정렬
        patient_epochs = defaultdict(list)
        for i, fn in enumerate(filenames):
            patient = fn.split("_")[0]  # "patient01"
            epoch_num = int(fn.split("epoch")[1].split(".")[0])  # 에포크 번호
            patient_epochs[patient].append((epoch_num, i, fn, labels[i]))

        # 에포크 번호 순서대로 정렬
        for patient in patient_epochs:
            patient_epochs[patient].sort(key=lambda x: x[0])

        # 시퀀스 생성 (슬라이딩 윈도우)
        self.sequences = []  # [(filenames_list, label), ...]

        for patient in sorted(patient_epochs.keys()):
            epochs = patient_epochs[patient]
            n = len(epochs)

            if n < seq_len:
                continue  # 에포크가 seq_len보다 적은 환자는 건너뜀

            for start in range(0, n - seq_len + 1, stride):
                seq_epochs = epochs[start:start + seq_len]

                # 연속성 확인 (에포크 번호가 순차적인지)
                epoch_nums = [e[0] for e in seq_epochs]
                is_continuous = all(
                    epoch_nums[i+1] - epoch_nums[i] == 1
                    for i in range(len(epoch_nums) - 1)
                )

                if not is_continuous:
                    continue

                seq_filenames = [e[2] for e in seq_epochs]
                mid_label = seq_epochs[seq_len // 2][3]  # 중간 에포크의 라벨
                self.sequences.append((seq_filenames, mid_label))

        self.data_num = len(self.sequences)
        logging.info('Sequence Dataset: {} sequences (seq_len={}, stride={}) from {}'.format(
            self.data_num, seq_len, stride, csv_path))

        # 라벨 분포 출력
        all_labels = [s[1] for s in self.sequences]
        label_counts = np.bincount(all_labels, minlength=config.classes_num)
        for i, name in enumerate(config.labels):
            logging.info('  {}: {} sequences'.format(name, label_counts[i]))

    def get_class_counts(self):
        all_labels = [s[1] for s in self.sequences]
        return np.bincount(all_labels, minlength=config.classes_num)

    def get_sample_weights(self):
        class_counts = self.get_class_counts()
        all_labels = [s[1] for s in self.sequences]
        weights_per_class = 1.0 / (class_counts + 1e-8)
        return np.array([weights_per_class[label] for label in all_labels])

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        filenames, label = self.sequences[index]

        waveforms = []
        for fn in filenames:
            patient_folder = fn.split("_")[0]
            audio_path = os.path.join(self.audio_dir, patient_folder, fn)
            if not os.path.exists(audio_path):
                audio_path = os.path.join(self.audio_dir, fn)

            waveform, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)

            if len(waveform) < self.clip_samples:
                waveform = np.concatenate([
                    waveform,
                    np.zeros(self.clip_samples - len(waveform), dtype=np.float32)
                ])
            else:
                waveform = waveform[:self.clip_samples]

            waveforms.append(waveform.astype(np.float32))

        # [seq_len, 480000]
        waveforms = np.stack(waveforms, axis=0)

        return {
            'waveform': waveforms,
            'target': label
        }


def collate_fn_lstm(list_data_dict):
    np_data_dict = {}
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([d[key] for d in list_data_dict])
    return np_data_dict
