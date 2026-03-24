import numpy as np
import csv
import os
import logging
import librosa

import config


class SleepDataset(object):
    """수면 오디오 데이터셋 클래스.

    CSV 파일에서 파일명과 라벨을 읽고,
    해당 WAV 파일을 로드하여 waveform과 라벨을 반환.

    폴더 구조 (환자별 서브폴더):
        data/
        ├── patient01/
        │   ├── patient01_epoch0001.wav
        │   └── ...
        ├── patient02/
        │   └── ...
        ├── train.csv
        └── val.csv
    """
    def __init__(self, csv_path, audio_dir, sample_rate=16000):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.clip_samples = sample_rate * 30  # 30초 = 480,000 샘플

        # CSV에서 파일 목록과 라벨 읽기
        self.filenames = []
        self.labels = []

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.filenames.append(row[0])
                self.labels.append(config.lb_to_ix[row[1]])

        self.data_num = len(self.filenames)
        logging.info('Dataset size: {} samples from {}'.format(self.data_num, csv_path))

        # 클래스별 샘플 수 출력
        label_counts = np.bincount(self.labels, minlength=config.classes_num)
        for i, name in enumerate(config.labels):
            logging.info('  {}: {} samples'.format(name, label_counts[i]))

    def get_class_counts(self):
        """클래스별 샘플 수 반환. Oversampling 가중치 계산에 사용."""
        return np.bincount(self.labels, minlength=config.classes_num)

    def get_sample_weights(self):
        """각 샘플의 Oversampling 가중치 반환.
        적은 클래스의 샘플에 높은 가중치 → WeightedRandomSampler에서 더 자주 뽑힘.

        예: rem=292, nrem=7322, wake=657
            rem 샘플 가중치  = 1/292  = 0.00342 (높음, 자주 뽑힘)
            nrem 샘플 가중치 = 1/7322 = 0.00014 (낮음, 덜 뽑힘)
            wake 샘플 가중치 = 1/657  = 0.00152
        """
        class_counts = self.get_class_counts()
        weights_per_class = 1.0 / class_counts
        sample_weights = np.array([weights_per_class[label] for label in self.labels])
        return sample_weights

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        """하나의 오디오 클립을 로드하여 반환."""
        filename = self.filenames[index]
        label = self.labels[index]

        # WAV 파일 로드 — 환자 서브폴더에서 찾기
        patient_folder = filename.split("_")[0]
        audio_path = os.path.join(self.audio_dir, patient_folder, filename)
        if not os.path.exists(audio_path):
            audio_path = os.path.join(self.audio_dir, filename)
        if not os.path.exists(audio_path):
            audio_path = os.path.join(self.audio_dir, "audio", filename)

        waveform, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # 길이 맞추기
        if len(waveform) < self.clip_samples:
            waveform = np.concatenate([
                waveform,
                np.zeros(self.clip_samples - len(waveform), dtype=np.float32)
            ])
        else:
            waveform = waveform[:self.clip_samples]

        data_dict = {
            'waveform': waveform.astype(np.float32),
            'target': label
        }

        return data_dict


def collate_fn(list_data_dict):
    """개별 데이터를 배치로 합치는 함수."""
    np_data_dict = {}
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    return np_data_dict
