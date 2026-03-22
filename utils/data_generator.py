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

    CSV 파일 형식:
        filename,label
        patient01_epoch0001.wav,wake
        patient01_epoch0002.wav,nrem
        patient01_epoch0003.wav,rem

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
        """
        Args:
            csv_path: CSV 파일 경로 (예: 'data/train.csv')
            audio_dir: WAV 루트 폴더 (환자별 서브폴더 포함, 예: 'data/')
            sample_rate: 샘플레이트 (기본 16000)
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.clip_samples = sample_rate * 30  # 30초 = 480,000 샘플

        # ===== CSV에서 파일 목록과 라벨 읽기 =====
        self.filenames = []   # WAV 파일명 리스트
        self.labels = []      # 라벨 인덱스 리스트 (0=rem, 1=nrem, 2=wake)

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # 헤더(첫 줄) 건너뛰기
            for row in reader:
                self.filenames.append(row[0])                   # 파일명
                self.labels.append(config.lb_to_ix[row[1]])     # 라벨 문자열 → 숫자

        self.data_num = len(self.filenames)
        logging.info('Dataset size: {} samples from {}'.format(self.data_num, csv_path))

        # 클래스별 샘플 수 출력
        label_counts = np.bincount(self.labels, minlength=config.classes_num)
        for i, name in enumerate(config.labels):
            logging.info('  {}: {} samples'.format(name, label_counts[i]))

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        """하나의 오디오 클립을 로드하여 반환.

        Args:
            index: 데이터 인덱스 (정수)

        Returns:
            data_dict: {
                'waveform': (clip_samples,) - float32 오디오 파형 (-1.0 ~ 1.0)
                'target': int              - 정답 라벨 인덱스 (0=rem, 1=nrem, 2=wake)
            }
        """
        filename = self.filenames[index]
        label = self.labels[index]

        # WAV 파일 로드 — 환자 서브폴더에서 찾기
        # patient01_epoch0001.wav → patient01/ 폴더에서 탐색
        patient_folder = filename.split("_")[0]
        audio_path = os.path.join(self.audio_dir, patient_folder, filename)
        if not os.path.exists(audio_path):
            # fallback: 직하 또는 audio/ 에서 찾기
            audio_path = os.path.join(self.audio_dir, filename)
        if not os.path.exists(audio_path):
            audio_path = os.path.join(self.audio_dir, "audio", filename)

        waveform, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # 길이 맞추기: 정확히 clip_samples(480,000)로 맞춤
        # - 짧으면: 뒤에 0으로 패딩 (zero-padding)
        # - 길면: 앞에서부터 clip_samples만큼만 자름
        if len(waveform) < self.clip_samples:
            # 짧은 오디오는 뒤를 0으로 채움
            waveform = np.concatenate([
                waveform,
                np.zeros(self.clip_samples - len(waveform), dtype=np.float32)
            ])
        else:
            # 긴 오디오는 앞에서부터 자름
            waveform = waveform[:self.clip_samples]

        data_dict = {
            'waveform': waveform.astype(np.float32),
            'target': label
        }

        return data_dict


def collate_fn(list_data_dict):
    """개별 데이터를 배치로 합치는 함수.

    DataLoader가 Dataset에서 개별 샘플을 여러 개 가져온 후,
    이 함수를 통해 하나의 배치 텐서로 합침.

    예시:
    입력: [{'waveform': (480000,), 'target': 1},
           {'waveform': (480000,), 'target': 0},
           ...] (batch_size개)

    출력: {'waveform': (batch_size, 480000),
           'target': (batch_size,)}
    """
    np_data_dict = {}

    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])

    return np_data_dict
