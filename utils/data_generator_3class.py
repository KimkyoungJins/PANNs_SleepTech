import numpy as np
import csv
import os
import logging
import librosa

import config

class SleepDataset(object):
    def __init__(self, csv_path, audio_dir, sample_rate=16000):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.clip_samples = sample_rate * 30

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

        label_counts = np.bincount(self.labels, minlength=config.classes_num)
        for i, name in enumerate(config.labels):
            logging.info('  {}: {} samples'.format(name, label_counts[i]))

    def get_class_counts(self):
        return np.bincount(self.labels, minlength=config.classes_num)

    def get_sample_weights(self):
        class_counts = self.get_class_counts()
        weights_per_class = 1.0 / class_counts
        return np.array([weights_per_class[label] for label in self.labels])

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        filename = self.filenames[index]
        label = self.labels[index]

        patient_folder = filename.split("_")[0]
        audio_path = os.path.join(self.audio_dir, patient_folder, filename)
        if not os.path.exists(audio_path):
            audio_path = os.path.join(self.audio_dir, filename)
        if not os.path.exists(audio_path):
            audio_path = os.path.join(self.audio_dir, "audio", filename)

        waveform, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        if len(waveform) < self.clip_samples:
            waveform = np.concatenate([
                waveform,
                np.zeros(self.clip_samples - len(waveform), dtype=np.float32)
            ])
        else:
            waveform = waveform[:self.clip_samples]

        return {
            'waveform': waveform.astype(np.float32),
            'target': label
        }

def collate_fn(list_data_dict):
    np_data_dict = {}
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([d[key] for d in list_data_dict])
    return np_data_dict
