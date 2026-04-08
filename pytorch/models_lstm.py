"""
SleepStageLSTM — CNN(ResNet22) + Bi-LSTM 수면 단계 분류 모델

구조:
  [30초 WAV] x seq_len → CNN(Freeze) → [seq_len, 2048] → Bi-LSTM → FC → 예측

CNN은 기존 학습된 ResNet22의 가중치를 그대로 사용하고 (Freeze),
LSTM과 FC만 새로 학습하여 시간 컨텍스트를 반영합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ResNet22


class SleepStageLSTM(nn.Module):
    def __init__(self, sample_rate=16000, window_size=512, hop_size=160,
                 mel_bins=64, fmin=50, fmax=8000, classes_num=3,
                 lstm_hidden=256, lstm_layers=2, lstm_dropout=0.3,
                 cnn_embedding_dim=2048):
        super(SleepStageLSTM, self).__init__()

        self.cnn_embedding_dim = cnn_embedding_dim

        # ===== CNN 특징 추출기 (ResNet22, Freeze) =====
        self.cnn = ResNet22(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num  # 임시, FC는 사용 안 함
        )

        # ===== Bi-LSTM =====
        self.lstm = nn.LSTM(
            input_size=cnn_embedding_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0
        )

        # ===== FC (분류기) =====
        lstm_output_dim = lstm_hidden * 2  # 양방향이라 x2
        self.fc1 = nn.Linear(lstm_output_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc_out = nn.Linear(128, classes_num)

    def load_cnn_weights(self, checkpoint_path, device='cpu'):
        """기존 학습된 CNN 가중치를 로드하고 Freeze"""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.cnn.load_state_dict(checkpoint['model'])

        # CNN 전체 Freeze
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.cnn.eval()

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f'CNN weights loaded and frozen. Trainable: {trainable:,} / {total:,}')

    def extract_embedding(self, waveform):
        """CNN으로 단일 에포크의 embedding 추출 (fc1까지, fc_audioset 전)"""
        with torch.no_grad():
            output_dict = self.cnn(waveform, None)
            embedding = output_dict['embedding']  # [B, 2048]
        return embedding

    def forward(self, waveforms_seq):
        """
        Args:
            waveforms_seq: [B, seq_len, 480000] 연속 에포크 시퀀스

        Returns:
            output_dict: {'clipwise_output': [B, classes_num]}
        """
        batch_size, seq_len, audio_len = waveforms_seq.shape

        # 1. 각 에포크를 CNN에 통과시켜 embedding 추출
        embeddings = []
        self.cnn.eval()  # CNN은 항상 eval 모드
        for t in range(seq_len):
            waveform_t = waveforms_seq[:, t, :]  # [B, 480000]
            emb = self.extract_embedding(waveform_t)  # [B, 2048]
            embeddings.append(emb)

        # [B, seq_len, 2048]
        embeddings = torch.stack(embeddings, dim=1)

        # 2. LSTM에 시퀀스 입력
        lstm_out, _ = self.lstm(embeddings)  # [B, seq_len, lstm_hidden*2]

        # 중간 에포크의 출력 사용 (시퀀스의 가운데)
        mid_idx = seq_len // 2
        features = lstm_out[:, mid_idx, :]  # [B, lstm_hidden*2]

        # 3. FC로 분류
        x = F.relu(self.fc1(features))
        x = self.dropout(x)
        clipwise_output = self.fc_out(x)  # [B, classes_num]

        output_dict = {'clipwise_output': clipwise_output}
        return output_dict
