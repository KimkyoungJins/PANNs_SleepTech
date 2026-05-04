"""
BiLSTM Classifier — REM/NREM 분류기.

입력: PANNs ResNet22로 추출한 epoch별 2048-d embedding 시퀀스
출력: 각 epoch에 대한 REM/NREM logits

구조:
    [B, T, 2048]
      → Linear(2048 → 256) + ReLU + Dropout       (차원 축소)
      → BiLSTM(256, hidden=128, 2-layer, dropout)  (시퀀스 처리)
      → Linear(256 → 2)                            (256 = 128*2 bidirectional)
    [B, T, 2]

사용:
    model = BiLSTMClassifier()
    logits = model(emb)   # emb: [B, 40, 2048] → logits: [B, 40, 2]
"""

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """소리 embedding 시퀀스 → REM/NREM 분류."""

    def __init__(
        self,
        embed_dim: int = 2048,
        proj_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()

        # ── 1. 차원 축소 (2048 → 256) ──
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── 2. BiLSTM ──
        self.bilstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── 3. 분류기 (256 = 128*2 양방향) ──
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, embed_dim]   (T=40 epoch sequence)

        Returns:
            logits: [B, T, num_classes]
        """
        x = self.projection(x)                  # [B, T, proj_dim]
        x, _ = self.bilstm(x)                   # [B, T, 2*hidden_dim]
        logits = self.classifier(x)             # [B, T, num_classes]
        return logits

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    # smoke test
    model = BiLSTMClassifier()
    print(f'Params: {model.num_parameters() / 1e6:.2f}M')

    B, T, D = 4, 40, 2048
    x = torch.randn(B, T, D)
    logits = model(x)
    print(f'Input  shape: {tuple(x.shape)}')
    print(f'Output shape: {tuple(logits.shape)}')
    assert logits.shape == (B, T, 2), f'Expected ({B},{T},2), got {tuple(logits.shape)}'
    print('✓ Forward pass OK')
