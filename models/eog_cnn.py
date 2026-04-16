"""EOG_CNN - 2채널 EOG 신호 기반 REM/NREM 분류 1D CNN."""

import torch
import torch.nn as nn


class EOG_CNN(nn.Module):
    """1D CNN for EOG-based REM vs NREM classification.

    Input:  (B, 2, 3000)  - 2채널 EOG, 100Hz × 30초
    Output: (B, 2)        - REM(0) / NREM(1) logits
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: (B, 2, 3000) → (B, 32, 750)
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            # Block 2: (B, 32, 750) → (B, 64, 187)
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            # Block 3: (B, 64, 187) → (B, 128, 93)
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            # Block 4: (B, 128, 93) → (B, 256, 1)
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 2, 3000) EOG tensor
        Returns:
            logits: (B, num_classes)
        """
        x = self.features(x)       # (B, 256, 1)
        x = x.squeeze(-1)          # (B, 256)
        x = self.classifier(x)     # (B, num_classes)
        return x


if __name__ == '__main__':
    model = EOG_CNN(num_classes=2)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    dummy = torch.randn(4, 2, 3000)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
