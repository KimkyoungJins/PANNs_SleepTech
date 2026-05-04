"""
Masked Focal Loss — REM/NREM 시퀀스 분류용.

EOG_CNN의 FocalLoss를 시퀀스(3D) 입력 + mask 지원으로 확장.
- mask=1 인 epoch에서만 loss 계산
- Wake epoch을 mask=0으로 처리해서 REM/NREM에만 학습 집중
- 가운데 20 epoch만 학습하고 싶으면 head/tail 10개도 mask=0
"""

import torch
import torch.nn as nn


class MaskedFocalLoss(nn.Module):
    """
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: class 0(REM)에 가중치 alpha, class 1(NREM)에 (1-alpha).
               REM이 적은 클래스이므로 0.75가 REM 강조.
        gamma: focal 강도 (값이 클수록 어려운 샘플에 집중).
        num_classes: 2.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 3.0, num_classes: int = 2):
        super().__init__()
        self.register_buffer(
            'alpha',
            torch.tensor([alpha, 1.0 - alpha], dtype=torch.float32),
        )
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits:  [B, T, C]
            targets: [B, T]      (값: 0=REM, 1=NREM, mask=0인 곳은 무시되므로 dummy 가능)
            mask:    [B, T]      (1=loss 계산, 0=무시 — Wake 또는 head/tail)

        Returns:
            scalar loss (mask=1인 위치들의 평균)
        """
        B, T, C = logits.shape
        assert C == self.num_classes, f'expected C={self.num_classes}, got {C}'
        assert targets.shape == (B, T)
        assert mask.shape == (B, T)

        logits_flat = logits.reshape(B * T, C)
        targets_flat = targets.reshape(B * T)
        mask_flat = mask.reshape(B * T).float()

        # 안전장치: mask=0 위치의 target이 범위 밖일 수 있으니 0으로 clamp
        targets_safe = targets_flat.clamp(min=0, max=C - 1)

        probs = torch.softmax(logits_flat, dim=1)
        pt = probs.gather(1, targets_safe.unsqueeze(1)).squeeze(1).clamp(min=1e-8)

        alpha_t = self.alpha.to(logits.device)[targets_safe]
        focal_weight = alpha_t * (1.0 - pt).pow(self.gamma)
        per_sample_loss = -focal_weight * torch.log(pt)   # [B*T]

        # mask 적용 평균 (mask=1인 위치들만)
        masked = per_sample_loss * mask_flat
        denom = mask_flat.sum().clamp(min=1.0)
        return masked.sum() / denom


if __name__ == '__main__':
    # smoke test
    torch.manual_seed(0)
    loss_fn = MaskedFocalLoss(alpha=0.75, gamma=3.0)

    B, T, C = 4, 40, 2
    logits = torch.randn(B, T, C, requires_grad=True)
    targets = torch.randint(0, 2, (B, T))

    # case 1: 가운데 20개만 학습 (앞뒤 10개씩 mask=0)
    mask = torch.zeros(B, T)
    mask[:, 10:30] = 1.0
    loss = loss_fn(logits, targets, mask)
    print(f'case 1 (center 20 only): loss = {loss.item():.4f}')
    loss.backward()
    print('  ✓ backward OK')

    # case 2: 일부 Wake 섞임 (random 30% mask=0)
    mask = (torch.rand(B, T) > 0.3).float()
    loss = loss_fn(logits, targets, mask)
    print(f'case 2 (random Wake mask): loss = {loss.item():.4f}')

    # case 3: edge — 모든 위치 mask=0 (denom 보호)
    mask_zero = torch.zeros(B, T)
    loss = loss_fn(logits, targets, mask_zero)
    print(f'case 3 (all masked): loss = {loss.item():.4f} (should be 0)')

    print('✓ MaskedFocalLoss tests passed')
