import torch
import torch.nn as nn
import torch.nn.functional as F


def clip_bce(output_dict, target_dict):
    """Binary Cross-Entropy 손실 (멀티라벨 분류용, 기존 AudioSet용)."""
    return F.binary_cross_entropy(
        output_dict['clipwise_output'], target_dict['target'])


def clip_ce(output_dict, target_dict):
    """Cross-Entropy 손실 (싱글라벨 분류용). 모든 클래스 동등 취급."""
    return F.cross_entropy(
        output_dict['clipwise_output'], target_dict['target'])


class FocalLoss(nn.Module):
    """Focal Loss — 쉬운 샘플의 손실을 줄이고, 어려운 샘플에 집중.

    원리:
        loss = -alpha * (1 - pt)^gamma * log(pt)
        pt가 높으면(쉬운 샘플) → (1-pt)^gamma이 매우 작아짐 → 손실 거의 0
        pt가 낮으면(어려운 샘플) → (1-pt)^gamma이 큼 → 손실 크게 유지

    Args:
        weight: 클래스별 가중치 텐서 (Weighted CE 역할 겸용)
        gamma: 포커싱 파라미터 (높을수록 어려운 샘플에 집중, 기본 2.0)
    """
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, output_dict, target_dict):
        logits = output_dict['clipwise_output']
        target = target_dict['target']

        ce_loss = F.cross_entropy(logits, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def get_loss_func(loss_type, class_weight=None, device=None):
    """손실 함수 팩토리.

    Args:
        loss_type:
            'clip_bce' — 기존 AudioSet용
            'clip_ce'  — 기본 CrossEntropy (1차 학습에서 사용)
            'focal'    — Focal Loss + Weighted CE (개선 버전)
        class_weight: 클래스별 가중치 리스트 (예: [9.44, 0.38, 4.20])
        device: torch.device
    """
    if loss_type == 'clip_bce':
        return clip_bce
    elif loss_type == 'clip_ce':
        return clip_ce
    elif loss_type == 'focal':
        weight = torch.FloatTensor(class_weight).to(device) if class_weight else None
        return FocalLoss(weight=weight, gamma=2.0)
