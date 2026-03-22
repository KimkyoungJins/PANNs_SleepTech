import torch
import torch.nn.functional as F


def clip_bce(output_dict, target_dict):
    """Binary Cross-Entropy 손실 (멀티라벨 분류용, 기존 AudioSet용).
    각 클래스를 독립적으로 판단 (여러 클래스가 동시에 1일 수 있음).
    """
    return F.binary_cross_entropy(
        output_dict['clipwise_output'], target_dict['target'])


def clip_ce(output_dict, target_dict):
    """Cross-Entropy 손실 (싱글라벨 분류용, 수면 단계 분류용).

    내부에서 softmax를 자동 적용하므로, 모델 출력은 raw logits여야 함.
    3개 클래스(REM, NREM, Wake) 중 하나만 정답인 경우에 사용.

    Args:
        output_dict['clipwise_output']: (batch_size, 3) - raw logits
        target_dict['target']: (batch_size,) - 정답 클래스 인덱스 (0, 1, 또는 2)
    """
    return F.cross_entropy(
        output_dict['clipwise_output'], target_dict['target'])


def get_loss_func(loss_type):
    if loss_type == 'clip_bce':
        return clip_bce
    elif loss_type == 'clip_ce':
        return clip_ce
