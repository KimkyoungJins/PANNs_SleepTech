import torch
import torch.nn as nn
import torch.nn.functional as F


def clip_bce(output_dict, target_dict):
    return F.binary_cross_entropy(
        output_dict['clipwise_output'], target_dict['target'])


def clip_ce(output_dict, target_dict):
    return F.cross_entropy(
        output_dict['clipwise_output'], target_dict['target'])


class FocalLoss(nn.Module):
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
    if loss_type == 'clip_bce':
        return clip_bce
    elif loss_type == 'clip_ce':
        return clip_ce
    elif loss_type == 'focal':
        weight = torch.FloatTensor(class_weight).to(device) if class_weight else None
        return FocalLoss(weight=weight, gamma=2.0)
