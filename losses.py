import torch
import torch.nn.functional as F

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)

    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    dice = (2 * intersection + eps) / (probs.sum(dim=1) + targets.sum(dim=1) + eps)

    return 1 - dice.mean()

def bce_dice_loss(logits, targets):
    pos_weight = torch.tensor([8.0]).to(logits.device)  # try 5–10
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

    dice = dice_loss(logits, targets)
    return bce + dice
