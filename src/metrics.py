import torch

EPS = 1e-7


def _binarize(logits):
    return (torch.sigmoid(logits) > 0.5).float()


def iou_score(logits, targets):
    preds = _binarize(logits)
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - inter
    return ((inter + EPS) / (union + EPS)).mean().item()


def dice_score(logits, targets):
    preds = _binarize(logits)
    inter = (preds * targets).sum(dim=(1, 2, 3))
    total = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2 * inter + EPS) / (total + EPS)).mean().item()
