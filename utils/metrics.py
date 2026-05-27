import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def compute_auc(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> float:
    pred = y_pred[mask].detach().cpu().numpy()
    true = y_true[mask].detach().cpu().numpy()
    if len(np.unique(true)) < 2:
        return float("nan")
    return roc_auc_score(true, pred)


def compute_acc(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> float:
    pred = (y_pred[mask] >= 0.5).float()
    true = y_true[mask].float()
    return (pred == true).float().mean().item()
