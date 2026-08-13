import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss for multi-class categorical classification.
    L_AF = - sum_{c=1}^4 alpha_c * (1 - p_c)^gamma_c * log(p_c)
    where gamma_pos = 1.0 and gamma_neg = 2.0.
    """

    def __init__(self, gamma_pos: float = 1.0, gamma_neg: float = 2.0, alpha: torch.Tensor = None):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, 4], targets: [B]
        probs = F.softmax(logits, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()

        # Probabilities for target class vs background classes
        p_t = (probs * targets_one_hot).sum(dim=-1)
        p_t = torch.clamp(p_t, min=1e-7, max=1.0 - 1e-7)

        # Modulating factor
        gamma_weights = torch.where(targets_one_hot == 1.0, self.gamma_pos, self.gamma_neg)
        focal_weight = torch.pow(1.0 - probs, gamma_weights)

        log_probs = torch.log(torch.clamp(probs, min=1e-7))
        loss_element = -targets_one_hot * focal_weight * log_probs

        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            loss_element = loss_element * self.alpha.unsqueeze(0)

        return loss_element.sum(dim=-1).mean()


class MultiTaskLoss(nn.Module):
    """
    Composite Multi-Task Loss combining Asymmetric Focal Loss (Classification)
    and Smooth L1 / Huber Loss (Regression).
    """

    def __init__(self, alpha_cls: torch.Tensor = None):
        super().__init__()
        self.focal_loss = AsymmetricFocalLoss(alpha=alpha_cls)
        self.smooth_l1 = nn.SmoothL1Loss(beta=1.0)

    def forward(self, logits: torch.Tensor, reg_scores: torch.Tensor, target_cls: torch.Tensor, target_reg: torch.Tensor):
        loss_cls = self.focal_loss(logits, target_cls)
        loss_reg = 2.5 * self.smooth_l1(reg_scores, target_reg)

        return loss_cls, loss_reg


class GradNormLossBalancer(nn.Module):
    """
    Dynamic Homoscedastic Uncertainty Loss Balancer (Kendall et al.).
    Dynamically balances classification and regression task losses:
    L_total = exp(-s_cls) * L_cls + s_cls + exp(-s_reg) * L_reg + s_reg
    where s_cls and s_reg are learnable log variance parameters.
    """

    def __init__(self, alpha: float = 0.12):
        super().__init__()
        self.alpha = alpha
        self.log_var_cls = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.log_var_reg = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def get_weighted_loss(self, loss_cls: torch.Tensor, loss_reg: torch.Tensor):
        precision_cls = torch.exp(-self.log_var_cls)
        precision_reg = torch.exp(-self.log_var_reg)
        total_loss = precision_cls * loss_cls + self.log_var_cls + precision_reg * loss_reg + self.log_var_reg
        return total_loss
