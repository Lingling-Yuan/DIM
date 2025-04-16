import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyAwareAdaptiveFocalLoss(nn.Module):
    def __init__(self, num_classes, base_alpha=1.0, base_beta=2.0, margin_const=0.5, reduction='mean'):
        """
        Args:
            num_classes (int): Number of classes.
            base_alpha (float): Hyperparameter α₀ for adaptive coefficient computation.
            base_beta (float): Hyperparameter β₀ for adaptive coefficient computation.
            margin_const (float): Hyperparameter m for frequency-aware offset fₖ.
            reduction (str): 'mean', 'sum', or 'none' to specify reduction method.
        """
        super(FrequencyAwareAdaptiveFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.base_alpha = base_alpha  # α₀ in Equation (23)
        self.base_beta = base_beta    # β₀ in Equation (23)
        self.margin_const = margin_const  # m in Equation (22)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Logits of shape (N, C) where N is the number of samples and C is the number of classes.
            targets (Tensor): Ground-truth labels of shape (N,).
        Returns:
            loss (Tensor): The computed Frequency-Aware Adaptive Focal Loss.
        """
        # Compute the original log-probabilities from the logits (zₖᵢ) 
        # and obtain pₖᵢ (Equation (21))
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        # pₖᵢ: the probability of the true class for each sample (from original logits)
        p_ki = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        # Calculate the frequency-aware offset fₖ for each class (Equation (22))
        # fₖ = m / (count(k))^(0.25)
        f_k = torch.zeros_like(inputs)
        class_counts = torch.bincount(targets, minlength=self.num_classes).float()
        for k in range(self.num_classes):
            if class_counts[k] > 0:
                f_k[:, k] = self.margin_const / (class_counts[k] ** 0.25)
        
        # Apply the frequency-aware offset to adjust the logits:
        # ẑₖᵢ = zₖᵢ + fₖ
        adjusted_logits = inputs + f_k
        
        # Recompute the log-probabilities using the adjusted logits and obtain p̂ₖᵢ
        adjusted_log_probs = F.log_softmax(adjusted_logits, dim=-1)
        p_hat = torch.exp(adjusted_log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1))
        
        # Compute the adaptive coefficients from the original probability pₖᵢ (Equation (23)):
        # αᵢ = α₀ · (1 − pₖᵢ) and βᵢ = β₀ · pₖᵢ
        alpha_i = self.base_alpha * (1 - p_ki)
        beta_i = self.base_beta * p_ki
        
        # Compute the final Frequency-Aware Adaptive Focal Loss for each sample (Equation (24)):
        # L_F = -αᵢ * (1 - p̂ₖᵢ)^(βᵢ) * log(p̂ₖᵢ)
        loss = -alpha_i * ((1 - p_hat) ** beta_i) * adjusted_log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        # Apply reduction as specified ('mean', 'sum', or no reduction)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
