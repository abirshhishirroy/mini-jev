"""
losses.py — Loss functions for Mini-Jev training.

Two losses:

  cross_entropy_loss:
    The baseline. Masked CE over candidate logits.

  rlcd_loss:
    An approximation of TypeSafe's RLCD (Reweighted Listwise Cross-Entropy
    Decision) loss. Adds a margin term on top of CE, where wrong candidates
    are reweighted by their current probability — harder negatives get more
    gradient. The exact RLCD formula is proprietary; this captures the two
    published design ideas (margin + hard-negative reweighting).

Both losses handle variable candidate counts via the `cand_valid` mask.
Padded candidates are set to -inf before softmax so they contribute zero
probability and zero gradient.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_loss(
    logits: torch.Tensor,      # [B, C] fp32
    cand_valid: torch.Tensor,  # [B, C] bool
    target: torch.Tensor,      # [B]    long
) -> torch.Tensor:
    """Plain masked cross-entropy over the candidate dimension."""
    masked = logits.masked_fill(~cand_valid, float("-inf"))
    return F.cross_entropy(masked, target)


def rlcd_loss(
    logits: torch.Tensor,       # [B, C] fp32
    cand_valid: torch.Tensor,   # [B, C] bool
    target: torch.Tensor,       # [B]    long
    margin: float = 0.5,
    lambda_margin: float = 0.3,
    temperature: float = 1.0,
    hard_neg_power: float = 1.0,
) -> torch.Tensor:
    """
    Reweighted Listwise Cross-Entropy Decision loss (approximation).

    L = CE(logits / T, target)
      + lambda * mean_over_examples( sum_i w_i * relu(margin - (c - w_i_logit)) )

    where:
        c           = correct logit for the example
        w_i_logit   = wrong candidate i's logit
        w_i         = softmax_prob_i ^ hard_neg_power, normalized per example
                      (higher for candidates the model currently confuses
                       with the correct one)

    Args:
        logits:         [B, C]
        cand_valid:     [B, C]
        target:         [B]
        margin:         desired minimum gap between correct and wrong logits
        lambda_margin:  weight of the margin term
        temperature:    softmax temperature for the CE term
        hard_neg_power: 0 = uniform weights, 1 = prob-weighted, 2 = more aggressive
    """
    B, C = logits.shape
    device = logits.device

    # ---- 1. listwise CE with masking --------------------------------
    masked = logits.masked_fill(~cand_valid, float("-inf"))
    ce = F.cross_entropy(masked / temperature, target)

    # ---- 2. margin term, reweighted by hard-negative probability ----
    # correct logit per example: [B]
    correct_logit = logits.gather(1, target.unsqueeze(1)).squeeze(1)

    # per-example softmax probabilities (padded entries are 0)
    probs = torch.softmax(masked, dim=-1)                          # [B, C]

    # correct - wrong for every candidate: [B, C]
    diff = correct_logit.unsqueeze(1) - logits

    # hinge: how much each wrong candidate violates the margin
    hinge = F.relu(margin - diff)                                  # [B, C]

    # boolean mask of wrong candidates (excludes padding and the correct one)
    idx = torch.arange(C, device=device).unsqueeze(0)              # [1, C]
    is_wrong = cand_valid & (idx != target.unsqueeze(1))           # [B, C]

    # reweight by current probability: hard negatives get more weight
    weights = probs.pow(hard_neg_power) * is_wrong.float()         # [B, C]

    # normalize per example so C doesn't scale the loss
    wsum = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
    weights = weights / wsum

    margin_term = (weights * hinge).sum(dim=1).mean()

    return ce + lambda_margin * margin_term


# ---------------------------------------------------------------------
# Dispatcher for train.py
# ---------------------------------------------------------------------
LOSS_REGISTRY = {
    "ce":   cross_entropy_loss,
    "rlcd": rlcd_loss,
}


def get_loss(name: str):
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss {name!r}. Options: {list(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name]