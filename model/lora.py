"""
lora.py — Hand-rolled LoRA (no peft).

We implement:
  LoRALinear: wraps a frozen nn.Linear with a low-rank trainable delta.
              Original output is preserved at init (B is zero-initialized).
  apply_lora: replaces q_proj/v_proj in the last N transformer blocks.

Math:
  y = W x + (alpha / r) * B @ A @ x
  where W is frozen, A: [r, in], B: [out, r], and B starts at zero.
"""

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = base.in_features
        out_features = base.out_features

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        # kaiming-style init for A; B stays zero so delta = 0 at start.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # cast lora weights to input dtype (backbone may be fp16)
        delta = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + delta


def _find_layers(root: nn.Module) -> nn.ModuleList:
    """Walk the module tree to find the decoder block ModuleList.

    Qwen2.5 exposes it as `model.layers`. Other architectures (Llama, Mistral)
    use the same name. If a wrapper adds `.model`, we handle that too.
    """
    # direct: root.layers
    if hasattr(root, "layers"):
        return root.layers
    # one level down: root.model.layers
    if hasattr(root, "model") and hasattr(root.model, "layers"):
        return root.model.layers
    raise AttributeError(f"Could not find `.layers` on {type(root).__name__}")


def apply_lora(model, last_n_blocks: int = 4, r: int = 8, alpha: int = 16) -> int:
    """
    Replace q_proj and v_proj in the last `last_n_blocks` transformer blocks
    with LoRALinear. Returns the number of trainable LoRA params added.
    """
    # Qwen2.5 architecture: model.model.layers is a ModuleList of decoder blocks
    layers = _find_layers(model.backbone.backbone)
    n = len(layers)
    start = max(0, n - last_n_blocks)

    added = 0
    for i in range(start, n):
        block = layers[i]
        for proj_name in ("q_proj", "v_proj"):
            base = getattr(block.self_attn, proj_name)
            if isinstance(base, LoRALinear):
                continue
            lora = LoRALinear(base, r=r, alpha=alpha).to(
                device=base.weight.device, dtype=base.weight.dtype
            )
            setattr(block.self_attn, proj_name, lora)
            added += lora.lora_A.numel() + lora.lora_B.numel()

    return added
