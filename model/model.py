"""
model.py — Mini-Jev backbone wrapper.

Step 1 scope: load Qwen2.5-0.5B-Instruct as a plain encoder (no LM head),
freeze it, and expose a clean forward() that returns last_hidden_state.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from data.data import pool_candidates
from model.lora import apply_lora


class MiniJevBackbone(nn.Module):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.backbone = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float16,
        )

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.hidden_size = self.backbone.config.hidden_size
        self.num_layers = self.backbone.config.num_hidden_layers

    def forward(
        self, input_ids: torch.Tensor, attention_mask=torch.Tensor
    ) -> torch.Tensor:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
        )

        return out.last_hidden_state


class DecisionHead(nn.Module):
    """
    Maps a per-candidate pooled hidden state to a single scalar logit.

    Input:  [B, C, hidden_size]   (B=batch, C=num candidates)
    Output: [B, C]                (one logit per candidate)

    This is intentionally tiny — a few hundred thousand parameters.
    All the heavy lifting happens in the frozen backbone.
    """

    def __init__(self, hidden_size: int = 896, head_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(pooled).squeeze(-1)


class MiniJevModel(nn.Module):
    """
    Backbone + decision head. For Step 2 we keep the backbone frozen and
    only the head is trainable.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        super().__init__()
        self.backbone = MiniJevBackbone(model_name)
        self.head = DecisionHead(
            hidden_size=self.backbone.hidden_size,
            head_dim=256,
        )

    @property
    def tokenizer(self):
        return self.backbone.tokenizer

    def enable_lora(self, last_n_blocks: int = 4, r: int = 8, alpha: int = 16) -> int:
        """Inject LoRA into the last N blocks. Returns added trainable params."""

        return apply_lora(self, last_n_blocks=last_n_blocks, r=r, alpha=alpha)

    def forward(self, input_ids, attention_mask, candidate_mask):
        """
        candidate_mask: [B, T, C] boolean — for each token, which candidate
        it belongs to. Tokens that belong to no candidate are all-False.
        """
        hidden = self.backbone(input_ids, attention_mask)  # [B, T, H] fp16
        pooled = pool_candidates(hidden, candidate_mask)  # [B, C, H] fp16
        pooled = pooled.to(next(self.head.parameters()).dtype)
        logits = self.head(pooled)  # [B, C] fp32
        return logits


if __name__ == "__main__":
    from data import Candidate, Example, build_prompt, tokenize_example

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}")

    model = MiniJevModel().to(device)
    model.eval()

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] total params     = {n_total / 1e6:.2f} M")
    print(f"[model] trainable params = {n_train / 1e6:.2f} M  (only the head)")

    # ---- a realistic example -------------------------------------
    example = Example(
        state="User says: 'I need to know if my package will arrive before Friday.'",
        candidates=[
            Candidate(id="shipping.track", description="track a shipment by order id"),
            Candidate(
                id="shipping.refund", description="start a refund for a lost package"
            ),
            Candidate(
                id="calendar.check", description="check the user's calendar for Friday"
            ),
            Candidate(
                id = "order.again", description="cancel the delivery and order again"
            ),
        ],
    )

    # ---- show the prompt -----------------------------------------
    text, spans = build_prompt(example.state, example.candidates)
    print("\n[prompt]")
    print(text)

    # ---- tokenize + mask -----------------------------------------
    input_ids, attn, cand_mask, spans = tokenize_example(
        example, model.tokenizer, device
    )

    print(f"Input id shape from the tokenizer output: {input_ids.shape}")
    print(f"The input ids are: {input_ids}")

    # ---- show which tokens were selected for each candidate ------
    toks = model.tokenizer.convert_ids_to_tokens(input_ids.tolist())
    for c, cand in enumerate(example.candidates):
        selected = [toks[t] for t in range(len(toks)) if cand_mask[t, c].item()]
        print(f"\n[candidate {c}] id={cand.id}")
        print(f"   description = {cand.description!r}")
        print(f"   pooled tokens = {selected}")

    # ---- forward pass --------------------------------------------
    with torch.no_grad():
        input_ids_b = input_ids.unsqueeze(0)  # [1, T]
        attn_b = attn.unsqueeze(0)  # [1, T]
        cand_mask_b = cand_mask.unsqueeze(0)  # [1, T, C]

        logits = model(input_ids_b, attn_b, cand_mask_b)  # [1, C]
        probs = torch.softmax(logits, dim=-1)[0]  # [C]

    print(f"after the dimension change input ids: {input_ids_b}")
    
    print(f"\n[forward] logits shape = {tuple(logits.shape)}  (expect [1, 3])")
    print("\n[output] candidate probabilities (untrained head, so ~random):")
    for c, cand in enumerate(example.candidates):
        print(f"   {cand.id:20s}  {probs[c].item():.4f}")

    if device == "cuda":
        print(
            f"\n[vram] peak allocated = {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
        )
