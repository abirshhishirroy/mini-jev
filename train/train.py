"""
train.py — Train only the DecisionHead on frozen Qwen2.5-0.5B.

Step 4 scope: head only. No LoRA. This gives us the baseline accuracy
for a frozen backbone, which Step 5 (LoRA) will try to beat.

Key mechanics:
  - Variable candidate counts: pad with -inf so softmax ignores padding.
  - fp16 backbone, fp32 head (mixed precision, Pascal-safe).
  - AdamW on head params only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

# ---- imports you need to add ----
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from data.data import Candidate, Example, tokenize_example
from model.model import MiniJevModel
from train.losses import cross_entropy_loss, rlcd_loss


# ---------------------------------------------------------------------
# Dataset — pre-tokenize once, cache tensors
# ---------------------------------------------------------------------
class JevDataset(Dataset):
    def __init__(self, path, tokenizer, max_len: int = 1024, subset: int = 0):
        self.path = Path(path)
        self.records: list[dict] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        if subset > 0:
            self.records = self.records[:subset]

        if not self.records:
            raise RuntimeError(f"No records loaded from {self.path}")

        self.max_len = max_len
        self.cache: list[tuple] = []
        print(
            f"[tok] pre-tokenizing {len(self.records)} examples (max_len={max_len}) ..."
        )

        for rec in self.records:
            example = Example(
                state=rec["state"],
                candidates=[
                    Candidate(id=c["id"], description=c["description"])
                    for c in rec["candidates"]
                ],
                correct_index=rec["correct_index"],
            )
            input_ids, attn, cand_mask, _ = tokenize_example(
                example, tokenizer, device="cpu"
            )
            # truncate
            if input_ids.shape[0] > max_len:
                input_ids = input_ids[:max_len]
                attn = attn[:max_len]
                cand_mask = cand_mask[:max_len]
            self.cache.append((input_ids, attn, cand_mask, rec["correct_index"]))

        # record lengths for bucketing
        self.lengths = [c[0].shape[0] for c in self.cache]
        print(
            f"[tok] done. len min={min(self.lengths)} "
            f"max={max(self.lengths)} mean={int(np.mean(self.lengths))} "
            f"p95={int(np.percentile(self.lengths, 95))}"
        )

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        return self.cache[idx]


# ---------------------------------------------------------------------
# Length-bucketed sampler — groups similar-length examples into batches
# ---------------------------------------------------------------------
class BucketSampler(Sampler):
    """
    Yields indices sorted by length, then shuffles within buckets of
    `bucket_size` for some randomness. This keeps padding minimal.
    """

    def __init__(self, lengths, batch_size, bucket_size=64, shuffle=True, seed=0):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        idx = np.argsort(self.lengths)
        # group into buckets of bucket_size, shuffle within bucket
        batches = []
        for i in range(0, len(idx), self.bucket_size):
            bucket = idx[i : i + self.bucket_size]
            if self.shuffle:
                self.rng.shuffle(bucket)
            batches.append(bucket)
        if self.shuffle:
            self.rng.shuffle(batches)
        # now chunk each bucket into batch_size pieces
        for bucket in batches:
            for j in range(0, len(bucket), self.batch_size):
                yield from bucket[j : j + self.batch_size].tolist()

    def __len__(self):
        return len(self.lengths)


# ---------------------------------------------------------------------
# Collate — pad token dim to max in batch, pad candidate dim to max in batch
# ---------------------------------------------------------------------
def collate(batch):
    """
    Each element: (input_ids[T_i], attn[T_i], cand_mask[T_i, C_i], target)
    Pads:
      - T dimension -> max_T
      - C dimension -> max_C
    Returns:
      input_ids:     [B, T_max]
      attn:          [B, T_max]
      cand_mask:     [B, T_max, C_max]
      cand_valid:    [B, C_max]   True for real candidates, False for padding
      target:        [B]          long
    """
    max_T = max(x[0].shape[0] for x in batch)
    max_C = max(x[2].shape[1] for x in batch)

    B = len(batch)
    input_ids = torch.zeros(B, max_T, dtype=torch.long)
    attn = torch.zeros(B, max_T, dtype=torch.long)
    cand_mask = torch.zeros(B, max_T, max_C, dtype=torch.bool)
    cand_valid = torch.zeros(B, max_C, dtype=torch.bool)
    targets = torch.zeros(B, dtype=torch.long)

    for i, (ids, a, cm, tgt) in enumerate(batch):
        T = ids.shape[0]
        C = cm.shape[1]
        input_ids[i, :T] = ids
        attn[i, :T] = a
        cand_mask[i, :T, :C] = cm
        cand_valid[i, :C] = True
        targets[i] = tgt

    return input_ids, attn, cand_mask, cand_valid, targets


def decision_loss(logits, cand_valid, target, loss_name="ce", **kwargs):
    """
    Dispatcher so the training loop can swap losses without knowing details.

    kwargs are forwarded to the loss. RLCD-specific keys are ignored by CE.
    """
    if loss_name == "ce":
        return cross_entropy_loss(logits, cand_valid, target)
    elif loss_name == "rlcd":
        return rlcd_loss(
            logits,
            cand_valid,
            target,
            margin=kwargs.get("rlcd_margin", 0.5),
            lambda_margin=kwargs.get("rlcd_lambda", 0.3),
            hard_neg_power=kwargs.get("rlcd_hard_neg_power", 1.0),
        )
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


# ---------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for input_ids, attn, cand_mask, cand_valid, target in loader:
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        cand_mask = cand_mask.to(device)
        cand_valid = cand_valid.to(device)
        target = target.to(device)

        logits = model(input_ids, attn, cand_mask)  # [B, C] fp32
        loss = cross_entropy_loss(logits, cand_valid, target)

        total_loss += loss.item() * target.shape[0]
        preds = logits.masked_fill(~cand_valid, float("-inf")).argmax(dim=-1)
        correct += (preds == target).sum().item()
        total += target.shape[0]

    return total_loss / total, correct / total


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=str, default="data/train.jsonl")
    p.add_argument("--val", type=str, default="data/val.jsonl")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument(
        "--lora", action="store_true", help="Enable LoRA on the last N blocks."
    )
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HF model name for the backbone.",
    )

    p.add_argument("--lora_last_n", type=int, default=4)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument(
        "--lr_lora",
        type=float,
        default=2e-4,
        help="LR for LoRA params (typically 5-10x lower than head LR).",
    )
    p.add_argument(
        "--subset",
        type=int,
        default=0,
        help="If >0, use only the first N training examples (smoke test).",
    )
    p.add_argument(
        "--grad_accum",
        type=int,
        default=1,
        help="Accumulate gradients over N micro-batches.",
    )

    p.add_argument(
        "--loss",
        type=str,
        default="ce",
        choices=["ce", "rlcd"],
        help="Training loss. `ce` = plain cross-entropy. "
        "`rlcd` = cross-entropy + reweighted margin term.",
    )
    p.add_argument(
        "--rlcd_lambda",
        type=float,
        default=0.3,
        help="Weight of the margin term in RLCD.",
    )
    p.add_argument(
        "--rlcd_margin",
        type=float,
        default=0.5,
        help="Desired minimum logit gap between correct and wrong.",
    )
    p.add_argument(
        "--rlcd_hard_neg_power",
        type=float,
        default=1.0,
        help="0 = uniform weights, 1 = prob-weighted, 2 = aggressive.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}")

    # ---- model ------------------------------------------------------
    model = MiniJevModel(model_name=args.model).to(device)

    # backbone -> fp16, head -> fp32
    model.backbone.backbone.float()
    model.head.float()

    # freeze everything, then unfreeze head
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    if args.lora:
        added = model.enable_lora(
            last_n_blocks=args.lora_last_n, r=args.lora_r, alpha=args.lora_alpha
        )
        print(
            f"[lora] injected into last {args.lora_last_n} blocks, "
            f"+{added / 1e6:.3f} M trainable params"
        )
        # make sure the lora params are actually trainable
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.requires_grad = True

        # Gradient checkpointing: recompute activations during backward
        # instead of storing them. ~30% slower, ~60% less activation VRAM.
        model.backbone.backbone.gradient_checkpointing_enable()
        # HF requires this to make the checkpointing graph build correctly
        # when the base model is frozen.
        model.backbone.backbone.enable_input_require_grads()
        print("[lora] gradient checkpointing enabled")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params = {n_trainable / 1e6:.3f} M")

    # ---- data -------------------------------------------------------
    train_ds = JevDataset(
        args.train, model.tokenizer, max_len=args.max_len, subset=args.subset
    )
    val_ds = JevDataset(
        args.val,
        model.tokenizer,
        max_len=args.max_len,
        subset=min(args.subset, 100) if args.subset else 0,
    )
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}")

    train_sampler = BucketSampler(train_ds.lengths, args.batch_size, shuffle=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collate,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        pin_memory=False,
    )

    head_params = list(model.head.parameters())
    lora_params = [
        p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad
    ]

    groups = [{"params": head_params, "lr": args.lr}]
    if lora_params:
        groups.append({"params": lora_params, "lr": args.lr_lora})

    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    n_trainable = sum(p.numel() for p in head_params + lora_params)
    print(
        f"[opt] head={sum(p.numel() for p in head_params) / 1e6:.3f} M "
        f"lora={sum(p.numel() for p in lora_params) / 1e6:.3f} M "
        f"total={n_trainable / 1e6:.3f} M"
    )

    print(
        f"[loss] training loss = {args.loss}"
        + (
            f"  (lambda={args.rlcd_lambda}, margin={args.rlcd_margin}, "
            f"hard_neg_power={args.rlcd_hard_neg_power})"
            if args.loss == "rlcd"
            else ""
        )
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    # ---- training loop ---------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        # backbone stays in eval mode (it's frozen; we don't want dropout etc.)
        model.backbone.eval()

        t0 = time.time()
        running_loss = 0.0
        seen = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}", ncols=100)
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar, start=1):
            input_ids, attn, cand_mask, cand_valid, target = batch
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            cand_mask = cand_mask.to(device)
            cand_valid = cand_valid.to(device)
            target = target.to(device)

            logits = model(input_ids, attn, cand_mask)
            loss = decision_loss(
                logits,
                cand_valid,
                target,
                loss_name=args.loss,
                rlcd_margin=args.rlcd_margin,
                rlcd_lambda=args.rlcd_lambda,
                rlcd_hard_neg_power=args.rlcd_hard_neg_power,
            )
            # divide by accumulation steps so the sum across accum steps
            # equals the loss of one big batch
            (loss / args.grad_accum).backward()

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * target.shape[0]
            seen += target.shape[0]

            if step % args.log_every == 0:
                pbar.set_postfix(loss=f"{running_loss / seen:.4f}")

        train_loss = running_loss / seen
        val_loss, val_acc = evaluate(model, val_loader, device)
        dt = time.time() - t0
        print(
            f"[epoch {epoch}] "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"({dt:.1f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.head.state_dict(), out_dir / "head_best.pt")
            if args.lora:
                lora_sd = {n: p for n, p in model.named_parameters() if "lora_" in n}
                torch.save(lora_sd, out_dir / "lora_best.pt")
            print(f"[ckpt] saved (val_acc={val_acc:.4f})")
            print(f"[ckpt] saved {out_dir / 'head_best.pt'}  (val_acc={val_acc:.4f})")

        if device == "cuda":
            print(f"[vram] peak = {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print(f"\n[done] best val_acc = {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
