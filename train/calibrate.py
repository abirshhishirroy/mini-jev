"""
calibrate.py — Temperature scaling for Mini-Jev.

Takes a trained model (head + optional LoRA), runs inference on the
calibration split, and learns a single scalar T > 0 such that
softmax(logits / T) is better calibrated on held-out data.

Also produces a reliability diagram (predicted prob vs. actual accuracy).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.model import MiniJevModel
from train.train import JevDataset, collate  # reuse


def load_trained_model(device: str, head_ckpt: str, lora_ckpt: str | None,
                       lora_last_n: int, lora_r: int, lora_alpha: int):
    model = MiniJevModel().to(device)
    model.backbone.backbone.bfloat16()
    model.head.float()

    if lora_ckpt:
        added = model.enable_lora(last_n_blocks=lora_last_n, r=lora_r, alpha=lora_alpha)
        print(f"[lora] re-injected into last {lora_last_n} blocks, +{added/1e6:.3f} M params")
        sd = torch.load(lora_ckpt, map_location=device)
        # strip any key prefix mismatches by matching on suffix
        own = dict(model.named_parameters())
        for k, v in sd.items():
            if k in own:
                own[k].data.copy_(v.to(own[k].device))
            else:
                print(f"[warn] lora key not found: {k}")
        # freeze backbone, keep only head + lora trainable? no — we're only inferring.

    sd = torch.load(head_ckpt, map_location=device)
    model.head.load_state_dict(sd)
    print(f"[load] head from {head_ckpt}")
    model.eval()
    return model


@torch.no_grad()
def collect_predictions(model, loader, device):
    """
    Return per-example arrays (not concatenated tensors, because candidate
    counts vary per batch).

    Returns:
        probs_list:  list of 1D tensors, each of shape [C_i]
        targets:     list of ints (correct index within the unpadded candidates)
    """
    probs_list = []
    targets = []
    for input_ids, attn, cand_mask, cand_valid, target in tqdm(loader, desc="infer"):
        input_ids  = input_ids.to(device)
        attn       = attn.to(device)
        cand_mask  = cand_mask.to(device)
        cand_valid = cand_valid.to(device)

        logits = model(input_ids, attn, cand_mask)              # [B, C_max] fp32
        masked = logits.masked_fill(~cand_valid, float("-inf")) # ignore padding
        probs  = torch.softmax(masked, dim=-1).cpu()            # [B, C_max]

        # split back into per-example rows, keeping only valid candidates
        for i in range(probs.shape[0]):
            c = int(cand_valid[i].sum().item())
            probs_list.append(probs[i, :c])
            targets.append(int(target[i].item()))

    return probs_list, targets


def expected_calibration_error(probs: np.ndarray, correct: np.ndarray,
                               n_bins: int = 15) -> float:
    """Standard ECE over the max-probability prediction."""
    max_p = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    is_correct = (preds == correct).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(max_p)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (max_p > lo) & (max_p <= hi)
        if mask.sum() == 0:
            continue
        acc = is_correct[mask].mean()
        conf = max_p[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_diagram(probs: np.ndarray, correct: np.ndarray,
                        n_bins: int = 15):
    """Print a text reliability diagram (works in a terminal on stream)."""
    max_p = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    is_correct = (preds == correct).astype(np.float32)

    print(f"\n{'bin':>12s}  {'n':>6s}  {'conf':>6s}  {'acc':>6s}  {'gap':>6s}  bar")
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (max_p > lo) & (max_p <= hi)
        if mask.sum() == 0:
            continue
        acc = is_correct[mask].mean()
        conf = max_p[mask].mean()
        bar = "█" * int(acc * 40)
        print(f"[{lo:.2f},{hi:.2f})  {mask.sum():>6d}  "
              f"{conf:.3f}  {acc:.3f}  {acc-conf:+.3f}  {bar}")



def _to_arrays(probs_list, targets):
    """
    Turn a list of variable-length probability tensors into a padded numpy
    array plus a flat correctness array. We pad to max_C and set invalid
    entries to 0 so argmax and max() behave correctly.
    """
    n = len(probs_list)
    max_c = max(p.shape[0] for p in probs_list)
    out = np.zeros((n, max_c), dtype=np.float32)
    for i, p in enumerate(probs_list):
        out[i, :p.shape[0]] = p.numpy()
    # correct array: does the argmax match the target?
    preds = out.argmax(axis=1)
    correct = np.array([1 if preds[i] == targets[i] else 0 for i in range(n)],
                       dtype=np.int64)
    return out, correct


def _safe_log(probs: torch.Tensor) -> torch.Tensor:
    """Log with a floor so we don't produce -inf for zero-probability classes."""
    return torch.log(probs.clamp(min=1e-12))


def fit_temperature_from_probs(probs_list, targets, lr: float = 0.01, max_iter: int = 100) -> float:
    """
    Fit T by minimizing NLL over temperature-scaled probabilities.

    We work from the already-softmaxed probabilities (log-space) rather than
    raw logits, since we don't keep the full padded logits tensor around.
    """
    log_T = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)

    # Precompute log-probs (unscaled) so closure is fast
    log_probs = [_safe_log(p) for p in probs_list]
    tgt = torch.tensor(targets, dtype=torch.long)

    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_T)
        # NLL = -sum over examples of log_softmax(log_p / T)[target]
        total = 0.0
        for i, lp in enumerate(log_probs):
            scaled = lp / T
            log_norm = torch.logsumexp(scaled, dim=-1)
            total = total + (log_norm - scaled[tgt[i]])
        loss = total / len(log_probs)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_T).item())




def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--calib", type=str, default="data/calibration.jsonl")
    p.add_argument("--head_ckpt", type=str, default="checkpoints/head_best.pt")
    p.add_argument("--lora_ckpt", type=str, default=None)
    p.add_argument("--lora_last_n", type=int, default=2)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--subset", type=int, default=0,
                   help="If >0, use only the first N calibration examples.")
    p.add_argument("--out", type=str, default="checkpoints/temperature.pt")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}")

    model = load_trained_model(
        device, args.head_ckpt, args.lora_ckpt,
        args.lora_last_n, args.lora_r, args.lora_alpha,
    )

    ds = JevDataset(args.calib, model.tokenizer, max_len=args.max_len,
                    subset=args.subset)
    print(f"[data] calibration n={len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate)

    probs_list, targets = collect_predictions(model, loader, device)
    print(f"[infer] n_examples={len(probs_list)}")


    # ECE before calibration
    probs_np, correct_np = _to_arrays(probs_list, targets)
    ece_before = expected_calibration_error(probs_np, correct_np)
    acc_before = float((probs_np.argmax(axis=1) == correct_np).mean())
    print(f"\n[before] acc={acc_before:.4f}  ECE={ece_before:.4f}")
    reliability_diagram(probs_np, correct_np)

    # Fit temperature
    T = fit_temperature_from_probs(probs_list, targets)
    print(f"\n[fit] optimal T = {T:.4f}")

    # ECE after calibration
    probs_cal_list = [torch.softmax(_safe_log(p) / T, dim=-1) for p in probs_list]
    probs_cal_np, _ = _to_arrays(probs_cal_list, targets)
    ece_after = expected_calibration_error(probs_cal_np, correct_np)
    acc_after = float((probs_cal_np.argmax(axis=1) == correct_np).mean())
    print(f"[after]  acc={acc_after:.4f}  ECE={ece_after:.4f}")
    reliability_diagram(probs_cal_np, correct_np)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"temperature": T}, out_path)
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()