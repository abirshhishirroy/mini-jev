"""
evaluate.py — Final evaluation for Mini-Jev.

Produces:
  - Per-task top-1 accuracy
  - Overall top-1 accuracy
  - Per-task mean confidence (after temperature scaling)
  - Latency benchmark (ms per example, p50 / p95)
  - Optional: gpt-4o-mini baseline on the same examples for comparison

This is the headline-number script. Run it right after training.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

from model.model import MiniJevModel
from train.train import JevDataset, collate

load_dotenv()


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------
def load_model(device: str, head_ckpt: str, lora_ckpt: str | None,
               lora_last_n: int, lora_r: int, lora_alpha: int,
               temperature_ckpt: str | None):
    model = MiniJevModel().to(device)
    model.backbone.backbone.bfloat16()
    model.head.float()

    if lora_ckpt:
        added = model.enable_lora(last_n_blocks=lora_last_n, r=lora_r, alpha=lora_alpha)
        print(f"[lora] re-injected into last {lora_last_n} blocks, +{added/1e6:.3f} M")
        sd = torch.load(lora_ckpt, map_location=device)
        own = dict(model.named_parameters())
        for k, v in sd.items():
            if k in own:
                own[k].data.copy_(v.to(own[k].device))

    sd = torch.load(head_ckpt, map_location=device)
    model.head.load_state_dict(sd)
    print(f"[load] head from {head_ckpt}")

    T = 1.0
    if temperature_ckpt and Path(temperature_ckpt).exists():
        t_sd = torch.load(temperature_ckpt, map_location="cpu")
        T = float(t_sd["temperature"])
        print(f"[load] temperature T = {T:.4f}")
    else:
        print(f"[load] no temperature checkpoint; using T = 1.0 (uncalibrated)")

    model.eval()
    return model, T


# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------
@torch.no_grad()
def run_inference(model, loader, device, temperature: float = 1.0):
    """
    Returns:
        results: list of dicts — one per example, with:
            probs        list[float]   (normalized over valid candidates)
            pred_idx     int
            target       int
            latency_ms   float
    """
    results = []
    for input_ids, attn, cand_mask, cand_valid, target in tqdm(loader, desc="infer"):
        input_ids  = input_ids.to(device)
        attn       = attn.to(device)
        cand_mask  = cand_mask.to(device)
        cand_valid = cand_valid.to(device)

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits = model(input_ids, attn, cand_mask)              # [B, C_max]
        logits = logits / temperature
        masked = logits.masked_fill(~cand_valid, float("-inf"))
        probs  = torch.softmax(masked, dim=-1).cpu()

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_ex = elapsed_ms / probs.shape[0]

        for i in range(probs.shape[0]):
            c = int(cand_valid[i].sum().item())
            p = probs[i, :c].tolist()
            results.append({
                "probs": p,
                "pred_idx": int(max(range(len(p)), key=lambda j: p[j])),
                "target": int(target[i].item()),
                "latency_ms": per_ex,
            })
    return results


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def compute_metrics(results, tasks):
    """
    results: list of dicts from run_inference
    tasks:   list of task names parallel to results
    """
    total = len(results)
    correct = sum(1 for r in results if r["pred_idx"] == r["target"])
    overall_acc = correct / total

    by_task = defaultdict(list)
    for r, t in zip(results, tasks):
        by_task[t].append(r)

    per_task = {}
    for t, rows in sorted(by_task.items()):
        n = len(rows)
        c = sum(1 for r in rows if r["pred_idx"] == r["target"])
        conf = statistics.mean(max(r["probs"]) for r in rows)
        per_task[t] = {"n": n, "acc": c / n, "mean_conf": conf}

    lats = [r["latency_ms"] for r in results]
    latency = {
        "p50": statistics.median(lats),
        "p95": sorted(lats)[int(0.95 * len(lats))],
        "mean": statistics.mean(lats),
    }

    return {
        "overall_acc": overall_acc,
        "n": total,
        "per_task": per_task,
        "latency_ms": latency,
    }


# ---------------------------------------------------------------------
# Optional gpt-4o-mini baseline
# ---------------------------------------------------------------------
LLM_PROMPT = (
    "You are a decision router. Given a state and a list of candidate "
    "options, pick the single best option. Respond with JSON only: "
    '{"choice": "<candidate id>"}.'
)


def build_llm_prompt(row):
    lines = [f"[TASK:{row.get('task', 'generic')}]", f"STATE: {row['state']}", ""]
    for c in row["candidates"]:
        lines.append(f"  - {c['id']}: {c['description']}")
    lines.append("")
    lines.append("Pick the best candidate id. Return JSON only.")
    return "\n".join(lines)


async def llm_predict(client, model_name, row, sem, retries=3):
    async with sem:
        for attempt in range(retries):
            try:
                r = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": LLM_PROMPT},
                        {"role": "user", "content": build_llm_prompt(row)},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=60,
                )
                data = json.loads(r.choices[0].message.content or "{}")
                choice = data.get("choice", "")
                ids = [c["id"] for c in row["candidates"]]
                if choice in ids:
                    return ids.index(choice)
                return -1
            except Exception:
                await asyncio.sleep(1.0)
    return -1


async def run_llm_baseline(rows, model_name, concurrency):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("[llm] OPENROUTER_API_KEY not set, skipping baseline")
        return None
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    sem = asyncio.Semaphore(concurrency)
    preds = await asyncio.gather(*[
        llm_predict(client, model_name, row, sem) for row in rows
    ])
    return preds


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test", type=str, default="data/multitask/val.jsonl")
    p.add_argument("--head_ckpt", type=str, default="checkpoints/head_best.pt")
    p.add_argument("--lora_ckpt", type=str, default=None)
    p.add_argument("--temperature_ckpt", type=str, default=None)
    p.add_argument("--lora_last_n", type=int, default=4)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--subset", type=int, default=0)
    p.add_argument("--compare_llm", action="store_true",
                   help="Also run gpt-4o-mini on the same examples.")
    p.add_argument("--llm_model", type=str, default="openai/gpt-4o-mini")
    p.add_argument("--llm_concurrency", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}")

    model, T = load_model(
        device, args.head_ckpt, args.lora_ckpt,
        args.lora_last_n, args.lora_r, args.lora_alpha,
        args.temperature_ckpt,
    )

    ds = JevDataset(args.test, model.tokenizer, max_len=args.max_len,
                    subset=args.subset)
    print(f"[data] test n={len(ds)}")
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate)

    # Load raw rows to get task labels and compare to LLM
    raw_rows = ds.records   # JevDataset keeps the originals
    tasks = [r.get("task", "generic") for r in raw_rows]

    results = run_inference(model, loader, device, temperature=T)
    m = compute_metrics(results, tasks)

    print("\n=== Mini-Jev results ===")
    print(f"overall accuracy: {m['overall_acc']:.4f}  (n={m['n']})")
    print(f"latency: p50={m['latency_ms']['p50']:.2f}ms  "
          f"p95={m['latency_ms']['p95']:.2f}ms  "
          f"mean={m['latency_ms']['mean']:.2f}ms")
    print()
    print(f"{'task':20s} {'n':>5s} {'acc':>7s} {'conf':>7s}")
    for t, v in m["per_task"].items():
        print(f"{t:20s} {v['n']:>5d} {v['acc']:>7.4f} {v['mean_conf']:>7.4f}")

    # LLM baseline
    if args.compare_llm:
        print(f"\n[llm] running {args.llm_model} baseline on {len(raw_rows)} examples")
        t0 = time.time()
        preds = asyncio.run(run_llm_baseline(raw_rows, args.llm_model, args.llm_concurrency))
        dt = time.time() - t0
        if preds is not None:
            correct = sum(1 for p, r in zip(preds, raw_rows)
                          if p == r["correct_index"])
            n = len(preds)
            print(f"[llm] {args.llm_model} accuracy: {correct/n:.4f}")
            print(f"[llm] total wall time: {dt:.1f}s  ({1000*dt/n:.1f}ms/example)")

            # per-task LLM breakdown
            by_task = defaultdict(list)
            for p, r in zip(preds, raw_rows):
                by_task[r.get("task", "generic")].append(p == r["correct_index"])
            print(f"\n{'task':20s} {'llm_acc':>9s} {'jev_acc':>9s}")
            for t in sorted(by_task):
                llm_acc = sum(by_task[t]) / len(by_task[t])
                jev_acc = m["per_task"][t]["acc"] if t in m["per_task"] else 0
                print(f"{t:20s} {llm_acc:>9.4f} {jev_acc:>9.4f}")


if __name__ == "__main__":
    main()