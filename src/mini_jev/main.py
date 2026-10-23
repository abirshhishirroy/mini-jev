import torch
import time
from model.model import MiniJevModel
from data.data import Candidate, Example, tokenize_example

def main():
    device = "cpu"
    print(f"[env] device={device} (Running 3B on CPU)")

    # 1. Load model in float32 for CPU
    model = MiniJevModel(model_name="Qwen/Qwen2.5-3B-Instruct").to(device)
    model.backbone.backbone.float()
    model.head.float()

    # 2. Load LoRA
    model.enable_lora(last_n_blocks=4, r=8, alpha=16)
    lora_sd = torch.load("checkpoints/lora_best.pt", map_location=device)
    own = dict(model.named_parameters())
    for k, v in lora_sd.items():
        if k in own:
            own[k].data.copy_(v.to(own[k].device))

    # 3. Load Head & Temperature
    head_sd = torch.load("checkpoints/head_best.pt", map_location=device)
    model.head.load_state_dict(head_sd)
    T = torch.load("checkpoints/temperature.pt", map_location="cpu")["temperature"]
    model.eval()

    # 4. Define the test case
    example = Example(
        task="tool_router",
        state="hey can you show me what's in main.py?",
        candidates=[
            Candidate("read_file", "read the full contents of a file by path; use when the user asks to view, show, or inspect a file"),
            Candidate("list_dir", "list files and directories at a path; use when the user asks what's in a folder"),
            Candidate("grep_search", "search file contents for a pattern; use when the user asks where a symbol is used or defined"),
            Candidate("abstain", "answer directly without calling a tool; use for questions answerable from context alone")
        ]
    )

    input_ids, attn, mask, _ = tokenize_example(example, model.tokenizer, device=device)

    print(f"\n[User State]: {example.state}")
    
    # 5. Inference
    t0 = time.time()
    with torch.no_grad():
        logits = model(input_ids.unsqueeze(0), attn.unsqueeze(0), mask.unsqueeze(0))
        probs = torch.softmax(logits / T, dim=-1)[0]
    latency = (time.time() - t0) * 1000

    print("\n[Output Probabilities]:")
    for cand, p in zip(example.candidates, probs.tolist()):
        print(f"  {cand.id:15s} {p:.4f}")
    print(f"\n[Latency]: {latency:.2f} ms")

if __name__ == "__main__":
    main()