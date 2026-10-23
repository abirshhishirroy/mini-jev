# Mini-Jev: A Jev-style Decision Model on Qwen2.5-3B

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

**Mini-Jev** is a specialized "System One" decision-making model built from scratch. Instead of generating text autoregressively, it ingests unstructured state (text, code, JSON) and returns a **typed probabilistic distribution over candidate actions** in a single forward pass. 

This project implements the core architecture of Jev (by TypeSafe) from scratch, using a frozen `Qwen2.5-3B-Instruct` backbone, a custom `DecisionHead`, and hand-rolled LoRA adapters.

## 🏆 Key Results

Trained on a custom 35,000-example dataset, Mini-Jev **outperforms OpenAI's GPT-4o-mini** on domain-specific routing and validation tasks, at a fraction of the latency and cost.

| Metric | Mini-Jev (3B) | GPT-4o-mini (API) | Advantage |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | **83.38%** | 73.00% | **+10.38%** |
| **Latency (p50)** | **23.95 ms** | ~600 ms | **~25x Faster** |
| **Cost per 1K Inferences** | **$0.00** (Local) | ~$0.30 | **Free** |
| **Calibration (ECE)** | **0.022** (Perfect) | N/A | **Trustworthy Probs** |

### Per-Task Breakdown
| Task | Mini-Jev Accuracy | GPT-4o-mini Accuracy |
| :--- | :--- | :--- |
| **Safety Gating** | **97.32%** | 78.19% |
| **Hallucination Validation** | **76.67%** | 61.00% |
| **Context Filtering** | **67.67%** | 55.67% |
| **Context Filtering (Noul)** | **71.00%** | 58.00% |
| **Safety Gating (Noul)** | **93.98%** | 86.29% |
| **Query Routing** | **100.00%** | 87.80% |
| **Tool Routing** | 82.33% | **89.75%** |

*(Note: GPT-4o-mini only wins on general tool routing, where its massive parameter count helps with broad semantic understanding. Mini-Jev dominates on structured validation, strict rule-following, and safety tasks).*

## 💡 Motivation: The Pivot from Generative LLM to Jev

Originally, my goal was to train a small (~135M parameter) generative LLM entirely from scratch that would simply output JSON for classification and routing. I wanted to understand the full stack of autoregressive language modeling. 

However, right as I began, TypeSafe AI showcased the **Jev architecture**. It completely shifted my perspective. Why force a small model to generate text token-by-token just to make a single routing decision, when I could build a model that outputs the decision directly in one pass?

I immediately pivoted. Instead of building a generative LLM, I decided to build a **Mini-Jev**. 

This pivot gave me three massive advantages:
1. **Extreme Speed:** No autoregressive generation loop means inference takes milliseconds, not seconds.
2. **Data Privacy:** By keeping the model local (3B parameters fits on consumer GPUs), no sensitive user data or proprietary code has to be sent to OpenAI or Anthropic APIs.
3. **Perfect Calibration:** A custom decision head allows for exact mathematical calibration, meaning the model "knows what it knows."

## 🛠️ Real-World Application: Where This Model is Used

This model is not a standalone toy; it is the core decision engine for two of my larger, ongoing projects:

1. **A CLI Coding Agent (Python Unit Testing):** I am building an agent that traverses existing Python projects and writes unit tests. Mini-Jev acts as the ultra-fast router, deciding whether the agent needs to `read_file`, `grep_search`, or `run_bash`. Crucially, Mini-Jev acts as a **Safety Gate**, intercepting the agent before it runs dangerous shell commands (like `rm -rf` or privilege escalations).
2. **An End-to-End RAG Pipeline:** Standard RAG systems blindly pass retrieved chunks to an LLM, causing hallucinations. Mini-Jev acts as a **Context Filter** (instantly classifying chunks as RELEVANT/IRRELEVANT) and a **Hallucination Validator** (checking if the final generated answer is FAITHFUL to the source chunks). 

## 🧠 Architecture

Standard LLMs use a `Linear(hidden_size, vocab_size)` head to predict the next token. Mini-Jev replaces this with a `Linear(hidden_size, 1)` head to score candidate actions.

1. **Prompt Construction:** The state and candidates are concatenated into a single sequence:
   `[TASK:tool_router] [STATE] ... [/STATE] [CANDIDATE id="read_file"] ... [/CANDIDATE]`
2. **Frozen Backbone:** The sequence is passed through a frozen `Qwen2.5-3B` backbone.
3. **Candidate Pooling:** A boolean mask is used to mean-pool the hidden states *only* over the tokens belonging to each specific candidate. (See `pool_candidates` in `model.py`).
4. **Decision Head:** A 2-layer MLP maps each pooled candidate vector to a single scalar logit.
5. **Softmax:** A grouped softmax over the candidates yields the final probabilities.

## 📊 The Dataset (35,769 Examples)

To train a multi-task router, a custom dataset was generated using `gpt-4o-mini` (for synthetic generation) and adapted from public benchmarks (SQuAD v2, HaluEval). It covers 7 distinct tasks. 

*(Note: The full 35,000-example dataset is excluded from this repository to keep it lightweight. You can view the exact format in `data/samples/train_sample.jsonl`*

### Data Format Example
Each line is a JSON object containing the task type, the state, the candidate options, and the correct index:

```json
{
  "task": "tool_router",
  "state": "hey can you show me what's in main.py?",
  "candidates": [
    {"id": "read_file", "description": "read the full contents of a file by path"},
    {"id": "list_dir", "description": "list files and directories at a path"},
    {"id": "abstain", "description": "answer directly without calling a tool"}
  ],
  "correct_index": 0
}