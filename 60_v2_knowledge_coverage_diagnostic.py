import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

VOCAB = 8000
D = 256
H = 8
LAYERS = 8
FF = 1024
CTX = 256

class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        hd = D // H
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.out = nn.Linear(D, D, bias=False)
        self.hd = hd

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, H, self.hd).transpose(1, 2)
        k = k.view(b, t, H, self.hd).transpose(1, 2)
        v = v.view(b, t, H, self.hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).contiguous().view(b, t, D))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.attn = Attn()
        self.ln2 = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, FF, bias=False)
        self.fc2 = nn.Linear(FF, D, bias=False)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D)
        self.pos = nn.Embedding(CTX, D)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, VOCAB, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, x):
        t = x.size(1)
        pos = torch.arange(t, device=x.device)
        h = self.tok(x) + self.pos(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))

def load_model(path, device):
    model = TinyGPT().to(device)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model

@torch.inference_mode()
def candidate_nll(model, tok, prompt, answer, device):
    full = prompt + answer
    prompt_ids = tok.encode(prompt).ids
    full_ids = tok.encode(full).ids
    if len(full_ids) > CTX or len(full_ids) <= len(prompt_ids):
        return float("inf"), 0
    x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    logits = model(x)[0]
    logp = torch.log_softmax(logits, dim=-1)
    targets = full_ids[1:]
    start = max(0, len(prompt_ids) - 1)
    vals = [logp[p, targets[p]].item() for p in range(start, len(targets))]
    return -sum(vals) / len(vals), len(vals)

@torch.inference_mode()
def greedy_completion(model, tok, prompt, device, max_new=32):
    ids = tok.encode(prompt).ids[-CTX:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    start = x.size(1)
    for _ in range(max_new):
        logits = model(x[:, -CTX:])[:, -1, :]
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        x = torch.cat([x, nxt], dim=1)
    return tok.decode(x[0].tolist()[start:]).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/content/drive/MyDrive/transformers_exercise_20260911/artifacts/step59/tiny_gpt_v2_best.pt")
    args, _ = parser.parse_known_args()

    drive = Path("/content/drive/MyDrive/transformers_exercise_20260911")
    tok_path = drive / "artifacts" / "step43" / "step43_bpe_8000.json"
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(str(tok_path))
    model = load_model(ckpt, device)

    tests = [
        {
            "name": "Li Bai — raw completion",
            "prompt": "李白是",
            "candidates": ["中国唐代诗人。", "一位中国古代诗人。", "一名现代作家。", "一位科学家。"],
        },
        {
            "name": "Li Bai — sentence completion",
            "prompt": "李白是一位",
            "candidates": ["中国唐代诗人。", "中国现代科学家。", "美国企业家。", "英国小说家。"],
        },
        {
            "name": "Einstein — raw completion",
            "prompt": "Albert Einstein was",
            "candidates": ["a German-born physicist.", "an American politician.", "a French painter.", "a Chinese poet."],
        },
        {
            "name": "Transformer — raw completion",
            "prompt": "A transformer is",
            "candidates": ["a neural network architecture.", "a type of database.", "a programming language.", "a web browser."],
        },
        {
            "name": "Artificial intelligence — raw completion",
            "prompt": "Artificial intelligence is",
            "candidates": ["technology that lets computers perform tasks that normally require human intelligence.", "a physical computer cable.", "a type of database table.", "a web browser."],
        },
        {
            "name": "Known v1 concept — chat prompt",
            "prompt": "User: 什么是人工智能？\\nAssistant: ",
            "candidates": ["人工智能是让计算机执行通常需要人类智能任务的一种技术。", "谢谢。", "我不知道。"],
        },
    ]

    print("=" * 112)
    print("Node 60 — TinyGPT v2 knowledge coverage diagnostic")
    print("=" * 112)
    print("checkpoint:", ckpt)
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    for test in tests:
        print("\n" + "-" * 112)
        print(test["name"])
        print("Prompt:", repr(test["prompt"]))
        print("Greedy completion:", greedy_completion(model, tok, test["prompt"], device))
        scored = []
        for answer in test["candidates"]:
            nll, tokens = candidate_nll(model, tok, test["prompt"], answer, device)
            scored.append((nll, answer, tokens))
        print("Candidate likelihood (lower NLL = stronger internal preference):")
        for nll, answer, tokens in sorted(scored):
            print(f"  NLL={nll:.4f} | tokens={tokens:>2} | {answer}")

    print("\nInterpretation:")
    print("- Correct answer ranked first on raw-completion prompts is direct evidence of learned knowledge signal.")
    print("- Correct raw completion but weak chat prompt indicates instruction-format weakness rather than missing knowledge.")
    print("- Correct answers below distractors on both raw and chat prompts suggest weak knowledge coverage or weak memorization.")
    print("- Compare this output with Step 54 diagnostic to separate corpus improvements from SFT effects.")
    print("\nNode 60 complete.")

if __name__ == "__main__":
    main()