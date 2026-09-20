import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

# Node 68 — raw knowledge diagnostic: Step 66 vs Step 59
#
# Important:
#   Step 59 and Step 66 are plain causal language models, not chat models.
#   Therefore this diagnostic deliberately uses raw completion prompts
#   ("Li Bai was", "A transformer is", etc.) instead of "User/Assistant".
#
# Goal:
#   Separate "knowledge coverage / association" from "instruction formatting".

VOCAB = 8000
CTX = 256

DRIVE = Path("/content/drive/MyDrive/transformers_exercise_20260911")
TOK_PATH = DRIVE / "artifacts/step43/step43_bpe_8000.json"
STEP59 = DRIVE / "artifacts/step59/tiny_gpt_v2_best.pt"
STEP66 = DRIVE / "artifacts/step66/tiny_gpt_v2_best.pt"


class Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        hd = d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.h = h
        self.hd = hd
        self.d = d

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.h, self.hd).transpose(1, 2)
        k = k.view(b, t, self.h, self.hd).transpose(1, 2)
        v = v.view(b, t, self.h, self.hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).contiguous().view(b, t, self.d))


class Block(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attn(d, h)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ff, bias=False)
        self.fc2 = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class TinyGPT(nn.Module):
    def __init__(self, d, h, layers, ff):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(CTX, d)
        self.blocks = nn.ModuleList([Block(d, h, ff) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, x):
        t = x.size(1)
        pos = torch.arange(t, device=x.device)
        h = self.tok(x) + self.pos(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))


def load_model(path, d, h, layers, ff, device):
    model = TinyGPT(d, h, layers, ff).to(device)
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
        return float("inf")

    x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    logits = model(x)[0]
    targets = full_ids[1:]

    start = max(0, len(prompt_ids) - 1)
    vals = []
    for p in range(start, len(targets)):
        lp = F.log_softmax(logits[p], dim=-1)
        vals.append(-float(lp[targets[p]].item()))

    return sum(vals) / max(1, len(vals))


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
    parser.add_argument("--device", default="cuda")
    args, _ = parser.parse_known_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(str(TOK_PATH))

    print("=" * 112)
    print("Node 68 — Step 66 vs Step 59 raw knowledge diagnostic")
    print("=" * 112)
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    step59 = load_model(STEP59, 256, 8, 8, 1024, device)
    step66 = load_model(STEP66, 512, 8, 12, 2048, device)

    tests = [
        {
            "name": "Li Bai",
            "prompt": "李白是",
            "candidates": [
                "中国唐代诗人。",
                "一位中国古代诗人。",
                "一位中国现代科学家。",
                "一位美国企业家。",
            ],
        },
        {
            "name": "Einstein",
            "prompt": "Albert Einstein was",
            "candidates": [
                "a German-born physicist.",
                "an American politician.",
                "a French painter.",
                "a Chinese poet.",
            ],
        },
        {
            "name": "Transformer",
            "prompt": "A transformer is",
            "candidates": [
                "a neural network architecture.",
                "a type of database.",
                "a web browser.",
                "a programming language.",
            ],
        },
        {
            "name": "Artificial intelligence",
            "prompt": "Artificial intelligence is",
            "candidates": [
                "technology that lets computers perform tasks that normally require human intelligence.",
                "a type of database table.",
                "a web browser.",
                "a physical computer cable.",
            ],
        },
        {
            "name": "Language model",
            "prompt": "A language model is",
            "candidates": [
                "a model that predicts likely tokens from context and can generate text.",
                "a database system.",
                "a web browser.",
                "a physical computer cable.",
            ],
        },
        {
            "name": "Attention",
            "prompt": "Attention in a transformer",
            "candidates": [
                "lets a model focus on relevant parts of the input.",
                "is a type of web browser.",
                "is a database protocol.",
                "is a graphics card.",
            ],
        },
    ]

    for name, model in [("Step 59 (8.4M)", step59), ("Step 66 (42M)", step66)]:
        print("\n" + "-" * 112)
        print(name)

        for test in tests:
            print("\n" + test["name"])
            print("Prompt:", repr(test["prompt"]))
            print("Greedy:", greedy_completion(model, tok, test["prompt"], device))

            scored = []
            for ans in test["candidates"]:
                nll = candidate_nll(model, tok, test["prompt"], ans, device)
                scored.append((nll, ans))

            for rank, (nll, ans) in enumerate(sorted(scored), 1):
                print(f"  #{rank} NLL={nll:.4f} | {ans}")

    print("\nNode 68 complete.")


if __name__ == "__main__":
    main()
