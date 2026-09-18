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


class GPT(nn.Module):
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

    @torch.no_grad()
    def greedy_generate(self, x, tok, max_new=40):
        start = x.size(1)
        for _ in range(max_new):
            logits = self(x[:, -CTX:])[:, -1, :]
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
            x = torch.cat([x, nxt], dim=1)
        return tok.decode(x[0].tolist()[start:])


@torch.no_grad()
def candidate_nll(model, tok, prompt, answer, device):
    full = prompt + " " + answer
    prompt_ids = tok.encode(prompt).ids
    full_ids = tok.encode(full).ids
    if len(full_ids) > CTX:
        return float("inf"), 0
    x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    logits = model(x)[0]
    logp = torch.log_softmax(logits, dim=-1)
    start = max(0, len(prompt_ids) - 1)
    targets = full_ids[1:]
    answer_positions = list(range(start, len(targets)))
    if not answer_positions:
        return float("inf"), 0
    vals = [logp[p, targets[p]].item() for p in answer_positions]
    return -sum(vals) / len(vals), len(vals)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drive = Path("/content/drive/MyDrive/transformers_exercise_20260911")
    ckpt = drive / "artifacts" / "step51" / "tiny_gpt_step51_best.pt"
    tok_path = drive / "artifacts" / "step43" / "step43_bpe_8000.json"

    print("=" * 112)
    print("Step 52: Step 51 diagnostic — knowledge vs decoding")
    print("=" * 112)
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    tok = Tokenizer.from_file(str(tok_path))
    model = GPT().to(device)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    tests = [
        (
            "User: What is 7 + 8?\nAssistant:",
            ["15.", "9.", "21.", "29."],
        ),
        (
            "User: 12 加 9 等于多少？\nAssistant:",
            ["21。", "29。", "15。", "12。"],
        ),
        (
            "User: Translate 'robot' into Chinese.\nAssistant:",
            ["机器人。", "软件。", "数据。"],
        ),
        (
            "User: 把“data”翻译成中文。\nAssistant:",
            ["数据。", "软件。", "机器人。"],
        ),
        (
            "User: What can you do?\nAssistant:",
            [
                "I can answer simple questions and follow short instructions.",
                "I am a small bilingual language model.",
            ],
        ),
        (
            "User: 你能做什么？\nAssistant:",
            [
                "我可以回答简单问题并执行简短指令。",
                "我是一个小型双语语言模型。",
            ],
        ),
        (
            "User: Explain artificial intelligence in simple terms.\nAssistant:",
            [
                "Artificial intelligence is technology that lets computers perform tasks that normally require human intelligence.",
                "Machine learning lets computers learn patterns from data.",
            ],
        ),
    ]

    print("\nPart 1: Greedy decoding")
    print("-" * 112)
    for prompt, _ in tests:
        ids = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
        out = model.greedy_generate(ids, tok)
        print("\n" + prompt + "\n" + out.strip())

    print("\nPart 2: Candidate answer likelihood")
    print("-" * 112)
    for prompt, candidates in tests:
        print("\n" + prompt)
        scored = []
        for ans in candidates:
            nll, tokens = candidate_nll(model, tok, prompt, ans, device)
            scored.append((nll, ans, tokens))
        for nll, ans, tokens in sorted(scored):
            print(f"  NLL={nll:.4f} | tokens={tokens:>2} | {ans}")

    print("\nStep 52 diagnostic complete.")


if __name__ == "__main__":
    main()
