"""
Node 67 — Step 66 vs Step 59 baseline diagnostic

Compares the 42M-param Step 66 model against the 8.4M Step 59 model
on the same QA preference groups from Step 63.
Uses the same scoring as Node 64.
"""

import sys
import urllib.request
from pathlib import Path

# Pull the data helpers from Step 63 (qa_groups, rows_from_groups, etc.)
NODE63_URL = (
    "https://raw.githubusercontent.com/"
    "Randyren2024/transformers_exercise_20260911/"
    "main/63_precision_knowledge_alignment_sft_v2.py"
    "?v=e13e36628744b0facf9087d6cd8e829a9934a44f"
)

from tokenizers import Tokenizer
import random

source = urllib.request.urlopen(NODE63_URL, timeout=60).read().decode("utf-8")
node63_env = {"__name__": "node63_lib", "Tokenizer": Tokenizer}
exec(
    compile(source, "63_precision_knowledge_alignment_sft_v2.py", "exec"),
    node63_env,
)
globals().update({k: v for k, v in node63_env.items() if k != "__name__"})

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DRIVE = Path("/content/drive/MyDrive/transformers_exercise_20260911")
TOK_PATH = DRIVE / "artifacts" / "step43" / "step43_bpe_8000.json"
STEP59 = DRIVE / "artifacts" / "step59" / "tiny_gpt_v2_best.pt"
STEP66_BEST = DRIVE / "artifacts" / "step66" / "tiny_gpt_v2_best.pt"

# ---------------------------------------------------------------------------
# Step 66 model architecture (scaled) — same classes as 66_scale_pretrain.py
# ---------------------------------------------------------------------------

V66, H66, L66, F66, C66 = 8000, 8, 12, 2048, 256

class Attn66(nn.Module):
    def __init__(self):
        super().__init__()
        hd = 512 // H66
        self.qkv = nn.Linear(512, 3 * 512, bias=False)
        self.out = nn.Linear(512, 512, bias=False)
        self.hd = hd
    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, H66, self.hd).transpose(1, 2)
        k = k.view(b, t, H66, self.hd).transpose(1, 2)
        v = v.view(b, t, H66, self.hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).contiguous().view(b, t, 512))

class Block66(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(512)
        self.attn = Attn66()
        self.ln2 = nn.LayerNorm(512)
        self.fc1 = nn.Linear(512, F66, bias=False)
        self.fc2 = nn.Linear(F66, 512, bias=False)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x

class TinyGPT66(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V66, 512)
        self.pos = nn.Embedding(C66, 512)
        self.blocks = nn.ModuleList([Block66() for _ in range(L66)])
        self.ln_f = nn.LayerNorm(512)
        self.head = nn.Linear(512, V66, bias=False)
        self.head.weight = self.tok.weight
    def forward(self, x):
        t = x.size(1)
        pos = torch.arange(t, device=x.device)
        h = self.tok(x) + self.pos(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))

def load_step66(path, device):
    model = TinyGPT66().to(device)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    return model

# ---------------------------------------------------------------------------
# Shared scoring (reuses Step 63 helpers where possible)
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = Tokenizer.from_file(str(TOK_PATH))

def split_groups_local():
    groups = qa_groups()
    rng = random.Random(63)
    rng.shuffle(groups)
    cut = max(1, int(len(groups) * 0.80))
    return groups[:cut], groups[cut:]

def rows_from_groups_local(groups):
    rows = []
    for en_prompts, en_good, en_bad, zh_prompts, zh_good, zh_bad in groups:
        for p in en_prompts:
            rows.append(("en", p, en_good, en_bad))
        for p in zh_prompts:
            rows.append(("zh", p, zh_good, zh_bad))
    return rows

train_groups, val_groups = split_groups_local()
all_groups = train_groups + val_groups
rows = rows_from_groups_local(all_groups)

def score(model, subset):
    good_nlls = []
    bad_nlls = []
    ranks = []
    margins = []

    for _, prompt, good, bad in subset:
        good_nll, bad_nll = answer_nll_batch(
            model,
            tok,
            [prompt, prompt],
            [good, bad],
            device,
        )
        g = float(good_nll.item())
        b = float(bad_nll.item())
        good_nlls.append(g)
        bad_nlls.append(b)
        margins.append(b - g)
        ranks.append(float(g < b))

    return {
        "good_nll": float(np.mean(good_nlls)),
        "bad_nll": float(np.mean(bad_nlls)),
        "margin": float(np.mean(margins)),
        "rank_accuracy": float(np.mean(ranks)),
    }

@torch.inference_mode()
def generate(model, tok, prompt, max_new=32, ctx=256):
    model.eval()
    ids = tok.encode(f"User: {prompt}\nAssistant:").ids
    x = torch.tensor([ids[-ctx:]], dtype=torch.long, device=next(model.parameters()).device)
    out = []
    for _ in range(max_new):
        logits = model(x[:, -ctx:])
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        x = torch.cat([x, nxt], dim=1)
        out.append(int(nxt.item()))
    return tok.decode(out).strip()

# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------

print("=" * 112)
print("Node 67 — Step 66 (42M) vs Step 59 (8.4M) baseline diagnostic")
print("=" * 112)
print("device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
print("Step 66 checkpoint:", STEP66_BEST)
print("Step 59 checkpoint:", STEP59)
print(f"Groups: {len(all_groups)} | train: {len(train_groups)} | held-out: {len(val_groups)}")

step59 = load(STEP59, device)  # from Step 63 helpers — same architecture
step66 = load_step66(STEP66_BEST, device)

train_rows = rows_from_groups_local(train_groups)
val_rows = rows_from_groups_local(val_groups)

probes = [
    ("李白是谁？", "李白是中国唐代诗人。"),
    ("谁是爱因斯坦？", "爱因斯坦是出生于德国的物理学家。"),
    ("什么是人工智能？", "人工智能是让计算机执行通常需要人类智能任务的一种技术。"),
    ("什么是 Transformer？", "Transformer 是一种以注意力机制为核心的神经网络架构。"),
    ("What is artificial intelligence?", "Artificial intelligence is technology that enables computers to perform tasks that normally require human intelligence."),
    ("What is a transformer in machine learning?", "A transformer is a neural network architecture built around attention."),
    ("Translate 'software' into Chinese.", "软件。"),
    ("What is 7 + 8?", "15."),
    ("What is 7 times 8?", "56."),
]

for name, model in [("Step 59 (8.4M)", step59), ("Step 66 (42M)", step66)]:
    all_score = score(model, rows)
    train_score = score(model, train_rows)
    val_score = score(model, val_rows)

    print("\n" + "-" * 112)
    print(name)
    print(
        f"ALL:      good={all_score['good_nll']:.4f} | "
        f"bad={all_score['bad_nll']:.4f} | "
        f"margin={all_score['margin']:.4f} | "
        f"rank={all_score['rank_accuracy'] * 100:.1f}%"
    )
    print(
        f"TRAIN:    good={train_score['good_nll']:.4f} | "
        f"bad={train_score['bad_nll']:.4f} | "
        f"margin={train_score['margin']:.4f} | "
        f"rank={train_score['rank_accuracy'] * 100:.1f}%"
    )
    print(
        f"HOLDOUT:  good={val_score['good_nll']:.4f} | "
        f"bad={val_score['bad_nll']:.4f} | "
        f"margin={val_score['margin']:.4f} | "
        f"rank={val_score['rank_accuracy'] * 100:.1f}%"
    )

    print("\nGeneration probes")
    for p, expected in probes:
        gen = generate(model, tok, p, ctx=256 if name.startswith("Step 66") else 256)
        print(f"  User: {p}")
        print(f"  Expected: {expected}")
        print(f"  {name.split()[0]}: {gen}")

print("\n" + "=" * 112)
print("Node 67 complete.")
