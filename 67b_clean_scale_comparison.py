import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

# Node 67B — clean baseline comparison
# Step 59: 8.4M, D=256, 8 layers, FF=1024
# Step 66: 42M, D=512, 12 layers, FF=2048
# Same tokenizer and same QA preference groups.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DRIVE = Path("/content/drive/MyDrive/transformers_exercise_20260911")
TOK_PATH = DRIVE / "artifacts/step43/step43_bpe_8000.json"
STEP59_PATH = DRIVE / "artifacts/step59/tiny_gpt_v2_best.pt"
STEP66_PATH = DRIVE / "artifacts/step66/tiny_gpt_v2_best.pt"

VOCAB = 8000
CTX = 256

# Load only the QA data definitions from the committed Node 63 source.
import urllib.request
NODE63_URL = (
    "https://raw.githubusercontent.com/"
    "Randyren2024/transformers_exercise_20260911/"
    "main/63_precision_knowledge_alignment_sft_v2.py"
    "?v=e13e36628744b0facf9087d6cd8e829a9934a44f"
)
source = urllib.request.urlopen(NODE63_URL, timeout=60).read().decode("utf-8")
qa_env = {"__name__": "qa_lib"}
exec(compile(source, "63_precision_knowledge_alignment_sft_v2.py", "exec"), qa_env)
qa_groups = qa_env["qa_groups"]


def rows_from_groups(groups):
    rows = []
    for en_prompts, en_good, en_bad, zh_prompts, zh_good, zh_bad in groups:
        for p in en_prompts:
            rows.append(("en", p, en_good, en_bad))
        for p in zh_prompts:
            rows.append(("zh", p, zh_good, zh_bad))
    return rows


def split_groups():
    groups = qa_groups()
    rng = random.Random(63)
    rng.shuffle(groups)
    cut = max(1, int(len(groups) * 0.80))
    return groups[:cut], groups[cut:]


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


def load_checkpoint(path, d, h, layers, ff):
    model = TinyGPT(d, h, layers, ff).to(DEVICE)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def encode_pair(tok, prompt, answer):
    prefix = f"User: {prompt}\nAssistant:"
    full = prefix + " " + answer + " [END]"
    prefix_ids = tok.encode(prefix).ids
    full_ids = tok.encode(full).ids
    if len(full_ids) > CTX:
        full_ids = full_ids[:CTX]
    if len(full_ids) <= len(prefix_ids) + 1:
        return None
    x = full_ids[:-1]
    y = full_ids[1:]
    answer_start = min(len(prefix_ids) - 1, len(y))
    mask = [False] * len(y)
    for i in range(answer_start, len(y)):
        mask[i] = True
    return x, y, mask


@torch.inference_mode()
def answer_nll(model, tok, prompt, answer):
    item = encode_pair(tok, prompt, answer)
    if item is None:
        return float("inf")
    x, y, mask = item
    xb = torch.tensor([x], dtype=torch.long, device=DEVICE)
    logits = model(xb)[0]
    losses = F.cross_entropy(
        logits,
        torch.tensor(y, dtype=torch.long, device=DEVICE),
        reduction="none",
    )
    mask_t = torch.tensor(mask, dtype=torch.bool, device=DEVICE)
    return float(losses[mask_t].mean().item())


def score(model, tok, rows):
    good, bad, margins, ranks = [], [], [], []
    for _, prompt, good_answer, bad_answer in rows:
        g = answer_nll(model, tok, prompt, good_answer)
        b = answer_nll(model, tok, prompt, bad_answer)
        good.append(g)
        bad.append(b)
        margins.append(b - g)
        ranks.append(float(g < b))
    return {
        "good_nll": float(np.mean(good)),
        "bad_nll": float(np.mean(bad)),
        "margin": float(np.mean(margins)),
        "rank": float(np.mean(ranks)),
    }


@torch.inference_mode()
def generate(model, tok, prompt, max_new=32):
    ids = tok.encode(f"User: {prompt}\nAssistant:").ids
    x = torch.tensor([ids[-CTX:]], dtype=torch.long, device=DEVICE)
    out = []
    for _ in range(max_new):
        logits = model(x[:, -CTX:])
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        x = torch.cat([x, nxt], dim=1)
        out.append(int(nxt.item()))
    return tok.decode(out).strip()


print("=" * 112)
print("Node 67B — clean Step 66 vs Step 59 baseline diagnostic")
print("=" * 112)
print("device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

print("Step 59:", STEP59_PATH)
print("Step 66:", STEP66_PATH)

tok = Tokenizer.from_file(str(TOK_PATH))
train_groups, val_groups = split_groups()
train_rows = rows_from_groups(train_groups)
val_rows = rows_from_groups(val_groups)
all_rows = train_rows + val_rows

print(f"Groups: 42 | train: {len(train_groups)} | held-out: {len(val_groups)}")
print("Loading Step 59...")
step59 = load_checkpoint(STEP59_PATH, 256, 8, 8, 1024)
print("Loading Step 66...")
step66 = load_checkpoint(STEP66_PATH, 512, 8, 12, 2048)

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
    all_s = score(model, tok, all_rows)
    train_s = score(model, tok, train_rows)
    val_s = score(model, tok, val_rows)

    print("\n" + "-" * 112)
    print(name)
    print(
        f"ALL:     good={all_s['good_nll']:.4f} | bad={all_s['bad_nll']:.4f} | "
        f"margin={all_s['margin']:.4f} | rank={all_s['rank'] * 100:.1f}%"
    )
    print(
        f"TRAIN:   good={train_s['good_nll']:.4f} | bad={train_s['bad_nll']:.4f} | "
        f"margin={train_s['margin']:.4f} | rank={train_s['rank'] * 100:.1f}%"
    )
    print(
        f"HOLDOUT: good={val_s['good_nll']:.4f} | bad={val_s['bad_nll']:.4f} | "
        f"margin={val_s['margin']:.4f} | rank={val_s['rank'] * 100:.1f}%"
    )

    print("\nGeneration probes")
    for prompt, expected in probes:
        print(f"User: {prompt}")
        print(f"Expected: {expected}")
        print(f"Generated: {generate(model, tok, prompt)}")

print("\nNode 67B complete.")
