import random
from pathlib import Path
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

# Node 70 — first-token / teacher-forcing diagnostic
#
# Compare clean Step 66 against Node 69 best SFT checkpoint.
# The key question:
#   Did SFT improve the complete-answer NLL while still failing to predict
#   the FIRST answer token under free-running generation?
#
# This separates teacher-forcing improvement from actual instruction-following.

DRIVE = Path("/content/drive/MyDrive/transformers_exercise_20260911")
TOK_PATH = DRIVE / "artifacts/step43/step43_bpe_8000.json"
STEP66 = DRIVE / "artifacts/step66/tiny_gpt_v2_best.pt"
STEP69 = DRIVE / "artifacts/step69/tiny_gpt_v2_42m_sft_best.pt"

VOCAB = 8000
D = 512
H = 8
LAYERS = 12
FF = 2048
CTX = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

QA_URL = (
    "https://raw.githubusercontent.com/"
    "Randyren2024/transformers_exercise_20260911/"
    "main/69_precision_data.py"
    "?v=d5fd2ec545b53ff0a7cc0b3d09bf447cc5358213"
)


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


def load(path):
    model = TinyGPT().to(DEVICE)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def get_qa_groups():
    src = urllib.request.urlopen(QA_URL, timeout=60).read().decode("utf-8")
    env = {"__name__": "precision_data_69"}
    exec(compile(src, "69_precision_data.py", "exec"), env)
    return env["qa_groups"]()


def rows_from_groups(groups):
    rows = []
    for en_prompts, en_good, _en_bad, zh_prompts, zh_good, _zh_bad in groups:
        for p in en_prompts:
            rows.append(("en", p, en_good))
        for p in zh_prompts:
            rows.append(("zh", p, zh_good))
    return rows


def split_groups():
    groups = get_qa_groups()
    rng = random.Random(69)
    rng.shuffle(groups)
    cut = max(1, int(len(groups) * 0.80))
    return groups[:cut], groups[cut:]


def encode(tok, prompt, answer):
    prefix = f"User: {prompt}\nAssistant:"
    full = prefix + " " + answer + " [END]"
    prefix_ids = tok.encode(prefix).ids
    full_ids = tok.encode(full).ids
    return prefix_ids, full_ids


@torch.inference_mode()
def analyze_item(model, tok, prompt, answer):
    prefix_ids, full_ids = encode(tok, prompt, answer)

    # Input ending immediately before the first answer-side token.
    prefix = torch.tensor([prefix_ids], dtype=torch.long, device=DEVICE)

    logits = model(prefix)[0, -1]
    logp = F.log_softmax(logits.float(), dim=-1)

    first_answer_id = full_ids[len(prefix_ids)]
    first_nll = -float(logp[first_answer_id].item())
    first_rank = int((logp > logp[first_answer_id]).sum().item()) + 1

    topv, topi = torch.topk(logp, 8)
    top_tokens = [
        (int(i.item()), tok.decode([int(i.item())]), float(v.item()))
        for v, i in zip(topv, topi)
    ]

    # Full answer NLL under teacher forcing.
    x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=DEVICE)
    logits_full = model(x)[0]
    targets = torch.tensor(full_ids[1:], dtype=torch.long, device=DEVICE)

    answer_start = len(prefix_ids) - 1
    losses = F.cross_entropy(
        logits_full,
        targets,
        reduction="none",
    )
    answer_losses = losses[answer_start:]
    full_nll = float(answer_losses.mean().item())

    # Free-running first several tokens.
    current = prefix.clone()
    generated = []
    for _ in range(10):
        nxt_logits = model(current[:, -CTX:])[:, -1, :]
        nxt = torch.argmax(nxt_logits, dim=-1, keepdim=True)
        current = torch.cat([current, nxt], dim=1)
        generated.append(int(nxt.item()))

    return {
        "first_nll": first_nll,
        "first_rank": first_rank,
        "first_token": tok.decode([first_answer_id]),
        "top_tokens": top_tokens,
        "full_nll": full_nll,
        "generated": tok.decode(generated).strip(),
    }


def main():
    tok = Tokenizer.from_file(str(TOK_PATH))
    train_groups, val_groups = split_groups()
    val_rows = rows_from_groups(val_groups)

    print("=" * 112)
    print("Node 70 — first-token / teacher-forcing diagnostic")
    print("=" * 112)
    print("device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Validation groups:", len(val_groups))
    print("Validation rows:", len(val_rows))

    step66 = load(STEP66)
    step69 = load(STEP69)

    for name, model in [("Step 66 baseline", step66), ("Step 69 best SFT", step69)]:
        print("\n" + "-" * 112)
        print(name)

        rows_to_test = val_rows[:]
        first_nlls = []
        first_ranks = []
        full_nlls = []

        for lang, prompt, answer in rows_to_test:
            a = analyze_item(model, tok, prompt, answer)
            first_nlls.append(a["first_nll"])
            first_ranks.append(a["first_rank"] == 1)
            full_nlls.append(a["full_nll"])

        print(
            f"First-token NLL:  {np.mean(first_nlls):.4f}\n"
            f"First-token top1: {np.mean(first_ranks) * 100:.1f}%\n"
            f"Full-answer NLL:  {np.mean(full_nlls):.4f}"
        )

        for prompt in [
            "李白是谁？",
            "谁是爱因斯坦？",
            "什么是人工智能？",
            "什么是 Transformer？",
            "What is artificial intelligence?",
            "What is a transformer in machine learning?",
            "Translate 'software' into Chinese.",
            "What is 7 + 8?",
            "What is 7 times 8?",
        ]:
            row = next((r for r in val_rows if r[1] == prompt), None)
            if row is None:
                row = next((r for r in rows_from_groups(train_groups) if r[1] == prompt), None)
            if row is None:
                continue

            a = analyze_item(model, tok, row[1], row[2])
            print(f"\nUser: {row[1]}")
            print(f"Expected first token: {a['first_token']!r}")
            print(f"First-token NLL: {a['first_nll']:.4f} | rank: {a['first_rank']}")
            print("Top candidates:")
            for tid, text, lp in a["top_tokens"]:
                print(f"  {tid:>5} | {text!r} | logP={lp:.4f}")
            print(f"Full-answer NLL: {a['full_nll']:.4f}")
            print(f"Free-run first 10: {a['generated']!r}")

    print("\nNode 70 complete.")


if __name__ == "__main__":
    main()
