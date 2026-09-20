import sys
import urllib.request
from pathlib import Path

# Node 64 — Step 63 checkpoint comparison diagnostic
# This does not train anything.
# It compares the clean Step 59 model against Node 63's saved best checkpoint
# on the exact same precision-QA groups, and separately probes generation.

NODE63_URL = (
    "https://raw.githubusercontent.com/"
    "Randyren2024/transformers_exercise_20260911/"
    "main/63_precision_knowledge_alignment_sft_v2.py"
    "?v=e13e36628744b0facf9087d6cd8e829a9934a44f"
)

source = urllib.request.urlopen(NODE63_URL, timeout=60).read().decode("utf-8")
exec(compile(source, "63_precision_knowledge_alignment_sft_v2.py", "exec"), {"__name__": "node63_lib"})

import numpy as np
import torch
from tokenizers import Tokenizer

DRIVE = Path("/content/drive/MyDrive/transformers_exercise_20260911")
TOK_PATH = DRIVE / "artifacts" / "step43" / "step43_bpe_8000.json"
STEP59 = DRIVE / "artifacts" / "step59" / "tiny_gpt_v2_best.pt"
STEP63_BEST = DRIVE / "artifacts" / "step63" / "tiny_gpt_v2_precision_best.pt"

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

def load(path):
    return load_model(path, device)

def score(model, subset):
    good_nlls = []
    bad_nlls = []
    ranks = []
    margins = []

    for _, prompt, good, bad in subset:
        good_nll, bad_nll = answer_nll_batch(
            model, tok, [prompt, prompt], [good, bad], device
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
def generate(model, prompt, max_new=32):
    model.eval()
    ids = tok.encode(f"User: {prompt}\nAssistant:").ids
    x = torch.tensor([ids[-CTX:]], dtype=torch.long, device=device)
    out = []
    for _ in range(max_new):
        logits = model(x[:, -CTX:])
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        x = torch.cat([x, nxt], dim=1)
        out.append(int(nxt.item()))
    return tok.decode(out).strip()

print("=" * 112)
print("Node 64 — Step 59 vs Node 63 checkpoint diagnostic")
print("=" * 112)
print("device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
print("Step 63 best checkpoint:", STEP63_BEST)
print("Step 63 best was selected by held-out rank accuracy.")
print("Groups:", len(all_groups), "| train:", len(train_groups), "| held-out:", len(val_groups))

step59 = load(STEP59)
step63 = load(STEP63_BEST)

for name, model in [("Step 59 baseline", step59), ("Step 63 BEST", step63)]:
    all_score = score(model, rows)
    train_rows = rows_from_groups_local(train_groups)
    val_rows = rows_from_groups_local(val_groups)
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

    probes = [
        "李白是谁？",
        "谁是爱因斯坦？",
        "什么是人工智能？",
        "什么是 Transformer？",
        "What is artificial intelligence?",
        "What is a transformer in machine learning?",
        "Translate 'software' into Chinese.",
        "What is 7 + 8?",
        "What is 7 times 8?",
    ]
    print("\nGeneration probes")
    for p in probes:
        print(f"User: {p}\nAssistant: {generate(model, p)}")

print("\nNode 64 complete.")
