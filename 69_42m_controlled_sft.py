import math
import random
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

# Load the exact QA data module. Prefer a local copy next to this file; fall
# back to GitHub raw at a pinned commit.
#
# The /<commit-sha>/ path form is the ONLY form GitHub actually pins. A
# "?v=<sha>" query parameter is ignored, and raw.githubusercontent then serves
# whatever is currently on main -- the pin is decorative. The previous version
# of this file had both problems: a "?v=" URL and a sha ending ...213 instead
# of the real ...214.
QA_DATA_SHA = "d5fd2ec545b53ff0a7cc0b3d09bf447cc5358214"
QA_DATA_URL = (
    "https://raw.githubusercontent.com/"
    "Randyren2024/transformers_exercise_20260911/"
    f"{QA_DATA_SHA}/69_precision_data.py"
)

try:
    _local_qa = Path(__file__).resolve().parent / "69_precision_data.py"
except NameError:  # exec'd without __file__
    _local_qa = Path("69_precision_data.py")

if _local_qa.exists():
    qa_source = _local_qa.read_text(encoding="utf-8")
    print(f"QA data: local {_local_qa}")
else:
    with urllib.request.urlopen(QA_DATA_URL, timeout=60) as _resp:
        qa_source = _resp.read().decode("utf-8")
    print(f"QA data: fetched {QA_DATA_SHA[:8]} from GitHub")

qa_env = {"__name__": "precision_data_69"}
exec(compile(qa_source, "69_precision_data.py", "exec"), qa_env)
qa_groups = qa_env["qa_groups"]

# Node 69 — controlled SFT on the 42M Step 66 base model.
#
# Design:
#   - Start from clean Step 66.
#   - Only teach the response behavior using short, precise answers.
#   - Augment prompts with multiple paraphrase templates.
#   - Hold out entire QA groups, so validation tests unseen facts.
#   - Freeze embeddings + bottom half of transformer blocks to protect
#     the pretrained language representation.
#   - Answer-only CE. No DPO, no pairwise ranking, no teacher KL.
#
# This is intentionally conservative.

SEED = 69
VOCAB = 8000
D = 512
H = 8
LAYERS = 12
FF = 2048
CTX = 256

STEPS = 800
BATCH = 32

LR = 1.0e-6
MIN_LR = 2.0e-7
WARMUP = 80
WEIGHT_DECAY = 0.01

DRIVE = Path("/content/drive/MyDrive/transformers_exercise_20260911")
BASE_CKPT = DRIVE / "artifacts/step66/tiny_gpt_v2_best.pt"
TOK_PATH = DRIVE / "artifacts/step43/step43_bpe_8000.json"
OUT_DIR = DRIVE / "artifacts/step69"


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


def seed_all():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def lr_at(step):
    if step <= WARMUP:
        return LR * step / WARMUP
    p = min(max((step - WARMUP) / max(1, STEPS - WARMUP), 0.0), 1.0)
    return MIN_LR + (LR - MIN_LR) * 0.5 * (1.0 + math.cos(math.pi * p))


EN_TEMPLATES = [
    "Who was {x}?",
    "What is {x}?",
    "What was {x} known for?",
    "Please briefly describe {x}.",
    "Can you briefly explain {x}?",
]

ZH_TEMPLATES = [
    "{x}是谁？",
    "什么是{x}？",
    "{x}以什么著名？",
    "请简要介绍{x}。",
    "请简单解释{x}。",
]


def make_rows(groups):
    rows = []
    for en_prompts, en_good, _en_bad, zh_prompts, zh_good, _zh_bad in groups:
        # Use original prompt variants already encoded in the groups.
        for p in en_prompts:
            rows.append(("en", p, en_good))
        for p in zh_prompts:
            rows.append(("zh", p, zh_good))
    return rows


def augmented_rows(groups):
    rows = []
    for en_prompts, en_good, _en_bad, zh_prompts, zh_good, _zh_bad in groups:
        # Preserve explicit prompts from the source data.
        for p in en_prompts:
            rows.append(("en", p, en_good))
        for p in zh_prompts:
            rows.append(("zh", p, zh_good))

        # Add paraphrases for robust instruction mapping.
        title = None
        if en_prompts:
            for prefix in ("Who was ", "What is ", "What was ", "Please briefly describe "):
                if en_prompts[0].startswith(prefix):
                    title = en_prompts[0][len(prefix):].rstrip(".?")
                    break

        if title:
            for template in EN_TEMPLATES:
                rows.append(("en", template.format(x=title), en_good))

        # Infer Chinese title from the Chinese prompts conservatively.
        zh_title = None
        if zh_prompts:
            raw = zh_prompts[0]
            for suffix in ("是谁？", "是什么？", "以什么著名？", "。"):
                if raw.endswith(suffix):
                    zh_title = raw[:-len(suffix)]
                    break
        if zh_title:
            for template in ZH_TEMPLATES:
                rows.append(("zh", template.format(x=zh_title), zh_good))

    # Deduplicate.
    seen = set()
    out = []
    for row in rows:
        key = (row[0], row[1], row[2])
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def encode_example(tok, prompt, answer):
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

    # y position i corresponds to x position i.
    answer_start = min(max(0, len(prefix_ids) - 1), len(y))
    mask = [False] * len(y)
    for i in range(answer_start, len(y)):
        mask[i] = True

    return x, y, mask


def batch_tensors(tok, rows, device):
    encoded = []
    for _lang, prompt, answer in rows:
        item = encode_example(tok, prompt, answer)
        if item is not None:
            encoded.append(item)

    max_len = max(len(item[0]) for item in encoded)
    xs, ys, ms = [], [], []

    for x, y, m in encoded:
        pad = max_len - len(x)
        xs.append(x + [0] * pad)
        ys.append(y + [-100] * pad)
        ms.append(m + [False] * pad)

    return (
        torch.tensor(xs, dtype=torch.long, device=device),
        torch.tensor(ys, dtype=torch.long, device=device),
        torch.tensor(ms, dtype=torch.bool, device=device),
    )


@torch.inference_mode()
def answer_nll(model, tok, prompt, answer, device):
    item = encode_example(tok, prompt, answer)
    if item is None:
        return float("inf")

    x_ids, y_ids, mask = item
    x = torch.tensor([x_ids], dtype=torch.long, device=device)
    logits = model(x)[0]
    targets = torch.tensor(y_ids, dtype=torch.long, device=device)
    losses = F.cross_entropy(logits, targets, reduction="none")
    mask_t = torch.tensor(mask, dtype=torch.bool, device=device)

    return float(losses[mask_t].mean().item())


@torch.inference_mode()
def generate(model, tok, prompt, device, max_new=48):
    model.eval()
    ids = tok.encode(f"User: {prompt}\nAssistant:").ids
    x = torch.tensor([ids[-CTX:]], dtype=torch.long, device=device)
    start = x.size(1)
    for _ in range(max_new):
        logits = model(x[:, -CTX:])
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        x = torch.cat([x, nxt], dim=1)

        text = tok.decode(x[0].tolist()[start:]).strip()
        if "[END]" in text:
            return text.split("[END]", 1)[0].strip()

    return tok.decode(x[0].tolist()[start:]).strip()


def evaluate(model, tok, rows, device):
    nlls = []
    for _lang, prompt, answer in rows:
        nlls.append(answer_nll(model, tok, prompt, answer, device))
    return float(sum(nlls) / len(nlls))


def main():
    seed_all()

    if not BASE_CKPT.exists():
        raise FileNotFoundError(BASE_CKPT)
    if not TOK_PATH.exists():
        raise FileNotFoundError(TOK_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(str(TOK_PATH))

    groups = qa_groups()
    rng = random.Random(SEED)
    rng.shuffle(groups)

    cut = max(1, int(len(groups) * 0.80))
    train_groups = groups[:cut]
    val_groups = groups[cut:]

    train_rows = augmented_rows(train_groups)
    val_rows = make_rows(val_groups)

    print("=" * 112)
    print("Node 69 — 42M controlled SFT")
    print("=" * 112)
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Base checkpoint:", BASE_CKPT)
    print("Train groups:", len(train_groups))
    print("Held-out groups:", len(val_groups))
    print("Train rows:", len(train_rows))
    print("Validation rows:", len(val_rows))

    model = TinyGPT().to(device)
    state = torch.load(BASE_CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])

    # Freeze the positional embedding and the lower half of the transformer.
    # This protects the pretrained language distribution while the upper
    # layers learn the instruction-response mapping.
    #
    # The input embedding is deliberately NOT frozen. head.weight IS tok.weight
    # (weight tying), so freezing tok also freezes the LM output projection,
    # leaving the model unable to move probability mass onto answer tokens.
    # That is what went wrong in the first Step 69 run: only 45% of the 42M
    # parameters stayed trainable, held-out facts did not improve, and
    # generation died at the first token. See RESULTS_node71_baseline.md.
    assert model.head.weight is model.tok.weight, (
        "weight tying was removed: 'freezing tok' no longer implies freezing "
        "the LM head, so the freeze policy below needs revisiting"
    )

    for p in model.pos.parameters():
        p.requires_grad_(False)
    for block in model.blocks[:6]:
        for p in block.parameters():
            p.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total_params:,}")
    print("Freeze map:")
    for label, module in [
        ("embedding (tied to head)", model.tok),
        ("positional", model.pos),
        ("blocks[0:6]", model.blocks[0]),
        ("blocks[6:12]", model.blocks[6]),
        ("ln_f", model.ln_f),
    ]:
        flags = {p.requires_grad for p in module.parameters()}
        state = {True: "TRAINABLE", False: "FROZEN"}.get(
            flags.pop() if len(flags) == 1 else None, "MIXED"
        )
        n = sum(p.numel() for p in module.parameters())
        print(f"  {label:26s} {n:>12,}  {state}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    tick = time.perf_counter()

    fixed_probes = [
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

    for step in range(1, STEPS + 1):
        model.train()
        lr = lr_at(step)
        optimizer.param_groups[0]["lr"] = lr

        batch = random.sample(train_rows, BATCH)
        x, y, mask = batch_tensors(tok, batch, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(x)
            targets = y.clone()
            targets[~mask] = -100

            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB),
                targets.reshape(-1),
                ignore_index=-100,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 100 == 0 or step == STEPS:
            speed = step * BATCH * CTX / max(time.perf_counter() - tick, 1e-6)
            print(
                f"step {step:>4}/{STEPS} | "
                f"loss {loss.item():.4f} | lr {lr:.2e} | {speed:,.0f} tok/s"
            )

        if step % 100 == 0 or step == STEPS:
            val_loss = evaluate(model, tok, val_rows, device)
            print(f"held-out answer NLL: {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "step": step,
                        "validation_loss": best_val,
                        "base_checkpoint": str(BASE_CKPT),
                        "tokenizer_path": str(TOK_PATH),
                        "config": {
                            "vocab": VOCAB,
                            "d_model": D,
                            "heads": H,
                            "layers": LAYERS,
                            "ffn": FF,
                            "context": CTX,
                        },
                        "frozen_blocks": 6,
                        "frozen_embedding": False,
                        "frozen_pos": True,
                        "weight_tying": "tok<->head (tied, both trained)",
                    },
                    OUT_DIR / "tiny_gpt_v2_42m_sft_best.pt",
                )
                print("saved new best:", best_val)

    model.eval()
    print("\nFixed probes")
    for prompt in fixed_probes:
        print(f"\nUser: {prompt}")
        print(f"Assistant: {generate(model, tok, prompt, device)}")

    print("\nNode 69 complete.")
    print("Best checkpoint:", OUT_DIR / "tiny_gpt_v2_42m_sft_best.pt")


if __name__ == "__main__":
    main()
