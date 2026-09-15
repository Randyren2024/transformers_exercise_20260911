import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer

# Step 47: Small bilingual instruction tuning (SFT)
# English: OpenAssistant/oasst1 (Apache-2.0)
# Chinese: Mxode/Chinese-Instruct / coig-cqia (CC-BY-SA-4.0)

SEED = 42
VOCAB_SIZE = 8000
D_MODEL = 192
N_HEADS = 6
N_LAYERS = 6
D_FF = 768
CONTEXT_LENGTH = 256
BATCH_SIZE = 64
DEFAULT_STEPS = 3000
LEARNING_RATE = 3e-5
MIN_LEARNING_RATE = 3e-6
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 150
EVAL_INTERVAL = 250
LOG_INTERVAL = 25
MAX_EXAMPLES_PER_LANGUAGE = 8000


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_text(row, *names):
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def chinese_ratio(text):
    zh = sum("\u4e00" <= c <= "\u9fff" for c in text)
    latin = sum(("A" <= c <= "Z") or ("a" <= c <= "z") for c in text)
    total = zh + latin
    return zh / total if total else 0.0


def clean_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text


def valid_pair(user, assistant, lang):
    user, assistant = clean_text(user), clean_text(assistant)
    if len(user) < 5 or len(assistant) < 5:
        return False
    if len(user) > 900 or len(assistant) > 1200:
        return False
    if "<html" in assistant.lower() or "</div>" in assistant.lower():
        return False
    if lang == "zh":
        return chinese_ratio(user + assistant) >= 0.35
    if lang == "en":
        return chinese_ratio(user + assistant) <= 0.20
    return True


def collect_oasst(max_examples):
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    rows = []
    by_id = {}
    for row in ds:
        mid = row.get("message_id")
        if mid:
            by_id[mid] = row
    for row in ds:
        role = str(row.get("role", "")).lower()
        parent_id = row.get("parent_id")
        if role != "assistant" or not parent_id:
            continue
        parent = by_id.get(parent_id)
        if not parent:
            continue
        parent_role = str(parent.get("role", "")).lower()
        if parent_role not in {"prompter", "user"}:
            continue
        user = pick_text(parent, "text")
        assistant = pick_text(row, "text")
        lang = str(row.get("lang", "en")).lower()
        if not lang.startswith("en"):
            continue
        if valid_pair(user, assistant, "en"):
            rows.append({"user": user, "assistant": assistant, "lang": "en"})
            if len(rows) >= max_examples:
                break
    return rows


def collect_coig(max_examples):
    # Mxode/Chinese-Instruct exposes each subset as a dataset config.
    # coig-cqia examples use instruction/input/output fields.
    ds = load_dataset("Mxode/Chinese-Instruct", "coig-cqia", split="train", streaming=True)
    rows = []
    for row in ds:
        instruction = pick_text(row, "instruction", "prompt", "question")
        extra = pick_text(row, "input")
        user = instruction
        if extra:
            user = f"{instruction}\n{extra}" if instruction else extra
        assistant = pick_text(row, "output", "response", "answer")
        if valid_pair(user, assistant, "zh"):
            rows.append({"user": user, "assistant": assistant, "lang": "zh"})
            if len(rows) >= max_examples:
                break
    return rows


def build_examples():
    print("Loading English instruction data from OpenAssistant/oasst1 ...")
    en = collect_oasst(MAX_EXAMPLES_PER_LANGUAGE)
    print(f"English pairs: {len(en):,}")
    print("Loading Chinese instruction data from Mxode/Chinese-Instruct/coig-cqia ...")
    zh = collect_coig(MAX_EXAMPLES_PER_LANGUAGE)
    print(f"Chinese pairs: {len(zh):,}")
    if not en or not zh:
        raise RuntimeError(
            f"SFT dataset is incomplete: English={len(en):,}, Chinese={len(zh):,}."
        )
    # Keep the two languages balanced instead of letting English dominate.
    n = min(len(en), len(zh))
    en = en[:n]
    zh = zh[:n]
    all_rows = en + zh
    random.shuffle(all_rows)
    split = max(1, int(len(all_rows) * 0.95))
    return all_rows[:split], all_rows[split:]


def make_sequence(tok, row):
    # Keep the full prompt in the context, but train only on Assistant tokens.
    prefix = f"User: {row['user']}\nAssistant:"
    full = prefix + " " + row["assistant"]
    prefix_ids = tok.encode(prefix).ids
    ids = tok.encode(full).ids
    if len(ids) > CONTEXT_LENGTH:
        ids = ids[:CONTEXT_LENGTH]
    response_start = min(len(prefix_ids), len(ids))
    return ids, response_start


def make_tensors(tok, rows):
    xs, ys = [], []
    for row in rows:
        ids, response_start = make_sequence(tok, row)
        if len(ids) < response_start + 2:
            continue
        x = ids[:-1]
        y = ids[1:]
        # Ignore prompt tokens. Only predict the answer portion.
        prompt_target_count = max(0, response_start - 1)
        y[:prompt_target_count] = [-100] * prompt_target_count
        pad = CONTEXT_LENGTH - len(x)
        if pad > 0:
            x = x + [0] * pad
            y = y + [-100] * pad
        xs.append(torch.tensor(x, dtype=torch.long))
        ys.append(torch.tensor(y, dtype=torch.long))
    if not xs:
        raise RuntimeError("No usable SFT examples remained after tokenization.")
    return torch.stack(xs), torch.stack(ys)


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        hd = D_MODEL // N_HEADS
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.out = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.hd = hd

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, N_HEADS, self.hd).transpose(1, 2)
        k = k.view(b, t, N_HEADS, self.hd).transpose(1, 2)
        v = v.view(b, t, N_HEADS, self.hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, D_MODEL)
        return self.out(y)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = Attn()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, D_FF, bias=False),
            nn.GELU(),
            nn.Linear(D_FF, D_MODEL, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Embedding(CONTEXT_LENGTH, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, x, y=None):
        t = x.size(1)
        p = torch.arange(t, device=x.device)
        h = self.tok(x) + self.pos(p)[None, :, :]
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln(h))
        loss = None
        if y is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, x, max_new_tokens=100, temperature=0.7, top_k=40):
        self.eval()
        start_len = x.size(1)
        for _ in range(max_new_tokens):
            ctx = x[:, -CONTEXT_LENGTH:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            x = torch.cat([x, nxt], dim=1)
        return x, start_len


def lr_at(step, total):
    if step <= WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS
    p = min(max((step - WARMUP_STEPS) / max(1, total - WARMUP_STEPS), 0), 1)
    return MIN_LEARNING_RATE + (LEARNING_RATE - MIN_LEARNING_RATE) * 0.5 * (
        1 + torch.cos(torch.tensor(torch.pi * p)).item()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    args, _ = parser.parse_known_args()
    seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 112)
    print("Step 47: Small bilingual instruction tuning (fixed)")
    print("=" * 112)
    print(f"device:              {device}")
    if device.type == "cuda":
        print(f"GPU:                 {torch.cuda.get_device_name(0)}")

    drive_root = Path("/content/drive/MyDrive/transformers_exercise_20260911")
    ckpt_path = drive_root / "artifacts" / "step45" / "tiny_gpt_step45.pt"
    tok_candidates = [drive_root / "artifacts" / "step43" / "step43_bpe_8000.json"]
    tok_candidates.append(drive_root / "artifacts" / "step45" / "step45_bpe_8000.json")
    tok_path = next((p for p in tok_candidates if p.exists()), None)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if tok_path is None:
        raise FileNotFoundError("Step 43B/45 tokenizer was not found in Drive.")

    tok = Tokenizer.from_file(str(tok_path))
    train_rows, val_rows = build_examples()
    train_x, train_y = make_tensors(tok, train_rows)
    val_x, val_y = make_tensors(tok, val_rows)

    print("\nPart 1: SFT dataset")
    print("-" * 112)
    print(f"Train examples:      {len(train_x):,}")
    print(f"Validation examples: {len(val_x):,}")
    print(f"English / Chinese:   {sum(r['lang']=='en' for r in train_rows):,} / {sum(r['lang']=='zh' for r in train_rows):,}")

    model = TinyGPT().to(device)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    model.train()
    n = len(train_x)
    tick = time.perf_counter()
    for step in range(1, args.steps + 1):
        lr = lr_at(step, args.steps)
        for g in optimizer.param_groups:
            g["lr"] = lr
        idx = torch.randint(0, n, (BATCH_SIZE,))
        x = train_x[idx].to(device)
        y = train_y[idx].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % LOG_INTERVAL == 0 or step == args.steps:
            speed = (step * BATCH_SIZE * CONTEXT_LENGTH) / max(
                time.perf_counter() - tick, 1e-6
            )
            print(
                f"step {step:>5}/{args.steps} | loss {loss.item():.4f} | "
                f"lr {lr:.2e} | {speed:,.0f} tok/s"
            )
        if step % EVAL_INTERVAL == 0 or step == args.steps:
            model.eval()
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                eval_idx = torch.arange(min(len(val_x), BATCH_SIZE), device="cpu")
                _, vloss = model(
                    val_x[eval_idx].to(device),
                    val_y[eval_idx].to(device),
                )
            model.train()
            print(f"           validation loss: {vloss.item():.4f}")

    out_dir = drive_root / "artifacts" / "step47"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = out_dir / "tiny_gpt_step47_sft_fixed.pt"
    out_tok = out_dir / "step47_tokenizer.json"
    tok.save(str(out_tok))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "tokenizer_path": str(out_tok),
            "base_checkpoint": str(ckpt_path),
            "step": args.steps,
            "train_examples": len(train_x),
            "validation_examples": len(val_x),
        },
        out_ckpt,
    )

    print("\nPart 2: Instruction probes")
    prompts = [
        "User: Explain what artificial intelligence is in simple terms.\nAssistant:",
        "User: 请用简单中文解释什么是人工智能。\nAssistant:",
        "User: 你好，请介绍一下你自己。\nAssistant:",
        "User: What is a transformer model?\nAssistant:",
    ]
    model.eval()
    for prompt in prompts:
        ids = tok.encode(prompt).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out, start_len = model.generate(x)
        generated = tok.decode(out[0].tolist()[start_len:])
        print(f"\n{prompt}\nAssistant: {generated}")

    print("\nStep 47 fixed complete.")
    print(f"Checkpoint: {out_ckpt}")


if __name__ == "__main__":
    main()
