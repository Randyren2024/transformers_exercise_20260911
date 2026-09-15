import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from datasets import load_dataset
    from tokenizers import Tokenizer
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    raise SystemExit(1)

SEED = 42
VOCAB_SIZE = 8_000
D_MODEL = 192
N_HEADS = 6
N_LAYERS = 6
D_FF = 768
CONTEXT_LENGTH = 256
BATCH_SIZE = 64
DEFAULT_TOTAL_CHARS = 30_000_000
DEFAULT_STEPS = 10_000
LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_STEPS = 500
EVAL_INTERVAL = 500
EVAL_BATCHES = 20
LOG_INTERVAL = 50
SAVE_INTERVAL = 1_000
EN_DATASET = "HuggingFaceFW/fineweb"
ZH_DATASET = "HuggingFaceFW/fineweb-2"
ZH_CONFIG = "cmn_Hani"


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize(text):
    text = str(text).replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def chinese_ratio(text):
    zh = sum("\u4e00" <= c <= "\u9fff" for c in text)
    en = sum(("A" <= c <= "Z") or ("a" <= c <= "z") for c in text)
    total = zh + en
    return zh / total if total else 0.0


def acceptable(text, lang):
    if len(text) < 250:
        return False
    lower = text.lower()
    if any(p in lower for p in ["enable javascript", "cookie settings", "all rights reserved", "privacy policy |"]):
        return False
    alpha = sum(c.isalpha() for c in text)
    if alpha / max(len(text), 1) < 0.20:
        return False
    ratio = chinese_ratio(text)
    if lang == "zh" and ratio < 0.30:
        return False
    if lang == "en" and ratio > 0.15:
        return False
    return True


def stream_docs(dataset_name, lang, target_chars, max_rows=200_000):
    kwargs = {"name": ZH_CONFIG} if lang == "zh" else {}
    ds = load_dataset(dataset_name, split="train", streaming=True, **kwargs)
    docs, chars, inspected, rejected, seen = [], 0, 0, 0, set()
    for row in ds:
        inspected += 1
        if inspected > max_rows or chars >= target_chars:
            break
        text = normalize(row.get("text", ""))
        if not text or not acceptable(text, lang):
            rejected += 1
            continue
        key = hash(text)
        if key in seen:
            rejected += 1
            continue
        seen.add(key)
        remaining = target_chars - chars
        if len(text) > remaining:
            text = text[:remaining]
        if len(text) < 250:
            break
        docs.append(text)
        chars += len(text)
        if len(docs) % 1_000 == 0:
            print(f"  {lang.upper()} accepted={len(docs):,} chars={chars:,}")
    print(f"{lang.upper()}: inspected={inspected:,} accepted={len(docs):,} chars={chars:,} rejected={rejected:,}")
    return docs


def split_docs(docs, val_ratio, rng):
    docs = list(docs)
    rng.shuffle(docs)
    n = max(1, int(len(docs) * val_ratio))
    return docs[n:], docs[:n]


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_dim = D_MODEL // N_HEADS
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.out = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.head_dim = head_dim

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, N_HEADS, self.head_dim).transpose(1, 2)
        k = k.view(b, t, N_HEADS, self.head_dim).transpose(1, 2)
        v = v.view(b, t, N_HEADS, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).contiguous().view(b, t, D_MODEL))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, D_FF, bias=False), nn.GELU(), nn.Linear(D_FF, D_MODEL, bias=False)
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
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
        return logits, loss


def get_batch(tokens, batch_size, device):
    max_start = len(tokens) - CONTEXT_LENGTH - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([tokens[int(s):int(s)+CONTEXT_LENGTH] for s in starts])
    y = torch.stack([tokens[int(s)+1:int(s)+CONTEXT_LENGTH+1] for s in starts])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def eval_loss(model, tokens, device, amp, batches=20):
    model.eval(); vals = []
    for _ in range(batches):
        x, y = get_batch(tokens, BATCH_SIZE, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            _, loss = model(x, y)
        vals.append(loss.item())
    model.train()
    return sum(vals) / len(vals)


def lr_at(step):
    if step <= WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS
    p = min(max((step - WARMUP_STEPS) / max(1, DEFAULT_STEPS - WARMUP_STEPS), 0), 1)
    return MIN_LEARNING_RATE + (LEARNING_RATE - MIN_LEARNING_RATE) * 0.5 * (1 + math.cos(math.pi * p))


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--total-chars", type=int, default=DEFAULT_TOTAL_CHARS)
    args, _ = parser.parse_known_args()
    seed_everything(SEED)

    drive_root = Path("/content/drive/MyDrive/transformers_exercise_20260911")
    src_ckpt = drive_root / "artifacts/step43/tiny_gpt_step43b.pt"
    tokenizer_path = drive_root / "artifacts/step43/step43_bpe_8000.json"
    out_dir = drive_root / "artifacts/step45"
    data_dir = drive_root / "data/step45"
    out_dir.mkdir(parents=True, exist_ok=True); data_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    print("=" * 112)
    print("Step 45: Larger-corpus continuation pretraining")
    print("=" * 112)
    print(f"device: {device}")
    if amp: print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"target chars: {args.total_chars:,}")
    print(f"steps: {args.steps:,}")

    if not src_ckpt.exists():
        raise FileNotFoundError(src_ckpt)
    if not tokenizer_path.exists():
        raise FileNotFoundError(tokenizer_path)

    meta = torch.load(src_ckpt, map_location="cpu")
    model = TinyGPT()
    model.load_state_dict(meta["model_state_dict"])
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    print("\nPart 1: Build larger fresh corpus")
    rng = random.Random(SEED)
    en_target = int(args.total_chars * 0.70); zh_target = args.total_chars - en_target
    en = stream_docs(EN_DATASET, "en", en_target); zh = stream_docs(ZH_DATASET, "zh", zh_target)
    en_train, en_val = split_docs(en, 0.10, rng); zh_train, zh_val = split_docs(zh, 0.10, rng)
    train_docs = en_train + zh_train; val_docs = en_val + zh_val
    rng.shuffle(train_docs); rng.shuffle(val_docs)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    train_text = "\n\n".join(train_docs); val_text = "\n\n".join(val_docs)
    train_tokens = torch.tensor(tokenizer.encode(train_text).ids, dtype=torch.long)
    val_tokens = torch.tensor(tokenizer.encode(val_text).ids, dtype=torch.long)
    torch.save(train_tokens, data_dir / "train_ids.pt"); torch.save(val_tokens, data_dir / "val_ids.pt")
    print(f"Train chars: {len(train_text):,}")
    print(f"Validation chars: {len(val_text):,}")
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Validation tokens: {len(val_tokens):,}")

    print("\nPart 2: Resume and train")
    start_seen = int(meta.get("tokens_seen", 0))
    print(f"Starting from Step 43 tokens seen: {start_seen:,}")
    initial_val = eval_loss(model, val_tokens, device, amp)
    print(f"Initial validation loss on new corpus: {initial_val:.4f}")
    timer = time.perf_counter(); last_log = 0
    for step in range(1, args.steps + 1):
        lr = LEARNING_RATE * (step / WARMUP_STEPS) if step <= WARMUP_STEPS else MIN_LEARNING_RATE + (LEARNING_RATE - MIN_LEARNING_RATE) * 0.5 * (1 + math.cos(math.pi * min(1, (step-WARMUP_STEPS)/max(1,args.steps-WARMUP_STEPS))))
        for g in opt.param_groups: g["lr"] = lr
        x, y = get_batch(train_tokens, BATCH_SIZE, device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            _, loss = model(x, y)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt); scaler.update()
        if step == 1 or step % LOG_INTERVAL == 0 or step == args.steps:
            elapsed = max(time.perf_counter()-timer, 1e-6); speed=(step-last_log)*BATCH_SIZE*CONTEXT_LENGTH/elapsed
            timer=time.perf_counter(); last_log=step
            print(f"step {step:>5}/{args.steps} | loss {loss.item():.4f} | lr {lr:.2e} | {speed:,.0f} tok/s")
        if step % EVAL_INTERVAL == 0 or step == args.steps:
            val_loss = eval_loss(model, val_tokens, device, amp)
            print(f"           validation loss: {val_loss:.4f}")
        if step % SAVE_INTERVAL == 0 or step == args.steps:
            ckpt = {"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(), "scaler_state_dict": scaler.state_dict() if amp else None, "step": step, "tokens_seen": start_seen + step*BATCH_SIZE*CONTEXT_LENGTH, "source_checkpoint": str(src_ckpt), "parameter_count": sum(p.numel() for p in model.parameters()), "vocab_size": VOCAB_SIZE, "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS, "d_ff": D_FF, "context_length": CONTEXT_LENGTH, "train_tokens": len(train_tokens), "validation_tokens": len(val_tokens), "val_loss": val_loss if 'val_loss' in locals() else None}
            torch.save(ckpt, out_dir / "tiny_gpt_step45.pt")
            print(f"           checkpoint saved: {out_dir / 'tiny_gpt_step45.pt'}")

    print("\nStep 45 complete.")


if __name__ == "__main__":
    main()
