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
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from tokenizers.trainers import BpeTrainer
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    print("Install with: pip install datasets tokenizers")
    raise SystemExit(1)

SEED = 42
VOCAB_SIZE = 8_000
D_MODEL = 192
N_HEADS = 6
N_LAYERS = 6
D_FF = 768
CONTEXT_LENGTH = 256
DROPOUT = 0.0
BATCH_SIZE_GPU = 64
BATCH_SIZE_CPU = 8
DEFAULT_TOTAL_CHARS = 10_000_000
TOKENIZER_TRAIN_CHARS = 2_000_000
DEFAULT_STEPS = 5_000
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_STEPS = 200
EVAL_INTERVAL = 250
EVAL_BATCHES = 20
LOG_INTERVAL = 25
SAVE_INTERVAL = 500

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
    if any(p in text.lower() for p in ["enable javascript", "cookie settings", "all rights reserved", "privacy policy |"]):
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


def stream_docs(dataset_name, lang, target_chars, max_rows=100_000):
    ds = load_dataset(dataset_name, split="train", streaming=True, **({"name": ZH_CONFIG} if lang == "zh" else {}))
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
    print(f"{lang.upper()}: inspected={inspected:,} accepted={len(docs):,} chars={chars:,} rejected={rejected:,}")
    return docs


def split_docs(docs, val_ratio, rng):
    docs = list(docs)
    rng.shuffle(docs)
    n = max(1, int(len(docs) * val_ratio))
    return docs[n:], docs[:n]


def build_bilingual(total_chars, seed):
    en_target = int(total_chars * 0.70)
    zh_target = total_chars - en_target
    rng = random.Random(seed)
    en = stream_docs(EN_DATASET, "en", en_target)
    zh = stream_docs(ZH_DATASET, "zh", zh_target)
    en_train, en_val = split_docs(en, 0.10, rng)
    zh_train, zh_val = split_docs(zh, 0.10, rng)
    train_docs = en_train + zh_train
    val_docs = en_val + zh_val
    rng.shuffle(train_docs)
    rng.shuffle(val_docs)
    return train_docs, val_docs


def train_tokenizer(tokenizer_text, out_path):
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tmp = out_path.with_suffix(".txt")
    tmp.write_text("\n\n".join(tokenizer_text), encoding="utf-8")
    tok.train([str(tmp)], trainer=trainer)
    tok.save(str(out_path))
    tmp.unlink(missing_ok=True)
    return tok


def encode_docs(tok, docs):
    text = "\n\n".join(docs)
    ids = tok.encode(text).ids
    return torch.tensor(ids, dtype=torch.long), len(text)


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
        y = y.transpose(1, 2).contiguous().view(b, t, D_MODEL)
        return self.out(y)


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
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

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
    x = torch.stack([tokens[int(s): int(s) + CONTEXT_LENGTH] for s in starts])
    y = torch.stack([tokens[int(s) + 1: int(s) + CONTEXT_LENGTH + 1] for s in starts])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def eval_loss(model, tokens, batch_size, device, batches, amp):
    model.eval()
    vals = []
    for _ in range(batches):
        x, y = get_batch(tokens, batch_size, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            _, loss = model(x, y)
        vals.append(loss.item())
    model.train()
    return sum(vals) / len(vals)


def lr_at(step, total):
    if step <= WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS
    p = min(max((step - WARMUP_STEPS) / max(1, total - WARMUP_STEPS), 0), 1)
    return MIN_LEARNING_RATE + (LEARNING_RATE - MIN_LEARNING_RATE) * 0.5 * (1 + math.cos(math.pi * p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--total-chars", type=int, default=DEFAULT_TOTAL_CHARS)
    args = parser.parse_args()

    seed_everything(SEED)
    drive_root = Path("/content/drive/MyDrive/transformers_exercise_20260911")
    drive_root.mkdir(parents=True, exist_ok=True)
    data_dir = drive_root / "data" / "step43"
    art_dir = drive_root / "artifacts" / "step43"
    data_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    batch_size = BATCH_SIZE_GPU if amp else BATCH_SIZE_CPU

    print("=" * 112)
    print("Step 43B: Self-contained Colab T4 pretraining")
    print("=" * 112)
    print(f"device:              {device}")
    if amp:
        print(f"GPU:                 {torch.cuda.get_device_name(0)}")
    print(f"target chars:        {args.total_chars:,}")
    print(f"tokenizer train chars: {TOKENIZER_TRAIN_CHARS:,}")
    print(f"steps:               {args.steps:,}")
    print(f"batch size:          {batch_size}")

    print("\nPart 1: Build fresh bilingual corpus")
    train_docs, val_docs = build_bilingual(args.total_chars, SEED)
    rng = random.Random(SEED)
    tokenizer_pool = list(train_docs)
    rng.shuffle(tokenizer_pool)
    selected = []
    chars = 0
    for doc in tokenizer_pool:
        if chars >= TOKENIZER_TRAIN_CHARS:
            break
        selected.append(doc)
        chars += len(doc)

    tok_path = art_dir / "step43_bpe_8000.json"
    print(f"Tokenizer sample chars: {chars:,}")
    tokenizer = train_tokenizer(selected, tok_path)

    print("\nPart 2: Encode corpus")
    train_tokens, train_chars = encode_docs(tokenizer, train_docs)
    val_tokens, val_chars = encode_docs(tokenizer, val_docs)
    torch.save(train_tokens, data_dir / "train_ids.pt")
    torch.save(val_tokens, data_dir / "val_ids.pt")
    print(f"Train chars:          {train_chars:,}")
    print(f"Validation chars:     {val_chars:,}")
    print(f"Train tokens:         {len(train_tokens):,}")
    print(f"Validation tokens:    {len(val_tokens):,}")

    model = TinyGPT().to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    print("\nPart 3: Model")
    print(f"Parameters:           {params:,}")
    print(f"vocab:                {VOCAB_SIZE:,}")
    print(f"d_model/heads/layers: {D_MODEL}/{N_HEADS}/{N_LAYERS}")
    print(f"context:              {CONTEXT_LENGTH}")
    print(f"tokens/step:          {batch_size * CONTEXT_LENGTH:,}")

    initial_train = eval_loss(model, train_tokens, batch_size, device, 5, amp)
    initial_val = eval_loss(model, val_tokens, batch_size, device, 5, amp)
    print(f"Initial train loss:   {initial_train:.4f}")
    print(f"Initial val loss:     {initial_val:.4f}")

    print("\nPart 4: Training")
    tokens_seen = 0
    latest_train, latest_val = initial_train, initial_val
    tick = time.perf_counter()
    last_log = 0
    model.train()
    for step in range(1, args.steps + 1):
        lr = lr_at(step, args.steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(train_tokens, batch_size, device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
        latest_train = loss.item()
        tokens_seen += batch_size * CONTEXT_LENGTH

        if step == 1 or step % LOG_INTERVAL == 0 or step == args.steps:
            elapsed = max(time.perf_counter() - tick, 1e-6)
            speed = (step - last_log) * batch_size * CONTEXT_LENGTH / elapsed
            tick = time.perf_counter(); last_log = step
            print(f"step {step:>5}/{args.steps} | loss {latest_train:.4f} | lr {lr:.2e} | {speed:,.0f} tok/s")

        if step % EVAL_INTERVAL == 0 or step == args.steps:
            latest_val = eval_loss(model, val_tokens, batch_size, device, EVAL_BATCHES, amp)
            print(f"           validation loss: {latest_val:.4f}")

        if step % SAVE_INTERVAL == 0 or step == args.steps:
            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scaler_state_dict": scaler.state_dict() if amp else None,
                "step": step,
                "train_loss": latest_train,
                "val_loss": latest_val,
                "tokens_seen": tokens_seen,
                "parameter_count": params,
                "vocab_size": VOCAB_SIZE,
                "d_model": D_MODEL,
                "n_heads": N_HEADS,
                "n_layers": N_LAYERS,
                "d_ff": D_FF,
                "context_length": CONTEXT_LENGTH,
                "tokenizer_path": str(tok_path),
            }
            torch.save(ckpt, art_dir / "tiny_gpt_step43b.pt")
            print(f"           checkpoint saved: {art_dir / 'tiny_gpt_step43b.pt'}")

    summary = {
        "steps": args.steps,
        "tokens_seen": tokens_seen,
        "parameter_count": params,
        "train_tokens": len(train_tokens),
        "validation_tokens": len(val_tokens),
        "initial_train_loss": initial_train,
        "initial_val_loss": initial_val,
        "final_train_loss": latest_train,
        "final_val_loss": latest_val,
        "tokenizer_path": str(tok_path),
        "checkpoint_path": str(art_dir / "tiny_gpt_step43b.pt"),
    }
    (art_dir / "step43b_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nStep 43B complete.")


if __name__ == "__main__":
    main()
