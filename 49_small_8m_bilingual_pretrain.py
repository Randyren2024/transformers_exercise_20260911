import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer

# Step 49: ~8.4M parameter bilingual GPT
# Fresh pretraining from scratch on the same 30M-character bilingual recipe used in Step 45.
# Goal: test whether model capacity, rather than SFT data alone, is the main bottleneck.

SEED = 49
VOCAB_SIZE = 8000
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 8
D_FF = 1024
CONTEXT_LENGTH = 256
BATCH_SIZE = 64
DEFAULT_STEPS = 12000
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WARMUP_STEPS = 500
EVAL_INTERVAL = 500
SAVE_INTERVAL = 2000
LOG_INTERVAL = 100
TARGET_CHARS = 30_000_000
EN_CHARS = 21_000_000
ZH_CHARS = 9_000_000


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_text(row):
    for k in ("text", "content"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def clean(text):
    return " ".join(text.split())


def build_corpus():
    print("\nPart 1: Build 30M-character bilingual corpus")
    print("-" * 112)
    en = []
    zh = []
    en_total = zh_total = 0

    print("Loading English FineWeb ...")
    ds_en = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
    for row in ds_en:
        text = clean(pick_text(row))
        if len(text) < 200:
            continue
        en.append(text)
        en_total += len(text)
        if en_total >= EN_CHARS:
            break

    print("Loading Chinese FineWeb2 (cmn_Hani) ...")
    ds_zh = load_dataset("HuggingFaceFW/fineweb-2", "cmn_Hani", split="train", streaming=True)
    for row in ds_zh:
        text = clean(pick_text(row))
        if len(text) < 120:
            continue
        zh.append(text)
        zh_total += len(text)
        if zh_total >= ZH_CHARS:
            break

    random.shuffle(en)
    random.shuffle(zh)
    en_text = "\n\n".join(en)[:EN_CHARS]
    zh_text = "\n\n".join(zh)[:ZH_CHARS]
    master = en_text + "\n\n" + zh_text
    cut = int(len(master) * 0.9)
    train_text = master[:cut]
    val_text = master[cut:]

    print(f"English chars:        {len(en_text):,}")
    print(f"Chinese chars:        {len(zh_text):,}")
    print(f"Total chars:          {len(master):,}")
    print(f"Train chars:          {len(train_text):,}")
    print(f"Validation chars:     {len(val_text):,}")
    return train_text, val_text


def encode_text(tok, text):
    return torch.tensor(tok.encode(text).ids, dtype=torch.long)


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
        self.fc1 = nn.Linear(D_MODEL, D_FF, bias=False)
        self.fc2 = nn.Linear(D_FF, D_MODEL, bias=False)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class TinyGPT8M(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Embedding(CONTEXT_LENGTH, D_MODEL)
        self.blocks = nn.ModuleList(Block() for _ in range(N_LAYERS))
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, y=None):
        t = x.size(1)
        p = torch.arange(t, device=x.device)
        h = self.tok(x) + self.pos(p)[None, :, :]
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln_f(h))
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, x, max_new_tokens=80, temperature=0.35, top_k=20, repetition_penalty=1.12):
        self.eval()
        for _ in range(max_new_tokens):
            ctx = x[:, -CONTEXT_LENGTH:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :]
            # Simple repetition control for tiny-model decoding.
            seen = torch.unique(ctx[0]).tolist()
            for token_id in seen:
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty
            logits = logits / max(temperature, 1e-5)
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            x = torch.cat([x, nxt], dim=1)
        return x


def get_batch(data, batch_size, context, device):
    starts = torch.randint(0, len(data) - context - 1, (batch_size,))
    x = torch.stack([data[i : i + context] for i in starts])
    y = torch.stack([data[i + 1 : i + context + 1] for i in starts])
    return x.to(device), y.to(device)


def lr_at(step, total):
    if step <= WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS
    p = min(max((step - WARMUP_STEPS) / max(1, total - WARMUP_STEPS), 0.0), 1.0)
    return MIN_LEARNING_RATE + (LEARNING_RATE - MIN_LEARNING_RATE) * 0.5 * (1 + math.cos(math.pi * p))


def evaluate(model, data, device, batches=8):
    model.eval()
    losses = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        for _ in range(batches):
            x, y = get_batch(data, BATCH_SIZE, CONTEXT_LENGTH, device)
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    args, _ = parser.parse_known_args()
    seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 112)
    print("Step 49: ~8.4M parameter bilingual GPT")
    print("=" * 112)
    print(f"device:              {device}")
    if device.type == "cuda":
        print(f"GPU:                 {torch.cuda.get_device_name(0)}")

    drive_root = Path("/content/drive/MyDrive/transformers_exercise_20260911")
    tok_path = drive_root / "artifacts" / "step43" / "step43_bpe_8000.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")

    tok = Tokenizer.from_file(str(tok_path))
    train_text, val_text = build_corpus()
    train_ids = encode_text(tok, train_text)
    val_ids = encode_text(tok, val_text)
    print(f"Train tokens:         {len(train_ids):,}")
    print(f"Validation tokens:    {len(val_ids):,}")

    model = TinyGPT8M().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters:     {params:,}")
    print(f"d_model / layers:     {D_MODEL} / {N_LAYERS}")
    print(f"heads / FFN:          {N_HEADS} / {D_FF}")
    print(f"tokens / step:        {BATCH_SIZE * CONTEXT_LENGTH:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    out_dir = drive_root / "artifacts" / "step49"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    tick = time.perf_counter()

    for step in range(1, args.steps + 1):
        lr = lr_at(step, args.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(train_ids, BATCH_SIZE, CONTEXT_LENGTH, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % LOG_INTERVAL == 0 or step == args.steps:
            speed = step * BATCH_SIZE * CONTEXT_LENGTH / max(time.perf_counter() - tick, 1e-6)
            print(f"step {step:>5}/{args.steps} | loss {loss.item():.4f} | lr {lr:.2e} | {speed:,.0f} tok/s")

        if step % EVAL_INTERVAL == 0 or step == args.steps:
            vloss = evaluate(model, val_ids, device)
            print(f"           validation loss: {vloss:.4f}")
            if vloss < best_loss:
                best_loss = vloss
                path = out_dir / "tiny_gpt_step49_best.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "step": step,
                    "validation_loss": best_loss,
                    "params": params,
                    "tokenizer_path": str(tok_path),
                    "config": {
                        "vocab_size": VOCAB_SIZE,
                        "d_model": D_MODEL,
                        "n_heads": N_HEADS,
                        "n_layers": N_LAYERS,
                        "d_ff": D_FF,
                        "context_length": CONTEXT_LENGTH,
                    },
                }, path)
                print(f"           new best checkpoint: {best_loss:.4f}")

        if step % SAVE_INTERVAL == 0 or step == args.steps:
            path = out_dir / f"tiny_gpt_step49_{step}.pt"
            torch.save({"model_state_dict": model.state_dict(), "step": step, "validation_loss": best_loss}, path)

    print("\nPart 2: Generation probes")
    prompts = [
        "User: Explain what artificial intelligence is in simple terms.\nAssistant:",
        "User: 请用简单中文解释什么是人工智能。\nAssistant:",
        "User: What is a transformer model?\nAssistant:",
        "User: 什么是 Transformer 模型？\nAssistant:",
        "User: What is 7 + 8?\nAssistant:",
        "User: 12 加 9 等于多少？\nAssistant:",
    ]
    model.eval()
    for prompt in prompts:
        ids = tok.encode(prompt).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(x)
        generated = out[0].tolist()[len(ids):]
        text = tok.decode(generated).strip()
        print(f"\n{prompt}\n{text}")

    meta = {
        "parameters": params,
        "best_validation_loss": best_loss,
        "steps": args.steps,
        "train_tokens": len(train_ids),
        "validation_tokens": len(val_ids),
        "architecture": {"d_model": D_MODEL, "heads": N_HEADS, "layers": N_LAYERS, "ffn": D_FF, "context": CONTEXT_LENGTH},
    }
    with open(out_dir / "step49_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("\nStep 49 complete.")
    print(f"Best checkpoint: {out_dir / 'tiny_gpt_step49_best.pt'}")


if __name__ == "__main__":
    main()
