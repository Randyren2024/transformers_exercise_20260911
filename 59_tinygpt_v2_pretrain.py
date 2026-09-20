import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer


SEED = 59
VOCAB = 8000
D = 256
H = 8
LAYERS = 8
FF = 1024
CTX = 256

BATCH = 64
STEPS = 8000
LR = 3e-4
MIN_LR = 3e-5
WARMUP = 400
WEIGHT_DECAY = 0.1

LOG_INTERVAL = 100
EVAL_INTERVAL = 500
SAVE_INTERVAL = 1000

CORPUS_DIR = Path(
    "/content/drive/MyDrive/transformers_exercise_20260911/"
    "data/step58_v2_corpus"
)
TOKENIZER_PATH = Path(
    "/content/drive/MyDrive/transformers_exercise_20260911/"
    "artifacts/step43/step43_bpe_8000.json"
)
OUT_DIR = Path(
    "/content/drive/MyDrive/transformers_exercise_20260911/"
    "artifacts/step59"
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, y=None):
        t = x.size(1)
        pos = torch.arange(t, device=x.device)
        h = self.tok(x) + self.pos(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln_f(h))
        loss = None
        if y is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB),
                y.reshape(-1),
            )
        return logits, loss


def iter_documents(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                yield text


def build_token_store(tokenizer, text_path, output_path):
    """
    Two-pass document tokenization.

    uint16 is sufficient because vocab=8000.
    A flat token store allows random context sampling without loading
    the entire corpus into GPU memory.
    """
    print(f"Tokenizing: {text_path}")

    docs = 0
    tokens = 0
    batch = []

    # Pass 1: count tokens.
    for text in iter_documents(text_path):
        batch.append(text + "\n\n")
        if len(batch) >= 256:
            encodings = tokenizer.encode_batch(batch)
            tokens += sum(len(e.ids) for e in encodings)
            docs += len(batch)
            batch = []

    if batch:
        encodings = tokenizer.encode_batch(batch)
        tokens += sum(len(e.ids) for e in encodings)
        docs += len(batch)

    print(f"Documents: {docs:,}")
    print(f"Tokens:    {tokens:,}")

    mmap = np.memmap(
        output_path,
        dtype=np.uint16,
        mode="w+",
        shape=(tokens,),
    )

    offset = 0
    batch = []

    # Pass 2: write tokens.
    for text in iter_documents(text_path):
        batch.append(text + "\n\n")
        if len(batch) >= 256:
            encodings = tokenizer.encode_batch(batch)
            for enc in encodings:
                ids = np.asarray(enc.ids, dtype=np.uint16)
                mmap[offset:offset + len(ids)] = ids
                offset += len(ids)
            batch = []

    if batch:
        encodings = tokenizer.encode_batch(batch)
        for enc in encodings:
            ids = np.asarray(enc.ids, dtype=np.uint16)
            mmap[offset:offset + len(ids)] = ids
            offset += len(ids)

    mmap.flush()
    del mmap

    return tokens, docs


def load_store(path):
    mmap = np.memmap(path, dtype=np.uint16, mode="r")
    return mmap


def get_batch(store, batch_size, context, device):
    max_start = len(store) - context - 1
    starts = np.random.randint(0, max_start, size=batch_size)

    x = np.stack(
        [np.asarray(store[s:s + context], dtype=np.int64) for s in starts]
    )
    y = np.stack(
        [np.asarray(store[s + 1:s + context + 1], dtype=np.int64) for s in starts]
    )

    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(y).to(device),
    )


def lr_at(step):
    if step <= WARMUP:
        return LR * step / WARMUP
    p = min(max((step - WARMUP) / max(1, STEPS - WARMUP), 0.0), 1.0)
    return MIN_LR + (LR - MIN_LR) * 0.5 * (1 + math.cos(math.pi * p))


@torch.inference_mode()
def evaluate(model, store, device, batches=8):
    model.eval()
    losses = []

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        for _ in range(batches):
            x, y = get_batch(store, BATCH, CTX, device)
            _, loss = model(x, y)
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    args, _ = parser.parse_known_args()

    seed_all(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 112)
    print("Node 59 — TinyGPT v2 pretraining")
    print("=" * 112)
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_text = CORPUS_DIR / "step58_train.txt"
    val_text = CORPUS_DIR / "step58_validation.txt"

    if not train_text.exists():
        raise FileNotFoundError(train_text)
    if not val_text.exists():
        raise FileNotFoundError(val_text)
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(TOKENIZER_PATH)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    train_bin = CORPUS_DIR / "step58_train_ids.uint16"
    val_bin = CORPUS_DIR / "step58_validation_ids.uint16"

    if not train_bin.exists():
        train_tokens, train_docs = build_token_store(
            tokenizer,
            train_text,
            train_bin,
        )
    else:
        train_tokens = train_bin.stat().st_size // 2
        train_docs = None

    if not val_bin.exists():
        val_tokens, val_docs = build_token_store(
            tokenizer,
            val_text,
            val_bin,
        )
    else:
        val_tokens = val_bin.stat().st_size // 2
        val_docs = None

    print("\nToken store")
    print("-" * 112)
    print(f"Train tokens:  {train_tokens:,}")
    print(f"Val tokens:    {val_tokens:,}")
    print(f"Train steps of corpus: ~{train_tokens / (BATCH * CTX):,.0f}")
    print(f"Planned steps:  {args.steps:,}")

    train_store = load_store(train_bin)
    val_store = load_store(val_bin)

    model = TinyGPT().to(device)
    params = sum(p.numel() for p in model.parameters())

    print(f"\nParameters:     {params:,}")
    print(f"Architecture:   D={D}, layers={LAYERS}, heads={H}, FFN={FF}")
    print(f"Context:        {CTX}")
    print(f"Batch:          {BATCH}")
    print(f"Tokens/step:    {BATCH * CTX:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    best_val = float("inf")
    tick = time.perf_counter()

    for step in range(1, args.steps + 1):
        lr = lr_at(step)
        optimizer.param_groups[0]["lr"] = lr

        x, y = get_batch(train_store, BATCH, CTX, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % LOG_INTERVAL == 0 or step == args.steps:
            speed = step * BATCH * CTX / max(time.perf_counter() - tick, 1e-6)
            print(
                f"step {step:>5}/{args.steps} | "
                f"loss {loss.item():.4f} | "
                f"lr {lr:.2e} | "
                f"{speed:,.0f} tok/s"
            )

        if step % EVAL_INTERVAL == 0 or step == args.steps:
            val_loss = evaluate(model, val_store, device)
            print(f"           validation loss: {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "step": step,
                        "validation_loss": best_val,
                        "params": params,
                        "tokenizer_path": str(TOKENIZER_PATH),
                        "config": {
                            "vocab": VOCAB,
                            "d_model": D,
                            "heads": H,
                            "layers": LAYERS,
                            "ffn": FF,
                            "context": CTX,
                        },
                        "corpus": {
                            "train_tokens": int(train_tokens),
                            "validation_tokens": int(val_tokens),
                        },
                    },
                    OUT_DIR / "tiny_gpt_v2_best.pt",
                )
                print(f"           new best checkpoint: {best_val:.4f}")

        if step % SAVE_INTERVAL == 0 or step == args.steps:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "step": step,
                    "validation_loss": best_val,
                    "params": params,
                    "config": {
                        "vocab": VOCAB,
                        "d_model": D,
                        "heads": H,
                        "layers": LAYERS,
                        "ffn": FF,
                        "context": CTX,
                    },
                },
                OUT_DIR / f"tiny_gpt_v2_step_{step}.pt",
            )

    with open(OUT_DIR / "step59_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "steps": args.steps,
                "best_validation_loss": best_val,
                "params": params,
                "train_tokens": int(train_tokens),
                "validation_tokens": int(val_tokens),
                "tokens_per_step": BATCH * CTX,
                "architecture": {
                    "vocab": VOCAB,
                    "d_model": D,
                    "heads": H,
                    "layers": LAYERS,
                    "ffn": FF,
                    "context": CTX,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 112)
    print("Node 59 complete")
    print("=" * 112)
    print("Best checkpoint:", OUT_DIR / "tiny_gpt_v2_best.pt")


if __name__ == "__main__":
    main()
