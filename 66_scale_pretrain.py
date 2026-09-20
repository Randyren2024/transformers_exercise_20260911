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


# Node 66 — Scaled TinyGPT pretraining
#
# Motivation:
#   Step 65 DPO failed to improve rank accuracy because the 8.4M-param
#   model (D=256, 8 layers) lacks the capacity to both maintain language
#   ability AND learn fine-grained preference distinctions.
#
# Scaling plan:
#   D=256 → 512        (hidden dim doubled)
#   LAYERS=8 → 12      (half as many layers again)
#   FF=1024 → 2048     (per-layer FFN doubled, D×4)
#   H=8 unchanged      (per-head dim stays 64, good for T4)
#   CTX=256 unchanged  (keeps memory bounded)
#
# Estimated params: ~42M (5× Step 59 v2 model)
# Corpus:            same step58 v2 (51.7M train tokens, 2.6M val)
# Tokenizer:         same step43 8K BPE
# Training recipe:   same LR cosine as step59, more steps
#
# Expected time on T4: ~3-4 hours for 12000 steps at ~50k tok/s.
# We run via `colab run --timeout 14400` for a dedicated VM.

SEED = 66
VOCAB = 8000
D = 512
H = 8
LAYERS = 12
FF = 2048
CTX = 256

BATCH = 32
STEPS = 12000
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
    "artifacts/step66"
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


def load_store(path):
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(store, batch_size, context, device):
    max_start = len(store) - context - 1
    starts = np.random.randint(0, max_start, size=batch_size)
    x = np.stack(
        [np.asarray(store[s:s + context], dtype=np.int64) for s in starts]
    )
    y = np.stack(
        [np.asarray(store[s + 1:s + context + 1], dtype=np.int64) for s in starts]
    )
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


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


@torch.inference_mode()
def quick_probe(model, tokenizer, device):
    model.eval()
    prompts = [
        "User: What is a transformer?\nAssistant:",
        "User: 李白是谁？\nAssistant:",
        "User: Translate 'software' into Chinese.\nAssistant:",
    ]
    for p in prompts:
        ids = tokenizer.encode(p).ids
        x = torch.tensor([ids[-CTX:]], dtype=torch.long, device=device)
        out = []
        for _ in range(24):
            logits, _ = model(x[:, -CTX:])
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            x = torch.cat([x, nxt], dim=1)
            out.append(int(nxt.item()))
        print(f"  {p[:-10]}... → {tokenizer.decode(out).strip()[:80]}")
    model.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    args, _ = parser.parse_known_args()

    seed_all(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 112)
    print("Node 66 — Scaled TinyGPT pretraining (D=512, L=12, FF=2048)")
    print("=" * 112)
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {props.total_memory / 1024**3:.1f} GB")

    train_bin = CORPUS_DIR / "step58_train_ids.uint16"
    val_bin = CORPUS_DIR / "step58_validation_ids.uint16"

    if not train_bin.exists():
        raise FileNotFoundError(
            f"{train_bin} not found. Run step58 corpus builder first."
        )
    if not val_bin.exists():
        raise FileNotFoundError(val_bin)
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(TOKENIZER_PATH)

    train_tokens = train_bin.stat().st_size // 2
    val_tokens = val_bin.stat().st_size // 2
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    print(f"\nTrain tokens:  {train_tokens:,}")
    print(f"Val tokens:    {val_tokens:,}")
    print(f"Corpus steps: ~{train_tokens / (BATCH * CTX):,.0f}")
    print(f"Planned steps: {args.steps:,}")

    train_store = load_store(train_bin)
    val_store = load_store(val_bin)

    model = TinyGPT().to(device)
    params = sum(p.numel() for p in model.parameters())

    print(f"\nParameters:     {params:,}")
    print(f"Architecture:   D={D}, layers={LAYERS}, heads={H}, FFN={FF}")
    print(f"Context:        {CTX}")
    print(f"Batch:          {BATCH}")
    print(f"Tokens/step:    {BATCH * CTX:,}")
    print(f"LR:             {LR} → {MIN_LR} cosine over {WARMUP} warmup")

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
                quick_probe(model, tokenizer, device)

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

    with open(OUT_DIR / "step66_metadata.json", "w", encoding="utf-8") as f:
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
                "scaling_vs_step59": {
                    "step59_params": 8413696,
                    "step66_params": params,
                    "ratio": params / 8413696,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 112)
    print("Node 66 complete")
    print("=" * 112)
    print("Best checkpoint:", OUT_DIR / "tiny_gpt_v2_best.pt")
    print(f"Params: {params:,} | Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
