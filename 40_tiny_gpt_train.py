import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tokenizers import Tokenizer
except ImportError:
    print("The 'tokenizers' package is not installed.")
    print("Install it with: pip install tokenizers")
    sys.exit(1)


# ============================================================
# Step 40: First real TinyGPT training run
#
# Goal:
#   Turn the Step 39 tokenizer + Step 38 real bilingual corpus into
#   an actual small decoder-only language model.
#
# This is intentionally a conservative first model:
#   vocab      = 8K
#   d_model    = 192
#   heads      = 6
#   layers     = 6
#   FFN        = 768
#   context    = 256 tokens
#   tied LM head / token embedding
#
# The target is roughly a 4-5M parameter model, large enough to show
# real language-model behavior but still practical on CPU for short runs.
# ============================================================

SEED = 42

REPO_ROOT = Path(__file__).resolve().parent
CORPUS_PATH = REPO_ROOT / "data" / "step38_real_corpus" / "step38b_master_70_30.txt"
TOKENIZER_PATH = REPO_ROOT / "artifacts" / "step39_bpe_8000.json"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "step40"
CHECKPOINT_PATH = ARTIFACT_DIR / "tiny_gpt_step40.pt"
CONFIG_PATH = ARTIFACT_DIR / "tiny_gpt_step40_config.json"

VOCAB_SIZE = 8_000
D_MODEL = 192
N_HEADS = 6
N_LAYERS = 6
D_FF = 768
CONTEXT_LENGTH = 256
DROPOUT = 0.0
TRAIN_RATIO = 0.90
BATCH_SIZE = 8
DEFAULT_STEPS = 300
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, context_length, dropout=0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, context_length, context_length))

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Use PyTorch's optimized scaled dot-product attention when available.
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = self.causal_mask[:, :, :seq_len, :seq_len]
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            y = weights @ v

        y = y.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, context_length, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, context_length, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        n_layers,
        d_ff,
        context_length,
        dropout=0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    d_ff,
                    context_length,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: saves parameters and is standard for compact language models.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        _, seq_len = input_ids.shape
        if seq_len > self.context_length:
            raise ValueError("Sequence length exceeds context length")

        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=0.8, top_k=50):
        self.eval()
        for _ in range(max_new_tokens):
            input_cond = input_ids[:, -self.context_length :]
            logits, _ = self(input_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                cutoff = values[:, [-1]]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def load_tokenized_corpus(tokenizer, text):
    encoding = tokenizer.encode(text)
    return torch.tensor(encoding.ids, dtype=torch.long)


def get_batch(tokens, batch_size, context_length, device):
    max_start = len(tokens) - context_length - 1
    starts = torch.randint(0, max_start + 1, (batch_size,)).tolist()
    x = torch.stack([tokens[start : start + context_length] for start in starts])
    y = torch.stack([tokens[start + 1 : start + context_length + 1] for start in starts])
    return x.to(device), y.to(device)


def estimate_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


@torch.no_grad()
def estimate_loss(model, tokens, batches, batch_size, context_length, device):
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = get_batch(tokens, batch_size, context_length, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser(description="Train the first real TinyGPT model.")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 112)
    print("Step 40: First real TinyGPT training run")
    print("=" * 112)

    if not CORPUS_PATH.exists():
        print(f"Corpus not found: {CORPUS_PATH}")
        print("Run Step 38b first.")
        sys.exit(1)

    if not TOKENIZER_PATH.exists():
        print(f"Tokenizer not found: {TOKENIZER_PATH}")
        print("Run Step 39 first.")
        sys.exit(1)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise RuntimeError(
            f"Expected an 8K tokenizer, got vocab size {tokenizer.get_vocab_size()}"
        )

    text = CORPUS_PATH.read_text(encoding="utf-8")
    split = int(len(text) * TRAIN_RATIO)
    train_text = text[:split]
    val_text = text[split:]

    train_tokens = load_tokenized_corpus(tokenizer, train_text)
    val_tokens = load_tokenized_corpus(tokenizer, val_text)

    if len(train_tokens) < CONTEXT_LENGTH + 2 or len(val_tokens) < CONTEXT_LENGTH + 2:
        raise RuntimeError("Corpus split is too small for the selected context length.")

    # Keep CPU as the default so the first run is easy to reproduce on the user's PC.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyGPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        context_length=CONTEXT_LENGTH,
        dropout=DROPOUT,
    ).to(device)

    parameter_count = estimate_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    print("\nPart 1: Dataset")
    print("-" * 112)
    print(f"Text chars:          {len(text):,}")
    print(f"Train chars:         {len(train_text):,}")
    print(f"Validation chars:    {len(val_text):,}")
    print(f"Train tokens:        {len(train_tokens):,}")
    print(f"Validation tokens:   {len(val_tokens):,}")
    print(f"Tokenizer vocab:     {tokenizer.get_vocab_size():,}")

    print("\nPart 2: Model")
    print("-" * 112)
    print(f"Parameters:          {parameter_count:,}")
    print(f"d_model:             {D_MODEL}")
    print(f"heads:               {N_HEADS}")
    print(f"layers:              {N_LAYERS}")
    print(f"FFN:                 {D_FF}")
    print(f"context length:      {CONTEXT_LENGTH}")
    print(f"weight tying:        True")
    print(f"device:              {device}")
    print(f"training steps:      {args.steps}")
    print(f"batch size:          {args.batch_size}")
    print(f"effective tokens/step: {args.batch_size * CONTEXT_LENGTH:,}")

    initial_train = estimate_loss(model, train_tokens, 10, args.batch_size, CONTEXT_LENGTH, device)
    initial_val = estimate_loss(model, val_tokens, 10, args.batch_size, CONTEXT_LENGTH, device)
    print(f"Initial train loss:  {initial_train:.4f}")
    print(f"Initial val loss:    {initial_val:.4f}")

    print("\nPart 3: Training")
    print("-" * 112)

    model.train()
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_tokens, args.batch_size, CONTEXT_LENGTH, device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step == 1 or step % 25 == 0 or step == args.steps:
            val_loss = estimate_loss(
                model,
                val_tokens,
                5,
                args.batch_size,
                CONTEXT_LENGTH,
                device,
            )
            print(
                f"step {step:>4}/{args.steps} | "
                f"train loss {loss.item():.4f} | val loss {val_loss:.4f}"
            )

    final_train = estimate_loss(model, train_tokens, 10, args.batch_size, CONTEXT_LENGTH, device)
    final_val = estimate_loss(model, val_tokens, 10, args.batch_size, CONTEXT_LENGTH, device)

    print("\nPart 4: First generation")
    print("-" * 112)
    model.eval()

    prompts = [
        "The model",
        "一个小型语言模型",
        "Partdro",
    ]

    generation_records = []
    for prompt in prompts:
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
        output_ids = model.generate(input_ids, max_new_tokens=60, temperature=0.8, top_k=40)
        generated = tokenizer.decode(output_ids[0].tolist())
        generation_records.append({"prompt": prompt, "text": generated})
        print(f"\nPrompt: {prompt}")
        print(generated)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "vocab_size": VOCAB_SIZE,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "d_ff": D_FF,
            "context_length": CONTEXT_LENGTH,
            "dropout": DROPOUT,
            "weight_tying": True,
            "tokenizer_path": str(TOKENIZER_PATH),
            "corpus_path": str(CORPUS_PATH),
        },
        "step": args.steps,
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "seed": args.seed,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)

    config = {
        **checkpoint["config"],
        "parameter_count": parameter_count,
        "train_tokens": len(train_tokens),
        "validation_tokens": len(val_tokens),
        "training_steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "initial_train_loss": initial_train,
        "initial_val_loss": initial_val,
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "generations": generation_records,
    }
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nPart 5: Results")
    print("-" * 112)
    print(f"Final train loss:    {final_train:.4f}")
    print(f"Final validation:    {final_val:.4f}")
    print(f"Checkpoint saved:    {CHECKPOINT_PATH}")
    print(f"Config saved:        {CONFIG_PATH}")
    print("\nStep 40 complete.")
    print("This is a real decoder-only language model, not a toy next-token demo.")


if __name__ == "__main__":
    main()
