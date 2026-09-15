import math
import random
import re
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 16. Larger corpus: give the same small GPT more data
# ============================================================
# The previous step used only a few thousand tokens. The model
# quickly memorized them, so validation loss became worse.
#
# In this step we keep the Transformer small, but replace the tiny
# hand-written corpus with Tiny Shakespeare (~1 MB of public-domain
# Shakespeare text). The file is downloaded once and then cached.
#
# Important: this example still uses a simple WORD-LEVEL tokenizer.
# We are changing the DATA SIZE, not the model architecture.

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DATA_FILE = DATA_DIR / "tinyshakespeare.txt"
DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)


# -----------------------------
# 1. Download/load the corpus
# -----------------------------
def load_corpus():
    if not DATA_FILE.exists():
        print("Downloading Tiny Shakespeare corpus...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        print(f"Saved to: {DATA_FILE}")

    text = DATA_FILE.read_text(encoding="utf-8")
    return text


text = load_corpus()

# Keep punctuation as separate tokens so the model can learn things
# such as: "ROMEO :" and "love ," instead of attaching punctuation
# to every word.
tokens = re.findall(r"\w+|[^\w\s]", text.lower())

# Limit the corpus to a manageable size for CPU training.
# This still gives us many times more data than step 15.
MAX_TOKENS = 50000
if len(tokens) > MAX_TOKENS:
    tokens = tokens[:MAX_TOKENS]

vocab = sorted(set(tokens))
stoi = {token: i for i, token in enumerate(vocab)}
itos = {i: token for token, i in stoi.items()}

encoded = torch.tensor([stoi[token] for token in tokens], dtype=torch.long)

# 90% training, 10% validation.
split_index = int(len(encoded) * 0.9)
train_data = encoded[:split_index]
val_data = encoded[split_index:]

print(f"Device: {DEVICE}")
print(f"Corpus characters: {len(text)}")
print(f"Corpus tokens used: {len(encoded)}")
print(f"Vocabulary size: {len(vocab)}")
print(f"Train tokens: {len(train_data)}")
print(f"Validation tokens: {len(val_data)}")


# -----------------------------
# 2. Mini-batch sampling
# -----------------------------
block_size = 64
batch_size = 16


def get_batch(split):
    data = train_data if split == "train" else val_data
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))

    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])

    return x.to(DEVICE), y.to(DEVICE)


# -----------------------------
# 3. Multi-head causal attention
# -----------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, block_size):
        super().__init__()

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        # Lower-triangular mask prevents looking into the future.
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, T, C] -> [B, H, T, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out = weights @ v

        # [B, H, T, head_dim] -> [B, T, C]
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.embedding_dim)

        out = self.out_proj(out)
        return out, weights


# -----------------------------
# 4. Transformer block
# -----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, block_size):
        super().__init__()

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            block_size=block_size,
        )
        self.ln2 = nn.LayerNorm(embedding_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embedding_dim),
        )

    def forward(self, x):
        attn_out, weights = self.attention(self.ln1(x))
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, weights


# -----------------------------
# 5. Tiny GPT
# -----------------------------
class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        ffn_dim=256,
    ):
        super().__init__()

        self.block_size = block_size
        self.embedding_dim = embedding_dim

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    block_size=block_size,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_ln = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None, return_attention=False):
        batch_size, seq_len = idx.shape

        if seq_len > self.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds block size {self.block_size}."
            )

        positions = torch.arange(seq_len, device=idx.device)

        x = self.token_embedding(idx) + self.position_embedding(positions)[None, :, :]

        all_attention = []
        for block in self.blocks:
            x, weights = block(x)
            if return_attention:
                all_attention.append(weights)

        x = self.final_ln(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if return_attention:
            return logits, loss, all_attention

        return logits, loss


# -----------------------------
# 6. Create model
# -----------------------------
model = TinyGPT(
    vocab_size=len(vocab),
    block_size=block_size,
    embedding_dim=64,
    num_heads=4,
    num_layers=2,
    ffn_dim=256,
).to(DEVICE)

num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\nModel:")
print(f"block_size = {block_size}")
print(f"batch_size = {batch_size}")
print("embedding_dim = 64")
print("heads = 4")
print("layers = 2")
print("ffn_dim = 256")
print(f"Total trainable parameters: {num_parameters}")


# -----------------------------
# 7. Check tensor shapes before training
# -----------------------------
x, y = get_batch("train")
logits, loss = model(x, y)

print("\nShape check:")
print(f"Input: {list(x.shape)}")
print(f"Target: {list(y.shape)}")
print(f"Logits: {list(logits.shape)}")
print(f"Initial loss: {loss.item():.4f}")


# -----------------------------
# 8. Validation helper
# -----------------------------
@torch.no_grad()
def estimate_loss():
    model.eval()
    result = {}

    for split in ["train", "val"]:
        losses = []
        for _ in range(5):
            xb, yb = get_batch(split)
            _, loss_value = model(xb, yb)
            losses.append(loss_value.item())

        result[split] = sum(losses) / len(losses)

    model.train()
    return result


# -----------------------------
# 9. Train
# -----------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

max_steps = 1500
print("\nTraining:")

for step in range(1, max_steps + 1):
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step == 1 or step % 100 == 0:
        losses = estimate_loss()
        print(
            f"Step {step:4d} | "
            f"train {losses['train']:.4f} | "
            f"val {losses['val']:.4f}"
        )


# -----------------------------
# 10. Text generation
# -----------------------------
def encode_prompt(prompt):
    prompt_tokens = re.findall(r"\w+|[^\w\s]", prompt.lower())
    unknown = [token for token in prompt_tokens if token not in stoi]
    if unknown:
        raise ValueError(f"Unknown prompt tokens: {unknown}")
    return torch.tensor([[stoi[token] for token in prompt_tokens]], dtype=torch.long, device=DEVICE)


def decode_tokens(ids):
    words = [itos[int(i)] for i in ids]

    text_out = ""
    for word in words:
        if word in {".", ",", "!", "?", ";", ":", ")", "]", "}"}:
            text_out += word
        elif word in {"(", "[", "{"}:
            text_out += (" " if text_out else "") + word
        else:
            text_out += (" " if text_out else "") + word

    return text_out


@torch.no_grad()
def generate(prompt, max_new_tokens=40, temperature=0.8):
    model.eval()
    idx = encode_prompt(prompt)

    for _ in range(max_new_tokens):
        context = idx[:, -block_size:]
        logits, _ = model(context)
        next_logits = logits[:, -1, :]
        next_logits = next_logits / temperature

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)

    model.train()
    return decode_tokens(idx[0].tolist())


print("\nGeneration examples:")
for prompt in ["romeo", "the king", "love"]:
    try:
        print(f"Prompt: {prompt}")
        print(generate(prompt))
        print()
    except ValueError as exc:
        print(f"Skipped '{prompt}': {exc}")


# -----------------------------
# 11. Attention inspection
# -----------------------------
@torch.no_grad()
def inspect_attention(prompt):
    model.eval()
    idx = encode_prompt(prompt)
    _, _, attention = model(idx, return_attention=True)

    weights = attention[0][0, 0]  # first layer, first batch, first head
    print(f"\nAttention inspection for: {prompt}")
    print("First layer, first head:")
    print(weights.cpu())
    model.train()


try:
    inspect_attention("romeo love")
except ValueError as exc:
    print(f"Attention inspection skipped: {exc}")

print("\nStep 16 complete.")
print("Main experiment: more data, same small Transformer.")
