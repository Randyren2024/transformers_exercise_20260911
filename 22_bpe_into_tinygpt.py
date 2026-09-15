import math
import os
import re
from collections import Counter
from urllib.request import urlopen

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Step 22: Put the tiny BPE tokenizer into TinyGPT
#
# Goal:
#   Compare the previous word-level TinyGPT tokenizer with a
#   small BPE/subword tokenizer on the same Tiny Shakespeare text.
#
# This is still a teaching implementation. It is NOT a production
# tokenizer and is intentionally much simpler than GPT-2/Qwen/Llama
# tokenizers.
# ============================================================

SEED = 42
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "tinyshakespeare.txt")
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    print("Downloading Tiny Shakespeare corpus...")
    with urlopen(DATA_URL, timeout=30) as response:
        text = response.read().decode("utf-8")
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        f.write(text)
else:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

print(f"Corpus characters: {len(text)}")

# ============================================================
# Part 1: Common word/punctuation tokenization
# ============================================================

def basic_tokens(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())


all_basic_tokens = basic_tokens(text)
MAX_WORD_LEVEL_TOKENS = 50_000
basic_tokens_used = all_basic_tokens[:MAX_WORD_LEVEL_TOKENS]

print(f"Word-level tokens used: {len(basic_tokens_used)}")

# ============================================================
# Part 2: Tiny BPE implementation
# ============================================================

NUM_MERGES = 200


def get_stats(vocab):
    stats = Counter()
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            stats[(symbols[i], symbols[i + 1])] += freq
    return stats


def merge_pair(pair, vocab):
    new_vocab = {}
    for symbols, freq in vocab.items():
        merged = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                merged.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        new_vocab[tuple(merged)] = freq
    return new_vocab


def train_bpe(words, num_merges):
    vocab = Counter()
    for word, freq in Counter(words).items():
        vocab[tuple(list(word) + ["</w>"])] = freq

    merges = []
    for _ in range(num_merges):
        stats = get_stats(vocab)
        if not stats:
            break
        best_pair, best_count = stats.most_common(1)[0]
        if best_count < 2:
            break
        merges.append(best_pair)
        vocab = merge_pair(best_pair, vocab)

    token_set = sorted({symbol for symbols in vocab for symbol in symbols})
    return merges, token_set


def apply_merges(word, merges):
    symbols = list(word) + ["</w>"]
    for pair in merges:
        merged = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                merged.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        symbols = merged
    return symbols


# Train BPE on the word types seen in the first 50k-token slice.
training_words = [t for t in basic_tokens_used if t.isalpha()]
merges, learned_bpe_vocab = train_bpe(training_words, NUM_MERGES)

# Encode every word/punctuation token into BPE token strings.
def bpe_encode_tokens(tokens, merges):
    result = []
    for token in tokens:
        if token.isalpha():
            result.extend(apply_merges(token, merges))
        else:
            result.append(token)
    return result


bpe_tokens_all = bpe_encode_tokens(basic_tokens_used, merges)

# Limit both experiments to 50k model tokens so the training loop is comparable.
MAX_MODEL_TOKENS = 50_000
bpe_tokens_used = bpe_tokens_all[:MAX_MODEL_TOKENS]

# Build vocabularies and IDs.
word_vocab = sorted(set(basic_tokens_used))
word_stoi = {token: i for i, token in enumerate(word_vocab)}
word_ids = torch.tensor([word_stoi[token] for token in basic_tokens_used], dtype=torch.long)

bpe_vocab = sorted(set(bpe_tokens_used))
bpe_stoi = {token: i for i, token in enumerate(bpe_vocab)}
bpe_ids = torch.tensor([bpe_stoi[token] for token in bpe_tokens_used], dtype=torch.long)

print("\nTokenizer comparison:")
print(f"Word-level vocabulary: {len(word_vocab)}")
print(f"BPE merges learned:    {len(merges)}")
print(f"BPE vocabulary:        {len(bpe_vocab)}")
print(f"BPE tokens available:  {len(bpe_tokens_used)}")
print(f"Word-level avg chars/token: {len(text[:]) / max(1, len(all_basic_tokens)):.2f} (rough corpus ratio)")

examples = ["king", "kings", "kingdom", "unusual", "honours", "banished"]
print("\nExample tokenization:")
for word in examples:
    print(f"{word:10s} -> {apply_merges(word, merges)}")

# ============================================================
# Part 3: TinyGPT model
# ============================================================

block_size = 64
batch_size = 16
embedding_dim = 64
heads = 4
layers = 2
ffn_dim = 256
train_steps = 400
learning_rate = 3e-4

print("\nTinyGPT configuration:")
print(f"block_size = {block_size}")
print(f"batch_size = {batch_size}")
print(f"embedding_dim = {embedding_dim}")
print(f"heads = {heads}")
print(f"layers = {layers}")
print(f"ffn_dim = {ffn_dim}")
print(f"training steps per tokenizer = {train_steps}")


def get_batch(data, batch_size, block_size):
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(DEVICE), y.to(DEVICE)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embedding_dim, heads, ffn_dim, block_size) for _ in range(layers)]
        )
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


# Train one model for each tokenizer. We deliberately keep everything
# else identical so tokenizer choice is the main changing variable.
def train_one(name, data, vocab_size):
    torch.manual_seed(SEED)
    model = TinyGPT(vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{name} model parameters: {parameter_count:,}")

    model.train()
    final_loss = None
    for step in range(1, train_steps + 1):
        x, y = get_batch(data, batch_size, block_size)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

        if step in (1, 100, 200, 300, 400):
            print(f"{name:11s} step {step:3d} | loss {final_loss:.4f}")

    return model, final_loss, parameter_count


print("\n============================================================")
print("Part 4: Train the same TinyGPT architecture with two tokenizers")
print("============================================================")

word_model, word_loss, word_params = train_one("Word-level", word_ids, len(word_vocab))
bpe_model, bpe_loss, bpe_params = train_one("BPE", bpe_ids, len(bpe_vocab))

# ============================================================
# Part 5: Compare what a fixed context window covers
# ============================================================

word_chars = sum(len(t) for t in basic_tokens_used[:block_size])
bpe_chars = 0
for t in bpe_tokens_used[:block_size]:
    bpe_chars += len(t.replace("</w>", ""))

print("\n============================================================")
print("Final comparison")
print("============================================================")
print(f"Word-level vocab:             {len(word_vocab):6d}")
print(f"BPE vocab:                    {len(bpe_vocab):6d}")
print(f"Word-level model parameters:  {word_params:6d}")
print(f"BPE model parameters:         {bpe_params:6d}")
print(f"Word-level final loss:        {word_loss:8.4f}")
print(f"BPE final loss:               {bpe_loss:8.4f}")
print(f"Approx chars in 64 word tokens:{word_chars:5d}")
print(f"Approx chars in 64 BPE tokens: {bpe_chars:5d}")

print("\nKey idea:")
print("- Tokenizer changes the sequence seen by the Transformer.")
print("- Vocabulary size changes embedding and LM-head parameter counts.")
print("- A fixed context window can cover a different amount of raw text.")
print("- This experiment keeps the Transformer architecture unchanged to isolate the tokenizer effect.")
print("- This BPE implementation is educational; real GPT tokenizers are more sophisticated.")

print("\nStep 22 complete.")
