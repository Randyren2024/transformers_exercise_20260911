import re
from collections import Counter

# ============================================================
# Step 21: A tiny BPE tokenizer
#
# Goal:
#   Compare word-level tokenization with a simple character-based
#   Byte Pair Encoding (BPE)-style tokenizer.
#
# This is a teaching implementation, not a production tokenizer.
# It deliberately keeps the algorithm small and visible.
# ============================================================

CORPUS = """
low lower lowest
low lower lowest
new newer newest
wide wider widest
reading reader reads
""".strip().lower()

NUM_MERGES = 20


def word_tokens(text):
    return re.findall(r"\w+|[^\w\s]", text)


def get_stats(vocab):
    """Count adjacent symbol pairs, weighted by word frequency."""
    stats = Counter()
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            stats[(symbols[i], symbols[i + 1])] += freq
    return stats


def merge_pair(pair, vocab):
    """Merge one selected pair everywhere in the training vocabulary."""
    new_vocab = {}
    bigram = pair

    for symbols, freq in vocab.items():
        merged = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == bigram:
                merged.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        new_vocab[tuple(merged)] = freq

    return new_vocab


def train_bpe(words, num_merges):
    """Train a tiny BPE vocabulary from characters + end-of-word marker."""
    vocab = Counter()
    for word, freq in Counter(words).items():
        symbols = tuple(list(word) + ["</w>"])
        vocab[symbols] = freq

    merges = []

    for step in range(num_merges):
        stats = get_stats(vocab)
        if not stats:
            break

        best_pair, best_count = stats.most_common(1)[0]
        if best_count < 2:
            break

        merges.append(best_pair)
        vocab = merge_pair(best_pair, vocab)

        print(f"Merge {step + 1:2d}: {best_pair}  (count={best_count})")

    token_set = sorted({symbol for symbols in vocab for symbol in symbols})
    return merges, token_set


def apply_merges(word, merges):
    """Apply learned BPE merges in the same order used during training."""
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


def pretty_bpe(tokens):
    """Remove the teaching end-of-word marker for display."""
    return [token.replace("</w>", "") for token in tokens]


print("=" * 64)
print("Step 21: Tiny BPE tokenizer experiment")
print("=" * 64)

print("\nOriginal corpus:")
print(CORPUS)

# ------------------------------------------------------------
# Part 1: Word-level tokenization
# ------------------------------------------------------------
words = word_tokens(CORPUS)
print("\nPart 1: Word-level tokens")
print("Token count:", len(words))
print("Tokens:", words)
print("Unique words:", len(set(words)))

# ------------------------------------------------------------
# Part 2: Train a tiny BPE vocabulary
# ------------------------------------------------------------
print("\nPart 2: BPE merges")
word_list = [word for word in words if word.isalpha()]
merges, bpe_vocab = train_bpe(word_list, NUM_MERGES)

print("\nLearned BPE vocabulary size:", len(bpe_vocab))
print("Learned tokens:", bpe_vocab)

# ------------------------------------------------------------
# Part 3: Compare tokenization
# ------------------------------------------------------------
examples = [
    "low",
    "lowest",
    "lower",
    "newest",
    "reader",
    "reading",
    "unseen",
]

print("\nPart 3: Word-level vs BPE")
for word in examples:
    word_level = [word]
    bpe_tokens = pretty_bpe(apply_merges(word, merges))
    print(f"{word:8s} | word-level: {word_level!s:14s} | BPE: {bpe_tokens}")

# ------------------------------------------------------------
# Part 4: Why this matters for modern LLMs
# ------------------------------------------------------------
print("\nPart 4: Key observations")
print("1. Word-level tokenization treats each whole word as one token.")
print("2. BPE can reuse pieces shared by many words.")
print("3. Rare or unseen words can often be represented by smaller pieces.")
print("4. This reduces the need to store every possible whole word in the vocabulary.")
print("5. Real GPT tokenizers use more sophisticated, production-grade variants of subword tokenization.")

print("\nStep 21 complete.")
