import re
from collections import Counter

# ============================================================
# Step 23: BPE merge-count tradeoff
#
# Keep the corpus fixed and change only the number of BPE merges.
# Goal: observe the trade-off between vocabulary size and token length.
# This step does NOT train TinyGPT; it isolates tokenizer behavior.
# ============================================================

CORPUS = """
low lower lowest
low lower lowest
new newer newest
wide wider widest
reading reader reads
king kings kingdom
kingdom kingdom kingdoms
unusual unusually useful useful
honours honour honoured honouring
banished banishing banishes
people peoples person personal personality
beautiful beautifully beauty
walking walked walker walking
running runner ran quickly
friend friendly friendship
nation national nationality
""".strip().lower()

MERGE_COUNTS = [50, 200, 500, 1000]


def words_from_text(text):
    return [w for w in re.findall(r"\w+|[^\w\s]", text) if w.isalpha()]


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


def clean_tokens(tokens):
    return [t.replace("</w>", "") for t in tokens]


def corpus_stats(words, merges, vocab):
    token_lists = [clean_tokens(apply_merges(word, merges)) for word in words]
    token_count = sum(len(tokens) for tokens in token_lists)
    char_count = sum(len(word) for word in words)
    avg_chars_per_token = char_count / token_count if token_count else 0.0
    return token_count, avg_chars_per_token, token_lists


print("=" * 72)
print("Step 23: BPE merge-count tradeoff")
print("=" * 72)

words = words_from_text(CORPUS)
char_count = sum(len(w) for w in words)
print(f"Corpus words: {len(words)}")
print(f"Corpus characters (letters only): {char_count}")
print(f"Merge counts tested: {MERGE_COUNTS}")

results = []
for merge_count in MERGE_COUNTS:
    merges, vocab = train_bpe(words, merge_count)
    token_count, chars_per_token, token_lists = corpus_stats(words, merges, vocab)

    # A rough parameter proxy for token embeddings + LM head.
    d_model = 64
    embedding_and_head_params = 2 * len(vocab) * d_model

    results.append(
        {
            "merges": merge_count,
            "vocab": len(vocab),
            "tokens": token_count,
            "chars_per_token": chars_per_token,
            "context_chars": 64 * chars_per_token,
            "embedding_head_params": embedding_and_head_params,
            "merges_learned": len(merges),
            "sample_tokens": token_lists,
        }
    )

print("\nComparison:")
print(
    "merges | vocab | corpus_tokens | chars/token | "
    "64-token chars | embedding+head params"
)
print("-------+-------+---------------+-------------+----------------+----------------------")
for r in results:
    print(
        f"{r['merges']:6d} | "
        f"{r['vocab']:5d} | "
        f"{r['tokens']:13d} | "
        f"{r['chars_per_token']:11.2f} | "
        f"{r['context_chars']:14.1f} | "
        f"{r['embedding_head_params']:20,d}"
    )

print("\nExample tokenization at different merge counts:")
examples = ["kingdom", "unusually", "honouring", "personality", "beautifully"]
for merge_count, r in zip(MERGE_COUNTS, results):
    merges, _ = train_bpe(words, merge_count)
    print(f"\nMerges = {merge_count}")
    for word in examples:
        tokens = clean_tokens(apply_merges(word, merges))
        print(f"  {word:12s} -> {tokens}")

print("\nKey idea:")
print("- More merges usually make larger reusable subword units.")
print("- More merges generally increase vocabulary size.")
print("- Larger vocabulary can reduce the number of tokens needed to represent text.")
print("- But a larger vocabulary increases embedding and LM-head parameter counts.")
print("- The best tokenizer is a trade-off, not simply 'more merges is better'.")
print("- This experiment isolates tokenizer behavior; it does not compare model quality.")

print("\nStep 23 complete.")
