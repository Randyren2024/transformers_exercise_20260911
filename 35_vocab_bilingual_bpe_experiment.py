from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel


# ============================================================
# Step 35: Vocabulary Size x English/Bilingual BPE experiment
#
# Goal:
#   Compare vocabulary sizes and training-corpus language mix before
#   we freeze the tokenizer for the final small conversational model.
#
# We compare:
#   - English-heavy corpus
#   - Bilingual English + Chinese corpus
#   - vocab sizes 2K / 4K / 8K / 16K
#
# Important:
#   The corpora here are still engineering rehearsal corpora.
#   The final tokenizer must later be trained on the actual modern
#   pretraining corpus.
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
ARTIFACT_DIR = REPO_ROOT / "artifacts"

SHADOW_TEXT = """
A language model learns useful patterns from diverse, high quality text.
A good assistant should explain ideas clearly, follow context, and answer practical questions.
Transformers use self attention to combine information from tokens in context.
Training predicts the next token, computes loss, and updates model parameters.
A tokenizer converts text into token IDs that a neural network can process.
Machine learning, software engineering, science, history, education, stories, and dialogue
are useful sources of language variation.
Python code uses functions, variables, loops, classes, and data structures.
Example: batch_size=16, sequence_length=512, learning_rate=3e-4.
A checkpoint stores model weights, optimizer state, tokenizer files, and training progress.
URLs such as https://example.com/docs and technical strings should remain representable.
Product specifications may contain payload 10 kg, flight time 35 min, IP54, 4K video, and 1200 W.
""".strip()

BILINGUAL_TEXT = """
中文也是现代对话模型需要处理的重要语言。
一个小型语言模型可以学习基本的中文表达、问答和上下文关系。
如果数据同时包含中文和英文，tokenizer 应该尽量让两种语言都得到合理的 subword 表示。
机器学习、自然语言处理、无人机、机器人、网站和广告都是专业领域中的常见词汇。
用户可以使用中文提问，也可以在一句话中混合 English 和 中文。
一个 bilingual assistant should be able to answer in the language used by the user.
中文和 English mixed text can contain URLs, code, product names, numbers, and symbols.
例如：这是一个 English 和中文混合的句子，用于测试 tokenizer 的行为。
Partdro D15R 是一个 window cleaning drone，payload 可以是 10 kg。
""".strip()


def load_shakespeare_slice(max_chars=250_000):
    path = DATA_DIR / "tinyshakespeare.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]


def build_corpora():
    shakespeare = load_shakespeare_slice()
    # Make the English-heavy corpus representative of normal English text.
    english = "\n\n".join([SHADOW_TEXT, shakespeare]) if shakespeare else SHADOW_TEXT

    # Keep corpus sizes comparable in broad scale while explicitly adding
    # a meaningful Chinese/English bilingual component.
    bilingual = "\n\n".join([SHADOW_TEXT, BILINGUAL_TEXT, shakespeare]) if shakespeare else "\n\n".join([SHADOW_TEXT, BILINGUAL_TEXT])

    return {
        "english_heavy": english,
        "bilingual": bilingual,
    }


def train_tokenizer(text, vocab_size):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        initial_alphabet=ByteLevel.alphabet(),
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        show_progress=False,
    )
    tokenizer.train_from_iterator([text], trainer=trainer)
    return tokenizer


def evaluate(tokenizer, evaluation_set):
    total_chars = 0
    total_tokens = 0
    unk_tokens = 0
    exact_roundtrips = 0

    for _, text in evaluation_set:
        encoding = tokenizer.encode(text)
        total_chars += len(text)
        total_tokens += len(encoding.ids)
        unk_id = tokenizer.token_to_id("<unk>")
        unk_tokens += sum(token_id == unk_id for token_id in encoding.ids)
        if tokenizer.decode(encoding.ids) == text:
            exact_roundtrips += 1

    chars_per_token = total_chars / total_tokens
    tokens_per_char = total_tokens / total_chars
    return {
        "chars": total_chars,
        "tokens": total_tokens,
        "chars_per_token": chars_per_token,
        "tokens_per_char": tokens_per_char,
        "unk": unk_tokens,
        "exact": exact_roundtrips,
        "examples": len(evaluation_set),
    }


def parameter_estimate(vocab_size):
    # Rough embedding + LM-head parameter count for d_model=128.
    d_model = 128
    return vocab_size * d_model * 2


def print_example(tokenizer, text, label):
    enc = tokenizer.encode(text)
    print(f"  {label}: {len(enc.ids):>4} tokens | {enc.tokens}")


def main():
    print("=" * 112)
    print("Step 35: Vocabulary Size x English/Bilingual BPE experiment")
    print("=" * 112)
    print("Purpose: choose a practical tokenizer vocabulary for our future small conversational model.")
    print("This is a design experiment, not the final tokenizer training run.")

    corpora = build_corpora()
    print("\nPart 1: Corpus setup")
    print("-" * 112)
    for name, text in corpora.items():
        chinese_chars = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
        ratio = chinese_chars / len(text) if text else 0.0
        print(f"{name:<16} chars={len(text):>9,}  Chinese-character ratio={ratio:>7.2%}")

    evaluation_set = [
        ("English", "The model should answer questions clearly and keep useful context."),
        ("Chinese", "一个小型语言模型也可以学习基本的问答和对话能力。"),
        ("Mixed", "一个 small model 可以处理 English 和 中文 mixed text。"),
        ("Code", "for step in range(1000): loss = model(x); loss.backward()"),
        ("Product", "Partdro D15R | payload 10 kg | flight time 35 min | IP54"),
        ("URL", "https://www.partdro.com/products/firefighting_drone/"),
        ("Numbers", "10 kg / 35 min / 15.2 V / 1200 W / 4K"),
        ("Dialogue", "User: What is a transformer? Assistant: It processes tokens in context."),
    ]

    vocab_sizes = [2000, 4000, 8000, 16000]
    results = []

    print("\nPart 2: Main benchmark")
    print("-" * 112)
    print(f"{'Corpus':<16} {'Vocab':>7} {'Tokens':>9} {'Chars/token':>12} {'Tok/char':>11} {'<unk>':>7} {'Roundtrip':>10}")
    print("-" * 112)

    for corpus_name, text in corpora.items():
        for vocab_size in vocab_sizes:
            tokenizer = train_tokenizer(text, vocab_size)
            stats = evaluate(tokenizer, evaluation_set)
            results.append((corpus_name, vocab_size, tokenizer, stats))
            print(
                f"{corpus_name:<16} {vocab_size:>7,} {stats['tokens']:>9,} "
                f"{stats['chars_per_token']:>12.2f} {stats['tokens_per_char']:>11.3f} "
                f"{stats['unk']:>7} {stats['exact']:>5}/{stats['examples']:<4}"
            )

    print("\nPart 3: Per-text comparison at each vocabulary size")
    print("-" * 112)
    for corpus_name, vocab_size, tokenizer, stats in results:
        if vocab_size not in (2000, 8000, 16000):
            continue
        print(f"\n[{corpus_name} | vocab={vocab_size:,}]")
        for label, text in evaluation_set:
            print_example(tokenizer, text, label)

    print("\nPart 4: Rough vocabulary parameter cost")
    print("-" * 112)
    print("Assumption: d_model=128, tied weights NOT assumed here.")
    print(f"{'Vocab':>8} {'Embedding params':>20} {'LM-head params':>18} {'Combined':>18}")
    print("-" * 70)
    for vocab_size in vocab_sizes:
        one = vocab_size * 128
        combined = parameter_estimate(vocab_size)
        print(f"{vocab_size:>8,} {one:>20,} {one:>18,} {combined:>18,}")

    print("\nPart 5: Interpretation")
    print("-" * 112)
    print("1. Larger vocabulary usually reduces the number of tokens needed to encode frequent text patterns.")
    print("2. Bilingual training data should improve Chinese and mixed-text efficiency compared with English-heavy training.")
    print("3. Larger vocabulary also increases embedding and output-layer parameter cost.")
    print("4. Token efficiency must be evaluated on the same type of text we expect the final model to see.")
    print("5. No single vocabulary size is universally best; the useful choice balances token efficiency, model size, and data diversity.")
    print("6. We should prefer a vocabulary that handles Chinese reasonably without wasting most entries on rare fragments.")

    # Pick a candidate automatically using a simple planning score.
    # This is deliberately only a heuristic, not a claim of optimality.
    scored = []
    for corpus_name, vocab_size, _tokenizer, stats in results:
        if corpus_name != "bilingual":
            continue
        # Reward fewer tokens, penalize vocab size modestly.
        score = stats["tokens"] + 0.05 * vocab_size
        scored.append((score, vocab_size, stats))
    scored.sort()
    best = scored[0]

    print("\nPart 6: Current planning recommendation")
    print("-" * 112)
    print(f"Heuristic best bilingual vocab in this rehearsal: {best[1]:,}")
    print("This is NOT the final choice.")
    print("The final vocabulary should be selected after we build a realistic modern English/Chinese pretraining corpus.")

    # Save one artifact for the best rehearsal configuration so later steps have
    # a concrete tokenizer to inspect, while clearly labeling it provisional.
    _, best_vocab, best_tokenizer, best_stats = next(
        item for item in results if item[0] == "bilingual" and item[1] == best[1]
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = ARTIFACT_DIR / f"step35_provisional_bilingual_bpe_{best_vocab}.json"
    best_tokenizer.save(str(artifact_path))
    print(f"Saved provisional tokenizer: {artifact_path}")
    print("Do not use this tokenizer for final pretraining yet.")

    print("\nStep 35 complete.")


if __name__ == "__main__":
    main()
