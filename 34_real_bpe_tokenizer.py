import os
import sys
from pathlib import Path


# ============================================================
# Step 34: Real BPE tokenizer experiment
#
# Goal:
#   Move from our hand-written BPE teaching example to a real
#   Byte-Level BPE tokenizer using the Hugging Face tokenizers
#   library.
#
# Important distinction:
#   This is a real tokenizer implementation, but the corpus used
#   here is still a small engineering/training rehearsal corpus.
#   Later, we will train the FINAL tokenizer on the actual modern
#   pretraining corpus.
#
# We test:
#   1. Training a Byte-Level BPE tokenizer.
#   2. Special tokens and vocabulary size.
#   3. English / Chinese / mixed / code / product text.
#   4. Token efficiency (characters per token).
#   5. Save + reload + round-trip correctness.
#
# Hugging Face Tokenizers supports BPE training from files or
# Python iterators. Byte-Level BPE is a useful GPT-style starting
# point because it can represent arbitrary text without relying on
# a fixed word vocabulary.
# ============================================================

try:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from tokenizers.trainers import BpeTrainer
except ImportError:
    print("The 'tokenizers' package is not installed.")
    print("Install it with:")
    print("  pip install tokenizers")
    sys.exit(1)


SEED = 42

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
CORPUS_PATH = DATA_DIR / "step34_bpe_training_corpus.txt"
TOKENIZER_PATH = ARTIFACT_DIR / "step34_byte_level_bpe.json"


# A compact, modern-style rehearsal corpus.
#
# We intentionally include:
#   - everyday English
#   - explanation / educational text
#   - dialogue-like text
#   - code / structured text
#   - product and technical language
#   - Chinese
#   - mixed Chinese + English
#
# This is NOT our final pretraining dataset.
MODERN_TEXT = """
A good language model should be able to explain an idea clearly, answer a practical question,
and continue a conversation without losing the context of what was already said.
A useful assistant does not need to sound complicated. It should be accurate, direct, and easy to understand.
When a problem is difficult, the model should break it into smaller steps and state its assumptions.
Machine learning systems learn statistical patterns from text, code, examples, and conversations.
A training dataset should contain diverse writing styles, but quality is usually more important than raw volume.
Removing duplicate documents helps reduce repeated patterns and gives the model more useful information per token.
Validation data must remain separate from gradient updates so that we can measure generalization.
A smaller model can still be useful when the tokenizer, data mixture, and training procedure are designed carefully.

User: Can you explain what a transformer does?
Assistant: A transformer processes tokens in context. Self-attention lets each position combine information from other positions.
User: Why does the model need a tokenizer?
Assistant: The tokenizer converts text into discrete token IDs that the neural network can process efficiently.
User: What happens during training?
Assistant: The model predicts the next token, compares the prediction with the target, computes loss, and updates its weights.
User: Can a small model really have a conversation?
Assistant: It can learn basic conversational patterns, but the quality depends strongly on model size, data quality, and training.

Python code often contains names such as tokenizer, model, optimizer, learning_rate, and batch_size.
def train_step(model, batch):
    logits = model(batch)
    loss = cross_entropy(logits[:, :-1], batch[:, 1:])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

Example configuration: batch_size=16, sequence_length=512, learning_rate=3e-4.
A checkpoint may contain model weights, optimizer state, tokenizer files, configuration, and training progress.
JSON looks like {"model": "tiny-gpt", "context": 512, "vocab_size": 16000}.

Modern technical text also contains URLs such as https://example.com/docs/model?step=34 and email-like strings such as hello@example.com.
Product specifications may look like: payload 10 kg, flight time 35 min, IP54, 4K video, 15.2 V, 1200 W.
A product page can contain names such as Partdro D15R, Movenew P1, FIMI X8T, and DJI Mini 4 Pro.

中文文本也需要被模型正确处理。
一个小型语言模型可以学习基本的中文表达、问答和上下文关系。
如果训练数据同时包含中文和英文，tokenizer需要尽可能合理地处理两种语言。
例如：这是一个 English 和中文混合的句子，用来测试 tokenizer 的行为。
机器学习、自然语言处理、无人机、机器人、网站和广告都是不同领域的专业词汇。

A bilingual assistant should be able to answer questions in the language used by the user.
中文和 English 混合的数据会让 vocabulary design 变得更加重要。
The goal is not to make the vocabulary as large as possible. The goal is to represent useful text efficiently.
""".strip()


def build_training_corpus():
    """Build a small local corpus, optionally augmented by Tiny Shakespeare."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    parts = [MODERN_TEXT]

    shakespeare_path = DATA_DIR / "tinyshakespeare.txt"
    if shakespeare_path.exists():
        # Only use a limited slice here so Step 34 remains quick.
        text = shakespeare_path.read_text(encoding="utf-8", errors="ignore")
        text = text[:250_000]
        parts.append(text)
        print(f"Found local Tiny Shakespeare: using first {len(text):,} characters")
    else:
        print("Tiny Shakespeare not found; using the built-in modern rehearsal corpus only.")
        print("This is intentional: Step 34 is a tokenizer engineering exercise, not the final corpus build.")

    corpus = "\n\n".join(parts)
    CORPUS_PATH.write_text(corpus, encoding="utf-8")
    return corpus


def train_bpe(corpus_path, vocab_size):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # ByteLevel pre-tokenization is a practical GPT-style choice:
    # arbitrary text can be represented through byte-level symbols.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=True,
    )

    tokenizer.train([str(corpus_path)], trainer=trainer)
    return tokenizer


def analyze_tokenizer(tokenizer, examples):
    print("\nPart 3: Tokenization examples")
    print("-" * 96)

    for name, text in examples:
        encoding = tokenizer.encode(text)
        tokens = encoding.tokens
        ids = encoding.ids
        decoded = tokenizer.decode(ids)
        round_trip = decoded == text

        print(f"[{name}]")
        print(f"Text:     {text}")
        print(f"Tokens:   {tokens}")
        print(f"IDs:      {ids}")
        print(f"Count:    {len(ids)}")
        print(f"Roundtrip exact: {round_trip}")
        print()


def efficiency_table(tokenizer, texts):
    print("Part 4: Token efficiency")
    print("-" * 96)
    print(f"{'Type':<24} {'Chars':>8} {'Tokens':>10} {'Chars/token':>14}")
    print("-" * 96)

    for name, text in texts:
        token_count = len(tokenizer.encode(text).ids)
        chars = len(text)
        ratio = chars / token_count if token_count else 0.0
        print(f"{name:<24} {chars:>8,} {token_count:>10,} {ratio:>14.2f}")


def show_vocab(tokenizer):
    print("\nPart 5: Vocabulary inspection")
    print("-" * 96)
    vocab = tokenizer.get_vocab()
    print(f"Vocabulary size: {len(vocab):,}")

    # Show a few low IDs and a few alphabetic-looking pieces.
    id_to_token = {idx: token for token, idx in vocab.items()}
    for idx in sorted(id_to_token)[:24]:
        print(f"  {idx:>5}: {id_to_token[idx]!r}")


def main():
    print("=" * 96)
    print("Step 34: Real BPE tokenizer experiment")
    print("=" * 96)

    corpus = build_training_corpus()
    print("\nPart 1: Training corpus")
    print("-" * 96)
    print(f"Corpus path:   {CORPUS_PATH}")
    print(f"Characters:    {len(corpus):,}")
    print(f"Lines:         {corpus.count(chr(10)) + 1:,}")
    print("This corpus is only a tokenizer rehearsal corpus.")
    print("The final tokenizer must later be trained on the actual pretraining corpus.")

    vocab_size = 2000
    print("\nPart 2: Train Byte-Level BPE")
    print("-" * 96)
    print(f"Requested vocabulary size: {vocab_size:,}")
    print("Special tokens: <pad>, <unk>, <bos>, <eos>")
    print("Pre-tokenizer: ByteLevel")

    tokenizer = train_bpe(CORPUS_PATH, vocab_size=vocab_size)

    examples = [
        ("English", "I love cats and I want to build a small language model."),
        ("Chinese", "我想训练一个小型中文语言模型。"),
        ("Mixed", "这是一个 English 和中文 mixed sentence。"),
        ("Code", "def train_step(model, batch): return loss.backward()"),
        ("Product", "Partdro D15R | payload 10 kg | flight time 35 min | IP54"),
        ("URL", "https://example.com/docs/model?step=34"),
        ("Unknown-looking", "antidisestablishmentarianism xyz123 🚁"),
    ]
    analyze_tokenizer(tokenizer, examples)

    efficiency_examples = [
        ("English", "The model should answer questions clearly and keep useful context."),
        ("Chinese", "一个小型语言模型也可以学习基本的问答和对话能力。"),
        ("Mixed", "一个 small model 可以处理 English 和 中文 mixed text。"),
        ("Code", "for step in range(1000): loss = model(x); loss.backward()"),
        ("URL", "https://www.partdro.com/products/firefighting_drone/"),
        ("Numbers", "10 kg / 35 min / 15.2 V / 1200 W / 4K"),
    ]
    efficiency_table(tokenizer, efficiency_examples)

    show_vocab(tokenizer)

    print("\nPart 6: Save and reload")
    print("-" * 96)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(TOKENIZER_PATH))
    print(f"Saved tokenizer: {TOKENIZER_PATH}")

    reloaded = Tokenizer.from_file(str(TOKENIZER_PATH))
    test_text = "这是一个测试: This tokenizer should survive save and reload."
    original_ids = tokenizer.encode(test_text).ids
    reloaded_ids = reloaded.encode(test_text).ids
    decoded = reloaded.decode(reloaded_ids)

    print(f"Reload test text: {test_text}")
    print(f"IDs identical:    {original_ids == reloaded_ids}")
    print(f"Decoded text:     {decoded}")

    print("\nPart 7: What this means for our final model")
    print("-" * 96)
    print("1. We are now using a real BPE implementation instead of our hand-written teaching BPE.")
    print("2. Byte-level BPE can represent English, Chinese, code, symbols, URLs, and product text in one vocabulary.")
    print("3. Vocabulary size is a design parameter: too small -> more tokens; too large -> more embedding/output parameters and possible fragmentation tradeoffs.")
    print("4. The final tokenizer should be trained on the same kind of text used for pretraining.")
    print("5. We should measure token efficiency on our actual English/Chinese mixture before fixing the final vocabulary size.")
    print("6. After this step, the tokenizer vocabulary and token IDs become part of the model interface and must be saved with checkpoints.")

    print("\nStep 34 complete.")


if __name__ == "__main__":
    main()
