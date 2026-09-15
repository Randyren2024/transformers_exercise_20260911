import os
import re
import sys
from pathlib import Path

try:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from tokenizers.trainers import BpeTrainer
except ImportError:
    print("The 'tokenizers' package is not installed.")
    print("Install it with:")
    print("  pip install tokenizers")
    sys.exit(1)


# ============================================================
# Step 37: Controlled BPE Benchmark
#
# Goal:
#   Compare language-ratio variants (80/20, 70/30, 50/50)
#   against vocabulary sizes (4K, 8K, 16K).
#
# Important:
#   Step 36 created the controlled rehearsal corpora.
#   This step uses those corpora without adding Tiny Shakespeare,
#   so we isolate the effect of language balance + vocabulary size.
#
# This is still a tokenizer design benchmark, not final training.
# ============================================================

SEED = 42
REPO_ROOT = Path(__file__).resolve().parent
CORPUS_DIR = REPO_ROOT / "data" / "step36_bilingual_corpora"
ARTIFACT_DIR = REPO_ROOT / "artifacts"

RATIOS = ["80_20", "70_30", "50_50"]
VOCAB_SIZES = [4000, 8000, 16000]

TESTS = [
    ("English", "The model should answer questions clearly and keep useful context."),
    ("Chinese", "一个小型语言模型也可以学习基本的问答和对话能力。"),
    ("Mixed", "一个 small model 可以处理 English 和 中文 mixed text。"),
    ("Code", "for step in range(1000): loss = model(x); loss.backward()"),
    ("Product", "Partdro D15R | payload 10 kg | flight time 35 min | IP54"),
    ("URL", "https://www.partdro.com/products/firefighting_drone/"),
    ("Dialogue", "User: 你好。Assistant: 你好，我可以帮助你分析这个问题。"),
]


def count_unicode_classes(text: str):
    chinese = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    latin = sum(("a" <= ch.lower() <= "z") for ch in text)
    return chinese, latin


def train_bpe(corpus_path: Path, vocab_size: int):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Full byte alphabet avoids unknowns for arbitrary UTF-8 byte sequences.
    initial_alphabet = pre_tokenizers.ByteLevel.alphabet()
    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        initial_alphabet=initial_alphabet,
        special_tokens=special_tokens,
        show_progress=False,
    )

    tokenizer.train([str(corpus_path)], trainer=trainer)
    return tokenizer


def token_stats(tokenizer, text: str):
    encoding = tokenizer.encode(text)
    ids = encoding.ids
    decoded = tokenizer.decode(ids)
    unk_id = tokenizer.token_to_id("<unk>")
    unk_count = sum(idx == unk_id for idx in ids)
    return len(ids), unk_count, decoded == text, encoding.tokens


def inspect_corpus(ratio: str):
    path = CORPUS_DIR / f"step36_{ratio}.txt"
    text = path.read_text(encoding="utf-8")
    chinese, latin = count_unicode_classes(text)
    total = chinese + latin
    chinese_ratio = chinese / total if total else 0.0
    return path, text, chinese_ratio, latin / total if total else 0.0


def parameter_costs(vocab_size: int, d_model: int = 128):
    embedding = vocab_size * d_model
    lm_head = vocab_size * d_model
    return embedding, lm_head, embedding + lm_head


def mean_chars_per_token(tokenizer, ratio: str):
    token_counts = []
    char_counts = []
    for _, text in TESTS:
        count, _, _, _ = token_stats(tokenizer, text)
        token_counts.append(count)
        char_counts.append(len(text))
    total_chars = sum(char_counts)
    total_tokens = sum(token_counts)
    return total_chars / total_tokens if total_tokens else 0.0


def main():
    print("=" * 112)
    print("Step 37: Controlled BPE Benchmark — Language Ratio × Vocabulary Size")
    print("=" * 112)
    print("Purpose: isolate the effect of bilingual corpus balance and vocabulary size.")
    print("No Tiny Shakespeare is added in this benchmark.")

    print("\nPart 1: Controlled corpora")
    print("-" * 112)
    corpus_meta = {}
    for ratio in RATIOS:
        path, text, chinese_ratio, latin_ratio = inspect_corpus(ratio)
        corpus_meta[ratio] = (path, text, chinese_ratio, latin_ratio)
        print(
            f"{ratio:<8} chars={len(text):>7,} "
            f"Chinese={chinese_ratio:>7.2%} Latin={latin_ratio:>7.2%} "
            f"path={path.name}"
        )

    print("\nPart 2: Main benchmark")
    print("-" * 112)
    header = (
        f"{'Corpus':<12}{'Vocab':>8}{'Total tok':>12}"
        f"{'Chars/tok':>12}{'Tok/char':>12}"
        f"{'Chinese':>12}{'English':>12}{'Code':>10}{'Product':>10}"
        f"{'<unk>':>8}{'RT':>6}"
    )
    print(header)
    print("-" * 112)

    all_results = []

    for ratio in RATIOS:
        path, _, _, _ = corpus_meta[ratio]
        for vocab_size in VOCAB_SIZES:
            tokenizer = train_bpe(path, vocab_size)

            per_text = {}
            total_tokens = 0
            total_chars = 0
            total_unk = 0
            roundtrip_ok = 0

            for name, text in TESTS:
                count, unk_count, roundtrip, _ = token_stats(tokenizer, text)
                per_text[name] = count
                total_tokens += count
                total_chars += len(text)
                total_unk += unk_count
                roundtrip_ok += int(roundtrip)

            chars_per_token = total_chars / total_tokens if total_tokens else 0.0
            tok_per_char = total_tokens / total_chars if total_chars else 0.0

            result = {
                "ratio": ratio,
                "vocab": vocab_size,
                "total_tokens": total_tokens,
                "chars_per_token": chars_per_token,
                "tok_per_char": tok_per_char,
                "per_text": per_text,
                "unk": total_unk,
                "roundtrip": f"{roundtrip_ok}/{len(TESTS)}",
                "vocab_actual": tokenizer.get_vocab_size(),
            }
            all_results.append(result)

            print(
                f"{ratio:<12}{vocab_size:>8,}{total_tokens:>12,}"
                f"{chars_per_token:>12.2f}{tok_per_char:>12.3f}"
                f"{per_text['Chinese']:>12,}{per_text['English']:>12,}"
                f"{per_text['Code']:>10,}{per_text['Product']:>10,}"
                f"{total_unk:>8,}{result['roundtrip']:>6}"
            )

    print("\nPart 3: Detailed bilingual comparison")
    print("-" * 112)
    print(f"{'Ratio':<10}{'Vocab':>8}{'English':>12}{'Chinese':>12}{'Mixed':>12}{'Dialogue':>12}{'Product':>12}{'URL':>10}")
    print("-" * 112)
    for result in all_results:
        p = result["per_text"]
        print(
            f"{result['ratio']:<10}{result['vocab']:>8,}"
            f"{p['English']:>12,}{p['Chinese']:>12,}{p['Mixed']:>12,}"
            f"{p['Dialogue']:>12,}{p['Product']:>12,}{p['URL']:>10,}"
        )

    print("\nPart 4: Vocabulary parameter cost")
    print("-" * 112)
    print("Assumption: d_model=128, untied embedding + LM head.")
    print(f"{'Vocab':>10}{'Embedding':>16}{'LM head':>16}{'Combined':>16}")
    print("-" * 112)
    for vocab_size in VOCAB_SIZES:
        emb, head, combined = parameter_costs(vocab_size)
        print(f"{vocab_size:>10,}{emb:>16,}{head:>16,}{combined:>16,}")

    print("\nPart 5: Best efficiency by total evaluation tokens")
    print("-" * 112)
    best = min(all_results, key=lambda x: x["total_tokens"])
    print(
        f"Best overall in this rehearsal: corpus={best['ratio']}, "
        f"vocab={best['vocab']:,}, total evaluation tokens={best['total_tokens']:,}."
    )

    print("\nPart 6: Saturation checks")
    print("-" * 112)
    for ratio in RATIOS:
        rows = [r for r in all_results if r["ratio"] == ratio]
        rows.sort(key=lambda x: x["vocab"])
        print(f"{ratio}:")
        for prev, cur in zip(rows, rows[1:]):
            reduction = 1.0 - cur["total_tokens"] / prev["total_tokens"]
            print(
                f"  {prev['vocab']:>5,} -> {cur['vocab']:>5,}: "
                f"token reduction = {reduction:>6.2%}"
            )

    print("\nPart 7: Interpretation")
    print("-" * 112)
    print("1. Higher vocabulary should reduce token count when useful multi-character patterns are learned.")
    print("2. A balanced bilingual corpus gives the tokenizer enough Chinese material to learn reusable Chinese byte patterns.")
    print("3. Vocabulary growth has diminishing returns; once token counts stop improving much, extra vocabulary mainly adds parameters.")
    print("4. The tokenizer should be trained on representative language/domain proportions, not merely convenient small files.")
    print("5. These results are only rehearsal evidence because the corpus is tiny; final choices require a realistic public-data sample.")

    print("\nPart 8: Save compact experiment metadata")
    print("-" * 112)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = ARTIFACT_DIR / "step37_results.txt"
    lines = [
        "Step 37 Controlled BPE Benchmark Results",
        "========================================",
    ]
    for result in all_results:
        lines.append(
            f"{result['ratio']} vocab={result['vocab']} total_tokens={result['total_tokens']} "
            f"chars_per_token={result['chars_per_token']:.4f} tok_per_char={result['tok_per_char']:.4f}"
        )
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {summary_path}")

    print("\nStep 37 complete.")


if __name__ == "__main__":
    main()
