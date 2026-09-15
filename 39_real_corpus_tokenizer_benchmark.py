import json
import sys
from pathlib import Path

try:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from tokenizers.trainers import BpeTrainer
except ImportError:
    print("The 'tokenizers' package is not installed.")
    print("Install it with: pip install tokenizers")
    sys.exit(1)


# ============================================================
# Step 39: Real corpus tokenizer benchmark
#
# Goal:
#   Move from tiny rehearsal corpora to the real public-data sample
#   produced by Step 38, then compare practical BPE vocabulary sizes.
#
# This step does NOT train a language model.
# It answers one design question first:
#   Which vocabulary size gives a useful balance of token efficiency
#   and model parameter cost on the actual English/Chinese corpus?
#
# Candidate vocabularies:
#   4K, 8K, 16K
#
# The Step 38 corpus is expected to contain a master bilingual text
# and separate English/Chinese samples. We benchmark the master text
# and a held-out evaluation file when available.
# ============================================================

SEED = 42

REPO_ROOT = Path(__file__).resolve().parent
CORPUS_DIR = REPO_ROOT / "data" / "step38_real_corpus"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
RESULTS_PATH = ARTIFACT_DIR / "step39_real_corpus_tokenizer_results.json"

MASTER_CANDIDATES = [
    CORPUS_DIR / "step38_master_70_30.txt",
    CORPUS_DIR / "step38_master.txt",
]
VALIDATION_CANDIDATES = [
    CORPUS_DIR / "step38_validation.txt",
    CORPUS_DIR / "step38_master_70_30_validation.txt",
]
ENGLISH_CANDIDATES = [
    CORPUS_DIR / "step38_english.txt",
    CORPUS_DIR / "step38_english_sample.txt",
]
CHINESE_CANDIDATES = [
    CORPUS_DIR / "step38_chinese.txt",
    CORPUS_DIR / "step38_chinese_sample.txt",
]

VOCAB_SIZES = [4_000, 8_000, 16_000]
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def first_existing(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


def load_text(path):
    return path.read_text(encoding="utf-8", errors="strict")


def chinese_ratio(text):
    chinese = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latin = sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    total = chinese + latin
    return chinese / total if total else 0.0, latin / total if total else 0.0


def build_tokenizer(corpus_path, vocab_size):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train([str(corpus_path)], trainer=trainer)
    return tokenizer


def count_unk(tokenizer, text):
    return sum(1 for token in tokenizer.encode(text).tokens if token == "<unk>")


def evaluate(tokenizer, name, text):
    encoding = tokenizer.encode(text)
    decoded = tokenizer.decode(encoding.ids)
    tokens = len(encoding.ids)
    chars = len(text)
    chinese_share, latin_share = chinese_ratio(text)
    return {
        "name": name,
        "chars": chars,
        "tokens": tokens,
        "chars_per_token": chars / tokens if tokens else 0.0,
        "tokens_per_char": tokens / chars if chars else 0.0,
        "unk_count": count_unk(tokenizer, text),
        "roundtrip": decoded == text,
        "chinese_ratio": chinese_share,
        "latin_ratio": latin_share,
    }


def estimated_vocab_cost(vocab_size, d_model=128, tied_weights=False):
    embedding = vocab_size * d_model
    lm_head = 0 if tied_weights else vocab_size * d_model
    return embedding, lm_head, embedding + lm_head


def print_header(title):
    print("\n" + title)
    print("-" * 112)


def main():
    print("=" * 112)
    print("Step 39: Real corpus tokenizer benchmark")
    print("=" * 112)
    print("Goal: evaluate 4K / 8K / 16K BPE on the real public-data sample from Step 38.")
    print("This is the first tokenizer benchmark intended to inform the final model.")

    master_path = first_existing(MASTER_CANDIDATES)
    if master_path is None:
        print("\nStep 38 corpus was not found.")
        print("Run first:")
        print("  python 38_build_real_modern_bilingual_corpus.py")
        print("Then rerun Step 39.")
        sys.exit(1)

    validation_path = first_existing(VALIDATION_CANDIDATES)
    english_path = first_existing(ENGLISH_CANDIDATES)
    chinese_path = first_existing(CHINESE_CANDIDATES)

    train_text = load_text(master_path)
    validation_text = load_text(validation_path) if validation_path else ""

    print("\nPart 1: Real corpus")
    print("-" * 112)
    print(f"Master path:       {master_path}")
    print(f"Training chars:    {len(train_text):,}")
    cn_ratio, en_ratio = chinese_ratio(train_text)
    print(f"Chinese char ratio: {cn_ratio * 100:.2f}%")
    print(f"Latin char ratio:   {en_ratio * 100:.2f}%")
    if validation_path:
        print(f"Validation path:    {validation_path}")
        print(f"Validation chars:  {len(validation_text):,}")
    else:
        print("Validation file:    not found (tokenizer evaluation will use the master text only)")

    print_header("Part 2: Train and benchmark BPE vocabularies")
    print(
        f"{'Vocab':>8} {'Train tok':>12} {'Chars/tok':>11} {'Tok/char':>11} "
        f"{'Eval tok':>11} {'Eval C/tok':>12} {'<unk>':>8} {'RT':>6}"
    )
    print("-" * 112)

    all_results = []
    tokenizer_paths = []

    for vocab_size in VOCAB_SIZES:
        tokenizer = build_tokenizer(master_path, vocab_size)

        train_eval = evaluate(tokenizer, "train_master", train_text)
        if validation_text:
            eval_result = evaluate(tokenizer, "validation", validation_text)
        else:
            eval_result = train_eval.copy()
            eval_result["name"] = "master_as_eval"

        embedding, lm_head, combined = estimated_vocab_cost(vocab_size, d_model=128, tied_weights=False)

        result = {
            "vocab_size": vocab_size,
            "train": train_eval,
            "evaluation": eval_result,
            "parameter_cost_d_model_128": {
                "embedding": embedding,
                "lm_head": lm_head,
                "combined": combined,
            },
        }
        all_results.append(result)

        print(
            f"{vocab_size:>8,} {train_eval['tokens']:>12,} "
            f"{train_eval['chars_per_token']:>11.2f} {train_eval['tokens_per_char']:>11.4f} "
            f"{eval_result['tokens']:>11,} {eval_result['chars_per_token']:>12.2f} "
            f"{eval_result['unk_count']:>8} {str(eval_result['roundtrip']):>6}"
        )

        tokenizer_path = ARTIFACT_DIR / f"step39_bpe_{vocab_size}.json"
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))
        tokenizer_paths.append(str(tokenizer_path))

    print_header("Part 3: Language-specific evaluation")
    print(
        f"{'Vocab':>8} {'English tok':>13} {'Chinese tok':>13} {'Mixed tok':>12} "
        f"{'Dialogue tok':>14} {'Product tok':>13}"
    )
    print("-" * 112)

    # These samples test capability categories without being part of tokenizer training.
    test_texts = [
        ("English", "The model should answer questions clearly and keep useful context."),
        ("Chinese", "一个小型语言模型也应该能够理解中文问题并给出清楚的回答。"),
        ("Mixed", "一个 small model 可以处理 English 和 中文 mixed text。"),
        ("Dialogue", "User: 你好。\nAssistant: 你好，我可以帮助你分析这个问题。"),
        ("Product", "Partdro D15R | payload 10 kg | flight time 35 min | IP54"),
    ]

    for vocab_size, result in zip(VOCAB_SIZES, all_results):
        # Reload from the saved tokenizer to prove artifacts are usable.
        tokenizer = Tokenizer.from_file(result_path := str(ARTIFACT_DIR / f"step39_bpe_{vocab_size}.json"))
        counts = [len(tokenizer.encode(text).ids) for _, text in test_texts]
        print(
            f"{vocab_size:>8,} {counts[0]:>13} {counts[1]:>13} {counts[2]:>12} "
            f"{counts[3]:>14} {counts[4]:>13}"
        )

    print_header("Part 4: Vocabulary parameter cost")
    print("Assumption: d_model=128, untied embedding + LM head.")
    print(f"{'Vocab':>10} {'Embedding':>15} {'LM head':>15} {'Combined':>15}")
    print("-" * 72)
    for vocab_size in VOCAB_SIZES:
        embedding, lm_head, combined = estimated_vocab_cost(vocab_size, 128, tied_weights=False)
        print(f"{vocab_size:>10,} {embedding:>15,} {lm_head:>15,} {combined:>15,}")

    print_header("Part 5: Decision guidance")
    print("This experiment does not force a final vocabulary yet.")
    print("Use these rules:")
    print("  1. Reject any vocabulary with unknown tokens or failed roundtrip.")
    print("  2. Prefer the smallest vocabulary that achieves near-saturated token efficiency.")
    print("  3. Evaluate Chinese and mixed-text efficiency separately from English.")
    print("  4. Prefer held-out validation text when available; do not tune on the validation text itself.")
    print("  5. If 8K is close to 16K, prefer 8K because it keeps the model smaller.")
    print("  6. The final choice will also depend on the eventual model width and whether embeddings/LM head are tied.")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": 39,
        "master_path": str(master_path),
        "validation_path": str(validation_path) if validation_path else None,
        "vocab_sizes": VOCAB_SIZES,
        "results": all_results,
        "tokenizer_artifacts": tokenizer_paths,
        "source_note": "Tokenizer trained on Step 38 real-data sample; final model corpus may later use a larger sampled corpus.",
    }
    RESULTS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved results: {RESULTS_PATH}")
    print("\nStep 39 complete.")


if __name__ == "__main__":
    main()
