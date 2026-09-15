import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

try:
    import torch
    from datasets import load_dataset
    from tokenizers import Tokenizer
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    print("Install with: pip install torch datasets tokenizers")
    sys.exit(1)


# ============================================================
# Step 42: Build a larger bilingual pretraining corpus
#
# Purpose:
#   Step 40/41 proved that the 4.24M-parameter model can learn.
#   The next bottleneck is data quantity, not model size.
#
# This script streams fresh public web text from:
#   English: HuggingFaceFW/fineweb
#   Chinese: HuggingFaceFW/fineweb-2, config=cmn_Hani
#
# It builds a deterministic 70/30 bilingual corpus, splits documents
# into train/validation BEFORE tokenization, and saves token IDs so
# the next GPU training step can start without repeating preprocessing.
# ============================================================

SEED = 42
ENGLISH_DATASET = "HuggingFaceFW/fineweb"
CHINESE_DATASET = "HuggingFaceFW/fineweb-2"
CHINESE_CONFIG = "cmn_Hani"

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "data" / "step42_training_corpus"
TOKENIZER_PATH = REPO_ROOT / "artifacts" / "step39_bpe_8000.json"

DEFAULT_TOTAL_CHARS = 10_000_000
DEFAULT_ENGLISH_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.10
DEFAULT_MIN_CHARS = 250
DEFAULT_MAX_CHARS = 30_000
DEFAULT_MAX_ROWS = 100_000


def normalize_text(text):
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def chinese_ratio(text):
    chinese = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latin = sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    total = chinese + latin
    return chinese / total if total else 0.0


def acceptable(text, language):
    if len(text) < DEFAULT_MIN_CHARS:
        return False
    if len(text) > DEFAULT_MAX_CHARS:
        text = text[:DEFAULT_MAX_CHARS]

    lower = text.lower()
    bad_patterns = [
        "enable javascript",
        "cookie settings",
        "all rights reserved",
        "privacy policy |",
        "terms of service |",
    ]
    if any(pattern in lower for pattern in bad_patterns):
        return False

    letters = sum(ch.isalpha() for ch in text)
    if letters / max(len(text), 1) < 0.20:
        return False

    if language == "zh" and chinese_ratio(text) < 0.30:
        return False
    if language == "en" and chinese_ratio(text) > 0.15:
        return False

    return True


def document_text(record):
    value = record.get("text", "")
    if value is None:
        return ""
    return normalize_text(str(value))


def fingerprint(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def stream_language(dataset_name, language, target_chars, max_rows, **kwargs):
    print(f"\nStreaming {language.upper()} from {dataset_name} ...")
    dataset = load_dataset(dataset_name, split="train", streaming=True, **kwargs)

    accepted = []
    seen = set()
    chars = 0
    inspected = 0
    rejected = 0

    for record in dataset:
        inspected += 1
        if inspected > max_rows:
            break

        text = document_text(record)
        if not text or not acceptable(text, language):
            rejected += 1
            continue

        key = fingerprint(text)
        if key in seen:
            rejected += 1
            continue
        seen.add(key)
        accepted.append(text)
        chars += len(text)

        if len(accepted) % 250 == 0:
            print(f"  accepted={len(accepted):,} chars={chars:,}")

        if chars >= target_chars:
            break

    print(f"  inspected={inspected:,}")
    print(f"  accepted docs={len(accepted):,}")
    print(f"  chars={chars:,}")
    print(f"  rejected={rejected:,}")
    return accepted


def trim_to_target(documents, target_chars):
    result = []
    chars = 0
    for doc in documents:
        if chars >= target_chars:
            break
        remaining = target_chars - chars
        if len(doc) > remaining:
            if remaining >= DEFAULT_MIN_CHARS:
                doc = doc[:remaining]
            else:
                break
        result.append(doc)
        chars += len(doc)
    return result


def split_documents(documents, val_ratio, rng):
    docs = list(documents)
    rng.shuffle(docs)
    val_count = max(1, int(len(docs) * val_ratio))
    return docs[val_count:], docs[:val_count]


def encode_documents(tokenizer, documents, output_path):
    # Keep document boundaries explicit. Newlines are part of the training text.
    text = "\n\n".join(documents)
    ids = tokenizer.encode(text).ids
    tensor = torch.tensor(ids, dtype=torch.int32)
    torch.save(tensor, output_path)
    return text, len(ids)


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build a larger bilingual pretraining corpus.")
    parser.add_argument("--total-chars", type=int, default=DEFAULT_TOTAL_CHARS)
    parser.add_argument("--english-ratio", type=float, default=DEFAULT_ENGLISH_RATIO)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    args = parser.parse_args()

    if not 0.5 <= args.english_ratio <= 0.9:
        raise ValueError("english-ratio should be between 0.5 and 0.9")
    if not 0.05 <= args.val_ratio <= 0.2:
        raise ValueError("val-ratio should be between 0.05 and 0.2")

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    total_en_chars = int(args.total_chars * args.english_ratio)
    total_zh_chars = args.total_chars - total_en_chars

    print("=" * 112)
    print("Step 42: Build larger bilingual pretraining corpus")
    print("=" * 112)
    print(f"Target chars:       {args.total_chars:,}")
    print(f"English target:     {total_en_chars:,}")
    print(f"Chinese target:     {total_zh_chars:,}")
    print(f"Train/validation:   {1.0 - args.val_ratio:.0%}/{args.val_ratio:.0%}")
    print(f"Seed:               {SEED}")

    english_docs = stream_language(
        ENGLISH_DATASET,
        "en",
        total_en_chars,
        args.max_rows,
    )
    chinese_docs = stream_language(
        CHINESE_DATASET,
        "zh",
        total_zh_chars,
        args.max_rows,
        name=CHINESE_CONFIG,
    )

    english_docs = trim_to_target(english_docs, total_en_chars)
    chinese_docs = trim_to_target(chinese_docs, total_zh_chars)

    print("\nPart 2: Split documents")
    print("-" * 112)
    en_train, en_val = split_documents(english_docs, args.val_ratio, rng)
    zh_train, zh_val = split_documents(chinese_docs, args.val_ratio, rng)

    # Shuffle each language before interleaving. This avoids large contiguous
    # English/Chinese blocks while preserving document boundaries.
    rng.shuffle(en_train)
    rng.shuffle(zh_train)
    rng.shuffle(en_val)
    rng.shuffle(zh_val)

    train_docs = []
    val_docs = []
    for en_doc, zh_doc in zip(en_train, zh_train):
        train_docs.append(en_doc)
        train_docs.append(zh_doc)
    train_docs.extend(en_train[len(zh_train):])
    train_docs.extend(zh_train[len(en_train):])

    for en_doc, zh_doc in zip(en_val, zh_val):
        val_docs.append(en_doc)
        val_docs.append(zh_doc)
    val_docs.extend(en_val[len(zh_val):])
    val_docs.extend(zh_val[len(en_val):])

    rng.shuffle(train_docs)
    rng.shuffle(val_docs)

    print(f"Train documents:    {len(train_docs):,}")
    print(f"Validation docs:    {len(val_docs):,}")
    print(f"Train characters:   {sum(len(x) for x in train_docs):,}")
    print(f"Validation chars:   {sum(len(x) for x in val_docs):,}")
    print(f"Train Chinese ratio: {chinese_ratio(''.join(train_docs)) * 100:.2f}%")

    train_text_path = OUTPUT_DIR / "step42_train.txt"
    val_text_path = OUTPUT_DIR / "step42_validation.txt"
    train_ids_path = OUTPUT_DIR / "step42_train_ids.pt"
    val_ids_path = OUTPUT_DIR / "step42_validation_ids.pt"

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    print("\nPart 3: Encode and save")
    print("-" * 112)
    train_text, train_tokens = encode_documents(tokenizer, train_docs, train_ids_path)
    val_text, val_tokens = encode_documents(tokenizer, val_docs, val_ids_path)
    train_text_path.write_text(train_text, encoding="utf-8")
    val_text_path.write_text(val_text, encoding="utf-8")

    metadata = {
        "step": 42,
        "seed": SEED,
        "english_dataset": ENGLISH_DATASET,
        "chinese_dataset": CHINESE_DATASET,
        "chinese_config": CHINESE_CONFIG,
        "target_total_chars": args.total_chars,
        "target_english_chars": total_en_chars,
        "target_chinese_chars": total_zh_chars,
        "actual_train_chars": len(train_text),
        "actual_validation_chars": len(val_text),
        "train_tokens": train_tokens,
        "validation_tokens": val_tokens,
        "tokenizer": str(TOKENIZER_PATH),
        "train_text": str(train_text_path),
        "validation_text": str(val_text_path),
        "train_ids": str(train_ids_path),
        "validation_ids": str(val_ids_path),
        "notes": "Train/validation split is document-level and performed before tokenization.",
    }
    metadata_path = OUTPUT_DIR / "step42_metadata.json"
    write_json(metadata_path, metadata)

    print(f"Train tokens:        {train_tokens:,}")
    print(f"Validation tokens:   {val_tokens:,}")
    print(f"Saved train IDs:     {train_ids_path}")
    print(f"Saved validation:    {val_ids_path}")
    print(f"Saved metadata:      {metadata_path}")
    print("\nStep 42 complete.")


if __name__ == "__main__":
    main()
