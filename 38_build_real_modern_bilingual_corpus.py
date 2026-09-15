import json
import re
from collections import Counter
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("The 'datasets' package is not installed.")
    print("Install it with:")
    print("  pip install datasets")
    raise SystemExit(1)

# ============================================================
# Step 38: Build a real modern bilingual corpus sample
#
# Goal:
#   Move from tiny hand-written rehearsal text to a repeatable
#   pipeline that samples real public modern text from Hugging Face.
#
# Sources selected for this first real-data pass:
#   - HuggingFaceFW/fineweb-2, eng_Latn
#   - HuggingFaceFW/fineweb-2, cmn_Hani
#
# FineWeb2 provides filtered/deduplicated multilingual web data and
# identifies languages by ISO 639-3 + script (for example cmn_Hani).
# It is distributed under ODC-By 1.0. We only stream a small sample;
# the full dataset is enormous and must NOT be downloaded here.
#
# Important separation:
#   Base pretraining corpus = general modern text.
#   Dialogue/instruction data = a later supervised fine-tuning stage.
#   OpenAssistant is intentionally NOT mixed into this pretraining
#   sample. It will be evaluated separately for the chat-tuning stage.
# ============================================================

SEED = 42
REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "data" / "step38_real_corpus"

# Keep this rehearsal deliberately small. We want to inspect the
# real distribution and tokenizer behavior before moving to 50M+ tokens.
TARGET_CHARS_PER_LANGUAGE = 800_000
MAX_ROWS_PER_LANGUAGE = 20_000
MIN_CHARS = 200
MAX_CHARS_PER_DOCUMENT = 20_000

SOURCES = {
    "english": {
        "dataset": "HuggingFaceFW/fineweb-2",
        "config": "eng_Latn",
        "split": "train",
    },
    "chinese": {
        "dataset": "HuggingFaceFW/fineweb-2",
        "config": "cmn_Hani",
        "split": "train",
    },
}


def normalize_text(text: str) -> str:
    text = str(text).replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def looks_like_noise(text: str) -> bool:
    if len(text) < MIN_CHARS:
        return True

    # Very high URL density is often navigation/boilerplate.
    url_count = len(re.findall(r"https?://", text, flags=re.IGNORECASE))
    if url_count > 12 and url_count / max(1, len(text) / 1000) > 2:
        return True

    # Very repetitive lines are not useful as a first corpus sample.
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) >= 8:
        counts = Counter(lines)
        most_common = counts.most_common(1)[0][1]
        if most_common / len(lines) > 0.5:
            return True

    return False


def language_stats(text: str):
    chinese = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    digits = len(re.findall(r"\d", text))
    spaces = len(re.findall(r"\s", text))
    return {
        "chars": len(text),
        "chinese": chinese,
        "latin": latin,
        "digits": digits,
        "spaces": spaces,
    }


def sample_language(name: str, config: str, target_chars: int):
    print(f"\nSampling {name}: {SOURCES[name]['dataset']} / {config}")
    print("Streaming mode: the full remote dataset is NOT downloaded.")

    ds = load_dataset(
        SOURCES[name]["dataset"],
        config,
        split=SOURCES[name]["split"],
        streaming=True,
    )
    ds = ds.shuffle(seed=SEED, buffer_size=10_000)

    documents = []
    total_chars = 0
    inspected = 0
    rejected = 0

    for row in ds:
        inspected += 1
        if inspected > MAX_ROWS_PER_LANGUAGE:
            break

        text = row.get("text", "")
        if not isinstance(text, str):
            rejected += 1
            continue

        text = normalize_text(text)
        if not text:
            rejected += 1
            continue

        if len(text) > MAX_CHARS_PER_DOCUMENT:
            text = text[:MAX_CHARS_PER_DOCUMENT]

        if looks_like_noise(text):
            rejected += 1
            continue

        documents.append(text)
        total_chars += len(text)

        if total_chars >= target_chars:
            break

    print(f"Inspected rows: {inspected:,}")
    print(f"Accepted docs:  {len(documents):,}")
    print(f"Rejected rows:  {rejected:,}")
    print(f"Characters:     {total_chars:,}")

    return documents


def deduplicate(documents):
    seen = set()
    unique = []
    for text in documents:
        key = re.sub(r"\s+", " ", text).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(text)
    return unique, len(documents) - len(unique)


def write_jsonl(path: Path, documents, language: str):
    with path.open("w", encoding="utf-8") as f:
        for idx, text in enumerate(documents):
            record = {
                "id": f"step38-{language}-{idx:06d}",
                "language": language,
                "text": text,
                "chars": len(text),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_text(path: Path, documents):
    with path.open("w", encoding="utf-8") as f:
        for text in documents:
            f.write(text)
            f.write("\n\n")


def main():
    print("=" * 112)
    print("Step 38: Real modern bilingual corpus sampler")
    print("=" * 112)
    print("This step uses a small STREAMED sample of real public data.")
    print("It is not the final 50M-500M-token pretraining corpus.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    english = sample_language("english", "eng_Latn", TARGET_CHARS_PER_LANGUAGE)
    chinese = sample_language("chinese", "cmn_Hani", TARGET_CHARS_PER_LANGUAGE)

    english, en_dupes = deduplicate(english)
    chinese, zh_dupes = deduplicate(chinese)

    print("\nPart 1: After local exact deduplication")
    print("-" * 112)
    print(f"English: {len(english):,} docs; removed duplicates={en_dupes:,}")
    print(f"Chinese: {len(chinese):,} docs; removed duplicates={zh_dupes:,}")

    en_text = "\n\n".join(english)
    zh_text = "\n\n".join(chinese)
    en_stats = language_stats(en_text)
    zh_stats = language_stats(zh_text)

    print("\nPart 2: Language/content statistics")
    print("-" * 112)
    for name, stats in [("English", en_stats), ("Chinese", zh_stats)]:
        print(
            f"{name:<10} chars={stats['chars']:>9,} "
            f"ChineseChars={stats['chinese']:>8,} "
            f"Latin={stats['latin']:>8,} "
            f"Digits={stats['digits']:>7,}"
        )

    # Build a deliberately conservative first tokenizer corpus:
    # 70% English / 30% Chinese by character budget.
    target_chars = min(len(en_text), len(zh_text) * 7 // 3)
    en_budget = int(target_chars * 0.70)
    zh_budget = target_chars - en_budget

    master_english = en_text[:en_budget]
    master_chinese = zh_text[:zh_budget]
    master = master_english + "\n\n" + master_chinese

    master_path = OUT_DIR / "step38_master_70_30.txt"
    master_path.write_text(master, encoding="utf-8")

    write_jsonl(OUT_DIR / "step38_english.jsonl", english, "english")
    write_jsonl(OUT_DIR / "step38_chinese.jsonl", chinese, "chinese")
    write_text(OUT_DIR / "step38_english.txt", english)
    write_text(OUT_DIR / "step38_chinese.txt", chinese)

    print("\nPart 3: First real tokenizer corpus")
    print("-" * 112)
    print("Chosen starting mixture: approximately 70% English / 30% Chinese by characters.")
    print(f"Master corpus characters: {len(master):,}")
    print(f"English characters:       {len(master_english):,}")
    print(f"Chinese characters:       {len(master_chinese):,}")
    print(f"Saved: {master_path}")

    print("\nPart 4: Sample inspection")
    print("-" * 112)
    for label, docs in [("English", english), ("Chinese", chinese)]:
        print(f"\n[{label} sample]")
        for text in docs[:2]:
            preview = text.replace("\n", " ")[:500]
            print(preview)

    metadata = {
        "step": 38,
        "seed": SEED,
        "sources": SOURCES,
        "license_note": "FineWeb2 is ODC-By 1.0; downstream use should follow the dataset card/license and rights of source material.",
        "target_chars_per_language": TARGET_CHARS_PER_LANGUAGE,
        "max_rows_per_language": MAX_ROWS_PER_LANGUAGE,
        "master_mixture": "70% English / 30% Chinese by character budget",
        "english_docs": len(english),
        "chinese_docs": len(chinese),
        "english_chars": len(en_text),
        "chinese_chars": len(zh_text),
        "master_chars": len(master),
        "next_stage": "Train 4K/8K/16K BPE tokenizers on this real-data sample and compare token efficiency before scaling the corpus.",
        "chat_stage_note": "Keep dialogue/instruction data separate from base pretraining; OpenAssistant will be evaluated later for supervised chat tuning.",
    }
    metadata_path = OUT_DIR / "step38_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nPart 5: Important project decision")
    print("-" * 112)
    print("1. We are now using real public modern text instead of Tiny Shakespeare or hand-written rehearsal text.")
    print("2. We stream the source datasets so the full multi-terabyte corpora are never downloaded.")
    print("3. The first working tokenizer corpus uses a conservative 70/30 English/Chinese character budget.")
    print("4. We do NOT treat 70/30 as the final language ratio; Step 39 will measure tokenizer behavior on real data.")
    print("5. General pretraining text stays separate from dialogue/instruction tuning data.")
    print("6. The final training corpus will be scaled only after the tokenizer and small-model pipeline work end to end.")
    print(f"Metadata: {metadata_path}")
    print("\nStep 38 complete.")


if __name__ == "__main__":
    main()
