import json
import re
from collections import Counter
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("The 'datasets' package is not installed.")
    print("Install it with: pip install datasets")
    raise SystemExit(1)

SEED = 42
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "step38_real_corpus"
TARGET_CHARS = 800_000
MAX_ROWS = 20_000
MIN_CHARS = 200
MAX_DOC_CHARS = 20_000

# Current public dataset split: English comes from FineWeb; Mandarin from FineWeb2.
SOURCES = {
    "english": ("HuggingFaceFW/fineweb", None),
    "chinese": ("HuggingFaceFW/fineweb-2", "cmn_Hani"),
}


def normalize(text: str) -> str:
    text = str(text).replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_noise(text: str) -> bool:
    if len(text) < MIN_CHARS:
        return True
    urls = len(re.findall(r"https?://", text, re.I))
    if urls > 12 and urls / max(1, len(text) / 1000) > 2:
        return True
    lines = [x.strip() for x in text.split("\n") if x.strip()]
    if len(lines) >= 8:
        top = Counter(lines).most_common(1)[0][1]
        if top / len(lines) > 0.5:
            return True
    return False


def sample(name: str):
    dataset, config = SOURCES[name]
    print(f"\nSampling {name}: {dataset}" + (f" / {config}" if config else ""))
    print("Streaming mode: the full remote dataset is NOT downloaded.")
    kwargs = {"split": "train", "streaming": True}
    if config:
        kwargs["name"] = config
    ds = load_dataset(dataset, **kwargs).shuffle(seed=SEED, buffer_size=10_000)

    docs = []
    total = 0
    inspected = 0
    rejected = 0
    for row in ds:
        inspected += 1
        if inspected > MAX_ROWS:
            break
        text = row.get("text", "")
        if not isinstance(text, str):
            rejected += 1
            continue
        text = normalize(text)
        if not text:
            rejected += 1
            continue
        if len(text) > MAX_DOC_CHARS:
            text = text[:MAX_DOC_CHARS]
        if is_noise(text):
            rejected += 1
            continue
        docs.append(text)
        total += len(text)
        if total >= TARGET_CHARS:
            break
    print(f"Inspected rows: {inspected:,}")
    print(f"Accepted docs:  {len(docs):,}")
    print(f"Rejected rows:  {rejected:,}")
    print(f"Characters:     {total:,}")
    return docs


def dedup(docs):
    seen = set()
    out = []
    for text in docs:
        key = re.sub(r"\s+", " ", text).strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(text)
    return out


def write_text(path: Path, docs):
    path.write_text("\n\n".join(docs) + "\n", encoding="utf-8")


def main():
    print("=" * 112)
    print("Step 38b: Real modern bilingual corpus sampler (fixed)")
    print("=" * 112)
    print("English: HuggingFaceFW/fineweb")
    print("Chinese: HuggingFaceFW/fineweb-2 / cmn_Hani")
    print("Streaming: full remote datasets are NOT downloaded.")
    OUT.mkdir(parents=True, exist_ok=True)

    en = dedup(sample("english"))
    zh = dedup(sample("chinese"))

    en_text = "\n\n".join(en)
    zh_text = "\n\n".join(zh)
    print("\nPart 1: Deduplicated sample")
    print(f"English docs={len(en):,} chars={len(en_text):,}")
    print(f"Chinese docs={len(zh):,} chars={len(zh_text):,}")

    # First controlled tokenizer corpus: 70/30 by character budget.
    target = min(len(en_text), len(zh_text) * 7 // 3)
    en_budget = int(target * 0.70)
    zh_budget = target - en_budget
    master_en = en_text[:en_budget]
    master_zh = zh_text[:zh_budget]
    master = master_en + "\n\n" + master_zh

    write_text(OUT / "step38b_english.txt", en)
    write_text(OUT / "step38b_chinese.txt", zh)
    (OUT / "step38b_master_70_30.txt").write_text(master, encoding="utf-8")

    records = []
    for lang, docs in (("english", en), ("chinese", zh)):
        for i, text in enumerate(docs):
            records.append({"id": f"step38b-{lang}-{i:06d}", "language": lang, "chars": len(text), "text": text})
    with (OUT / "step38b_documents.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metadata = {
        "step": "38b",
        "seed": SEED,
        "sources": SOURCES,
        "master_ratio": "70% English / 30% Chinese by character budget",
        "english_docs": len(en),
        "chinese_docs": len(zh),
        "english_chars": len(en_text),
        "chinese_chars": len(zh_text),
        "master_chars": len(master),
        "note": "This is a real-data tokenizer rehearsal sample, not the final pretraining corpus.",
    }
    (OUT / "step38b_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nPart 2: Master tokenizer corpus")
    print(f"English characters: {len(master_en):,}")
    print(f"Chinese characters: {len(master_zh):,}")
    print(f"Total characters:   {len(master):,}")
    print(f"Saved: {OUT / 'step38b_master_70_30.txt'}")
    print("\nStep 38b complete.")


if __name__ == "__main__":
    main()
