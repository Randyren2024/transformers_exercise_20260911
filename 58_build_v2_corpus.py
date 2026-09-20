import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from datasets import load_dataset


SEED = 58
TRAIN_RATIO = 0.95

TARGETS = {
    "fineweb_en": 30_000_000,
    "fineweb2_zh": 20_000_000,
    "wiki_en": 30_000_000,
    "wiki_zh": 20_000_000,
}

MIN_LEN = {
    "fineweb_en": 300,
    "fineweb2_zh": 120,
    "wiki_en": 250,
    "wiki_zh": 120,
}


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fingerprint(text):
    normalized = text.lower().strip()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def get_field(row, *names):
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def stream_source(name):
    if name == "fineweb_en":
        # Avoid FineWeb default: it currently expands to tens of thousands
        # of Parquet files before streaming can begin.
        return load_dataset(
            "HuggingFaceFW/fineweb",
            "sample-10BT",
            split="train",
            streaming=True,
        )
    if name == "fineweb2_zh":
        # FineWeb-2 currently has no sample-* configs.
        # Keep streaming and start consuming rows directly.
        return load_dataset(
            "HuggingFaceFW/fineweb-2",
            "cmn_Hani",
            split="train",
            streaming=True,
        )

    if name == "wiki_en":
        return load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=True,
        ).shuffle(seed=SEED, buffer_size=10_000)

    if name == "wiki_zh":
        return load_dataset(
            "wikimedia/wikipedia",
            "20231101.zh",
            split="train",
            streaming=True,
        ).shuffle(seed=SEED, buffer_size=10_000)

    raise ValueError(name)


def prepare_text(name, row):
    if name in ("wiki_en", "wiki_zh"):
        title = clean_text(get_field(row, "title"))
        body = clean_text(get_field(row, "text", "content"))
        if title and body:
            return f"{title}\n\n{body}"
        return body or title

    return clean_text(get_field(row, "text", "content"))


def collect_source(name, target_chars):
    accepted = []
    seen = set()
    total = 0
    scanned = 0

    dataset = stream_source(name)

    for row in dataset:
        scanned += 1
        text = prepare_text(name, row)

        if len(text) < MIN_LEN[name]:
            continue

        fp = fingerprint(text)
        if fp in seen:
            continue

        seen.add(fp)
        accepted.append(text)
        total += len(text)

        if total >= target_chars:
            break

    return {
        "name": name,
        "texts": accepted,
        "accepted_docs": len(accepted),
        "accepted_chars": total,
        "scanned_rows": scanned,
    }


def split_documents(texts, seed, ratio):
    items = list(texts)
    random.Random(seed).shuffle(items)
    cut = int(len(items) * ratio)
    return items[:cut], items[cut:]


def write_corpus(out_dir, train_texts, val_texts):
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "step58_train.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(train_texts))

    with open(out_dir / "step58_validation.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(val_texts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="/content/drive/MyDrive/transformers_exercise_20260911/"
                "data/step58_v2_corpus",
    )
    args, _ = parser.parse_known_args()

    random.seed(SEED)
    out_dir = Path(args.output)

    print("=" * 112)
    print("Node 58 — TinyGPT v2 knowledge-focused corpus builder")
    print("=" * 112)

    all_train = []
    all_val = []
    manifest = {
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "targets": TARGETS,
        "sources": {},
    }

    for name, target in TARGETS.items():
        print(f"\nCollecting: {name}")
        print("-" * 112)
        result = collect_source(name, target)

        train_docs, val_docs = split_documents(
            result["texts"],
            seed=SEED + len(manifest["sources"]),
            ratio=TRAIN_RATIO,
        )

        all_train.extend((name, x) for x in train_docs)
        all_val.extend((name, x) for x in val_docs)

        manifest["sources"][name] = {
            "target_chars": target,
            "accepted_docs": result["accepted_docs"],
            "accepted_chars": result["accepted_chars"],
            "scanned_rows": result["scanned_rows"],
            "train_docs": len(train_docs),
            "validation_docs": len(val_docs),
            "train_chars": sum(len(x) for x in train_docs),
            "validation_chars": sum(len(x) for x in val_docs),
        }

        print(f"Scanned rows:      {result['scanned_rows']:,}")
        print(f"Accepted docs:     {result['accepted_docs']:,}")
        print(f"Accepted chars:    {result['accepted_chars']:,}")
        print(f"Train docs:        {len(train_docs):,}")
        print(f"Validation docs:   {len(val_docs):,}")

    random.shuffle(all_train)
    random.shuffle(all_val)

    train_texts = [x for _, x in all_train]
    val_texts = [x for _, x in all_val]

    total_train_chars = sum(len(x) for x in train_texts)
    total_val_chars = sum(len(x) for x in val_texts)
    total_chars = total_train_chars + total_val_chars

    manifest["totals"] = {
        "train_docs": len(train_texts),
        "validation_docs": len(val_texts),
        "train_chars": total_train_chars,
        "validation_chars": total_val_chars,
        "total_chars": total_chars,
        "english_chars": (
            manifest["sources"]["fineweb_en"]["accepted_chars"]
            + manifest["sources"]["wiki_en"]["accepted_chars"]
        ),
        "chinese_chars": (
            manifest["sources"]["fineweb2_zh"]["accepted_chars"]
            + manifest["sources"]["wiki_zh"]["accepted_chars"]
        ),
    }

    write_corpus(out_dir, train_texts, val_texts)

    with open(out_dir / "step58_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 112)
    print("Node 58 corpus complete")
    print("=" * 112)
    print(f"Train docs:        {len(train_texts):,}")
    print(f"Validation docs:   {len(val_texts):,}")
    print(f"Train chars:       {total_train_chars:,}")
    print(f"Validation chars:  {total_val_chars:,}")
    print(f"Total chars:       {total_chars:,}")
    print(f"English chars:     {manifest['totals']['english_chars']:,}")
    print(f"Chinese chars:     {manifest['totals']['chinese_chars']:,}")
    print(f"Output:            {out_dir}")


if __name__ == "__main__":
    main()