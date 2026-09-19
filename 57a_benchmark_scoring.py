import argparse
import json
import re
from pathlib import Path


def normalize(text):
    text = str(text or "").lower().strip()
    text = text.replace("。", ".").replace("！", "!").replace("？", "?")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"""["'“”‘’]""", "", text)
    text = re.sub(r"[.,!?;:，。！？；：]+$", "", text)
    return text


def keyword_coverage(output, keywords):
    if not keywords:
        return 1.0
    text = normalize(output)
    hits = sum(1 for k in keywords if normalize(k) in text)
    return hits / len(keywords)


def strict_score(item):
    expected = item.get("expected")
    if expected is None:
        return None
    return normalize(item.get("output")) == normalize(expected)


def semantic_proxy_score(item, threshold=0.66):
    """
    A transparent semantic proxy, not a neural semantic similarity model.

    For items with curated keywords:
        score = fraction of expected concepts/phrases present.
        pass when coverage >= threshold.

    For items without keywords but with an expected answer:
        fall back to strict normalized exact match.

    This keeps the benchmark deterministic and dependency-free.
    """
    keywords = item.get("keywords") or []
    if keywords:
        coverage = keyword_coverage(item.get("output"), keywords)
        return {
            "score": coverage,
            "passed": coverage >= threshold,
            "method": "keyword_coverage",
        }

    strict = strict_score(item)
    return {
        "score": 1.0 if strict else 0.0,
        "passed": bool(strict),
        "method": "exact_fallback",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/content/drive/MyDrive/transformers_exercise_20260911/"
                "artifacts/step56/tiny_gpt_step54_best_benchmark.json",
    )
    parser.add_argument("--threshold", type=float, default=0.66)
    parser.add_argument("--save-json", action="store_true")
    args, _ = parser.parse_known_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Benchmark JSON not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    results = payload.get("results", [])
    if not results:
        raise ValueError("No benchmark results found in JSON.")

    enriched = []
    categories = {}

    for item in results:
        strict = strict_score(item)
        semantic = semantic_proxy_score(item, args.threshold)

        rec = dict(item)
        rec["strict_pass"] = bool(strict) if strict is not None else None
        rec["strict_score"] = (
            1.0 if strict else 0.0
        ) if strict is not None else None
        rec["semantic_proxy_score"] = semantic["score"]
        rec["semantic_proxy_pass"] = semantic["passed"]
        rec["semantic_proxy_method"] = semantic["method"]

        enriched.append(rec)

        cat = item["category"]
        stats = categories.setdefault(
            cat,
            {
                "total": 0,
                "strict_total": 0,
                "strict_passed": 0,
                "semantic_passed": 0,
                "semantic_score_sum": 0.0,
            },
        )
        stats["total"] += 1
        if strict is not None:
            stats["strict_total"] += 1
            stats["strict_passed"] += int(strict)
        stats["semantic_passed"] += int(semantic["passed"])
        stats["semantic_score_sum"] += semantic["score"]

    strict_items = [r for r in enriched if r["strict_pass"] is not None]
    strict_passed = sum(r["strict_pass"] for r in strict_items)
    semantic_passed = sum(r["semantic_proxy_pass"] for r in enriched)
    semantic_mean = sum(r["semantic_proxy_score"] for r in enriched) / len(enriched)

    print("=" * 112)
    print("Node 57A — TinyGPT dual-metric benchmark scoring")
    print("=" * 112)
    print("input:", input_path)
    print("items:", len(enriched))
    print(f"semantic proxy threshold: {args.threshold:.0%}")

    print("\nOverall")
    print("-" * 112)
    if strict_items:
        print(
            f"Strict exact-match:   {strict_passed}/{len(strict_items)} = "
            f"{strict_passed / len(strict_items):.2%}"
        )
    else:
        print("Strict exact-match:   no exact-answer items")
    print(
        f"Semantic-proxy pass:   {semantic_passed}/{len(enriched)} = "
        f"{semantic_passed / len(enriched):.2%}"
    )
    print(f"Semantic-proxy mean:   {semantic_mean:.2%}")

    print("\nBy category")
    print("-" * 112)
    for cat in sorted(categories):
        s = categories[cat]
        strict_text = "n/a"
        if s["strict_total"]:
            strict_text = f"{s['strict_passed']}/{s['strict_total']} = {s['strict_passed']/s['strict_total']:.2%}"
        sem_rate = s["semantic_passed"] / s["total"]
        sem_mean = s["semantic_score_sum"] / s["total"]
        print(
            f"{cat:<30} "
            f"strict {strict_text:<18} | "
            f"semantic {s['semantic_passed']:>3}/{s['total']:<3} = {sem_rate:.2%} "
            f"| mean {sem_mean:.2%}"
        )

    print("\nOpen-ended semantic examples")
    print("-" * 112)
    shown = 0
    for r in enriched:
        if r["keywords"] and not r["strict_pass"]:
            print(
                f"[{r['category']}] {r['prompt']}\n"
                f"  output:  {r['output']}\n"
                f"  keywords: {r['keywords']}\n"
                f"  coverage: {r['semantic_proxy_score']:.0%} "
                f"-> {'PASS' if r['semantic_proxy_pass'] else 'FAIL'}"
            )
            shown += 1
            if shown >= 12:
                break

    if args.save_json:
        out_dir = input_path.parent
        out_path = out_dir / (input_path.stem + "_dual_metrics.json")
        output_payload = {
            "source_benchmark": str(input_path),
            "semantic_proxy_threshold": args.threshold,
            "items": len(enriched),
            "strict_exact_match_rate": (
                strict_passed / len(strict_items) if strict_items else None
            ),
            "semantic_proxy_pass_rate": semantic_passed / len(enriched),
            "semantic_proxy_mean": semantic_mean,
            "by_category": categories,
            "results": enriched,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=2)
        print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
