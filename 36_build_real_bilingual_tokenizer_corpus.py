import re
from collections import Counter
from pathlib import Path


# ============================================================
# Step 36: Build a realistic bilingual tokenizer corpus
#
# Goal:
#   Create a small, reproducible English/Chinese corpus with
#   controlled language balance for the next tokenizer experiment.
#
# This is NOT the final pretraining dataset. It is a data-engineering
# rehearsal that teaches us how to:
#   1. Define data sources/categories.
#   2. Control English/Chinese proportions.
#   3. Detect obvious low-quality text.
#   4. Remove exact duplicates.
#   5. Keep a held-out validation slice.
#   6. Produce separate 80/20, 70/30, and 50/50 bilingual corpora.
#
# Later, the same pipeline can be replaced with real downloaded
# public datasets without changing the downstream tokenizer code.
# ============================================================

SEED = 42
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "step36_bilingual_corpora"


# A deliberately small but diverse rehearsal corpus.
# The important point is the structure and balancing logic, not corpus size.
ENGLISH_TEXTS = [
    "A useful language model should explain ideas clearly and answer practical questions.",
    "Transformers process sequences of tokens using self attention and feed forward layers.",
    "The training loop predicts the next token, computes loss, and updates the model weights.",
    "Good educational text should be precise, readable, and organized around concrete ideas.",
    "A small model can still learn useful patterns when the data is clean and diverse.",
    "Validation data should remain separate from the training examples used for gradient updates.",
    "Dialogue data teaches the model how questions and answers are structured in a conversation.",
    "Technical documentation contains explanations, examples, configuration values, and code.",
    "A tokenizer converts text into token IDs that a neural network can process.",
    "The goal of this project is not to build the largest model, but a small model that can chat.",
    "Code examples include Python functions, JSON configuration, URLs, and numerical specifications.",
    "A good corpus mixes everyday language, educational writing, dialogue, technical text, and some code.",
    "Quality filtering should remove obvious spam, navigation fragments, repeated boilerplate, and broken text.",
    "The same tokenizer should be used consistently during training, validation, and inference.",
    "A conversational model needs both language modeling data and later instruction or dialogue tuning.",
    "Modern language changes over time, so the corpus should contain relatively recent language where practical.",
]

CHINESE_TEXTS = [
    "一个有用的语言模型应该能够清楚地解释概念，并回答实际问题。",
    "Transformer 通过自注意力机制和前馈网络处理上下文中的 token。",
    "训练过程中，模型预测下一个 token，计算损失，然后更新网络参数。",
    "高质量的教育文本应该准确、清晰，并围绕具体问题组织内容。",
    "只要训练数据足够干净而且具有多样性，小模型也可以学习很多有用的语言模式。",
    "验证集不能参与梯度更新，否则我们就无法可靠地判断模型的泛化能力。",
    "对话数据能够帮助模型学习问题、回答以及多轮上下文之间的关系。",
    "技术文档通常包含解释、示例、配置参数、代码和结构化信息。",
    "Tokenizer 的作用是把文本转换成神经网络可以处理的 token ID。",
    "我们的目标不是训练最大的模型，而是在有限算力下训练一个真正能够基础对话的小模型。",
    "代码、JSON、URL、产品参数和数字信息也是现代文本的重要组成部分。",
    "一个合理的数据集应该混合日常语言、教育内容、对话、技术文本以及少量代码。",
    "数据清洗需要去掉明显的垃圾内容、导航文字、重复模板以及损坏的文本。",
    "训练、验证和推理阶段必须使用一致的 tokenizer，否则 token ID 会失去对应关系。",
    "基础语言模型学习语言规律之后，还需要通过指令数据进一步学习如何回答用户。",
    "现代语言会不断变化，因此在条件允许的情况下应该加入相对较新的文本。",
]

MIXED_TEXTS = [
    "这个 small model should be able to answer 简单的问题。",
    "我们先在 CPU 上测试 training speed，然后再决定是否使用 Colab GPU。",
    "Tokenizer 的 vocab size 会影响 token 数量，也会影响 embedding 和 LM head 的参数量。",
    "一个 bilingual model 需要同时处理 English、中文、code、URL 和 product specifications。",
    "训练 100M tokens 可能需要较长时间，所以我们会先做 10M 和 50M token 实验。",
    "Step 36 的目标是 build a realistic corpus，而不是马上下载巨大的数据集。",
    "产品信息例如 payload 10 kg、flight time 35 min、IP54 都属于 structured technical text。",
    "对话格式可以写成 User: 你好。Assistant: 你好，我可以帮助你分析这个问题。",
]

CODE_AND_STRUCTURED = [
    "def train_step(model, batch):\n    logits = model(batch)\n    loss = cross_entropy(logits[:, :-1], batch[:, 1:])\n    loss.backward()\n    optimizer.step()",
    'config = {"vocab_size": 8000, "context_length": 512, "d_model": 128, "layers": 4}',
    "https://www.example.com/docs/model?step=36&lang=en",
    "payload=10kg; flight_time=35min; voltage=15.2V; power=1200W; ip54=True",
    "Partdro D15R | Movenew P1 | FIMI X8T | DJI Mini 4 Pro",
    "User: What does self-attention do?\nAssistant: It lets each token combine information from relevant positions in its context.",
    "用户：什么是 tokenizer？\n助手：Tokenizer 会把文本切分成模型可以处理的 token，并转换成对应的 token ID。",
]


def is_chinese_char(ch):
    return "\u4e00" <= ch <= "\u9fff"


def text_language_ratio(text):
    letters = sum(ch.isalpha() for ch in text)
    latin = sum(("a" <= ch.lower() <= "z") for ch in text)
    chinese = sum(is_chinese_char(ch) for ch in text)
    denominator = max(1, letters + chinese)
    return chinese / denominator, latin / denominator


def contains_obvious_noise(text):
    stripped = text.strip()
    if len(stripped) < 20:
        return True
    if len(set(stripped)) <= 3:
        return True
    url_count = len(re.findall(r"https?://", stripped))
    if url_count > 5:
        return True
    if re.search(r"(.)\1{12,}", stripped):
        return True
    return False


def normalize(text):
    # Keep content readable while making whitespace consistent.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def deduplicate(items):
    seen = set()
    result = []
    duplicate_count = 0
    for item in items:
        key = re.sub(r"\s+", " ", item.strip()).lower()
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        result.append(item)
    return result, duplicate_count


def prepare_items(items):
    normalized = [normalize(x) for x in items]
    filtered = [x for x in normalized if not contains_obvious_noise(x)]
    deduped, duplicates = deduplicate(filtered)
    return deduped, duplicates, len(normalized) - len(filtered)


def write_lines(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(items) + "\n", encoding="utf-8")


def char_stats(items):
    text = "\n\n".join(items)
    chars = len(text)
    chinese = sum(is_chinese_char(ch) for ch in text)
    latin = sum(("a" <= ch.lower() <= "z") for ch in text)
    return chars, chinese, latin


def approximate_token_language_balance(items):
    # A rough character-level view only. We intentionally do not tokenize here.
    chars, chinese, latin = char_stats(items)
    useful = max(1, chinese + latin)
    return {
        "chars": chars,
        "chinese_ratio": chinese / useful,
        "latin_ratio": latin / useful,
    }


def build_balanced_corpus(en_items, zh_items, mixed_items, structured_items, en_weight):
    """Build a corpus with an approximate English/Chinese language balance.

    en_weight refers to the desired share of the *natural-language* text.
    Mixed and structured examples are always included in smaller amounts.
    """
    zh_weight = 1.0 - en_weight

    en_count = max(1, round(len(en_items) * en_weight))
    zh_count = max(1, round(len(zh_items) * zh_weight))

    # Because the rehearsal lists are short, repeat deterministically and then
    # deduplicate at the document level. In a real pipeline, sampling would be
    # done by source/document rather than repeating the same strings.
    def take_repeated(items, count):
        out = []
        for i in range(count):
            out.append(items[i % len(items)])
        return out

    selected_en = take_repeated(en_items, en_count)
    selected_zh = take_repeated(zh_items, zh_count)

    # Add mixed + structured text equally across all language-ratio variants.
    combined = selected_en + selected_zh + mixed_items + structured_items
    combined, _, _ = prepare_items(combined)

    # Shuffle deterministically without importing random global state.
    # A simple sort by a stable hash-like key keeps runs reproducible.
    combined.sort(key=lambda x: (len(x), x))
    return combined


def report(name, items):
    stats = approximate_token_language_balance(items)
    print(f"{name:<18} docs={len(items):>4} chars={stats['chars']:>7,} "
          f"Chinese={stats['chinese_ratio'] * 100:>6.2f}% "
          f"Latin={stats['latin_ratio'] * 100:>6.2f}%")


def main():
    print("=" * 112)
    print("Step 36: Build a realistic bilingual tokenizer corpus")
    print("=" * 112)
    print("This is a corpus-engineering rehearsal, not the final pretraining dataset.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    en, en_dupes, en_noise = prepare_items(ENGLISH_TEXTS)
    zh, zh_dupes, zh_noise = prepare_items(CHINESE_TEXTS)
    mixed, mixed_dupes, mixed_noise = prepare_items(MIXED_TEXTS)
    structured, structured_dupes, structured_noise = prepare_items(CODE_AND_STRUCTURED)

    print("\nPart 1: Source categories after cleaning")
    print("-" * 112)
    report("English", en)
    report("Chinese", zh)
    report("Mixed", mixed)
    report("Structured", structured)
    print(f"Removed duplicates: English={en_dupes}, Chinese={zh_dupes}, Mixed={mixed_dupes}, Structured={structured_dupes}")
    print(f"Removed obvious noise: English={en_noise}, Chinese={zh_noise}, Mixed={mixed_noise}, Structured={structured_noise}")

    print("\nPart 2: Build language-ratio variants")
    print("-" * 112)
    ratios = [("80_20", 0.80), ("70_30", 0.70), ("50_50", 0.50)]
    built = {}

    for name, en_weight in ratios:
        items = build_balanced_corpus(en, zh, mixed, structured, en_weight)
        built[name] = items
        report(name, items)
        path = OUT_DIR / f"step36_{name}.txt"
        write_lines(path, items)
        print(f"Saved: {path}")

    print("\nPart 3: Inspect representative samples")
    print("-" * 112)
    for name in ["80_20", "70_30", "50_50"]:
        print(f"\n[{name}]")
        for item in built[name][:5]:
            print(f"  {item[:180].replace(chr(10), ' | ')}")

    print("\nPart 4: Validation split")
    print("-" * 112)
    for name, items in built.items():
        split = max(1, len(items) // 5)
        train_items = items[split:]
        val_items = items[:split]
        train_path = OUT_DIR / f"step36_{name}_train.txt"
        val_path = OUT_DIR / f"step36_{name}_validation.txt"
        write_lines(train_path, train_items)
        write_lines(val_path, val_items)
        print(f"{name}: train_docs={len(train_items):>3}, validation_docs={len(val_items):>3}")
        print(f"       train={train_path}")
        print(f"       val  ={val_path}")

    print("\nPart 5: Data-quality checks for the future real pipeline")
    print("-" * 112)
    checks = [
        "Unicode should be preserved; do not accidentally decode UTF-8 bytes as text.",
        "Exact and near-duplicate removal should happen before tokenizer training.",
        "Boilerplate, navigation fragments, spam, broken pages, and repeated templates should be filtered.",
        "Train/validation sources must be separated so validation is not leaked into training.",
        "Language balance should be measured on the actual corpus, not assumed from document counts.",
        "Tokenizer training data should represent the same languages and domains the model will see later.",
        "The final pretraining corpus will be sampled by source/document quality, not by repeating tiny examples.",
    ]
    for idx, check in enumerate(checks, start=1):
        print(f"{idx}. {check}")

    print("\nPart 6: Project direction")
    print("-" * 112)
    print("For the final small conversational model, we will likely use a bilingual corpus rather than English-only data.")
    print("Step 35 showed that bilingual tokenizer training can greatly improve Chinese efficiency.")
    print("Step 36 now gives us a repeatable way to test different English/Chinese corpus ratios.")
    print("The next tokenizer experiment can train 4K / 8K / 16K BPEs on these controlled corpora.")
    print("After that, we can replace the rehearsal text with real public datasets while keeping the same pipeline structure.")

    print("\nStep 36 complete.")


if __name__ == "__main__":
    main()
