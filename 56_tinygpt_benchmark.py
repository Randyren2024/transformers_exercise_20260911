import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer


VOCAB = 8000
D = 256
H = 8
LAYERS = 8
FF = 1024
CTX = 256


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        hd = D // H
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.out = nn.Linear(D, D, bias=False)
        self.hd = hd

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, H, self.hd).transpose(1, 2)
        k = k.view(b, t, H, self.hd).transpose(1, 2)
        v = v.view(b, t, H, self.hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).contiguous().view(b, t, D))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.attn = Attn()
        self.ln2 = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, FF, bias=False)
        self.fc2 = nn.Linear(FF, D, bias=False)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D)
        self.pos = nn.Embedding(CTX, D)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, VOCAB, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, x):
        t = x.size(1)
        pos = torch.arange(t, device=x.device)
        h = self.tok(x) + self.pos(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))


def normalize(text):
    text = text.lower().strip()
    text = text.replace("。", ".").replace("！", "!").replace("？", "?")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"["'“”‘’]", "", text)
    text = re.sub(r"[.,!?;:，。！？；：]+$", "", text)
    return text


def exact_match(text, expected):
    return normalize(text) == normalize(expected)


def contains_all(text, keywords):
    n = normalize(text)
    return all(normalize(k) in n for k in keywords)


def starts_with(text, prefix):
    return normalize(text).startswith(normalize(prefix))


def build_benchmark():
    rows = []

    def add(cat, lang, prompt, expected=None, keywords=None, prefix=None):
        rows.append({
            "category": cat,
            "language": lang,
            "prompt": prompt,
            "expected": expected,
            "keywords": keywords,
            "prefix": prefix,
        })

    # 1. Exact bilingual translation
    translations = [
        ("robot", "机器人"),
        ("data", "数据"),
        ("software", "软件"),
        ("hardware", "硬件"),
        ("computer", "电脑"),
        ("language model", "语言模型"),
        ("machine learning", "机器学习"),
        ("attention", "注意力"),
        ("question", "问题"),
        ("answer", "答案"),
        ("price", "价格"),
        ("product", "产品"),
        ("customer", "客户"),
        ("market", "市场"),
        ("network", "网络"),
        ("image", "图像"),
        ("video", "视频"),
        ("browser", "浏览器"),
        ("website", "网站"),
        ("dataset", "数据集"),
    ]
    for en, zh in translations:
        add("translation_en_zh", "en", f"Translate '{en}' into Chinese.", expected=zh + "。")
        add("translation_zh_zh", "zh", f"把“{en}”翻译成中文。", expected=zh + "。")

    # 2. Short factual QA
    facts_en = [
        ("How many days are in a week?", "Seven.", ["seven"]),
        ("How many months are in a year?", "Twelve.", ["twelve"]),
        ("What color is grass usually?", "Green.", ["green"]),
        ("What do bees make?", "Honey.", ["honey"]),
        ("What is the opposite of hot?", "Cold.", ["cold"]),
        ("What is the opposite of big?", "Small.", ["small"]),
        ("What do we use to see?", "Our eyes.", ["eyes"]),
        ("What do we use to hear?", "Our ears.", ["ears"]),
        ("What is a robot?", "A robot is a machine.", ["robot", "machine"]),
        ("What is software?", "Software is the programs and data that run on a computer.", ["software", "programs", "data"]),
        ("What is data?", "Data is information that can be stored and processed.", ["data", "information"]),
        ("What is a database?", "A database stores and organizes data.", ["database", "stores", "data"]),
        ("What is an API?", "An API is an interface that lets software systems communicate.", ["api", "interface", "software"]),
        ("What is a GPU?", "A GPU is a processor designed for parallel numerical operations.", ["gpu", "processor", "parallel"]),
        ("What is a CPU?", "A CPU is a general-purpose processor.", ["cpu", "processor"]),
    ]
    for q, expected, kws in facts_en:
        add("factual_qa_en", "en", q, expected=expected, keywords=kws)

    facts_zh = [
        ("一周有几天？", "七天。", ["七天"]),
        ("一年有几个月？", "十二个月。", ["十二个月"]),
        ("草通常是什么颜色？", "绿色。", ["绿色"]),
        ("蜜蜂生产什么？", "蜂蜜。", ["蜂蜜"]),
        ("热的反义词是什么？", "冷。", ["冷"]),
        ("大的反义词是什么？", "小。", ["小"]),
        ("我们用什么看东西？", "眼睛。", ["眼睛"]),
        ("我们用什么听声音？", "耳朵。", ["耳朵"]),
        ("什么是机器人？", "机器人是机器。", ["机器人", "机器"]),
        ("什么是数据？", "数据是可以存储和处理的信息。", ["数据", "信息"]),
        ("什么是数据库？", "数据库用于存储和组织数据。", ["数据库", "存储", "数据"]),
        ("什么是 API？", "API 是让不同软件系统进行通信的接口。", ["api", "软件", "接口"]),
    ]
    for q, expected, kws in facts_zh:
        add("factual_qa_zh", "zh", q, expected=expected, keywords=kws)

    # 3. Identity / instruction following
    identity = [
        ("What can you do?", "I can answer simple questions and follow short instructions.", ["answer simple questions", "short instructions"]),
        ("你能做什么？", "我可以回答简单问题并执行简短指令。", ["回答简单问题", "简短指令"]),
        ("Who are you?", "I am a small bilingual language model.", ["small bilingual language model"]),
        ("你是谁？", "我是一个小型双语语言模型。", ["小型双语语言模型"]),
        ("Are you a human?", "No. I am a language model.", ["language model"]),
        ("你是人类吗？", "不是。我是一个语言模型。", ["语言模型"]),
        ("Do you understand Chinese?", "Yes. I can understand and generate Chinese and English.", ["understand", "Chinese", "English"]),
        ("你懂中文吗？", "是的。我可以理解并生成中文和英文。", ["理解", "中文", "英文"]),
        ("Do you understand English?", "Yes. I can understand and generate English.", ["understand", "English"]),
        ("What should you do when unsure?", "Say that you are not sure instead of pretending to know.", ["not sure"]),
        ("不确定答案时应该怎么办？", "应该说明自己不确定，而不是假装知道。", ["不确定", "假装"]),
        ("Please answer briefly: What is a robot?", "A robot is a machine.", ["robot", "machine"]),
    ]
    for q, expected, kws in identity:
        lang = "zh" if re.search(r"[\u4e00-\u9fff]", q) else "en"
        add("identity_instruction", lang, q, expected=expected, keywords=kws)

    # 4. Definitions and concepts
    concepts = [
        ("Explain artificial intelligence in simple terms.", ["artificial intelligence", "technology", "computers"]),
        ("What is machine learning?", ["machine learning", "learn", "data"]),
        ("What is a language model?", ["language model", "context", "tokens"]),
        ("What is attention?", ["attention", "input"]),
        ("What is a transformer?", ["transformer", "attention", "neural"]),
        ("请用简单中文解释什么是人工智能。", ["人工智能", "计算机", "技术"]),
        ("什么是机器学习？", ["机器学习", "数据", "规律"]),
        ("什么是语言模型？", ["语言模型", "上下文", "词元"]),
        ("什么是注意力？", ["注意力", "输入"]),
        ("什么是 Transformer？", ["transformer", "注意力", "神经网络"]),
    ]
    for q, kws in concepts:
        lang = "zh" if re.search(r"[\u4e00-\u9fff]", q) else "en"
        add("concept_explanation", lang, q, keywords=kws)

    # 5. Arithmetic: exact short answers. These are intentionally held out from the
    # most obvious Step 53/54 probe values to test generalization.
    arithmetic = [
        ("What is 3 + 14?", "17."),
        ("What is 8 + 27?", "35."),
        ("What is 19 + 24?", "43."),
        ("What is 36 + 17?", "53."),
        ("What is 42 + 29?", "71."),
        ("11 加 23 等于多少？", "34。"),
        ("18 加 26 等于多少？", "44。"),
        ("27 加 35 等于多少？", "62。"),
        ("54 减 17 等于多少？", "37。"),
        ("What is 63 - 28?", "35."),
        ("What is 79 - 34?", "45."),
        ("41 减 19 等于多少？", "22。"),
        ("What is 6 times 7?", "42."),
        ("What is 8 times 9?", "72."),
        ("What is 12 times 6?", "72."),
        ("7 乘 8 等于多少？", "56。"),
        ("9 乘 7 等于多少？", "63。"),
        ("72 divided by 8 equals what?", "9."),
        ("What is 81 divided by 9?", "9."),
        ("72 除以 8 等于多少？", "9。"),
    ]
    for q, expected in arithmetic:
        lang = "zh" if re.search(r"[\u4e00-\u9fff]", q) else "en"
        add("arithmetic_exact", lang, q, expected=expected)

    # 6. Simple instruction transformations
    tasks = [
        ("Say hello in a friendly way.", ["hello"]),
        ("Say thank you politely.", ["thank", "you"]),
        ("Give me one simple study tip.", ["study"]),
        ("Give me one simple programming tip.", ["code"]),
        ("Why is sleep important? Answer in one sentence.", ["sleep"]),
        ("为什么睡眠重要？用一句话回答。", ["睡眠"]),
        ("为什么运动有用？用一句话回答。", ["运动"]),
        ("Tell me one benefit of exercise in one sentence.", ["exercise"]),
    ]
    for q, kws in tasks:
        lang = "zh" if re.search(r"[\u4e00-\u9fff]", q) else "en"
        add("short_instruction", lang, q, keywords=kws)

    # 7. Cross-lingual consistency: ask for the same concept in both languages.
    cross = [
        ("What is a robot?", ["robot", "machine"]),
        ("什么是机器人？", ["机器人", "机器"]),
        ("What is software?", ["software", "programs"]),
        ("什么是软件？", ["软件", "程序"]),
        ("What is data?", ["data", "information"]),
        ("什么是数据？", ["数据", "信息"]),
    ]
    for q, kws in cross:
        lang = "zh" if re.search(r"[\u4e00-\u9fff]", q) else "en"
        add("cross_lingual_consistency", lang, q, keywords=kws)

    return rows


@torch.inference_mode()
def generate(model, tokenizer, prompt, device, max_new=48):
    ids = tokenizer.encode(f"User: {prompt}\nAssistant:").ids
    ids = ids[-CTX:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    start = x.size(1)

    for _ in range(max_new):
        logits = model(x[:, -CTX:])[:, -1, :]
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        x = torch.cat([x, nxt], dim=1)
        text = tokenizer.decode(x[0].tolist()[start:]).strip()
        if "[END]" in text:
            return text.split("[END]", 1)[0].strip()
    return tokenizer.decode(x[0].tolist()[start:]).strip()


def load_model(ckpt_path, device):
    model = TinyGPT().to(device)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def score_item(output, item):
    if item["expected"] is not None:
        exact = exact_match(output, item["expected"])
    else:
        exact = False

    keyword_pass = False
    if item["keywords"]:
        keyword_pass = contains_all(output, item["keywords"])

    prefix_pass = False
    if item["prefix"]:
        prefix_pass = starts_with(output, item["prefix"])

    if item["expected"] is not None:
        passed = exact
    elif item["keywords"]:
        passed = keyword_pass
    elif item["prefix"]:
        passed = prefix_pass
    else:
        passed = False

    return {
        "passed": passed,
        "exact": exact,
        "keyword_pass": keyword_pass,
        "prefix_pass": prefix_pass,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="step54")
    parser.add_argument("--max-new", type=int, default=48)
    parser.add_argument("--save-json", action="store_true")
    args, _ = parser.parse_known_args()

    drive = Path("/content/drive/MyDrive/transformers_exercise_20260911")
    tok_path = drive / "artifacts" / "step43" / "step43_bpe_8000.json"

    if args.checkpoint.endswith(".pt"):
        ckpt = Path(args.checkpoint)
    else:
        ckpt = drive / "artifacts" / args.checkpoint / f"tiny_gpt_{args.checkpoint}_best.pt"

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(str(tok_path))
    benchmark = build_benchmark()

    print("=" * 112)
    print("Node 56 — TinyGPT Benchmark")
    print("=" * 112)
    print("checkpoint:", ckpt)
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("benchmark items:", len(benchmark))

    model = load_model(ckpt, device)

    by_category = {}
    results = []

    for i, item in enumerate(benchmark, 1):
        output = generate(model, tokenizer, item["prompt"], device, args.max_new)
        score = score_item(output, item)

        rec = {
            "id": i,
            **item,
            "output": output,
            **score,
        }
        results.append(rec)

        cat = item["category"]
        by_category.setdefault(cat, {"total": 0, "passed": 0})
        by_category[cat]["total"] += 1
        by_category[cat]["passed"] += int(score["passed"])

        print(f"[{i:03d}/{len(benchmark)}] {item['category']:<28} {'PASS' if score['passed'] else 'FAIL':4} | {item['prompt']}")
        print(f"    -> {output}")

    total = len(results)
    passed = sum(r["passed"] for r in results)
    exact_total = sum(r["exact"] for r in results)
    keyword_total = sum(r["keyword_pass"] for r in results)

    print("\n" + "=" * 112)
    print("Benchmark summary")
    print("=" * 112)
    print(f"Overall:          {passed}/{total} = {passed / total:.2%}")
    print(f"Exact-match:      {exact_total}/{total} = {exact_total / total:.2%}")
    print(f"Keyword checks:   {keyword_total}/{total} = {keyword_total / total:.2%}")

    for cat in sorted(by_category):
        s = by_category[cat]
        print(f"{cat:<30} {s['passed']:>3}/{s['total']:<3} = {s['passed'] / s['total']:.2%}")

    if args.save_json:
        out_dir = drive / "artifacts" / "step56"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(ckpt).stem
        out_path = out_dir / f"{safe_name}_benchmark.json"
        payload = {
            "checkpoint": str(ckpt),
            "tokenizer": str(tok_path),
            "device": str(device),
            "items": len(results),
            "overall_pass_rate": passed / total,
            "exact_match_rate": exact_total / total,
            "keyword_pass_rate": keyword_total / total,
            "by_category": by_category,
            "results": results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
