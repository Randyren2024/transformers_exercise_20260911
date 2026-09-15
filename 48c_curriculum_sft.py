import argparse
import copy
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

# Step 48c: Curated bilingual curriculum SFT
# Start from the Step 45 base model, not from the overfit Step 48 model.
# The dataset is generated locally in a controlled, short-answer curriculum.

SEED = 42
VOCAB_SIZE = 8000
D_MODEL = 192
N_HEADS = 6
N_LAYERS = 6
D_FF = 768
CONTEXT_LENGTH = 256

BATCH_SIZE = 64
DEFAULT_STEPS = 900
LEARNING_RATE = 5e-6
MIN_LEARNING_RATE = 1e-6
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.01

LOG_INTERVAL = 25
EVAL_INTERVAL = 100


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_pair(rows, en_user, en_answer, zh_user, zh_answer):
    rows.append({"user": en_user, "assistant": en_answer, "lang": "en"})
    rows.append({"user": zh_user, "assistant": zh_answer, "lang": "zh"})


def build_curriculum():
    rows = []

    concepts = [
        ("artificial intelligence", "technology that lets computers perform tasks that normally require human intelligence", "人工智能", "让计算机执行通常需要人类智能的任务的一种技术"),
        ("machine learning", "a method that lets computers learn patterns from data", "机器学习", "让计算机从数据中学习规律的一种方法"),
        ("a language model", "a model that predicts likely tokens from context", "语言模型", "根据上下文预测下一个可能词元的模型"),
        ("a token", "a small unit of text processed by a language model", "token", "语言模型处理的一小段文本单位"),
        ("a dataset", "a collection of examples used for training or evaluation", "数据集", "一组用于训练或评估的样本"),
        ("training", "the process of updating model parameters to improve performance", "训练", "更新模型参数以提高模型表现的过程"),
        ("inference", "using a trained model to produce an answer for new input", "推理", "使用训练好的模型处理新输入并生成答案"),
        ("overfitting", "learning the training examples too closely and performing poorly on new data", "过拟合", "过度记住训练样本而在新数据上表现不好的现象"),
        ("attention", "a mechanism that lets a model focus on relevant parts of the input", "注意力", "让模型关注输入中相关部分的一种机制"),
        ("a transformer", "a neural network architecture built around attention", "Transformer", "一种以注意力机制为核心的神经网络架构"),
        ("a GPU", "a processor designed to handle many calculations in parallel", "GPU", "适合并行执行大量计算的处理器"),
        ("a CPU", "a general-purpose processor for many kinds of computation", "CPU", "可以执行多种计算任务的通用处理器"),
        ("an algorithm", "a defined procedure for solving a problem", "算法", "用于解决问题的一套明确步骤"),
        ("a variable", "a named value that can change in a program", "变量", "程序中可以改变其值的命名数据"),
        ("a function", "a reusable block of code that performs a task", "函数", "执行特定任务的一段可重复使用的代码"),
        ("a database", "a system used to store and organize data", "数据库", "用于存储和组织数据的系统"),
        ("a file", "a named container for storing digital information", "文件", "用于保存数字信息的命名数据容器"),
        ("a network", "a system that connects computers or devices so they can communicate", "网络", "连接计算机或设备并让它们通信的系统"),
        ("a browser", "software used to access and view websites", "浏览器", "用于访问和查看网站的软件"),
        ("an API", "an interface that lets software systems communicate", "API", "让不同软件系统进行通信的接口"),
    ]

    for en, ans, zh, zans in concepts:
        variants_en = [
            f"What is {en}?",
            f"Explain {en} in simple terms.",
            f"Give a short definition of {en}.",
            f"In one sentence, what is {en}?",
        ]
        variants_zh = [
            f"什么是{zh}？",
            f"请用简单的话解释{zh}。",
            f"请简要定义{zh}。",
            f"请用一句话说明什么是{zh}。",
        ]
        for qe, qz in zip(variants_en, variants_zh):
            add_pair(rows, qe, ans.capitalize() + ".", qz, zans + "。")

    for a in range(1, 41):
        b = (a * 7) % 19 + 1
        add_pair(rows, f"What is {a} + {b}?", f"{a + b}.", f"{a} 加 {b} 等于多少？", f"{a + b}。")
        add_pair(rows, f"Calculate {a} plus {b}.", f"{a + b}.", f"请计算 {a} 加 {b}。", f"{a + b}。")

    for a in range(2, 21):
        b = (a * 3) % 9 + 2
        add_pair(rows, f"What is {a} times {b}?", f"{a * b}.", f"{a} 乘 {b} 等于多少？", f"{a * b}。")

    for a, b in [(5, 2), (7, 3), (9, 4), (10, 6), (12, 5),
                 (15, 7), (18, 9), (20, 8), (25, 10), (30, 15),
                 (32, 12), (40, 18), (50, 25)]:
        add_pair(rows, f"What is {a} - {b}?", f"{a - b}.", f"{a} 减 {b} 等于多少？", f"{a - b}。")
        add_pair(rows, f"Calculate {a} minus {b}.", f"{a - b}.", f"请计算 {a} 减 {b}。", f"{a - b}。")

    for a, b in [(6, 2), (8, 2), (9, 3), (10, 2), (12, 3),
                 (14, 2), (15, 3), (16, 4), (18, 3), (20, 4),
                 (24, 6), (30, 5)]:
        add_pair(rows, f"What is {a} divided by {b}?", f"{a // b}.", f"{a} 除以 {b} 等于多少？", f"{a // b}。")

    words = [
        ("hello", "你好"), ("thank you", "谢谢"), ("good morning", "早上好"),
        ("computer", "电脑"), ("language model", "语言模型"),
        ("machine learning", "机器学习"), ("training", "训练"),
        ("data", "数据"), ("attention", "注意力"), ("question", "问题"),
        ("answer", "答案"), ("book", "书"), ("water", "水"),
        ("fire", "火"), ("friend", "朋友"), ("school", "学校"),
        ("city", "城市"), ("robot", "机器人"), ("cloud", "云"),
        ("future", "未来"),
    ]
    for en, zh in words:
        add_pair(rows, f"Translate '{en}' into Chinese.", f"{zh}。", f"把“{en}”翻译成中文。", f"{zh}。")
        add_pair(rows, f"Translate '{zh}' into English.", f"{en}.", f"把“{zh}”翻译成英语。", f"{en}。")
        add_pair(rows, f"What is the Chinese word for '{en}'?", f"{zh}。", f"“{en}”的中文是什么？", f"{zh}。")
        add_pair(rows, f"What is the English word for '{zh}'?", f"{en}.", f"“{zh}”的英文是什么？", f"{en}。")

    identity = [
        ("Hello.", "Hello! I am a small bilingual language model.", "你好。", "你好！我是一个小型双语语言模型。"),
        ("Who are you?", "I am a small bilingual language model.", "你是谁？", "我是一个小型双语语言模型。"),
        ("What can you do?", "I can answer simple questions and follow short instructions.", "你能做什么？", "我可以回答简单问题并执行简短指令。"),
        ("Do you understand Chinese?", "Yes. I can understand and generate Chinese and English.", "你懂中文吗？", "是的。我可以理解并生成中文和英文。"),
        ("Do you understand English?", "Yes. I can understand and generate English.", "你懂英语吗？", "是的。我可以理解并生成英语。"),
        ("Are you a human?", "No. I am a language model.", "你是人类吗？", "不是。我是一个语言模型。"),
    ]
    for _ in range(10):
        for item in identity:
            add_pair(rows, *item)

    facts = [
        ("What color is grass usually?", "Green.", "草通常是什么颜色？", "绿色。"),
        ("What do bees make?", "Honey.", "蜜蜂生产什么？", "蜂蜜。"),
        ("What do plants need for photosynthesis?", "Light, water, and carbon dioxide.", "植物进行光合作用需要什么？", "光、水和二氧化碳。"),
        ("What is the opposite of hot?", "Cold.", "热的反义词是什么？", "冷。"),
        ("What is the opposite of big?", "Small.", "大的反义词是什么？", "小。"),
        ("What is the opposite of fast?", "Slow.", "快的反义词是什么？", "慢。"),
        ("How many days are in a week?", "Seven.", "一周有几天？", "七天。"),
        ("How many months are in a year?", "Twelve.", "一年有几个月？", "十二个月。"),
        ("What comes after Monday?", "Tuesday.", "星期一之后是什么？", "星期二。"),
        ("What comes after Friday?", "Saturday.", "星期五之后是什么？", "星期六。"),
        ("What do we use to see?", "Our eyes.", "我们用什么看东西？", "眼睛。"),
        ("What do we use to hear?", "Our ears.", "我们用什么听声音？", "耳朵。"),
        ("What is ice made from?", "Frozen water.", "冰是由什么组成的？", "冰冻的水。"),
        ("What is rain?", "Water that falls from clouds.", "什么是雨？", "从云中落下的水。"),
        ("What is a robot?", "A machine that can perform programmed or controlled actions.", "什么是机器人？", "能够执行程序或控制动作的机器。"),
        ("What is software?", "Programs and data that run on a computer.", "什么是软件？", "在计算机上运行的程序和数据。"),
        ("What is hardware?", "The physical parts of a computer or device.", "什么是硬件？", "计算机或设备的物理部件。"),
        ("What is the internet?", "A global network connecting computers and devices.", "什么是互联网？", "连接全球计算机和设备的网络。"),
        ("What is an email?", "A message sent electronically over a network.", "什么是电子邮件？", "通过网络以电子方式发送的信息。"),
        ("What is a website?", "A collection of web pages available on the internet.", "什么是网站？", "互联网上的一组网页。"),
    ]
    for q, a, zq, za in facts:
        add_pair(rows, q, a, zq, za)
        add_pair(rows, q.replace("What is", "Explain"), a, zq.replace("什么是", "请解释什么是"), za)
        add_pair(rows, "Give a short answer: " + q, a, "请简短回答：" + zq, za)

    rewrites = [
        ("Make this sentence shorter: The meeting will start at nine o'clock.", "The meeting starts at nine.", "把这句话缩短：会议将在九点钟开始。", "会议九点开始。"),
        ("Make this sentence shorter: Please send the document to me.", "Please send me the document.", "把这句话缩短：请把这份文件发送给我。", "请把文件发给我。"),
        ("Rewrite politely: Help me.", "Could you please help me?", "把这句话写得更礼貌：帮我一下。", "可以请你帮我一下吗？"),
        ("Rewrite politely: Send me the file.", "Please send me the file.", "把这句话写得更礼貌：把文件发给我。", "请把文件发给我，谢谢。"),
        ("Rewrite clearly: The test finished yesterday.", "The test finished yesterday.", "把这句话写得更清楚：测试昨天完成了。", "测试于昨天完成。"),
    ]
    for _ in range(6):
        for item in rewrites:
            add_pair(rows, *item)

    logic = [
        ("Which is larger, 9 or 4?", "9.", "9 和 4 哪个更大？", "9。"),
        ("Which is smaller, 3 or 8?", "3.", "3 和 8 哪个更小？", "3。"),
        ("If today is Monday, what day is tomorrow?", "Tuesday.", "如果今天是星期一，明天是星期几？", "星期二。"),
        ("If today is Friday, what day is tomorrow?", "Saturday.", "如果今天是星期五，明天是星期几？", "星期六。"),
        ("If a box has 3 red balls and 2 blue balls, how many balls are there?", "5.", "一个盒子里有3个红球和2个蓝球，一共有多少个球？", "5个。"),
        ("If you have 10 apples and eat 2, how many remain?", "8.", "你有10个苹果，吃掉2个，还剩多少？", "8个。"),
        ("Which is a fruit: apple or chair?", "Apple.", "哪个是水果：苹果还是椅子？", "苹果。"),
        ("Which is an animal: dog or table?", "Dog.", "哪个是动物：狗还是桌子？", "狗。"),
        ("Which is a programming language: Python or Paris?", "Python.", "哪个是编程语言：Python 还是巴黎？", "Python。"),
    ]
    for _ in range(6):
        for item in logic:
            add_pair(rows, *item)

    strict = [
        ("Answer with one word: What color is snow?", "White.", "只用一个词回答：雪是什么颜色？", "白色。"),
        ("Answer with one word: What color is coal?", "Black.", "只用一个词回答：煤是什么颜色？", "黑色。"),
        ("Answer with a number: How many days are in a week?", "7.", "只用数字回答：一周有几天？", "7。"),
        ("Answer with a number: How many months are in a year?", "12.", "只用数字回答：一年有几个月？", "12。"),
        ("List three colors.", "Red, blue, and green.", "列出三种颜色。", "红、蓝、绿。"),
        ("List three animals.", "Cat, dog, and bird.", "列出三种动物。", "猫、狗、鸟。"),
        ("List three fruits.", "Apple, banana, and orange.", "列出三种水果。", "苹果、香蕉、橙子。"),
        ("List three programming languages.", "Python, C, and Java.", "列出三种编程语言。", "Python、C 和 Java。"),
    ]
    for _ in range(6):
        for item in strict:
            add_pair(rows, *item)

    random.shuffle(rows)

    en = [r for r in rows if r["lang"] == "en"]
    zh = [r for r in rows if r["lang"] == "zh"]
    n = min(len(en), len(zh))
    en, zh = en[:n], zh[:n]
    rows = en + zh
    random.shuffle(rows)

    unique = []
    seen = set()
    for row in rows:
        key = (row["lang"], row["user"], row["assistant"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    rows = unique

    split = int(len(rows) * 0.90)
    return rows[:split], rows[split:]


def make_sequence(tok, row):
    prefix = f"User: {row['user']}\nAssistant:"
    full = prefix + " " + row["assistant"]
    prefix_ids = tok.encode(prefix).ids
    ids = tok.encode(full).ids
    if len(ids) > CONTEXT_LENGTH:
        return None
    response_start = len(prefix_ids)
    return ids, response_start


def make_tensors(tok, rows):
    xs, ys = [], []
    for row in rows:
        item = make_sequence(tok, row)
        if item is None:
            continue
        ids, response_start = item
        if len(ids) < response_start + 2:
            continue

        x = ids[:-1]
        y = ids[1:]
        prompt_target_count = max(0, response_start - 1)
        y[:prompt_target_count] = [-100] * prompt_target_count

        pad = CONTEXT_LENGTH - len(x)
        if pad > 0:
            x = x + [0] * pad
            y = y + [-100] * pad

        xs.append(torch.tensor(x, dtype=torch.long))
        ys.append(torch.tensor(y, dtype=torch.long))

    if not xs:
        raise RuntimeError("No usable curriculum examples remained.")
    return torch.stack(xs), torch.stack(ys)


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        head_dim = D_MODEL // N_HEADS
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.out = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.head_dim = head_dim

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, N_HEADS, self.head_dim).transpose(1, 2)
        k = k.view(b, t, N_HEADS, self.head_dim).transpose(1, 2)
        v = v.view(b, t, N_HEADS, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, D_MODEL)
        return self.out(y)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = Attn()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, D_FF, bias=False),
            nn.GELU(),
            nn.Linear(D_FF, D_MODEL, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Embedding(CONTEXT_LENGTH, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, x, y=None):
        t = x.size(1)
        pos = torch.arange(t, device=x.device)
        h = self.tok(x) + self.pos(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln(h))
        loss = None
        if y is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss


@torch.no_grad()
def generate(model, tok, prompt, device, max_new_tokens=80):
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    generated = []
    recent = []

    model.eval()
    for _ in range(max_new_tokens):
        ctx = x[:, -CONTEXT_LENGTH:]
        logits, _ = model(ctx)
        logits = logits[:, -1, :]
        next_id = int(torch.argmax(logits, dim=-1).item())

        if len(recent) >= 4 and all(v == next_id for v in recent[-4:]):
            break
        if len(recent) >= 3 and recent[-3:] == [next_id, next_id, next_id]:
            break

        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
        generated.append(next_id)
        recent.append(next_id)

    return tok.decode(generated)


def evaluate(model, x, y, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
    ):
        for start in range(0, len(x), BATCH_SIZE):
            xb = x[start:start + BATCH_SIZE].to(device)
            yb = y[start:start + BATCH_SIZE].to(device)
            _, loss = model(xb, yb)
            total_loss += float(loss.item())
            total_batches += 1
    return total_loss / max(total_batches, 1)


def lr_at(step, total):
    if step <= WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS
    p = min(max((step - WARMUP_STEPS) / max(1, total - WARMUP_STEPS), 0.0), 1.0)
    return MIN_LEARNING_RATE + (LEARNING_RATE - MIN_LEARNING_RATE) * 0.5 * (
        1.0 + torch.cos(torch.tensor(torch.pi * p)).item()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    args, _ = parser.parse_known_args()

    seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 112)
    print("Step 48c: Curated bilingual curriculum SFT")
    print("=" * 112)
    print(f"device:              {device}")
    if device.type == "cuda":
        print(f"GPU:                 {torch.cuda.get_device_name(0)}")

    drive_root = Path("/content/drive/MyDrive/transformers_exercise_20260911")
    ckpt_path = drive_root / "artifacts" / "step45" / "tiny_gpt_step45.pt"
    tok_path = drive_root / "artifacts" / "step43" / "step43_bpe_8000.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))
    if not tok_path.exists():
        raise FileNotFoundError(str(tok_path))

    tok = Tokenizer.from_file(str(tok_path))
    train_rows, val_rows = build_curriculum()
    train_x, train_y = make_tensors(tok, train_rows)
    val_x, val_y = make_tensors(tok, val_rows)

    print("\nPart 1: Curriculum dataset")
    print("-" * 112)
    print(f"Train examples:      {len(train_x):,}")
    print(f"Validation examples: {len(val_x):,}")
    print(
        f"English / Chinese:   "
        f"{sum(r['lang'] == 'en' for r in train_rows):,} / "
        f"{sum(r['lang'] == 'zh' for r in train_rows):,}"
    )

    model = TinyGPT().to(device)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val = float("inf")
    best_state = None
    n = len(train_x)
    tick = time.perf_counter()

    model.train()
    for step in range(1, args.steps + 1):
        lr = lr_at(step, args.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        idx = torch.randint(0, n, (BATCH_SIZE,))
        xb = train_x[idx].to(device)
        yb = train_y[idx].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            _, loss = model(xb, yb)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % LOG_INTERVAL == 0 or step == args.steps:
            speed = (step * BATCH_SIZE * CONTEXT_LENGTH) / max(
                time.perf_counter() - tick, 1e-6
            )
            print(
                f"step {step:>5}/{args.steps} | "
                f"loss {loss.item():.4f} | lr {lr:.2e} | {speed:,.0f} tok/s"
            )

        if step % EVAL_INTERVAL == 0 or step == args.steps:
            val_loss = evaluate(model, val_x, val_y, device)
            print(f"           validation loss: {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                print(f"           new best checkpoint: {best_val:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    out_dir = drive_root / "artifacts" / "step48c"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = out_dir / "tiny_gpt_step48c_curriculum_sft.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "tokenizer_path": str(tok_path),
            "base_checkpoint": str(ckpt_path),
            "step": args.steps,
            "train_examples": len(train_x),
            "validation_examples": len(val_x),
            "best_validation_loss": best_val,
        },
        out_ckpt,
    )

    print("\nPart 2: Generalization probes")
    probes = [
        "User: Explain what artificial intelligence is in simple terms.\nAssistant:",
        "User: 请用简单中文解释什么是人工智能。\nAssistant:",
        "User: What is a transformer model?\nAssistant:",
        "User: 什么是 Transformer 模型？\nAssistant:",
        "User: What is 7 + 8?\nAssistant:",
        "User: 12 加 9 等于多少？\nAssistant:",
        "User: What can you do?\nAssistant:",
        "User: 你能做什么？\nAssistant:",
        "User: Translate 'robot' into Chinese.\nAssistant:",
        "User: 把“data”翻译成中文。\nAssistant:",
    ]

    for prompt in probes:
        answer = generate(model, tok, prompt, device)
        print(f"\n{prompt}")
        print(answer)

    print("\nStep 48c complete.")
    print(f"Checkpoint: {out_ckpt}")


if __name__ == "__main__":
    main()
