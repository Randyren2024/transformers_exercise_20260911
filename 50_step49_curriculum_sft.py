import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

# Step 50: Instruction tuning for the ~8.4M Step 49 model.
# Reuses a controlled bilingual curriculum but starts from the Step 49 best base checkpoint.

SEED = 50
VOCAB_SIZE = 8000
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 8
D_FF = 1024
CONTEXT = 256
BATCH = 64
STEPS = 1400
LR = 3e-6
MIN_LR = 5e-7
WARM = 100
WD = 0.01


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add(rows, lang, user, answer):
    rows.append((lang, user.strip(), answer.strip()))


def build_dataset():
    rows = []

    concepts = [
        ("artificial intelligence", "Artificial intelligence is technology that lets computers perform tasks that normally require human intelligence.", "人工智能", "人工智能是让计算机执行通常需要人类智能任务的一种技术。"),
        ("machine learning", "Machine learning lets computers learn patterns from data instead of using a separate rule for every case.", "机器学习", "机器学习让计算机从数据中学习规律，而不是为每种情况都手写规则。"),
        ("a language model", "A language model predicts likely tokens from context and can generate text.", "语言模型", "语言模型根据上下文预测可能的词元，并生成文本。"),
        ("a token", "A token is a small unit of text processed by a language model.", "token", "Token 是语言模型处理的一小段文本单位。"),
        ("training", "Training updates model parameters so that predictions better match examples.", "模型训练", "模型训练通过更新参数，让模型的预测逐渐更符合训练样本。"),
        ("inference", "Inference means using a trained model to produce an output for new input.", "模型推理", "模型推理是使用训练好的模型处理新输入并生成输出。"),
        ("overfitting", "Overfitting happens when a model memorizes training examples too closely and performs worse on new data.", "过拟合", "过拟合是模型过度记住训练样本，导致它在新数据上的表现变差。"),
        ("attention", "Attention lets a model focus on relevant parts of the input.", "注意力", "注意力机制让模型重点关注输入中更相关的部分。"),
        ("a transformer", "A transformer is a neural network architecture built around attention.", "Transformer 模型", "Transformer 是一种以注意力机制为核心的神经网络架构。"),
        ("a GPU", "A GPU is a processor designed for many parallel numerical operations.", "GPU", "GPU 是适合并行执行大量数值计算的处理器。"),
        ("a CPU", "A CPU is a general-purpose processor used for many kinds of computation.", "CPU", "CPU 是用于多种计算任务的通用处理器。"),
        ("an algorithm", "An algorithm is a clear sequence of steps for solving a problem.", "算法", "算法是一组解决问题的清晰步骤。"),
        ("a variable", "A variable is a named value that can change during program execution.", "变量", "变量是程序运行过程中值可以变化的命名数据。"),
        ("a function", "A function is a reusable block of code that performs a task.", "函数", "函数是一段可以重复使用、用于完成某项任务的代码。"),
        ("a database", "A database stores and organizes data so it can be queried and updated.", "数据库", "数据库用于存储和组织数据，并支持查询和更新。"),
        ("an API", "An API is an interface that lets software systems communicate.", "API", "API 是让不同软件系统进行通信的接口。"),
        ("the internet", "The internet is a global network connecting computers and devices.", "互联网", "互联网是连接全球计算机和设备的网络。"),
        ("a robot", "A robot is a machine that can perform programmed or controlled actions.", "机器人", "机器人是能够执行程序或控制动作的机器。"),
    ]

    en_variants = [
        "What is {}?",
        "Explain {} in simple terms.",
        "Give a short definition of {}.",
        "In one sentence, what is {}?",
    ]
    zh_variants = [
        "什么是{}？",
        "请用简单的话解释{}。",
        "请简要定义{}。",
        "请用一句话说明什么是{}。",
    ]
    for term_en, ans_en, term_zh, ans_zh in concepts:
        for qe, qz in zip(en_variants, zh_variants):
            add(rows, "en", qe.format(term_en), ans_en)
            add(rows, "zh", qz.format(term_zh), ans_zh)

    for a in range(2, 42):
        b = (a * 7) % 17 + 1
        add(rows, "en", f"What is {a} + {b}?", f"{a+b}.")
        add(rows, "zh", f"{a} 加 {b} 等于多少？", f"{a+b}。")
        add(rows, "en", f"Calculate {a} plus {b}.", f"{a+b}.")
        add(rows, "zh", f"请计算 {a} 加 {b}。", f"{a+b}。")
    for a in range(3, 31):
        b = (a * 3) % 9 + 2
        add(rows, "en", f"What is {a} times {b}?", f"{a*b}.")
        add(rows, "zh", f"{a} 乘 {b} 等于多少？", f"{a*b}。")
    for a, b in [(7,2),(9,4),(12,5),(15,7),(18,9),(20,8),(25,10),(32,12),(40,18),(50,25)]:
        add(rows, "en", f"What is {a} - {b}?", f"{a-b}.")
        add(rows, "zh", f"{a} 减 {b} 等于多少？", f"{a-b}。")
        add(rows, "en", f"What is {a} divided by {b} if the result is an integer?", f"{a//b}.")
        if a % b == 0:
            add(rows, "zh", f"{a} 除以 {b} 等于多少？", f"{a//b}。")

    words = [
        ("hello", "你好"), ("thank you", "谢谢"), ("good morning", "早上好"),
        ("computer", "电脑"), ("language model", "语言模型"), ("machine learning", "机器学习"),
        ("training", "训练"), ("data", "数据"), ("attention", "注意力"),
        ("question", "问题"), ("answer", "答案"), ("book", "书"), ("water", "水"),
        ("fire", "火"), ("friend", "朋友"), ("school", "学校"), ("city", "城市"),
        ("robot", "机器人"), ("cloud", "云"), ("future", "未来"),
    ]
    for en, zh in words:
        add(rows, "en", f"Translate '{en}' into Chinese.", zh + "。")
        add(rows, "zh", f"把“{en}”翻译成中文。", zh + "。")
        add(rows, "en", f"Translate '{zh}' into English.", en + ".")
        add(rows, "zh", f"把“{zh}”翻译成英语。", en + ".")

    identity = [
        ("Hello.", "Hello! I am a small bilingual language model.", "你好。", "你好！我是一个小型双语语言模型。"),
        ("Who are you?", "I am a small bilingual language model.", "你是谁？", "我是一个小型双语语言模型。"),
        ("What can you do?", "I can answer simple questions and follow short instructions.", "你能做什么？", "我可以回答简单问题并执行简短指令。"),
        ("Do you understand Chinese?", "Yes. I can understand and generate Chinese and English.", "你懂中文吗？", "是的。我可以理解并生成中文和英文。"),
        ("Are you a human?", "No. I am a language model.", "你是人类吗？", "不是。我是一个语言模型。"),
        ("Can you know every fact?", "No. A language model can make mistakes and does not know every fact.", "你知道所有事实吗？", "不知道。语言模型可能犯错，也不可能知道所有事实。"),
        ("What should you do when you are unsure?", "Say that you are not sure instead of pretending to know.", "不确定答案时应该怎么办？", "应该说明自己不确定，而不是假装知道。"),
    ]
    for en_q, en_a, zh_q, zh_a in identity:
        for _ in range(8):
            add(rows, "en", en_q, en_a)
            add(rows, "zh", zh_q, zh_a)

    facts = [
        ("What color is grass usually?", "Green.", "草通常是什么颜色？", "绿色。"),
        ("What do bees make?", "Honey.", "蜜蜂生产什么？", "蜂蜜。"),
        ("How many days are in a week?", "Seven.", "一周有几天？", "七天。"),
        ("How many months are in a year?", "Twelve.", "一年有几个月？", "十二个月。"),
        ("What is the opposite of hot?", "Cold.", "热的反义词是什么？", "冷。"),
        ("What is the opposite of big?", "Small.", "大的反义词是什么？", "小。"),
        ("What do we use to see?", "Our eyes.", "我们用什么看东西？", "眼睛。"),
        ("What do we use to hear?", "Our ears.", "我们用什么听声音？", "耳朵。"),
        ("What is software?", "Programs and data that run on a computer.", "什么是软件？", "在计算机上运行的程序和数据。"),
        ("What is hardware?", "The physical parts of a computer or device.", "什么是硬件？", "计算机或设备的物理部件。"),
    ]
    for q, a, zq, za in facts:
        add(rows, "en", q, a); add(rows, "zh", zq, za)
        add(rows, "en", "Answer briefly: " + q, a); add(rows, "zh", "请简短回答：“" + zq + "”", za)

    # Short behavior tasks.
    tasks = [
        ("Say hello in a friendly way.", "Hello! It is nice to meet you.", "用友好的方式说你好。", "你好！很高兴认识你。"),
        ("Say thank you politely.", "Thank you very much.", "礼貌地说谢谢。", "非常感谢你。"),
        ("Give me one simple study tip.", "Study one small topic at a time and review it regularly.", "给我一个简单的学习建议。", "一次学习一个小主题，并定期复习。"),
        ("Give me one simple programming tip.", "Test small pieces of code before combining them into a larger program.", "给我一个简单的编程建议。", "先测试小段代码，再把它们组合成更大的程序。"),
        ("Answer in one sentence: Why is sleep important?", "Sleep helps the body and brain recover and function well.", "用一句话回答：为什么睡眠重要？", "睡眠可以帮助身体和大脑恢复，并保持良好状态。"),
        ("Answer in one sentence: Why is exercise useful?", "Regular exercise can improve fitness, mood, and overall health.", "用一句话回答：为什么运动有用？", "规律运动可以改善体能、情绪和整体健康。"),
    ]
    for q, a, zq, za in tasks:
        for _ in range(8):
            add(rows, "en", q, a); add(rows, "zh", zq, za)

    # Remove exact duplicates, then make a deterministic balanced split.
    unique = list(dict.fromkeys(rows))
    en = [r for r in unique if r[0] == "en"]
    zh = [r for r in unique if r[0] == "zh"]
    random.shuffle(en); random.shuffle(zh)
    n = min(len(en), len(zh))
    data = en[:n] + zh[:n]
    random.shuffle(data)
    cut = int(len(data) * 0.9)
    return data[:cut], data[cut:]


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        hd = D_MODEL // N_HEADS
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.out = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.hd = hd

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, N_HEADS, self.hd).transpose(1, 2)
        k = k.view(b, t, N_HEADS, self.hd).transpose(1, 2)
        v = v.view(b, t, N_HEADS, self.hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).contiguous().view(b, t, D_MODEL))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = Attn()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.fc1 = nn.Linear(D_MODEL, D_FF, bias=False)
        self.fc2 = nn.Linear(D_FF, D_MODEL, bias=False)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class TinyGPT8M(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Embedding(CONTEXT, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, x, y=None):
        t = x.size(1)
        pos = torch.arange(t, device=x.device)
        h = self.tok(x) + self.pos(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln_f(h))
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, x, max_new=80, temperature=0.2, top_k=12):
        self.eval()
        start = x.size(1)
        previous = []
        for _ in range(max_new):
            ctx = x[:, -CONTEXT:]
            logits, _ = self(ctx)
            z = logits[:, -1, :] / max(temperature, 1e-5)
            vals, _ = torch.topk(z, min(top_k, VOCAB_SIZE))
            z = z.masked_fill(z < vals[:, [-1]], float("-inf"))
            nxt = torch.multinomial(torch.softmax(z, dim=-1), 1)
            x = torch.cat([x, nxt], dim=1)
            token_list = x[0].tolist()[start:]
            text = TOKENIZER.decode(token_list).strip()
            if "\nUser:" in text or "\nAssistant:" in text:
                break
            if len(token_list) >= 12:
                tail = token_list[-12:]
                if previous and tail == previous[-1]:
                    break
                previous.append(tail)
        return TOKENIZER.decode(x[0].tolist()[start:]).strip()


def make_tensors(tok, rows):
    xs, ys = [], []
    for _, user, answer in rows:
        prefix = f"User: {user}\nAssistant:"
        full = prefix + " " + answer
        ids = tok.encode(full).ids[:CONTEXT]
        pids = tok.encode(prefix).ids
        response_start = min(len(pids), len(ids))
        if len(ids) <= response_start + 1:
            continue
        x = ids[:-1]
        y = ids[1:]
        y[:max(0, response_start - 1)] = [-100] * max(0, response_start - 1)
        pad = CONTEXT - len(x)
        if pad > 0:
            x += [0] * pad
            y += [-100] * pad
        xs.append(torch.tensor(x, dtype=torch.long))
        ys.append(torch.tensor(y, dtype=torch.long))
    return torch.stack(xs), torch.stack(ys)


def lr_at(step, total):
    if step <= WARM:
        return LR * step / WARM
    p = min(1.0, max(0.0, (step - WARM) / max(1, total - WARM)))
    return MIN_LR + (LR - MIN_LR) * 0.5 * (1 + math.cos(math.pi * p))


def evaluate(model, x, y, device, batches=4):
    model.eval(); losses = []
    n = len(x)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        for _ in range(batches):
            idx = torch.randint(0, n, (min(BATCH, n),))
            _, loss = model(x[idx].to(device), y[idx].to(device))
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    args, _ = parser.parse_known_args()
    seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 112)
    print("Step 50: SFT for ~8.4M bilingual GPT")
    print("=" * 112)
    print(f"device:              {device}")
    if device.type == "cuda":
        print(f"GPU:                 {torch.cuda.get_device_name(0)}")

    drive = Path("/content/drive/MyDrive/transformers_exercise_20260911")
    ckpt = drive / "artifacts" / "step49" / "tiny_gpt_step49_best.pt"
    tok_path = drive / "artifacts" / "step43" / "step43_bpe_8000.json"
    if not ckpt.exists():
        raise FileNotFoundError(f"Step 49 checkpoint not found: {ckpt}")
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")

    global TOKENIZER
    TOKENIZER = Tokenizer.from_file(str(tok_path))
    train_rows, val_rows = build_dataset()
    train_x, train_y = make_tensors(TOKENIZER, train_rows)
    val_x, val_y = make_tensors(TOKENIZER, val_rows)

    print("\nPart 1: Curriculum SFT dataset")
    print("-" * 112)
    print(f"Train examples:      {len(train_x):,}")
    print(f"Validation examples: {len(val_x):,}")
    print(f"English / Chinese:   {sum(r[0]=='en' for r in train_rows):,} / {sum(r[0]=='zh' for r in train_rows):,}")

    model = TinyGPT8M().to(device)
    checkpoint = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = float("inf")
    out_dir = drive / "artifacts" / "step50"
    out_dir.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()

    for step in range(1, args.steps + 1):
        lr = lr_at(step, args.steps)
        optimizer.param_groups[0]["lr"] = lr
        idx = torch.randint(0, len(train_x), (BATCH,))
        x = train_x[idx].to(device)
        y = train_y[idx].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 50 == 0 or step == args.steps:
            speed = step * BATCH * CONTEXT / max(time.perf_counter() - tick, 1e-6)
            print(f"step {step:>5}/{args.steps} | loss {loss.item():.4f} | lr {lr:.2e} | {speed:,.0f} tok/s")

        if step % 200 == 0 or step == args.steps:
            vloss = evaluate(model, val_x, val_y, device)
            print(f"           validation loss: {vloss:.4f}")
            if vloss < best:
                best = vloss
                path = out_dir / "tiny_gpt_step50_best.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "step": step,
                    "validation_loss": best,
                    "base_checkpoint": str(ckpt),
                    "tokenizer_path": str(tok_path),
                }, path)
                print(f"           new best checkpoint: {best:.4f}")

    print("\nPart 2: Instruction probes")
    prompts = [
        "User: Explain artificial intelligence in simple terms.\nAssistant:",
        "User: 请用简单中文解释什么是人工智能。\nAssistant:",
        "User: What is a transformer model?\nAssistant:",
        "User: 什么是 Transformer 模型？\nAssistant:",
        "User: What is 7 + 8?\nAssistant:",
        "User: 12 加 9 等于多少？\nAssistant:",
        "User: Translate 'robot' into Chinese.\nAssistant:",
        "User: 把“data”翻译成中文。\nAssistant:",
        "User: What can you do?\nAssistant:",
        "User: 你能做什么？\nAssistant:",
    ]
    model.eval()
    for prompt in prompts:
        ids = TOKENIZER.encode(prompt).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        text = model.generate(x)
        print(f"\n{prompt}\n{text}")

    print("\nStep 50 complete.")
    print(f"Best checkpoint: {out_dir / 'tiny_gpt_step50_best.pt'}")


if __name__ == "__main__":
    main()
