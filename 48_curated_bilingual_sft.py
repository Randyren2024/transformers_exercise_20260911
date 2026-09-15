import argparse
import math
import random
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

SEED = 48
VOCAB_SIZE = 8000
D_MODEL = 192
N_HEADS = 6
N_LAYERS = 6
D_FF = 768
CONTEXT_LENGTH = 256
BATCH_SIZE = 64
DEFAULT_STEPS = 2200
LEARNING_RATE = 2e-5
MIN_LEARNING_RATE = 2e-6
WARMUP_STEPS = 150
WEIGHT_DECAY = 0.01
EVAL_INTERVAL = 200
LOG_INTERVAL = 25


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add(rows, lang, user, answer):
    user = re.sub(r"\s+", " ", user).strip()
    answer = re.sub(r"\s+", " ", answer).strip()
    if len(user) >= 3 and len(answer) >= 4 and len(answer) <= 420:
        rows.append({"lang": lang, "user": user, "assistant": answer})


def build_examples():
    rows = []

    # Stable, short bilingual knowledge examples.
    facts = [
        ("What is artificial intelligence?", "Artificial intelligence is the field of building computer systems that can perform tasks that normally require human intelligence, such as understanding language, recognizing images, and reasoning.", "什么是人工智能？", "人工智能是让计算机执行通常需要人类智能的任务，例如理解语言、识别图像和进行推理。"),
        ("What is machine learning?", "Machine learning is a method that lets computers learn patterns from data instead of being explicitly programmed for every case.", "什么是机器学习？", "机器学习是一种让计算机从数据中学习规律的方法，而不是为每一种情况都手写规则。"),
        ("What is a transformer model?", "A transformer is a neural network architecture built around attention. It can process relationships between tokens and is widely used for modern language models.", "什么是 Transformer 模型？", "Transformer 是一种以注意力机制为核心的神经网络架构，它可以学习不同词元之间的关系，并广泛用于现代语言模型。"),
        ("What is an algorithm?", "An algorithm is a clear sequence of steps used to solve a problem or complete a task.", "什么是算法？", "算法是一组清晰、有顺序的步骤，用来解决问题或完成任务。"),
        ("What is data?", "Data is information that can be stored, processed, and analyzed by people or computers.", "什么是数据？", "数据是可以被存储、处理和分析的信息。"),
        ("What is a database?", "A database is an organized collection of data that can be stored, searched, and updated efficiently.", "什么是数据库？", "数据库是有组织的数据集合，可以高效地存储、查询和更新信息。"),
        ("What is Python?", "Python is a general-purpose programming language known for readable syntax and a large ecosystem of libraries.", "Python 是什么？", "Python 是一种通用编程语言，以语法易读和丰富的库生态而闻名。"),
        ("What is an API?", "An API is an interface that allows different software systems to communicate and exchange data or functions.", "什么是 API？", "API 是一种接口，让不同的软件系统能够通信，并交换数据或调用功能。"),
        ("What is a token in a language model?", "A token is a small unit of text processed by a language model. It may be a word, part of a word, punctuation mark, or other symbol.", "语言模型中的 token 是什么？", "Token 是语言模型处理的文本小单元，可以是一个词、词的一部分、标点或其他符号。"),
        ("What is training?", "Training is the process of adjusting a model's parameters so that its predictions better match the training data.", "什么是模型训练？", "模型训练是调整模型参数，使模型的预测逐渐更符合训练数据的过程。"),
        ("What is inference?", "Inference is using a trained model to produce predictions or generate outputs for new input.", "什么是推理？", "推理是使用已经训练好的模型，根据新的输入产生预测或输出的过程。"),
        ("What is a neural network?", "A neural network is a model made of layers of connected numerical transformations that can learn patterns from data.", "什么是神经网络？", "神经网络是一种由多层数值变换组成的模型，可以从数据中学习规律。"),
        ("What is overfitting?", "Overfitting happens when a model learns the training examples too closely and performs worse on new data.", "什么是过拟合？", "过拟合是模型过度记住训练样本，导致它在新数据上的表现变差。"),
        ("What is a GPU?", "A GPU is a processor designed to perform many numerical operations in parallel and is widely used for machine learning.", "什么是 GPU？", "GPU 是一种擅长并行执行大量数值运算的处理器，因此广泛用于机器学习。"),
        ("What is an operating system?", "An operating system is software that manages computer hardware and provides services for applications and users.", "什么是操作系统？", "操作系统负责管理计算机硬件，并为应用程序和用户提供基础服务。"),
        ("What is the internet?", "The internet is a global network of interconnected computer networks that exchange information using standard protocols.", "什么是互联网？", "互联网是由许多相互连接的计算机网络组成的全球网络，通过标准协议交换信息。"),
        ("What is a file?", "A file is a named collection of digital information stored on a computer or other storage device.", "什么是文件？", "文件是存储在计算机或其他存储设备中的一组有名称的数字信息。"),
        ("What is cloud computing?", "Cloud computing means using computing resources such as servers and storage over a network instead of owning all the hardware locally.", "什么是云计算？", "云计算是通过网络使用服务器、存储等计算资源，而不是把所有硬件都部署在本地。"),
        ("What is a programming language?", "A programming language is a formal language used to describe instructions that a computer can execute.", "什么是编程语言？", "编程语言是一种形式化语言，用来描述计算机可以执行的指令。"),
    ]
    for en_u, en_a, zh_u, zh_a in facts:
        add(rows, "en", en_u, en_a)
        add(rows, "zh", zh_u, zh_a)

    # Arithmetic and simple reasoning: deterministic answers are ideal for a tiny model.
    for a in range(2, 21):
        for b in range(2, 11):
            add(rows, "en", f"What is {a} + {b}?", f"{a + b}.")
            add(rows, "zh", f"{a} 加 {b} 等于多少？", f"{a + b}。")
            add(rows, "en", f"What is {a} - {b}?", f"{a - b}.")
            add(rows, "zh", f"{a} 减 {b} 等于多少？", f"{a - b}。")
    for a in range(2, 13):
        for b in range(2, 9):
            add(rows, "en", f"What is {a} times {b}?", f"{a * b}.")
            add(rows, "zh", f"{a} 乘以 {b} 等于多少？", f"{a * b}。")

    # Simple instruction following and rewriting.
    pairs = [
        ("Say hello in a friendly way.", "Hello! It is nice to meet you.", "用友好的方式说你好。", "你好！很高兴认识你。"),
        ("Say thank you politely.", "Thank you very much.", "礼貌地说谢谢。", "非常感谢你。"),
        ("Translate 'good morning' into Chinese.", "早上好。", "把“good morning”翻译成英文。", "Good morning."),
        ("Translate '你好' into English.", "Hello.", "把“thank you”翻译成中文。", "谢谢。"),
        ("Make this sentence shorter: The device is able to operate safely at high altitude.", "The device can operate safely at high altitude.", "把这句话缩短：这个设备能够在高空安全运行。", "这个设备可以在高空安全运行。"),
        ("Give me one simple study tip.", "Study one small topic at a time and review it regularly.", "给我一个简单的学习建议。", "一次学习一个小主题，并定期复习。"),
        ("Give me one simple programming tip.", "Test small pieces of code before combining them into a larger program.", "给我一个简单的编程建议。", "先测试小段代码，再把它们组合成更大的程序。"),
        ("Answer in one sentence: Why is sleep important?", "Sleep helps the body and brain recover and function well.", "用一句话回答：为什么睡眠重要？", "睡眠可以帮助身体和大脑恢复，并保持良好状态。"),
        ("Answer in one sentence: Why is exercise useful?", "Regular exercise can improve fitness, mood, and overall health.", "用一句话回答：为什么运动有用？", "规律运动可以改善体能、情绪和整体健康。"),
        ("Write a short definition of honesty.", "Honesty means telling the truth and avoiding deliberate deception.", "用一句话解释诚实。", "诚实就是说真话，并避免故意欺骗。"),
    ]
    for en_u, en_a, zh_u, zh_a in pairs:
        add(rows, "en", en_u, en_a)
        add(rows, "zh", zh_u, zh_a)

    # Conversational identity and safe uncertainty.
    chats = [
        ("Hello, introduce yourself briefly.", "Hello! I am a small language model designed to answer simple questions and follow short instructions.", "你好，请简单介绍一下你自己。", "你好！我是一个小型语言模型，用来回答简单问题并执行简短指令。"),
        ("Can you help me learn?", "Yes. Ask one clear question at a time, and we can work through it step by step.", "你可以帮助我学习吗？", "可以。你可以一次问一个清晰的问题，我们可以一步一步解决。"),
        ("What should I do if you are unsure?", "I should say that I am not sure instead of pretending to know the answer.", "如果你不确定答案，应该怎么办？", "我应该明确说明不确定，而不是假装知道答案。"),
        ("Can you know every fact?", "No. A language model can make mistakes and does not know every fact.", "你知道所有事实吗？", "不知道。语言模型可能犯错，也不可能知道所有事实。"),
        ("What is a good way to ask an AI a question?", "State the goal clearly and include the important details the AI needs.", "怎样向 AI 提一个好问题？", "把目标说清楚，并提供 AI 完成任务所需要的重要信息。"),
    ]
    for en_u, en_a, zh_u, zh_a in chats:
        add(rows, "en", en_u, en_a)
        add(rows, "zh", zh_u, zh_a)

    # More controlled concept comparisons.
    compare = [
        ("What is the difference between training and inference?", "Training changes model parameters using data. Inference uses the trained parameters to produce outputs.", "训练和推理有什么区别？", "训练通过数据调整模型参数；推理使用已经训练好的参数产生输出。"),
        ("What is the difference between RAM and storage?", "RAM is fast temporary working memory, while storage keeps data for longer periods.", "内存和存储有什么区别？", "内存是快速的临时工作空间，而存储用于长期保存数据。"),
        ("What is the difference between a file and a folder?", "A file stores information, while a folder organizes files and other folders.", "文件和文件夹有什么区别？", "文件用于保存信息，文件夹用于组织文件和其他文件夹。"),
        ("What is the difference between a CPU and a GPU?", "A CPU is a general-purpose processor, while a GPU is optimized for many parallel computations.", "CPU 和 GPU 有什么区别？", "CPU 是通用处理器，而 GPU 更擅长大量并行计算。"),
        ("What is the difference between English and Chinese tokenization?", "Tokenization splits text into model units, and the resulting token boundaries can differ greatly between English and Chinese.", "英文和中文的分词有什么区别？", "分词会把文本切成模型处理的单元，而英文和中文产生的 token 边界可能有很大不同。"),
    ]
    for en_u, en_a, zh_u, zh_a in compare:
        add(rows, "en", en_u, en_a)
        add(rows, "zh", zh_u, zh_a)

    # Deterministic list tasks.
    for n in range(1, 11):
        nums = ", ".join(str(i) for i in range(1, n + 1))
        zh_nums = "、".join(str(i) for i in range(1, n + 1))
        add(rows, "en", f"List the numbers from 1 to {n}.", nums + ".")
        add(rows, "zh", f"列出从 1 到 {n} 的数字。", zh_nums + "。")
        words = ["apple", "book", "cat", "dog", "earth", "fish", "green", "home", "idea", "joy"][:n]
        add(rows, "en", f"Give {n} simple English words.", ", ".join(words) + ".")
        zh_words = ["苹果", "书", "猫", "狗", "地球", "鱼", "绿色", "家", "想法", "快乐"][:n]
        add(rows, "zh", f"给出 {n} 个简单的中文词语。", "、".join(zh_words) + "。")

    # Deduplicate and balance exactly.
    seen = set()
    unique = []
    for r in rows:
        key = (r["lang"], r["user"], r["assistant"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    random.shuffle(unique)
    en = [r for r in unique if r["lang"] == "en"]
    zh = [r for r in unique if r["lang"] == "zh"]
    n = min(len(en), len(zh), 1400)
    data = en[:n] + zh[:n]
    random.shuffle(data)
    split = int(len(data) * 0.9)
    return data[:split], data[split:]


def make_tensors(tok, rows):
    xs, ys = [], []
    for row in rows:
        prefix = f"User: {row['user']}\nAssistant:"
        full = prefix + " " + row["assistant"]
        ids = tok.encode(full).ids
        prefix_ids = tok.encode(prefix).ids
        if len(ids) < len(prefix_ids) + 2:
            continue
        ids = ids[:CONTEXT_LENGTH]
        response_start = min(len(prefix_ids), len(ids))
        x = ids[:-1]
        y = ids[1:]
        y[: max(0, response_start - 1)] = [-100] * max(0, response_start - 1)
        pad = CONTEXT_LENGTH - len(x)
        if pad:
            x += [0] * pad
            y += [-100] * pad
        xs.append(torch.tensor(x, dtype=torch.long))
        ys.append(torch.tensor(y, dtype=torch.long))
    return torch.stack(xs), torch.stack(ys)


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
        return x + self.attn(self.ln1(x)) + self.ffn(self.ln2(x))


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
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, x, tokenizer, max_new_tokens=80, temperature=0.35, top_k=12):
        self.eval()
        start_len = x.size(1)
        text = ""
        for _ in range(max_new_tokens):
            ctx = x[:, -CONTEXT_LENGTH:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            x = torch.cat([x, nxt], dim=1)
            tail = tokenizer.decode(x[0].tolist()[start_len:])
            text = tail
            if "\nUser:" in text or "\nAssistant:" in text or text.count("你") > 25 or text.count("the") > 15:
                break
        return text.strip()


def lr_at(step, total):
    if step <= WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS
    p = min(max((step - WARMUP_STEPS) / max(1, total - WARMUP_STEPS), 0), 1)
    return MIN_LEARNING_RATE + (LEARNING_RATE - MIN_LEARNING_RATE) * 0.5 * (1 + math.cos(math.pi * p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    args, _ = parser.parse_known_args()
    seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 112)
    print("Step 48: Curated bilingual instruction alignment")
    print("=" * 112)
    print(f"device:              {device}")
    if device.type == "cuda":
        print(f"GPU:                 {torch.cuda.get_device_name(0)}")

    drive = Path("/content/drive/MyDrive/transformers_exercise_20260911")
    ckpt = drive / "artifacts" / "step45" / "tiny_gpt_step45.pt"
    tok_path = drive / "artifacts" / "step43" / "step43_bpe_8000.json"
    tok = Tokenizer.from_file(str(tok_path))
    train_rows, val_rows = build_examples()
    train_x, train_y = make_tensors(tok, train_rows)
    val_x, val_y = make_tensors(tok, val_rows)
    print("\nPart 1: Curated SFT dataset")
    print("-" * 112)
    print(f"Train examples:      {len(train_x):,}")
    print(f"Validation examples: {len(val_x):,}")
    print(f"English / Chinese:   {sum(r['lang']=='en' for r in train_rows):,} / {sum(r['lang']=='zh' for r in train_rows):,}")

    model = TinyGPT().to(device)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model.train()
    n = len(train_x)
    tick = time.perf_counter()
    for step in range(1, args.steps + 1):
        lr = lr_at(step, args.steps)
        for g in optimizer.param_groups:
            g["lr"] = lr
        idx = torch.randint(0, n, (BATCH_SIZE,))
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
        if step == 1 or step % LOG_INTERVAL == 0 or step == args.steps:
            speed = (step * BATCH_SIZE * CONTEXT_LENGTH) / max(time.perf_counter() - tick, 1e-6)
            print(f"step {step:>5}/{args.steps} | loss {loss.item():.4f} | lr {lr:.2e} | {speed:,.0f} tok/s")
        if step % EVAL_INTERVAL == 0 or step == args.steps:
            model.eval()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                m = min(len(val_x), BATCH_SIZE)
                _, vloss = model(val_x[:m].to(device), val_y[:m].to(device))
            model.train()
            print(f"           validation loss: {vloss.item():.4f}")

    out_dir = drive / "artifacts" / "step48"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = out_dir / "tiny_gpt_step48_curated_sft.pt"
    out_tok = out_dir / "step48_tokenizer.json"
    tok.save(str(out_tok))
    torch.save({
        "model_state_dict": model.state_dict(),
        "tokenizer_path": str(out_tok),
        "base_checkpoint": str(ckpt),
        "step": args.steps,
        "train_examples": len(train_x),
        "validation_examples": len(val_x),
    }, out_ckpt)

    print("\nPart 2: deterministic instruction probes")
    prompts = [
        "User: Explain what artificial intelligence is in simple terms.\nAssistant:",
        "User: 请用简单中文解释什么是人工智能。\nAssistant:",
        "User: 你好，请介绍一下你自己。\nAssistant:",
        "User: What is a transformer model?\nAssistant:",
        "User: What is 7 + 8?\nAssistant:",
        "User: 12 加 9 等于多少？\nAssistant:",
    ]
    model.eval()
    for prompt in prompts:
        ids = tok.encode(prompt).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        answer = model.generate(x, tok)
        print(f"\n{prompt}\n{answer}")
    print("\nStep 48 complete.")
    print(f"Checkpoint: {out_ckpt}")


if __name__ == "__main__":
    main()
