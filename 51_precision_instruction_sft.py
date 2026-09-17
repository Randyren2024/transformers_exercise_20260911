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

# Step 51: Precision instruction SFT for the ~8.4M model.
# Starts from Step 50's best checkpoint.
# Main goals:
#   1) remove exact duplicate instruction examples
#   2) teach a consistent [END] answer boundary
#   3) add varied arithmetic / translation / short QA examples
#   4) block repeated 3-grams during generation

SEED = 51
VOCAB_SIZE = 8000
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 8
D_FF = 1024
CONTEXT = 256
BATCH = 64
STEPS = 1000
LR = 1.5e-6
MIN_LR = 2.0e-7
WARM = 80
WD = 0.01
END_MARK = " [END]"
LOG_INTERVAL = 25
EVAL_INTERVAL = 100


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add(rows, lang, user, answer):
    user = re.sub(r"\s+", " ", user).strip()
    answer = re.sub(r"\s+", " ", answer).strip()
    if 3 <= len(user) <= 240 and 1 <= len(answer) <= 260:
        rows.append((lang, user, answer))


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
        ("software", "Software is the programs and data that run on a computer.", "软件", "软件是在计算机上运行的程序和数据。"),
        ("hardware", "Hardware is the physical part of a computer or device.", "硬件", "硬件是计算机或设备的物理部件。"),
        ("a dataset", "A dataset is a collection of examples used for training or evaluation.", "数据集", "数据集是一组用于训练或评估的样本。"),
        ("a browser", "A browser is software used to access and view websites.", "浏览器", "浏览器是用于访问和查看网站的软件。"),
        ("an email", "An email is a message sent electronically over a network.", "电子邮件", "电子邮件是通过网络以电子方式发送的信息。"),
        ("a website", "A website is a collection of web pages available on the internet.", "网站", "网站是互联网上的一组网页。"),
    ]
    en_variants = ["What is {}?", "Explain {} in simple terms.", "Give a short definition of {}.", "In one sentence, what is {}?"]
    zh_variants = ["什么是{}？", "请用简单的话解释{}。", "请简要定义{}。", "请用一句话说明什么是{}。"]
    for term_en, ans_en, term_zh, ans_zh in concepts:
        for qe, qz in zip(en_variants, zh_variants):
            add(rows, "en", qe.format(term_en), ans_en)
            add(rows, "zh", qz.format(term_zh), ans_zh)

    for a in range(1, 81):
        b = (a * 11) % 31 + 1
        add(rows, "en", f"What is {a} + {b}?", f"{a + b}.")
        add(rows, "en", f"Calculate {a} plus {b}.", f"{a + b}.")
        add(rows, "zh", f"{a} 加 {b} 等于多少？", f"{a + b}。")
        add(rows, "zh", f"请计算 {a} 加 {b}。", f"{a + b}。")
    for a in range(10, 81):
        b = (a * 5) % 27 + 1
        add(rows, "en", f"What is {a} - {b}?", f"{a - b}.")
        add(rows, "en", f"Calculate {a} minus {b}.", f"{a - b}.")
        add(rows, "zh", f"{a} 减 {b} 等于多少？", f"{a - b}。")
        add(rows, "zh", f"请计算 {a} 减 {b}。", f"{a - b}。")
    for a in range(2, 41):
        b = (a * 3) % 11 + 2
        add(rows, "en", f"What is {a} times {b}?", f"{a * b}.")
        add(rows, "zh", f"{a} 乘 {b} 等于多少？", f"{a * b}。")
    for b in range(2, 11):
        for q in range(2, 16):
            a = b * q
            add(rows, "en", f"What is {a} divided by {b}?", f"{q}.")
            add(rows, "zh", f"{a} 除以 {b} 等于多少？", f"{q}。")

    words = [
        ("hello", "你好"), ("thank you", "谢谢"), ("good morning", "早上好"), ("computer", "电脑"), ("language model", "语言模型"),
        ("machine learning", "机器学习"), ("training", "训练"), ("data", "数据"), ("attention", "注意力"), ("question", "问题"),
        ("answer", "答案"), ("book", "书"), ("water", "水"), ("fire", "火"), ("friend", "朋友"), ("school", "学校"),
        ("city", "城市"), ("robot", "机器人"), ("cloud", "云"), ("future", "未来"), ("window", "窗户"), ("house", "房子"),
        ("road", "道路"), ("car", "汽车"), ("phone", "手机"), ("model", "模型"), ("network", "网络"), ("image", "图像"),
        ("video", "视频"), ("price", "价格"), ("product", "产品"), ("customer", "客户"), ("market", "市场"), ("simple", "简单"),
        ("small", "小型"), ("large", "大型"), ("fast", "快速"), ("slow", "缓慢"),
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
        ("Do you understand English?", "Yes. I can understand and generate English.", "你懂英语吗？", "是的。我可以理解并生成英语。"),
        ("Are you a human?", "No. I am a language model.", "你是人类吗？", "不是。我是一个语言模型。"),
        ("Do you know every fact?", "No. I can make mistakes and I do not know every fact.", "你知道所有事实吗？", "不知道。我可能犯错，也不可能知道所有事实。"),
        ("What should you do when unsure?", "Say that you are not sure instead of pretending to know.", "不确定答案时应该怎么办？", "应该说明自己不确定，而不是假装知道。"),
    ]
    for en_q, en_a, zh_q, zh_a in identity:
        variants_en = [en_q, "Please answer briefly: " + en_q, "Answer in one sentence: " + en_q]
        variants_zh = [zh_q, "请简短回答：" + zh_q, "请用一句话回答：" + zh_q]
        for qe, qz in zip(variants_en, variants_zh):
            add(rows, "en", qe, en_a)
            add(rows, "zh", qz, zh_a)

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
        ("What is a robot?", "A machine that can perform programmed or controlled actions.", "什么是机器人？", "能够执行程序或控制动作的机器。"),
    ]
    for q, a, zq, za in facts:
        add(rows, "en", q, a); add(rows, "zh", zq, za)
        add(rows, "en", "Answer briefly: " + q, a); add(rows, "zh", "请简短回答：“" + zq + "”", za)

    tasks = [
        ("Say hello in a friendly way.", "Hello! It is nice to meet you.", "用友好的方式说你好。", "你好！很高兴认识你。"),
        ("Say thank you politely.", "Thank you very much.", "礼貌地说谢谢。", "非常感谢你。"),
        ("Give me one simple study tip.", "Study one small topic at a time and review it regularly.", "给我一个简单的学习建议。", "一次学习一个小主题，并定期复习。"),
        ("Give me one simple programming tip.", "Test small pieces of code before combining them into a larger program.", "给我一个简单的编程建议。", "先测试小段代码，再把它们组合成更大的程序。"),
        ("Why is sleep important? Answer in one sentence.", "Sleep helps the body and brain recover and function well.", "为什么睡眠重要？用一句话回答。", "睡眠可以帮助身体和大脑恢复，并保持良好状态。"),
        ("Why is exercise useful? Answer in one sentence.", "Regular exercise can improve fitness, mood, and overall health.", "为什么运动有用？用一句话回答。", "规律运动可以改善体能、情绪和整体健康。"),
    ]
    for q, a, zq, za in tasks:
        add(rows, "en", q, a); add(rows, "zh", zq, za)
        add(rows, "en", "Please answer briefly: " + q, a); add(rows, "zh", "请简短回答：" + zq, za)

    unique = list(dict.fromkeys(rows))
    en = [r for r in unique if r[0] == "en"]
    zh = [r for r in unique if r[0] == "zh"]
    random.shuffle(en); random.shuffle(zh)
    n = min(len(en), len(zh))
    data = en[:n] + zh[:n]
    random.shuffle(data)
    cut = int(len(data) * 0.90)
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


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Embedding(CONTEXT, D_MODEL)
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
    def generate(self, x, tok, max_new=64, temperature=0.15, top_k=5, no_repeat_ngram=3):
        self.eval(); generated = []
        for _ in range(max_new):
            logits, _ = self(x[:, -CONTEXT:])
            z = logits[:, -1, :] / max(temperature, 1e-5)
            recent = x[0, -64:].tolist()
            for token_id in set(recent):
                if z[0, token_id] > 0: z[0, token_id] /= 1.12
                else: z[0, token_id] *= 1.12
            if no_repeat_ngram >= 2 and len(x[0]) >= no_repeat_ngram - 1:
                prefix = tuple(x[0, -(no_repeat_ngram - 1):].tolist())
                banned = set(); ids = x[0].tolist()
                for i in range(len(ids) - no_repeat_ngram + 1):
                    gram = tuple(ids[i:i + no_repeat_ngram])
                    if gram[:-1] == prefix: banned.add(gram[-1])
                if banned: z[0, list(banned)] = float("-inf")
            values, _ = torch.topk(z, min(top_k, z.size(-1)))
            z = z.masked_fill(z < values[:, [-1]], float("-inf"))
            nxt = torch.multinomial(torch.softmax(z, dim=-1), 1)
            x = torch.cat([x, nxt], dim=1); generated.append(int(nxt.item()))
            text = tok.decode(generated).strip()
            if "[END]" in text: return text.split("[END]", 1)[0].strip()
        return tok.decode(generated).strip()


def make_tensors(tok, rows):
    xs, ys = [], []
    for _, user, answer in rows:
        prefix = f"User: {user}\nAssistant:"
        full = prefix + " " + answer + END_MARK
        prefix_ids = tok.encode(prefix).ids
        ids = tok.encode(full).ids[:CONTEXT]
        response_start = min(len(prefix_ids), len(ids))
        if len(ids) <= response_start + 1: continue
        x = ids[:-1]; y = ids[1:]
        ignore = max(0, response_start - 1); y[:ignore] = [-100] * ignore
        pad = CONTEXT - len(x)
        if pad > 0: x += [0] * pad; y += [-100] * pad
        xs.append(torch.tensor(x, dtype=torch.long)); ys.append(torch.tensor(y, dtype=torch.long))
    if not xs: raise RuntimeError("No usable SFT examples.")
    return torch.stack(xs), torch.stack(ys)


def lr_at(step, total):
    if step <= WARM: return LR * step / WARM
    p = min(max((step - WARM) / max(1, total - WARM), 0.0), 1.0)
    return MIN_LR + (LR - MIN_LR) * 0.5 * (1 + math.cos(math.pi * p))


def evaluate(model, x, y, device):
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        n = min(BATCH, len(x)); _, loss = model(x[:n].to(device), y[:n].to(device))
    model.train(); return loss.item()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--steps", type=int, default=STEPS); args, _ = parser.parse_known_args(); seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 112); print("Step 51: Precision instruction SFT"); print("=" * 112); print(f"device:              {device}")
    if device.type == "cuda": print(f"GPU:                 {torch.cuda.get_device_name(0)}")
    drive = Path("/content/drive/MyDrive/transformers_exercise_20260911")
    ckpt = drive / "artifacts" / "step50" / "tiny_gpt_step50_best.pt"
    tok_path = drive / "artifacts" / "step43" / "step43_bpe_8000.json"
    if not ckpt.exists(): raise FileNotFoundError(ckpt)
    if not tok_path.exists(): raise FileNotFoundError(tok_path)
    tok = Tokenizer.from_file(str(tok_path))
    train_rows, val_rows = build_dataset(); train_x, train_y = make_tensors(tok, train_rows); val_x, val_y = make_tensors(tok, val_rows)
    print("\nPart 1: Precision SFT dataset"); print("-" * 112); print(f"Train examples:      {len(train_x):,}"); print(f"Validation examples: {len(val_x):,}"); print(f"English / Chinese:   {sum(r[0] == 'en' for r in train_rows):,} / {sum(r[0] == 'zh' for r in train_rows):,}")
    model = GPT().to(device); model.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95)); scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = float("inf"); tick = time.perf_counter(); n = len(train_x); model.train()
    for step in range(1, args.steps + 1):
        lr = lr_at(step, args.steps); optimizer.param_groups[0]["lr"] = lr; idx = torch.randint(0, n, (BATCH,)); x = train_x[idx].to(device); y = train_y[idx].to(device); optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"): _, loss = model(x, y)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update()
        if step == 1 or step % LOG_INTERVAL == 0 or step == args.steps:
            speed = step * BATCH * CONTEXT / max(time.perf_counter() - tick, 1e-6); print(f"step {step:>5}/{args.steps} | loss {loss.item():.4f} | lr {lr:.2e} | {speed:,.0f} tok/s")
        if step % EVAL_INTERVAL == 0 or step == args.steps:
            vloss = evaluate(model, val_x, val_y, device); print(f"           validation loss: {vloss:.4f}")
            if vloss < best:
                best = vloss; out = drive / "artifacts" / "step51"; out.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state_dict": model.state_dict(), "step": step, "validation_loss": best, "base_checkpoint": str(ckpt), "tokenizer_path": str(tok_path)}, out / "tiny_gpt_step51_best.pt")
                print(f"           new best checkpoint: {best:.4f}")
    print("\nPart 2: Precision probes")
    prompts = [
        "User: Explain artificial intelligence in simple terms.\nAssistant:", "User: 请用简单中文解释什么是人工智能。\nAssistant:",
        "User: What is a transformer model?\nAssistant:", "User: 什么是 Transformer 模型？\nAssistant:",
        "User: What is 7 + 8?\nAssistant:", "User: 12 加 9 等于多少？\nAssistant:",
        "User: Translate 'robot' into Chinese.\nAssistant:", "User: 把“data”翻译成中文。\nAssistant:",
        "User: What can you do?\nAssistant:", "User: 你能做什么？\nAssistant:",
    ]
    model.eval()
    for prompt in prompts:
        ids = tok.encode(prompt).ids; x = torch.tensor([ids], dtype=torch.long, device=device); text = model.generate(x, tok); print(f"\n{prompt}\n{text}")
    print("\nStep 51 complete."); print(f"Best checkpoint: {drive / 'artifacts' / 'step51' / 'tiny_gpt_step51_best.pt'}")


if __name__ == "__main__": main()
