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

SEED = 53
VOCAB = 8000
D = 256
H = 8
LAYERS = 8
FF = 1024
CTX = 256
BATCH = 64
STEPS = 1800
LR = 3e-6
MIN_LR = 5e-7
WARM = 120
END = " [END]"


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add(rows, lang, q, a):
    q = re.sub(r"\s+", " ", q).strip()
    a = re.sub(r"\s+", " ", a).strip()
    if 2 <= len(q) <= 220 and 1 <= len(a) <= 180:
        rows.append((lang, q, a))


def build_rows():
    rows = []

    # Arithmetic: exact-answer training dominates this step.
    for a in range(0, 100):
        for b in range(0, 100, 5):
            s = a + b
            qs = [
                (f"What is {a} + {b}?", f"{s}."),
                (f"Calculate {a} plus {b}.", f"{s}."),
                (f"{a} plus {b} equals what?", f"{s}."),
                (f"{a} 加 {b} 等于多少？", f"{s}。"),
                (f"请计算 {a} 加 {b}。", f"{s}。"),
            ]
            for q, ans in qs[:3]:
                add(rows, "en", q, ans)
            for q, ans in qs[3:]:
                add(rows, "zh", q, ans)

    for a in range(10, 100):
        for b in range(0, min(a, 45), 5):
            s = a - b
            for q in [
                f"What is {a} - {b}?",
                f"Calculate {a} minus {b}.",
                f"{a} 减 {b} 等于多少？",
                f"请计算 {a} 减 {b}。",
            ]:
                lang = "zh" if "等于" in q or "请计算" in q else "en"
                add(rows, lang, q, f"{s}。" if lang == "zh" else f"{s}.")

    for a in range(2, 40):
        for b in range(2, 16):
            s = a * b
            for q in [f"What is {a} times {b}?", f"Calculate {a} times {b}."]:
                add(rows, "en", q, f"{s}.")
            for q in [f"{a} 乘 {b} 等于多少？", f"请计算 {a} 乘 {b}。"]:
                add(rows, "zh", q, f"{s}。")

    for b in range(2, 13):
        for qv in range(2, 21):
            a = b * qv
            for q in [f"What is {a} divided by {b}?", f"Calculate {a} divided by {b}."]:
                add(rows, "en", q, f"{qv}.")
            for q in [f"{a} 除以 {b} 等于多少？", f"请计算 {a} 除以 {b}。"]:
                add(rows, "zh", q, f"{qv}。")

    # Exact translation mappings. Repeat each mapping under varied prompts.
    words = [
        ("hello", "你好"), ("thank you", "谢谢"), ("computer", "电脑"),
        ("language model", "语言模型"), ("machine learning", "机器学习"),
        ("training", "训练"), ("data", "数据"), ("attention", "注意力"),
        ("question", "问题"), ("answer", "答案"), ("robot", "机器人"),
        ("cloud", "云"), ("window", "窗户"), ("house", "房子"),
        ("car", "汽车"), ("phone", "手机"), ("model", "模型"),
        ("network", "网络"), ("image", "图像"), ("video", "视频"),
        ("price", "价格"), ("product", "产品"), ("customer", "客户"),
        ("market", "市场"), ("software", "软件"), ("hardware", "硬件"),
        ("dataset", "数据集"), ("browser", "浏览器"), ("website", "网站"),
        ("simple", "简单"), ("small", "小型"), ("large", "大型"),
        ("fast", "快速"), ("slow", "缓慢"),
    ]
    for en, zh in words:
        for q in [
            f"Translate '{en}' into Chinese.",
            f"What is '{en}' in Chinese?",
            f"Give the Chinese translation of '{en}'.",
            f"Please translate '{en}' to Chinese.",
        ]:
            add(rows, "en", q, zh + "。")
        for q in [
            f"把“{en}”翻译成中文。",
            f"“{en}”的中文是什么？",
            f"请把“{en}”翻译成中文。",
        ]:
            add(rows, "zh", q, zh + "。")
        for q in [
            f"Translate '{zh}' into English.",
            f"What is '{zh}' in English?",
            f"把“{zh}”翻译成英语。",
        ]:
            lang = "zh" if q.startswith("把") else "en"
            add(rows, lang, q, en + ".")

    # Small, stable factual and identity answers.
    pairs = [
        ("What can you do?", "I can answer simple questions and follow short instructions.",
         "你能做什么？", "我可以回答简单问题并执行简短指令。"),
        ("Who are you?", "I am a small bilingual language model.",
         "你是谁？", "我是一个小型双语语言模型。"),
        ("Are you a human?", "No. I am a language model.",
         "你是人类吗？", "不是。我是一个语言模型。"),
        ("What is a robot?", "A robot is a machine that can perform programmed or controlled actions.",
         "什么是机器人？", "机器人是能够执行程序或控制动作的机器。"),
        ("What is software?", "Software is the programs and data that run on a computer.",
         "什么是软件？", "软件是在计算机上运行的程序和数据。"),
        ("What is data?", "Data is information that can be stored and processed.",
         "什么是数据？", "数据是可以存储和处理的信息。"),
        ("How many days are in a week?", "Seven.", "一周有几天？", "七天。"),
        ("How many months are in a year?", "Twelve.", "一年有几个月？", "十二个月。"),
    ]
    for enq, ena, zhq, zha in pairs:
        for q in [enq, "Please answer briefly: " + enq, "Answer in one sentence: " + enq]:
            add(rows, "en", q, ena)
        for q in [zhq, "请简短回答：" + zhq, "请用一句话回答：" + zhq]:
            add(rows, "zh", q, zha)

    concepts = [
        ("Explain artificial intelligence in simple terms.",
         "Artificial intelligence is technology that lets computers perform tasks that normally require human intelligence.",
         "请用简单中文解释什么是人工智能。",
         "人工智能是让计算机执行通常需要人类智能任务的一种技术。"),
        ("What is a transformer model?",
         "A transformer is a neural network architecture built around attention.",
         "什么是 Transformer 模型？",
         "Transformer 是一种以注意力机制为核心的神经网络架构。"),
    ]
    for enq, ena, zhq, zha in concepts:
        add(rows, "en", enq, ena)
        add(rows, "en", "Please answer briefly: " + enq, ena)
        add(rows, "zh", zhq, zha)
        add(rows, "zh", "请简短回答：" + zhq, zha)

    unique = list(dict.fromkeys(rows))
    en = [r for r in unique if r[0] == "en"]
    zh = [r for r in unique if r[0] == "zh"]
    random.shuffle(en)
    random.shuffle(zh)
    n = min(len(en), len(zh))
    data = en[:n] + zh[:n]
    random.shuffle(data)
    cut = int(len(data) * 0.92)
    return data[:cut], data[cut:]


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


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D)
        self.pos = nn.Embedding(CTX, D)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, VOCAB, bias=False)
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
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, x, tok, max_new=48):
        self.eval()
        start = x.size(1)
        for _ in range(max_new):
            logits, _ = self(x[:, -CTX:])
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            x = torch.cat([x, nxt], dim=1)
            text = tok.decode(x[0].tolist()[start:]).strip()
            if "[END]" in text:
                return text.split("[END]", 1)[0].strip()
        return tok.decode(x[0].tolist()[start:]).strip()


def make_tensors(tok, rows):
    xs, ys = [], []
    for _, user, answer in rows:
        prefix = f"User: {user}\nAssistant:"
        full = prefix + " " + answer + END
        prefix_ids = tok.encode(prefix).ids
        ids = tok.encode(full).ids[:CTX]
        start = min(len(prefix_ids), len(ids))
        if len(ids) <= start + 1:
            continue
        x = ids[:-1]
        y = ids[1:]
        y[:max(0, start - 1)] = [-100] * max(0, start - 1)
        pad = CTX - len(x)
        if pad > 0:
            x += [0] * pad
            y += [-100] * pad
        xs.append(torch.tensor(x, dtype=torch.long))
        ys.append(torch.tensor(y, dtype=torch.long))
    return torch.stack(xs), torch.stack(ys)


def lr_at(step, total):
    if step <= WARM:
        return LR * step / WARM
    p = min(max((step - WARM) / max(1, total - WARM), 0), 1)
    return MIN_LR + (LR - MIN_LR) * 0.5 * (1 + math.cos(math.pi * p))


def evaluate(model, x, y, device):
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        _, loss = model(x[:BATCH].to(device), y[:BATCH].to(device))
    model.train()
    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    args, _ = parser.parse_known_args()
    seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drive = Path("/content/drive/MyDrive/transformers_exercise_20260911")
    ckpt = drive / "artifacts" / "step51" / "tiny_gpt_step51_best.pt"
    tok_path = drive / "artifacts" / "step43" / "step43_bpe_8000.json"

    print("=" * 112)
    print("Step 53: Focused precision SFT")
    print("=" * 112)
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    tok = Tokenizer.from_file(str(tok_path))
    train_rows, val_rows = build_rows()
    train_x, train_y = make_tensors(tok, train_rows)
    val_x, val_y = make_tensors(tok, val_rows)
    print("\nTrain examples:", len(train_x))
    print("Validation examples:", len(val_x))
    print("English / Chinese:", sum(r[0] == "en" for r in train_rows), "/", sum(r[0] == "zh" for r in train_rows))

    model = GPT().to(device)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = float("inf")
    tick = time.perf_counter()

    for step in range(1, args.steps + 1):
        model.train()
        lr = lr_at(step, args.steps)
        opt.param_groups[0]["lr"] = lr
        idx = torch.randint(0, len(train_x), (BATCH,))
        x = train_x[idx].to(device)
        y = train_y[idx].to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if step == 1 or step % 100 == 0 or step == args.steps:
            speed = step * BATCH * CTX / max(time.perf_counter() - tick, 1e-6)
            print(f"step {step:>4}/{args.steps} | loss {loss.item():.4f} | lr {lr:.2e} | {speed:,.0f} tok/s")

        if step % 150 == 0 or step == args.steps:
            v = evaluate(model, val_x, val_y, device)
            print(f"validation loss: {v:.4f}")
            if v < best:
                best = v
                out = drive / "artifacts" / "step53"
                out.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state_dict": model.state_dict(), "step": step, "validation_loss": best}, out / "tiny_gpt_step53_best.pt")
                print("saved new best:", best)

    print("\nExact probes")
    probes = [
        ("What is 7 + 8?", "15."),
        ("12 加 9 等于多少？", "21。"),
        ("What is 17 + 26?", "43."),
        ("What is 7 times 8?", "56."),
        ("What is 72 divided by 8?", "9."),
        ("Translate 'robot' into Chinese.", "机器人。"),
        ("把“data”翻译成中文。", "数据。"),
        ("Translate 'software' into Chinese.", "软件。"),
        ("What can you do?", "I can answer simple questions and follow short instructions."),
        ("你能做什么？", "我可以回答简单问题并执行简短指令。"),
        ("Explain artificial intelligence in simple terms.", "Artificial intelligence is technology that lets computers perform tasks that normally require human intelligence."),
    ]
    for user, expected in probes:
        prompt = f"User: {user}\nAssistant:"
        x = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
        out = model.generate(x, tok)
        print(f"\nUser: {user}\nAssistant: {out}\nExpected: {expected}")

    print("\nStep 53 complete.")
    print("Best checkpoint:", drive / "artifacts" / "step53" / "tiny_gpt_step53_best.pt")


if __name__ == "__main__":
    main()
