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

# Step 54:
# Contrastive precision tuning.
#
# Step 53 showed:
#   - exact language answers can be strong
#   - arithmetic and some short mappings are still confused
#
# This step adds a pairwise ranking objective:
#   the correct answer must receive higher likelihood than distractors.
#
# We keep the model size unchanged and use the Step 53 best checkpoint.
# This is intentionally a final targeted neural attempt before we decide
# whether arithmetic should be handled by a tiny external calculator tool.

SEED = 54
VOCAB = 8000
D = 256
H = 8
LAYERS = 8
FF = 1024
CTX = 256
GROUPS_PER_BATCH = 16
CANDIDATES = 4
STEPS = 1200
LR = 1.2e-6
MIN_LR = 2e-7
WARM = 100
MARGIN = 0.30
RANK_WEIGHT = 0.70
END = " [END]"


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean(s):
    return re.sub(r"\s+", " ", s).strip()


def add_group(groups, user, correct, distractors):
    row = [clean(user), clean(correct)] + [clean(x) for x in distractors[:3]]
    if len(row) == 5 and all(row[1:]):
        groups.append(row)


def build_groups():
    groups = []

    # Arithmetic groups: exact result vs plausible distractors.
    for a in range(0, 60):
        for b in range(0, 40):
            s = a + b
            wrong = [s + 1, s - 1, a * b]
            add_group(groups, f"What is {a} + {b}?", f"{s}.", [f"{wrong[0]}.", f"{wrong[1]}.", f"{wrong[2]}." ])
            add_group(groups, f"{a} 加 {b} 等于多少？", f"{s}。", [f"{wrong[0]}。", f"{wrong[1]}。", f"{wrong[2]}。"])

    for a in range(10, 70):
        for b in range(0, min(a, 30)):
            s = a - b
            wrong = [s + 1, s - 1, a + b]
            add_group(groups, f"What is {a} - {b}?", f"{s}.", [f"{wrong[0]}.", f"{wrong[1]}.", f"{wrong[2]}." ])
            add_group(groups, f"{a} 减 {b} 等于多少？", f"{s}。", [f"{wrong[0]}。", f"{wrong[1]}。", f"{wrong[2]}。"])

    for a in range(2, 35):
        for b in range(2, 13):
            s = a * b
            wrong = [s + a, s + b, a + b]
            add_group(groups, f"What is {a} times {b}?", f"{s}.", [f"{wrong[0]}.", f"{wrong[1]}.", f"{wrong[2]}." ])
            add_group(groups, f"{a} 乘 {b} 等于多少？", f"{s}。", [f"{wrong[0]}。", f"{wrong[1]}。", f"{wrong[2]}。"])

    for b in range(2, 10):
        for q in range(2, 16):
            a = b * q
            wrong = [q + 1, q - 1, b + q]
            add_group(groups, f"What is {a} divided by {b}?", f"{q}.", [f"{wrong[0]}.", f"{wrong[1]}.", f"{wrong[2]}." ])
            add_group(groups, f"{a} 除以 {b} 等于多少？", f"{q}。", [f"{wrong[0]}。", f"{wrong[1]}。", f"{wrong[2]}。"])

    words = [
        ("robot", "机器人", ["软件", "数据", "模型"]),
        ("data", "数据", ["软件", "机器人", "模型"]),
        ("software", "软件", ["数据", "机器人", "硬件"]),
        ("hardware", "硬件", ["软件", "数据", "机器人"]),
        ("computer", "电脑", ["手机", "网络", "模型"]),
        ("language model", "语言模型", ["机器学习", "数据集", "软件"]),
        ("machine learning", "机器学习", ["人工智能", "语言模型", "数据"]),
        ("attention", "注意力", ["训练", "网络", "答案"]),
        ("question", "问题", ["答案", "数据", "训练"]),
        ("answer", "答案", ["问题", "数据", "模型"]),
        ("price", "价格", ["产品", "市场", "客户"]),
        ("product", "产品", ["价格", "客户", "市场"]),
    ]
    for en, zh, distractors in words:
        add_group(groups, f"Translate '{en}' into Chinese.", zh + "。", [x + "。" for x in distractors])
        add_group(groups, f"把“{en}”翻译成中文。", zh + "。", [x + "。" for x in distractors])

    qa = [
        ("What can you do?", "I can answer simple questions and follow short instructions.",
         ["I am a robot.", "I only speak English.", "I know every fact."]),
        ("你能做什么？", "我可以回答简单问题并执行简短指令。",
         ["我是一个机器人。", "我只会说英语。", "我知道所有事实。"]),
        ("Who are you?", "I am a small bilingual language model.",
         ["I am a human.", "I am a search engine.", "I am a calculator."]),
        ("你是谁？", "我是一个小型双语语言模型。",
         ["我是一个人类。", "我是一个搜索引擎。", "我是一个计算器。"]),
        ("What is a robot?", "A robot is a machine that can perform programmed or controlled actions.",
         ["A robot is a type of food.", "A robot is a website.", "A robot is a number."]),
        ("什么是机器人？", "机器人是能够执行程序或控制动作的机器。",
         ["机器人是一种食物。", "机器人是一个网站。", "机器人是一个数字。"]),
        ("Explain artificial intelligence in simple terms.",
         "Artificial intelligence is technology that lets computers perform tasks that normally require human intelligence.",
         ["Machine learning is a spreadsheet.", "Artificial intelligence is a type of battery.", "A transformer is a database."]),
    ]
    for q, correct, wrong in qa:
        add_group(groups, q, correct, wrong)

    random.shuffle(groups)
    return groups


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

    def forward(self, x):
        t = x.size(1)
        pos = torch.arange(t, device=x.device)
        h = self.tok(x) + self.pos(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))


def encode_groups(tok, groups):
    encoded = []
    for user, correct, *wrong in groups:
        prompt = f"User: {user}\nAssistant:"
        candidates = [correct] + wrong
        seqs = []
        starts = []
        for answer in candidates:
            full = prompt + " " + answer + END
            pids = tok.encode(prompt).ids
            ids = tok.encode(full).ids[:CTX]
            start = min(len(pids), len(ids))
            if len(ids) <= start + 1:
                break
            seqs.append(ids)
            starts.append(start)
        if len(seqs) != 4:
            continue
        encoded.append((seqs, starts))
    return encoded


def batch_group_tensors(encoded, indices, device):
    seqs = []
    starts = []
    group_ids = []
    for gi, idx in enumerate(indices):
        four, four_starts = encoded[idx]
        for seq, start in zip(four, four_starts):
            seqs.append(seq)
            starts.append(start)
            group_ids.append(gi)
    tmax = max(len(x) for x in seqs)
    x = []
    targets = []
    mask = []
    for seq, start in zip(seqs, starts):
        xi = seq[:-1]
        yi = seq[1:]
        m = [False] * len(yi)
        for p in range(max(0, start - 1), len(yi)):
            m[p] = True
        pad = (tmax - 1) - len(xi)
        if pad > 0:
            xi = xi + [0] * pad
            yi = yi + [0] * pad
            m = m + [False] * pad
        x.append(xi)
        targets.append(yi)
        mask.append(m)
    return (
        torch.tensor(x, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
        torch.tensor(mask, dtype=torch.bool, device=device),
        torch.tensor(group_ids, dtype=torch.long, device=device),
    )


def answer_nll(logits, targets, mask):
    logp = torch.log_softmax(logits, dim=-1)
    vals = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    vals = vals.masked_fill(~mask, 0.0)
    denom = mask.sum(dim=1).clamp_min(1)
    return -vals.sum(dim=1) / denom


def lr_at(step, total):
    if step <= WARM:
        return LR * step / WARM
    p = min(max((step - WARM) / max(1, total - WARM), 0), 1)
    return MIN_LR + (LR - MIN_LR) * 0.5 * (1 + math.cos(math.pi * p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    args, _ = parser.parse_known_args()
    seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drive = Path("/content/drive/MyDrive/transformers_exercise_20260911")
    ckpt = drive / "artifacts" / "step53" / "tiny_gpt_step53_best.pt"
    tok_path = drive / "artifacts" / "step43" / "step43_bpe_8000.json"

    print("=" * 112)
    print("Step 54: Contrastive precision SFT")
    print("=" * 112)
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    tok = Tokenizer.from_file(str(tok_path))
    groups = build_groups()
    encoded = encode_groups(tok, groups)
    print("Prompt groups:", len(encoded))

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

        idx = random.sample(range(len(encoded)), GROUPS_PER_BATCH)
        x, y, mask, gids = batch_group_tensors(encoded, idx, device)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
            nll = answer_nll(logits, y, mask)
            correct = nll[0::4]
            distractors = torch.stack([nll[1::4], nll[2::4], nll[3::4]], dim=1)
            rank_loss = F.softplus(MARGIN + correct.unsqueeze(1) - distractors).mean()
            ce_loss = correct.mean()
            loss = ce_loss + RANK_WEIGHT * rank_loss

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if step == 1 or step % 100 == 0 or step == args.steps:
            speed = step * x.size(0) * x.size(1) / max(time.perf_counter() - tick, 1e-6)
            print(f"step {step:>4}/{args.steps} | loss {loss.item():.4f} | ce {ce_loss.item():.4f} | rank {rank_loss.item():.4f} | lr {lr:.2e} | {speed:,.0f} tok/s")

        if step % 200 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                score = float(loss.item())
            if score < best:
                best = score
                out = drive / "artifacts" / "step54"
                out.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state_dict": model.state_dict(), "step": step, "loss": best}, out / "tiny_gpt_step54_best.pt")
                print("saved new best:", best)

    print("\nFocused greedy probes")
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
    ]
    for user, expected in probes:
        prompt = f"User: {user}\nAssistant:"
        ids = tok.encode(prompt).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        text = []
        for _ in range(32):
            logits = model(x[:, -CTX:])[:, -1, :]
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
            x = torch.cat([x, nxt], dim=1)
            text_now = tok.decode(x[0].tolist()[len(ids):]).strip()
            if "[END]" in text_now:
                text_now = text_now.split("[END]", 1)[0].strip()
                break
        else:
            text_now = tok.decode(x[0].tolist()[len(ids):]).strip()
        print(f"\nUser: {user}\nAssistant: {text_now}\nExpected: {expected}")

    print("\nStep 54 complete.")
    print("Best checkpoint:", drive / "artifacts" / "step54" / "tiny_gpt_step54_best.pt")


if __name__ == "__main__":
    main()
