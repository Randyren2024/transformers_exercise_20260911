import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer


# Node 62
# Goal:
#   1) start again from the clean Step 59 knowledge base
#   2) teach a very small set of high-confidence bilingual facts/concepts
#   3) explicitly rank correct answers above common distractors
#   4) preserve the Step 59 token distribution on the prompt via teacher KL
#   5) add a small amount of raw-corpus LM loss to reduce drift
#
# This is intentionally SMALL and repeated. The objective is alignment,
# not another large knowledge dump.

SEED = 62
VOCAB = 8000
D = 256
H = 8
LAYERS = 8
FF = 1024
CTX = 256

BATCH = 24
STEPS = 1200

LR = 3.0e-7
MIN_LR = 8.0e-8
WARMUP = 100
WD = 0.01

RANK_WEIGHT = 0.35
KL_WEIGHT = 0.08
RAW_WEIGHT = 0.10
MARGIN = 0.20

DRIVE = Path("/content/drive/MyDrive/transformers_exercise_20260911")
BASE_CKPT = DRIVE / "artifacts" / "step59" / "tiny_gpt_v2_best.pt"
TOKENIZER_PATH = DRIVE / "artifacts" / "step43" / "step43_bpe_8000.json"
RAW_TRAIN_BIN = DRIVE / "data" / "step58_v2_corpus" / "step58_train_ids.uint16"
OUT_DIR = DRIVE / "artifacts" / "step62"


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


def load_model(path, device):
    model = TinyGPT().to(device)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    return model


def lr_at(step):
    if step <= WARMUP:
        return LR * step / WARMUP
    p = min(max((step - WARMUP) / max(1, STEPS - WARMUP), 0.0), 1.0)
    return MIN_LR + (LR - MIN_LR) * 0.5 * (1.0 + math.cos(math.pi * p))


def qa_groups():
    # Each group:
    #   english prompt variants, correct answer, wrong answer
    #   chinese prompt variants, correct answer, wrong answer
    #
    # Answers are intentionally short so the tiny model gets a clean signal.
    groups = [
        (
            ["Who was Li Bai?", "What was Li Bai known for?"],
            "Li Bai was a Chinese poet of the Tang dynasty.",
            "Li Bai was a modern scientist.",
            ["李白是谁？", "李白以什么著名？"],
            "李白是中国唐代诗人。",
            "李白是现代科学家。",
        ),
        (
            ["Who was Albert Einstein?", "What was Einstein known for?"],
            "Albert Einstein was a German-born physicist.",
            "Albert Einstein was an American politician.",
            ["爱因斯坦是谁？", "爱因斯坦以什么著名？"],
            "爱因斯坦是出生于德国的物理学家。",
            "爱因斯坦是美国政治家。",
        ),
        (
            ["Who was Isaac Newton?", "What was Newton known for?"],
            "Isaac Newton was an English physicist and mathematician.",
            "Isaac Newton was a French painter.",
            ["牛顿是谁？", "牛顿以什么著名？"],
            "牛顿是英国物理学家和数学家。",
            "牛顿是法国画家。",
        ),
        (
            ["Who was Ada Lovelace?", "What was Ada Lovelace known for?"],
            "Ada Lovelace was a pioneer of computer programming.",
            "Ada Lovelace was a Renaissance painter.",
            ["阿达·洛夫莱斯是谁？", "阿达·洛夫莱斯以什么著名？"],
            "阿达·洛夫莱斯是计算机编程先驱。",
            "阿达·洛夫莱斯是文艺复兴时期画家。",
        ),
        (
            ["Who was Alan Turing?", "What was Alan Turing known for?"],
            "Alan Turing was a British mathematician and computer scientist.",
            "Alan Turing was a Chinese poet.",
            ["图灵是谁？", "图灵以什么著名？"],
            "图灵是英国数学家和计算机科学家。",
            "图灵是中国诗人。",
        ),
        (
            ["What is artificial intelligence?", "Define artificial intelligence."],
            "Artificial intelligence is technology that enables computers to perform tasks that normally require human intelligence.",
            "Artificial intelligence is a database table.",
            ["什么是人工智能？", "请定义人工智能。"],
            "人工智能是让计算机执行通常需要人类智能任务的一种技术。",
            "人工智能是一张数据库表。",
        ),
        (
            ["What is machine learning?", "Define machine learning."],
            "Machine learning lets computers learn patterns from data.",
            "Machine learning is a web browser.",
            ["什么是机器学习？", "请定义机器学习。"],
            "机器学习让计算机从数据中学习规律。",
            "机器学习是一种网页浏览器。",
        ),
        (
            ["What is a language model?", "Define a language model."],
            "A language model predicts likely tokens from context and can generate text.",
            "A language model is a physical computer cable.",
            ["什么是语言模型？", "请定义语言模型。"],
            "语言模型根据上下文预测可能的词元，并生成文本。",
            "语言模型是一根电脑电缆。",
        ),
        (
            ["What is a transformer in machine learning?", "Define a transformer model."],
            "A transformer is a neural network architecture built around attention.",
            "A transformer is a database system.",
            ["机器学习中的 Transformer 是什么？", "请定义 Transformer 模型。"],
            "Transformer 是一种以注意力机制为核心的神经网络架构。",
            "Transformer 是一种数据库系统。",
        ),
        (
            ["What is attention in a transformer?", "What does attention do?"],
            "Attention lets a model focus on relevant parts of the input.",
            "Attention is a type of web browser.",
            ["Transformer 中的注意力是什么？", "注意力机制有什么作用？"],
            "注意力机制让模型重点关注输入中更相关的部分。",
            "注意力机制是一种网页浏览器。",
        ),
        (
            ["What is a tokenizer?", "What does a tokenizer do?"],
            "A tokenizer converts text into tokens that a model can process.",
            "A tokenizer is a graphics card.",
            ["什么是 tokenizer？", "tokenizer 有什么作用？"],
            "tokenizer 把文本转换成模型可以处理的词元。",
            "tokenizer 是一种显卡。",
        ),
        (
            ["What is BPE?", "What does BPE stand for in tokenization?"],
            "BPE is a subword tokenization method.",
            "BPE is a database protocol.",
            ["什么是 BPE？", "BPE 在分词中是什么？"],
            "BPE 是一种子词分词方法。",
            "BPE 是一种数据库协议。",
        ),
        (
            ["What is a neural network?", "Define a neural network."],
            "A neural network is a model made of connected computational layers.",
            "A neural network is a file format.",
            ["什么是神经网络？", "请定义神经网络。"],
            "神经网络是由相互连接的计算层组成的模型。",
            "神经网络是一种文件格式。",
        ),
        (
            ["What is a GPU?", "What does GPU stand for?"],
            "A GPU is a processor designed for highly parallel computation.",
            "A GPU is a text file format.",
            ["什么是 GPU？", "GPU 是什么？"],
            "GPU 是适合大规模并行计算的处理器。",
            "GPU 是一种文本文件格式。",
        ),
        (
            ["What is a CPU?", "What does CPU stand for?"],
            "A CPU is a general-purpose processor.",
            "A CPU is an image file format.",
            ["什么是 CPU？", "CPU 是什么？"],
            "CPU 是一种通用处理器。",
            "CPU 是一种图片文件格式。",
        ),
        (
            ["What is Python?", "What is the Python programming language?"],
            "Python is a general-purpose programming language.",
            "Python is a database engine.",
            ["什么是 Python？", "Python 是什么编程语言？"],
            "Python 是一种通用编程语言。",
            "Python 是一种数据库引擎。",
        ),
        (
            ["What is Git?", "What is Git used for?"],
            "Git is a version control system.",
            "Git is a web browser.",
            ["什么是 Git？", "Git 用来做什么？"],
            "Git 是一种版本控制系统。",
            "Git 是一种网页浏览器。",
        ),
        (
            ["What is GitHub?", "What is GitHub used for?"],
            "GitHub is a platform for hosting and collaborating on software repositories.",
            "GitHub is a CPU architecture.",
            ["什么是 GitHub？", "GitHub 用来做什么？"],
            "GitHub 是用于托管和协作软件代码仓库的平台。",
            "GitHub 是一种 CPU 架构。",
        ),
        (
            ["What is HTML?", "What is HTML used for?"],
            "HTML is the markup language used to structure web pages.",
            "HTML is a machine learning optimizer.",
            ["什么是 HTML？", "HTML 用来做什么？"],
            "HTML 是用于组织网页结构的标记语言。",
            "HTML 是一种机器学习优化器。",
        ),
        (
            ["What is CSS?", "What is CSS used for?"],
            "CSS is used to style web pages.",
            "CSS is a programming language for operating systems.",
            ["什么是 CSS？", "CSS 用来做什么？"],
            "CSS 用于设置网页样式。",
            "CSS 是一种操作系统编程语言。",
        ),
        (
            ["What is JSON?", "What is JSON used for?"],
            "JSON is a text format commonly used to represent structured data.",
            "JSON is a graphics card.",
            ["什么是 JSON？", "JSON 用来做什么？"],
            "JSON 是一种常用于表示结构化数据的文本格式。",
            "JSON 是一种显卡。",
        ),
        (
            ["What is a database?", "Define a database."],
            "A database is a system for storing and organizing data.",
            "A database is a programming language.",
            ["什么是数据库？", "请定义数据库。"],
            "数据库是用于存储和组织数据的系统。",
            "数据库是一种编程语言。",
        ),
        (
            ["What is the Internet?", "Define the Internet."],
            "The Internet is a global network of connected computer systems.",
            "The Internet is a single computer.",
            ["什么是互联网？", "请定义互联网。"],
            "互联网是由相互连接的计算机系统组成的全球网络。",
            "互联网是一台单独的电脑。",
        ),
        (
            ["What is gravity?", "Define gravity."],
            "Gravity is a force that attracts masses toward each other.",
            "Gravity is a type of programming language.",
            ["什么是重力？", "请定义重力。"],
            "重力是使物体相互吸引的一种力。",
            "重力是一种编程语言。",
        ),
        (
            ["What is water made of?", "What is the chemical formula of water?"],
            "Water is H2O.",
            "Water is CO2.",
            ["水由什么组成？", "水的化学式是什么？"],
            "水的化学式是 H2O。",
            "水的化学式是 CO2。",
        ),
        (
            ["How many days are in a week?", "How long is one week?"],
            "Seven days.",
            "Ten days.",
            ["一周有几天？", "一星期有多长？"],
            "七天。",
            "十天。",
        ),
        (
            ["How many months are in a year?", "How many months make one year?"],
            "Twelve months.",
            "Ten months.",
            ["一年有几个月？", "一年包含多少个月？"],
            "十二个月。",
            "十个月。",
        ),
        (
            ["What do bees make?", "What do bees produce?"],
            "Bees make honey.",
            "Bees make steel.",
            ["蜜蜂生产什么？", "蜜蜂会制造什么？"],
            "蜜蜂生产蜂蜜。",
            "蜜蜂生产钢铁。",
        ),
        (
            ["What is the opposite of hot?", "What is the opposite of hot?"],
            "Cold.",
            "Tall.",
            ["热的反义词是什么？", "什么是热的反义词？"],
            "冷。",
            "高。",
        ),
        (
            ["What is the opposite of big?", "What is opposite to big?"],
            "Small.",
            "Fast.",
            ["大的反义词是什么？", "什么是大的反义词？"],
            "小。",
            "快。",
        ),
        (
            ["What is 7 + 8?", "Calculate 7 + 8."],
            "15.",
            "9.",
            ["7 加 8 等于多少？", "计算 7 加 8。"],
            "15。",
            "9。",
        ),
        (
            ["What is 12 + 9?", "Calculate 12 + 9."],
            "21.",
            "29.",
            ["12 加 9 等于多少？", "计算 12 加 9。"],
            "21。",
            "29。",
        ),
        (
            ["What is 7 times 8?", "Calculate 7 * 8."],
            "56.",
            "84.",
            ["7 乘 8 等于多少？", "计算 7 乘 8。"],
            "56。",
            "84。",
        ),
        (
            ["What is 72 divided by 8?", "Calculate 72 / 8."],
            "9.",
            "18.",
            ["72 除以 8 等于多少？", "计算 72 除以 8。"],
            "9。",
            "18。",
        ),
        (
            ["Translate 'robot' into Chinese.", "What is the Chinese word for robot?"],
            "机器人。",
            "数据库。",
            ["把“robot”翻译成中文。", "robot 的中文是什么？"],
            "机器人。",
            "数据库。",
        ),
        (
            ["Translate 'data' into Chinese.", "What is the Chinese word for data?"],
            "数据。",
            "软件。",
            ["把“data”翻译成中文。", "data 的中文是什么？"],
            "数据。",
            "软件。",
        ),
        (
            ["Translate 'software' into Chinese.", "What is the Chinese word for software?"],
            "软件。",
            "硬件。",
            ["把“software”翻译成中文。", "software 的中文是什么？"],
            "软件。",
            "硬件。",
        ),
        (
            ["Translate 'hardware' into Chinese.", "What is the Chinese word for hardware?"],
            "硬件。",
            "软件。",
            ["把“hardware”翻译成中文。", "hardware 的中文是什么？"],
            "硬件。",
            "软件。",
        ),
        (
            ["Translate 'computer' into Chinese.", "What is the Chinese word for computer?"],
            "电脑。",
            "机器人。",
            ["把“computer”翻译成中文。", "computer 的中文是什么？"],
            "电脑。",
            "机器人。",
        ),
        (
            ["Translate 'language model' into Chinese.", "What is the Chinese translation of language model?"],
            "语言模型。",
            "机器视觉。",
            ["把“language model”翻译成中文。", "language model 的中文是什么？"],
            "语言模型。",
            "机器视觉。",
        ),
        (
            ["Translate 'attention' into Chinese.", "What is the Chinese translation of attention?"],
            "注意力。",
            "数据库。",
            ["把“attention”翻译成中文。", "attention 的中文是什么？"],
            "注意力。",
            "数据库。",
        ),
        (
            ["Translate 'drone' into Chinese.", "What is the Chinese word for drone?"],
            "无人机。",
            "数据库。",
            ["把“drone”翻译成中文。", "drone 的中文是什么？"],
            "无人机。",
            "数据库。",
        ),
    ]
    return groups


def build_examples():
    rows = []
    for en_prompts, en_good, en_bad, zh_prompts, zh_good, zh_bad in qa_groups():
        for p in en_prompts:
            rows.append(("en", p, en_good, en_bad))
        for p in zh_prompts:
            rows.append(("zh", p, zh_good, zh_bad))
    random.shuffle(rows)
    return rows


def encode_pair(tok, prompt, answer, device):
    prefix = f"User: {prompt}\nAssistant:"
    full = prefix + " " + answer + " [END]"
    prefix_ids = tok.encode(prefix).ids
    full_ids = tok.encode(full).ids
    if len(full_ids) > CTX:
        full_ids = full_ids[:CTX]
    if len(full_ids) <= len(prefix_ids) + 1:
        return None
    x = full_ids[:-1]
    y = full_ids[1:]
    answer_start = min(len(prefix_ids) - 1, len(y))
    target = y[answer_start:]
    if not target:
        return None
    x_t = torch.tensor([x], dtype=torch.long, device=device)
    y_t = torch.tensor([y], dtype=torch.long, device=device)
    mask = torch.zeros_like(y_t, dtype=torch.bool)
    mask[:, answer_start:] = True
    return x_t, y_t, mask


def make_supervised_batch(tok, rows, batch_idx, device):
    encoded = []
    for i in batch_idx:
        item = encode_pair(tok, rows[i][1], rows[i][2], device)
        if item is not None:
            encoded.append(item)
    if not encoded:
        raise RuntimeError("No valid SFT rows in batch.")

    max_len = max(e[0].size(1) for e in encoded)
    xs, ys, masks = [], [], []
    for x, y, mask in encoded:
        pad = max_len - x.size(1)
        if pad:
            x = F.pad(x, (0, pad), value=0)
            y = F.pad(y, (0, pad), value=-100)
            mask = F.pad(mask, (0, pad), value=False)
        xs.append(x)
        ys.append(y)
        masks.append(mask)
    return torch.cat(xs, 0), torch.cat(ys, 0), torch.cat(masks, 0)


def answer_nll_batch(model, tok, prompts, answers, device):
    xs, ys, masks = [], [], []
    for p, a in zip(prompts, answers):
        item = encode_pair(tok, p, a, device)
        if item is None:
            raise RuntimeError("Prompt/answer too long.")
        x, y, mask = item
        xs.append(x)
        ys.append(y)
        masks.append(mask)

    max_len = max(x.size(1) for x in xs)
    xb, yb, mb = [], [], []
    for x, y, m in zip(xs, ys, masks):
        pad = max_len - x.size(1)
        if pad:
            x = F.pad(x, (0, pad), value=0)
            y = F.pad(y, (0, pad), value=-100)
            m = F.pad(m, (0, pad), value=False)
        xb.append(x)
        yb.append(y)
        mb.append(m)

    x = torch.cat(xb, 0)
    y = torch.cat(yb, 0)
    m = torch.cat(mb, 0)

    # encode_pair returns:
    #   x = full_ids[:-1]
    #   y = full_ids[1:]
    # Therefore model(x) predicts exactly y.
    logits = model(x)

    losses = F.cross_entropy(
        logits.reshape(-1, VOCAB),
        y.reshape(-1),
        reduction="none",
    ).view(x.size(0), -1)

    valid = m.float()
    nll = (losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    return nll

def sft_loss(model, tok, prompts, answers):
    xs = []
    ys = []
    masks = []
    for p, a in zip(prompts, answers):
        item = encode_pair(tok, p, a, next(model.parameters()).device)
        if item is None:
            continue
        x, y, mask = item
        xs.append(x)
        ys.append(y)
        masks.append(mask)

    if not xs:
        raise RuntimeError("No valid examples.")

    max_len = max(x.size(1) for x in xs)
    xb, yb = [], []
    for x, y in zip(xs, ys):
        pad = max_len - x.size(1)
        if pad:
            x = F.pad(x, (0, pad), value=0)
            y = F.pad(y, (0, pad), value=-100)
        xb.append(x)
        yb.append(y)

    x = torch.cat(xb, 0)
    y = torch.cat(yb, 0)
    logits = model(x)[:, :-1, :]
    targets = y[:, 1:]
    loss = F.cross_entropy(
        logits.reshape(-1, VOCAB),
        targets.reshape(-1),
        ignore_index=-100,
    )
    return loss


def prompt_kl_loss(student, teacher, tok, prompts, device):
    xs = []
    masks = []
    for p in prompts:
        prefix = f"User: {p}\nAssistant:"
        ids = tok.encode(prefix).ids
        if len(ids) < 2:
            continue
        ids = ids[:CTX]
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        # model(x) predicts ids[1:], so each position in x can be
        # matched directly between student and frozen Step 59 teacher.
        m = torch.ones_like(x, dtype=torch.bool)
        xs.append(x)
        masks.append(m)

    if not xs:
        return torch.tensor(0.0, device=device)

    max_len = max(x.size(1) for x in xs)
    xb, mb = [], []
    for x, m in zip(xs, masks):
        pad = max_len - x.size(1)
        if pad:
            x = F.pad(x, (0, pad), value=0)
            m = F.pad(m, (0, pad), value=False)
        xb.append(x)
        mb.append(m)

    x = torch.cat(xb, 0)
    m = torch.cat(mb, 0)

    student_logits = student(x)
    with torch.no_grad():
        teacher_logits = teacher(x)

    s_logp = F.log_softmax(student_logits.float(), dim=-1)
    t_prob = F.softmax(teacher_logits.float(), dim=-1)
    token_kl = F.kl_div(s_logp, t_prob, reduction="none").sum(dim=-1)

    valid = m.float()
    return (token_kl * valid).sum() / valid.sum().clamp_min(1.0)

def load_raw_store():
    if not RAW_TRAIN_BIN.exists():
        raise FileNotFoundError(RAW_TRAIN_BIN)
    return np.memmap(RAW_TRAIN_BIN, dtype=np.uint16, mode="r")


def raw_batch(store, device):
    max_start = len(store) - CTX - 1
    starts = np.random.randint(0, max_start, size=BATCH)
    x = np.stack(
        [np.asarray(store[s:s + CTX], dtype=np.int64) for s in starts]
    )
    y = np.stack(
        [np.asarray(store[s + 1:s + CTX + 1], dtype=np.int64) for s in starts]
    )
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(y).to(device),
    )


def raw_lm_loss(model, x, y):
    logits = model(x)
    return F.cross_entropy(
        logits.reshape(-1, VOCAB),
        y.reshape(-1),
    )


def evaluate(model, teacher, tok, rows, device):
    model.eval()
    teacher.eval()
    nlls = []
    ranks = []
    with torch.no_grad():
        for _, prompt, good, bad in rows[: min(120, len(rows))]:
            nll_good, nll_bad = answer_nll_batch(
                model,
                tok,
                [prompt, prompt],
                [good, bad],
                device,
            )
            nlls.append(float(nll_good.item()))
            ranks.append(float(nll_good.item() < nll_bad.item()))
    model.train()
    return float(np.mean(nlls)), float(np.mean(ranks))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    args, _ = parser.parse_known_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    if not BASE_CKPT.exists():
        raise FileNotFoundError(BASE_CKPT)
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(TOKENIZER_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))

    print("=" * 112)
    print("Node 62 — TinyGPT v2 precision knowledge-alignment SFT")
    print("=" * 112)
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    rows = build_examples()
    print("Precision QA rows:", len(rows))
    print("QA groups:", len(qa_groups()))
    print("Starting from:", BASE_CKPT)
    print("Objective: supervised answer + ranking + teacher KL + small raw-LM preservation")

    student = load_model(BASE_CKPT, device)
    teacher = load_model(BASE_CKPT, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=LR,
        weight_decay=WD,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    raw_store = load_raw_store()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    best_rank = -1.0
    best_good_nll = float("inf")
    tick = time.perf_counter()
    n = len(rows)

    for step in range(1, args.steps + 1):
        student.train()
        lr = lr_at(step)
        optimizer.param_groups[0]["lr"] = lr

        batch = random.sample(rows, min(BATCH, len(rows)))
        prompts = [r[1] for r in batch]
        good = [r[2] for r in batch]
        bad = [r[3] for r in batch]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            ce = sft_loss(student, tok, prompts, good)

            good_nll, bad_nll = answer_nll_batch(
                student,
                tok,
                prompts,
                good,
                device,
            ), answer_nll_batch(
                student,
                tok,
                prompts,
                bad,
                device,
            )
            rank = F.relu(good_nll - bad_nll + MARGIN).mean()

            kl = prompt_kl_loss(student, teacher, tok, prompts, device)

            rx, ry = raw_batch(raw_store, device)
            raw = raw_lm_loss(student, rx, ry)

            total = ce + RANK_WEIGHT * rank + KL_WEIGHT * kl + RAW_WEIGHT * raw

        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 100 == 0 or step == args.steps:
            speed = step * BATCH * CTX / max(time.perf_counter() - tick, 1e-6)
            print(
                f"step {step:>4}/{args.steps} | "
                f"total {total.item():.4f} | ce {ce.item():.4f} | "
                f"rank {rank.item():.4f} | kl {kl.item():.4f} | raw {raw.item():.4f} | "
                f"lr {lr:.2e} | {speed:,.0f} tok/s"
            )

        if step % 200 == 0 or step == args.steps:
            good_nll, rank_acc = evaluate(
                student,
                teacher,
                tok,
                rows,
                device,
            )
            print(
                f"validation precision: good-NLL={good_nll:.4f} | "
                f"rank-accuracy={rank_acc * 100:.1f}%"
            )

            if rank_acc > best_rank or (rank_acc == best_rank and good_nll < best_good_nll):
                best_rank = rank_acc
                best_good_nll = good_nll
                torch.save(
                    {
                        "model_state_dict": student.state_dict(),
                        "step": step,
                        "rank_accuracy": best_rank,
                        "good_nll": best_good_nll,
                        "base_checkpoint": str(BASE_CKPT),
                        "tokenizer_path": str(TOKENIZER_PATH),
                    },
                    OUT_DIR / "tiny_gpt_v2_precision_best.pt",
                )
                print("saved new best:", best_rank, best_good_nll)

    # Fixed diagnostic probes
    student.eval()
    probes = [
        ("李白是谁？", "李白是中国唐代诗人。"),
        ("谁是爱因斯坦？", "爱因斯坦是出生于德国的物理学家。"),
        ("什么是人工智能？", "人工智能是让计算机执行通常需要人类智能任务的一种技术。"),
        ("什么是 Transformer？", "Transformer 是一种以注意力机制为核心的神经网络架构。"),
        ("What is artificial intelligence?", "Artificial intelligence is technology that enables computers to perform tasks that normally require human intelligence."),
        ("What is a transformer in machine learning?", "A transformer is a neural network architecture built around attention."),
        ("Translate 'software' into Chinese.", "软件。"),
        ("What is 7 + 8?", "15."),
        ("What is 7 times 8?", "56."),
    ]

    print("\nFixed probes")
    for prompt, expected in probes:
        good_nll, bad_nll = answer_nll_batch(
            student,
            tok,
            [prompt, prompt],
            [expected, "This is an unrelated answer."],
            device,
        )
        ids = tok.encode(f"User: {prompt}\nAssistant:").ids
        x = torch.tensor([ids[-CTX:]], dtype=torch.long, device=device)
        generated = []
        for _ in range(32):
            logits = student(x[:, -CTX:])
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            x = torch.cat([x, nxt], dim=1)
            generated.append(int(nxt.item()))
        print(
            f"\nUser: {prompt}\n"
            f"Expected: {expected}\n"
            f"Expected-answer NLL: {good_nll.item():.4f}\n"
            f"Generated: {tok.decode(generated).strip()}"
        )

    print("\nNode 62 complete.")
    print("Best checkpoint:", OUT_DIR / "tiny_gpt_v2_precision_best.pt")


if __name__ == "__main__":
    main()
