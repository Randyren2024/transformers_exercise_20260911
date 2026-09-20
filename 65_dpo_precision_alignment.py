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


# Node 65 — Direct Preference Optimization (DPO) precision alignment
#
# Step 64 diagnostic revealed:
#   - Step 63 rank accuracy = 44.0% (below random, margin is negative)
#   - The pairwise margin loss in Steps 54/63 failed to teach preference
#   - Generation degenerates into repetition
#
# Root cause: margin loss has no normalization, no reference constraint,
# and uses a hard threshold (softplus + margin) instead of a smooth,
# probability-ratio-based objective.
#
# DPO fixes all three:
#   1) Uses probability RATIO (policy / reference) — length-normalized
#   2) Frozen reference model — implicit KL constraint prevents drift
#   3) Sigmoid — smooth gradient even when margin is large
#
# Loss:
#   L_DPO = -log σ( β * ( log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x) ) )
#
# Where:
#   π_θ  = trainable policy (initialized from Step 63 best)
#   π_ref = frozen reference (also Step 63 best)
#   y_w  = preferred (correct) answer
#   y_l  = dispreferred (wrong) answer
#   β    = 0.1 (controls divergence from reference)
#
# A small SFT regularization term (weight 0.15) keeps the policy anchored
# to the correct answer distribution, preventing the reference model's
# broken preferences from being amplified unchecked.

SEED = 65
VOCAB = 8000
D = 256
H = 8
LAYERS = 8
FF = 1024
CTX = 256

BATCH = 16
STEPS = 400
LR = 1e-6
MIN_LR = 2e-7
WARMUP = 50
WD = 0.01

BETA = 0.2        # moderate DPO temperature
SFT_WEIGHT = 0.05 # light SFT anchor to prevent collapse

DRIVE = Path("/content/drive/MyDrive/transformers_exercise_20260911")
BASE_CKPT = DRIVE / "artifacts" / "step63" / "tiny_gpt_v2_precision_best.pt"
TOKENIZER_PATH = DRIVE / "artifacts" / "step43" / "step43_bpe_8000.json"
OUT_DIR = DRIVE / "artifacts" / "step65"


# ---------------------------------------------------------------------------
# Architecture (identical to Step 59/63)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data — same QA groups as Step 63
# ---------------------------------------------------------------------------

def qa_groups():
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
            ['把"robot"翻译成中文。', "robot 的中文是什么？"],
            "机器人。",
            "数据库。",
        ),
        (
            ["Translate 'data' into Chinese.", "What is the Chinese word for data?"],
            "数据。",
            "软件。",
            ['把"data"翻译成中文。', "data 的中文是什么？"],
            "数据。",
            "软件。",
        ),
        (
            ["Translate 'software' into Chinese.", "What is the Chinese word for software?"],
            "软件。",
            "硬件。",
            ['把"software"翻译成中文。', "software 的中文是什么？"],
            "软件。",
            "硬件。",
        ),
        (
            ["Translate 'hardware' into Chinese.", "What is the Chinese word for hardware?"],
            "硬件。",
            "软件。",
            ['把"hardware"翻译成中文。', "hardware 的中文是什么？"],
            "硬件。",
            "软件。",
        ),
        (
            ["Translate 'computer' into Chinese.", "What is the Chinese word for computer?"],
            "电脑。",
            "机器人。",
            ['把"computer"翻译成中文。', "computer 的中文是什么？"],
            "电脑。",
            "机器人。",
        ),
        (
            ["Translate 'language model' into Chinese.", "What is the Chinese translation of language model?"],
            "语言模型。",
            "机器视觉。",
            ['把"language model"翻译成中文。', "language model 的中文是什么？"],
            "语言模型。",
            "机器视觉。",
        ),
        (
            ["Translate 'attention' into Chinese.", "What is the Chinese translation of attention?"],
            "注意力。",
            "数据库。",
            ['把"attention"翻译成中文。', "attention 的中文是什么？"],
            "注意力。",
            "数据库。",
        ),
        (
            ["Translate 'drone' into Chinese.", "What is the Chinese word for drone?"],
            "无人机。",
            "数据库。",
            ['把"drone"翻译成中文。', "drone 的中文是什么？"],
            "无人机。",
            "数据库。",
        ),
    ]
    return groups


def rows_from_groups(groups):
    rows = []
    for en_prompts, en_good, en_bad, zh_prompts, zh_good, zh_bad in groups:
        for p in en_prompts:
            rows.append(("en", p, en_good, en_bad))
        for p in zh_prompts:
            rows.append(("zh", p, zh_good, zh_bad))
    return rows


def split_groups():
    groups = qa_groups()
    rng = random.Random(SEED)
    rng.shuffle(groups)
    cut = max(1, int(len(groups) * 0.80))
    return groups[:cut], groups[cut:]


# ---------------------------------------------------------------------------
# Encoding utilities (identical to Step 63)
# ---------------------------------------------------------------------------

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

    logits = model(x)
    losses = F.cross_entropy(
        logits.reshape(-1, VOCAB),
        y.reshape(-1),
        reduction="none",
    ).view(x.size(0), -1)

    valid = m.float()
    nll = (losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    return nll


def sft_loss(model, tok, prompts, answers, device):
    xs, ys, masks = [], [], []
    for p, a in zip(prompts, answers):
        item = encode_pair(tok, p, a, device)
        if item is None:
            continue
        x, y, mask = item
        xs.append(x)
        ys.append(y)
        masks.append(mask)
    if not xs:
        raise RuntimeError("No valid examples.")

    max_len = max(x.size(1) for x in xs)
    xb, yb, mb = [], [], []
    for x, y, mask in zip(xs, ys, masks):
        pad = max_len - x.size(1)
        if pad:
            x = F.pad(x, (0, pad), value=0)
            y = F.pad(y, (0, pad), value=-100)
            mask = F.pad(mask, (0, pad), value=False)
        xb.append(x)
        yb.append(y)
        mb.append(mask)

    x = torch.cat(xb, 0)
    y = torch.cat(yb, 0)
    mask = torch.cat(mb, 0)

    logits = model(x)
    targets = y.clone()
    targets[~mask] = -100
    return F.cross_entropy(
        logits.reshape(-1, VOCAB),
        targets.reshape(-1),
        ignore_index=-100,
    )


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------

def dpo_loss(policy, reference, tok, prompts, good_answers, bad_answers, device):
    """Compute DPO loss.

    L = -log σ( β * ( (log π_θ(y_w) - log π_ref(y_w)) - (log π_θ(y_l) - log π_ref(y_l)) ) )

    Since NLL = -log π, this becomes:
    L = -log σ( β * ( (nll_ref_w - nll_θ_w) - (nll_ref_l - nll_θ_l) ) )

    Returns loss, and metrics dict.
    """
    # Policy NLLs (with gradient)
    policy_good = answer_nll_batch(policy, tok, prompts, good_answers, device)
    policy_bad = answer_nll_batch(policy, tok, prompts, bad_answers, device)

    # Reference NLLs (no gradient)
    with torch.no_grad():
        ref_good = answer_nll_batch(reference, tok, prompts, good_answers, device)
        ref_bad = answer_nll_batch(reference, tok, prompts, bad_answers, device)

    # Log-ratios:  log π_θ(y) - log π_ref(y) = nll_ref - nll_θ
    logratio_w = ref_good - policy_good   # want this to INCREASE
    logratio_l = ref_bad - policy_bad     # want this to DECREASE

    # DPO loss
    logits_diff = BETA * (logratio_w - logratio_l)
    loss = -F.logsigmoid(logits_diff).mean()

    # Metrics
    with torch.no_grad():
        chosen_reward = BETA * logratio_w
        rejected_reward = BETA * logratio_l
        reward_margin = (chosen_reward - rejected_reward).mean()
        reward_acc = (chosen_reward > rejected_reward).float().mean()

    return loss, {
        "policy_good_nll": policy_good.mean().item(),
        "policy_bad_nll": policy_bad.mean().item(),
        "ref_good_nll": ref_good.mean().item(),
        "ref_bad_nll": ref_bad.mean().item(),
        "reward_margin": reward_margin.item(),
        "reward_acc": reward_acc.item(),
        "logratio_w": logratio_w.mean().item(),
        "logratio_l": logratio_l.mean().item(),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, tok, rows, device):
    model.eval()
    nlls = []
    ranks = []
    with torch.no_grad():
        for _, prompt, good, bad in rows[: min(120, len(rows))]:
            nll_good, nll_bad = answer_nll_batch(
                model, tok, [prompt, prompt], [good, bad], device
            )
            nlls.append(float(nll_good.item()))
            ranks.append(float(nll_good.item() < nll_bad.item()))
    model.train()
    return float(np.mean(nlls)), float(np.mean(ranks))


@torch.inference_mode()
def generate(model, tok, prompt, max_new=32):
    model.eval()
    ids = tok.encode(f"User: {prompt}\nAssistant:").ids
    x = torch.tensor([ids[-CTX:]], dtype=torch.long, device=next(model.parameters()).device)
    out = []
    for _ in range(max_new):
        logits = model(x[:, -CTX:])
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        x = torch.cat([x, nxt], dim=1)
        out.append(int(nxt.item()))
    return tok.decode(out).strip()


def lr_at(step, total):
    if step <= WARMUP:
        return LR * step / WARMUP
    p = min(max((step - WARMUP) / max(1, total - WARMUP), 0.0), 1.0)
    return MIN_LR + (LR - MIN_LR) * 0.5 * (1.0 + math.cos(math.pi * p))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    print("Node 65 — DPO precision alignment")
    print("=" * 112)
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print(f"β = {BETA} | SFT weight = {SFT_WEIGHT} | LR = {LR:.1e} | steps = {args.steps}")
    print("Reference (frozen):", BASE_CKPT)
    print("Policy (trainable): initialized from same checkpoint")

    train_groups, val_groups = split_groups()
    train_rows = rows_from_groups(train_groups)
    val_rows = rows_from_groups(val_groups)
    print(f"Train rows: {len(train_rows)} | Holdout rows: {len(val_rows)}")

    # Load policy and reference from the same checkpoint
    policy = load_model(BASE_CKPT, device)
    reference = load_model(BASE_CKPT, device)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=LR,
        weight_decay=WD,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Baseline evaluation
    base_good_nll, base_rank = evaluate(reference, tok, val_rows, device)
    print(f"\nBaseline (reference) holdout: good-NLL={base_good_nll:.4f} | rank={base_rank*100:.1f}%")

    best_rank = -1.0
    best_good_nll = float("inf")
    tick = time.perf_counter()

    for step in range(1, args.steps + 1):
        policy.train()
        lr = lr_at(step, args.steps)
        optimizer.param_groups[0]["lr"] = lr

        batch = random.sample(train_rows, min(BATCH, len(train_rows)))
        prompts = [r[1] for r in batch]
        good = [r[2] for r in batch]
        bad = [r[3] for r in batch]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            dpo, metrics = dpo_loss(
                policy, reference, tok, prompts, good, bad, device
            )
            sft = sft_loss(policy, tok, prompts, good, device)
            total = dpo + SFT_WEIGHT * sft

        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 50 == 0 or step == args.steps:
            speed = step * BATCH * CTX / max(time.perf_counter() - tick, 1e-6)
            print(
                f"step {step:>4}/{args.steps} | "
                f"dpo {dpo.item():.4f} | sft {sft.item():.4f} | "
                f"rwd_acc {metrics['reward_acc']*100:.1f}% | "
                f"rwd_margin {metrics['reward_margin']:.4f} | "
                f"lr {lr:.2e} | {speed:,.0f} tok/s"
            )

        if step % 100 == 0 or step == args.steps:
            good_nll, rank_acc = evaluate(policy, tok, val_rows, device)
            print(
                f"  holdout: good-NLL={good_nll:.4f} | rank={rank_acc*100:.1f}%"
            )
            if rank_acc > best_rank or (rank_acc == best_rank and good_nll < best_good_nll):
                best_rank = rank_acc
                best_good_nll = good_nll
                torch.save(
                    {
                        "model_state_dict": policy.state_dict(),
                        "step": step,
                        "rank_accuracy": best_rank,
                        "good_nll": best_good_nll,
                        "base_checkpoint": str(BASE_CKPT),
                        "tokenizer_path": str(TOKENIZER_PATH),
                        "beta": BETA,
                        "sft_weight": SFT_WEIGHT,
                    },
                    OUT_DIR / "tiny_gpt_v2_dpo_best.pt",
                )
                print(f"  saved new best: rank={best_rank*100:.1f}% nll={best_good_nll:.4f}")

    # Final probes
    print("\n" + "=" * 112)
    print("Final probes — Step 65 DPO vs Step 63 reference")
    print("=" * 112)

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

    for prompt, expected in probes:
        ref_gen = generate(reference, tok, prompt)
        pol_gen = generate(policy, tok, prompt)
        print(f"\nUser: {prompt}")
        print(f"Expected: {expected}")
        print(f"Step 63:  {ref_gen}")
        print(f"Step 65:  {pol_gen}")

    print("\nNode 65 complete.")
    print("Best checkpoint:", OUT_DIR / "tiny_gpt_v2_dpo_best.pt")
    print(f"Best holdout rank accuracy: {best_rank*100:.1f}%")


if __name__ == "__main__":
    main()
