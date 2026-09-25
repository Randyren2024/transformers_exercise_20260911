import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer

# Node 71 — semantic-token / exposure-bias diagnostic
# Purpose:
#   1) Ignore the formatting-only leading space token.
#   2) Measure the rank/NLL of the first semantic answer token.
#   3) Find the first position where greedy free-run generation diverges
#      from the gold answer token sequence.
#   4) Compare Step 66 base vs Step 69 SFT.

DRIVE = Path("/content/drive/MyDrive/transformers_exercise_20260911")
TOK_PATH = DRIVE / "artifacts/step43/step43_bpe_8000.json"
STEP66 = DRIVE / "artifacts/step66/tiny_gpt_v2_best.pt"
STEP69 = DRIVE / "artifacts/step69/tiny_gpt_v2_42m_sft_best.pt"
STEP69_FROZEN = DRIVE / "artifacts/step69/tiny_gpt_v2_42m_sft_frozenhead_baseline.pt"

VOCAB = 8000
D = 512
H = 8
LAYERS = 12
FF = 2048
CTX = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Probe sets are generated from the same split 69_42m_controlled_sft.py
# uses (qa_groups() shuffled with SEED=69, 80/20). Regenerate rather than
# hand-edit. PROBES_HELDOUT is the headline metric: those facts are never
# trained on. The gap between the two sets measures memorization.
# Auto-generated probe sets for the Node 70B/71 diagnostics.
# Regenerate with the split in 69_42m_controlled_sft.py (SEED=69, 80/20).

# 9 QA groups Node 69 never trains on (its validation split).
PROBES_HELDOUT = [
    ('What does a tokenizer do?', 'A tokenizer converts text into tokens that a model can process.'),
    ('What is a tokenizer?', 'A tokenizer converts text into tokens that a model can process.'),
    ('What was Alan Turing known for?', 'Alan Turing was a British mathematician and computer scientist.'),
    ('Who was Alan Turing?', 'Alan Turing was a British mathematician and computer scientist.'),
    ('What was Newton known for?', 'Isaac Newton was an English physicist and mathematician.'),
    ('Who was Isaac Newton?', 'Isaac Newton was an English physicist and mathematician.'),
    ('What is JSON used for?', 'JSON is a text format commonly used to represent structured data.'),
    ('What is JSON?', 'JSON is a text format commonly used to represent structured data.'),
    ('JSON 用来做什么？', 'JSON 是一种常用于表示结构化数据的文本格式。'),
    ('什么是 JSON？', 'JSON 是一种常用于表示结构化数据的文本格式。'),
    ('Define machine learning.', 'Machine learning lets computers learn patterns from data.'),
    ('What is machine learning?', 'Machine learning lets computers learn patterns from data.'),
    ('What is opposite to big?', 'Small.'),
    ('What is the opposite of big?', 'Small.'),
    ('Define the Internet.', 'The Internet is a global network of connected computer systems.'),
    ('What is the Internet?', 'The Internet is a global network of connected computer systems.'),
    ('How many months are in a year?', 'Twelve months.'),
    ('How many months make one year?', 'Twelve months.'),
    ('tokenizer 有什么作用？', 'tokenizer 把文本转换成模型可以处理的词元。'),
    ('什么是 tokenizer？', 'tokenizer 把文本转换成模型可以处理的词元。'),
    ('什么是互联网？', '互联网是由相互连接的计算机系统组成的全球网络。'),
    ('请定义互联网。', '互联网是由相互连接的计算机系统组成的全球网络。'),
    ('一年包含多少个月？', '十二个月。'),
    ('一年有几个月？', '十二个月。'),
    ('图灵以什么著名？', '图灵是英国数学家和计算机科学家。'),
    ('图灵是谁？', '图灵是英国数学家和计算机科学家。'),
    ('什么是大的反义词？', '小。'),
    ('大的反义词是什么？', '小。'),
    ("Translate 'data' into Chinese.", '数据。'),
    ('What is the Chinese word for data?', '数据。'),
    ('data 的中文是什么？', '数据。'),
    ('把“data”翻译成中文。', '数据。'),
    ('什么是机器学习？', '机器学习让计算机从数据中学习规律。'),
    ('请定义机器学习。', '机器学习让计算机从数据中学习规律。'),
    ('牛顿以什么著名？', '牛顿是英国物理学家和数学家。'),
    ('牛顿是谁？', '牛顿是英国物理学家和数学家。'),
]

# 6 QA groups drawn from Node 69's training split, same construction, for contrast.
PROBES_SEEN = [
    ('Calculate 72 / 8.', '9.'),
    ('What is 72 divided by 8?', '9.'),
    ('72 除以 8 等于多少？', '9。'),
    ('计算 72 除以 8。', '9。'),
    ('Define a language model.', 'A language model predicts likely tokens from context and can generate text.'),
    ('What is a language model?', 'A language model predicts likely tokens from context and can generate text.'),
    ('Define artificial intelligence.', 'Artificial intelligence is technology that enables computers to perform tasks that normally require human intelligence.'),
    ('What is artificial intelligence?', 'Artificial intelligence is technology that enables computers to perform tasks that normally require human intelligence.'),
    ('What was Li Bai known for?', 'Li Bai was a Chinese poet of the Tang dynasty.'),
    ('Who was Li Bai?', 'Li Bai was a Chinese poet of the Tang dynasty.'),
    ('What is the chemical formula of water?', 'Water is H2O.'),
    ('What is water made of?', 'Water is H2O.'),
    ('什么是人工智能？', '人工智能是让计算机执行通常需要人类智能任务的一种技术。'),
    ('请定义人工智能。', '人工智能是让计算机执行通常需要人类智能任务的一种技术。'),
    ('李白以什么著名？', '李白是中国唐代诗人。'),
    ('李白是谁？', '李白是中国唐代诗人。'),
    ('水由什么组成？', '水的化学式是 H2O。'),
    ('水的化学式是什么？', '水的化学式是 H2O。'),
    ('什么是语言模型？', '语言模型根据上下文预测可能的词元，并生成文本。'),
    ('请定义语言模型。', '语言模型根据上下文预测可能的词元，并生成文本。'),
    ("Translate 'software' into Chinese.", '软件。'),
    ('What is the Chinese word for software?', '软件。'),
    ('software 的中文是什么？', '软件。'),
    ('把“software”翻译成中文。', '软件。'),
]



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
        pos = torch.arange(x.size(1), device=x.device)
        h = self.tok(x) + self.pos(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))


def load_model(path):
    model = TinyGPT().to(DEVICE)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def decode_one(tok, token_id):
    return tok.decode([int(token_id)])


@torch.inference_mode()
def diagnose(model, tok, prompt, answer, max_new=24):
    prefix = f"User: {prompt}\nAssistant:"
    # Same training format as Node 69.
    full = prefix + " " + answer + " [END]"

    prefix_ids = tok.encode(prefix).ids
    full_ids = tok.encode(full).ids

    # Gold answer token sequence starts immediately after the prefix.
    gold = full_ids[len(prefix_ids):]
    answer_ids = gold

    # Find first token that contains non-whitespace: semantic first token.
    semantic_idx = None
    for i, tid in enumerate(answer_ids):
        s = decode_one(tok, tid)
        if s.strip():
            semantic_idx = i
            break

    prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=DEVICE)
    logits = model(prefix_tensor)[0, -1]
    logp = F.log_softmax(logits.float(), dim=-1)

    first_id = answer_ids[0]
    first_rank = int((logp > logp[first_id]).sum().item()) + 1
    first_nll = -float(logp[first_id].item())

    # Teacher-forced semantic first token.
    sem_rank = None
    sem_nll = None
    sem_token = None
    if semantic_idx is not None:
        # Context includes prefix + all earlier gold answer tokens.
        ctx_ids = prefix_ids + answer_ids[:semantic_idx]
        ctx = torch.tensor([ctx_ids], dtype=torch.long, device=DEVICE)
        sem_logits = model(ctx)[0, -1]
        sem_logp = F.log_softmax(sem_logits.float(), dim=-1)
        sem_id = answer_ids[semantic_idx]
        sem_rank = int((sem_logp > sem_logp[sem_id]).sum().item()) + 1
        sem_nll = -float(sem_logp[sem_id].item())
        sem_token = decode_one(tok, sem_id)

    # Free-run greedy generation.
    current = prefix_tensor.clone()
    generated = []
    for _ in range(max_new):
        logits = model(current[:, -CTX:])[:, -1, :]
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        current = torch.cat([current, nxt], dim=1)
        generated.append(int(nxt.item()))

    # First divergence against the gold answer path.
    div = None
    m = min(len(generated), len(answer_ids))
    for i in range(m):
        if generated[i] != answer_ids[i]:
            div = i
            break
    if div is None and len(generated) != len(answer_ids):
        div = m

    generated_text = tok.decode(generated).strip()
    gold_text = tok.decode(answer_ids).strip()

    return {
        "format_first": decode_one(tok, first_id),
        "format_rank": first_rank,
        "format_nll": first_nll,
        "semantic_idx": semantic_idx,
        "semantic_token": sem_token,
        "semantic_rank": sem_rank,
        "semantic_nll": sem_nll,
        "divergence_token_index": div,
        "generated_text": generated_text,
        "gold_text": gold_text,
    }


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


@torch.inference_mode()
def sweep(model, tok, probes, label):
    results = [diagnose(model, tok, p, a) for p, a in probes]
    sem = [r for r in results if r["semantic_rank"] is not None]
    div = [
        r["divergence_token_index"]
        for r in results
        if r["divergence_token_index"] is not None
    ]
    stats = {
        "label": label,
        "n": len(results),
        "format_nll": mean(r["format_nll"] for r in results),
        "format_top1": mean(r["format_rank"] == 1 for r in results) * 100,
        "sem_nll": mean(r["semantic_nll"] for r in sem),
        "sem_rank": mean(r["semantic_rank"] for r in sem),
        "sem_top1": mean(r["semantic_rank"] == 1 for r in sem) * 100,
        "divergence": mean(div),
        "any_token_correct": mean(r["divergence_token_index"] != 0 for r in results) * 100,
    }

    print(f"\n{label}  (n={stats['n']})")
    print(f"  format-first  NLL {stats['format_nll']:.4f} | top1 {stats['format_top1']:.1f}%")
    print(f"  semantic-first NLL {stats['sem_nll']:.4f} | rank {stats['sem_rank']:.1f} "
          f"| top1 {stats['sem_top1']:.1f}%")
    print(f"  mean first free-run divergence token: {stats['divergence']:.2f}")
    print(f"  probes whose FIRST generated token is correct: {stats['any_token_correct']:.1f}%")

    if label == "HELD-OUT":
        print("\n  per-probe detail (held-out):")
        for (prompt, answer), r in zip(probes, results):
            print(f"    {prompt}")
            print(f"      gold {answer!r}")
            print(f"      sem first {r['semantic_token']!r} rank {r['semantic_rank']} "
                  f"nll {r['semantic_nll']:.3f} | diverge @ {r['divergence_token_index']}")
            print(f"      free-run {r['generated_text']!r}")

    return stats


def main():
    tok = Tokenizer.from_file(str(TOK_PATH))

    print("=" * 112)
    print("Node 71 - semantic-token / exposure-bias diagnostic")
    print("=" * 112)
    print("device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print(f"HELD-OUT probes: {len(PROBES_HELDOUT)}  (facts Node 69 never trains on)")
    print(f"SEEN probes:     {len(PROBES_SEEN)}  (drawn from Node 69 training split)")

    print("\nLoading Step 66 base...")
    step66 = load_model(STEP66)
    print("Loading Step 69 (frozen head)...")
    step69_frozen = load_model(STEP69_FROZEN)
    print("Loading Step 69 (head trained)...")
    step69_fixed = load_model(STEP69)

    models = [
        ("Step 66 base", step66),
        ("Step 69 frozen-head", step69_frozen),
        ("Step 69 head-trained", step69_fixed),
    ]

    table = {}
    for name, model in models:
        print("\n" + "=" * 112)
        print(name)
        print("=" * 112)
        table[name] = {
            "HELD-OUT": sweep(model, tok, PROBES_HELDOUT, "HELD-OUT"),
            "SEEN": sweep(model, tok, PROBES_SEEN, "SEEN"),
        }

    width = 22
    names = [n for n, _ in models]

    def grid(title, fmt, field, suffix=""):
        print(f"\n{title}")
        print(f"{'subset':10s}" + "".join(f"{n:>{width}s}" for n in names))
        for key in ("HELD-OUT", "SEEN"):
            cells = "".join(
                f"{format(table[n][key][field], fmt) + suffix:>{width}s}" for n in names
            )
            print(f"{key:10s}" + cells)

    print("\n" + "=" * 112)
    print("SUMMARY")
    print("=" * 112)
    grid("semantic-first rank (lower is better)", ".1f", "sem_rank")
    grid("semantic-first top1", ".1f", "sem_top1", "%")
    grid("format-first NLL", ".4f", "format_nll")
    grid("mean first free-run divergence token (0 = wrong at the very first token)",
         ".2f", "divergence")
    grid("first generated token correct", ".1f", "any_token_correct", "%")

    print("\nNode 71 complete.")


if __name__ == "__main__":
    main()
