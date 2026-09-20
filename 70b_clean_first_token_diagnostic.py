import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer

# Node 70B — clean first-token diagnostic
# No external QA module, no dynamic exec, no train/validation split.
# Compare Step 66 base vs Step 69 best SFT on fixed probes.

DRIVE = Path("/content/drive/MyDrive/transformers_exercise_20260911")
TOK_PATH = DRIVE / "artifacts/step43/step43_bpe_8000.json"
STEP66 = DRIVE / "artifacts/step66/tiny_gpt_v2_best.pt"
STEP69 = DRIVE / "artifacts/step69/tiny_gpt_v2_42m_sft_best.pt"

VOCAB = 8000
D = 512
H = 8
LAYERS = 12
FF = 2048
CTX = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROBES = [
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


def load_model(path):
    model = TinyGPT().to(DEVICE)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def encode(tok, prompt, answer):
    prefix = f"User: {prompt}\nAssistant:"
    full = prefix + " " + answer + " [END]"
    prefix_ids = tok.encode(prefix).ids
    full_ids = tok.encode(full).ids
    return prefix_ids, full_ids


@torch.inference_mode()
def analyze(model, tok, prompt, answer):
    prefix_ids, full_ids = encode(tok, prompt, answer)

    prefix = torch.tensor([prefix_ids], dtype=torch.long, device=DEVICE)
    first_logits = model(prefix)[0, -1]
    logp = F.log_softmax(first_logits.float(), dim=-1)

    first_id = full_ids[len(prefix_ids)]
    first_nll = -float(logp[first_id].item())
    first_rank = int((logp > logp[first_id]).sum().item()) + 1

    topv, topi = torch.topk(logp, 8)
    top_tokens = [
        (tok.decode([int(i.item())]), float(v.item()))
        for v, i in zip(topv, topi)
    ]

    x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=DEVICE)
    logits = model(x)[0]
    targets = torch.tensor(full_ids[1:], dtype=torch.long, device=DEVICE)
    losses = F.cross_entropy(logits, targets, reduction="none")

    answer_start = len(prefix_ids) - 1
    full_nll = float(losses[answer_start:].mean().item())

    current = prefix.clone()
    generated = []
    for _ in range(12):
        nxt_logits = model(current[:, -CTX:])[:, -1, :]
        nxt = torch.argmax(nxt_logits, dim=-1, keepdim=True)
        current = torch.cat([current, nxt], dim=1)
        generated.append(int(nxt.item()))

    return {
        "first_token": tok.decode([first_id]),
        "first_nll": first_nll,
        "first_rank": first_rank,
        "full_nll": full_nll,
        "top": top_tokens,
        "generated": tok.decode(generated).strip(),
    }


def main():
    tok = Tokenizer.from_file(str(TOK_PATH))
    print("=" * 112)
    print("Node 70B — clean first-token / teacher-forcing diagnostic")
    print("=" * 112)
    print("device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Loading Step 66...")
    step66 = load_model(STEP66)
    print("Loading Step 69...")
    step69 = load_model(STEP69)

    for name, model in [
        ("Step 66 baseline", step66),
        ("Step 69 best SFT", step69),
    ]:
        results = []
        print("\n" + "-" * 112)
        print(name)

        for prompt, answer in PROBES:
            r = analyze(model, tok, prompt, answer)
            results.append(r)

        print(
            f"Mean first-token NLL: {sum(r['first_nll'] for r in results)/len(results):.4f}"
        )
        print(
            f"Mean first-token rank: {sum(r['first_rank'] for r in results)/len(results):.1f}"
        )
        print(
            f"First-token top1: "
            f"{sum(r['first_rank'] == 1 for r in results) / len(results) * 100:.1f}%"
        )
        print(
            f"Mean full-answer NLL: {sum(r['full_nll'] for r in results)/len(results):.4f}"
        )

        for (prompt, answer), r in zip(PROBES, results):
            print(f"\nUser: {prompt}")
            print(f"Expected first token: {r['first_token']!r}")
            print(f"First NLL: {r['first_nll']:.4f} | rank: {r['first_rank']}")
            print("Top candidates:")
            for text, lp in r["top"]:
                print(f"  {text!r} | logP={lp:.4f}")
            print(f"Full-answer NLL: {r['full_nll']:.4f}")
            print(f"Free-run 12: {r['generated']!r}")

    print("\nNode 70B complete.")


if __name__ == "__main__":
    main()
