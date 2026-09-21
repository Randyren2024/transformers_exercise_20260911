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


def main():
    tok = Tokenizer.from_file(str(TOK_PATH))

    print("=" * 112)
    print("Node 71 — semantic-token / exposure-bias diagnostic")
    print("=" * 112)
    print("device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Loading Step 66...")
    step66 = load_model(STEP66)
    print("Loading Step 69...")
    step69 = load_model(STEP69)

    for name, model in [("Step 66 baseline", step66), ("Step 69 best SFT", step69)]:
        results = []
        print("\n" + "-" * 112)
        print(name)

        for prompt, answer in PROBES:
            results.append(diagnose(model, tok, prompt, answer))

        sem = [r for r in results if r["semantic_rank"] is not None]
        print(f"Mean format-first NLL: {sum(r['format_nll'] for r in results)/len(results):.4f}")
        print(f"Format-first top1: {sum(r['format_rank'] == 1 for r in results)/len(results)*100:.1f}%")
        print(f"Mean semantic-first NLL: {sum(r['semantic_nll'] for r in sem)/len(sem):.4f}")
        print(f"Mean semantic-first rank: {sum(r['semantic_rank'] for r in sem)/len(sem):.1f}")
        print(f"Semantic-first top1: {sum(r['semantic_rank'] == 1 for r in sem)/len(sem)*100:.1f}%")
        valid_div = [r["divergence_token_index"] for r in results if r["divergence_token_index"] is not None]
        print(f"Mean first free-run divergence token: {sum(valid_div)/len(valid_div):.1f}")

        for (prompt, answer), r in zip(PROBES, results):
            print(f"\nUser: {prompt}")
            print(f"Gold: {answer}")
            print(f"Format first: {r['format_first']!r} | rank {r['format_rank']} | NLL {r['format_nll']:.4f}")
            print(f"Semantic first: {r['semantic_token']!r} | rank {r['semantic_rank']} | NLL {r['semantic_nll']:.4f}")
            print(f"First free-run divergence token: {r['divergence_token_index']}")
            print(f"Free-run: {r['generated_text']!r}")

    print("\nNode 71 complete.")


if __name__ == "__main__":
    main()
