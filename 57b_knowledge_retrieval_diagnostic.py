import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

VOCAB = 8000
D = 256
H = 8
LAYERS = 8
FF = 1024
CTX = 256


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


@torch.inference_mode()
def answer_nll(model, tokenizer, prompt, answer, device):
    prefix = f"User: {prompt}\nAssistant:"
    full = prefix + " " + answer
    prefix_ids = tokenizer.encode(prefix).ids
    full_ids = tokenizer.encode(full).ids

    if len(full_ids) > CTX:
        return float("inf"), 0

    x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    logits = model(x)[0]
    logp = torch.log_softmax(logits, dim=-1)
    targets = full_ids[1:]

    start = max(0, len(prefix_ids) - 1)
    vals = []
    for pos in range(start, len(targets)):
        vals.append(logp[pos, targets[pos]].item())

    if not vals:
        return float("inf"), 0
    return -sum(vals) / len(vals), len(vals)


@torch.inference_mode()
def greedy(model, tokenizer, prompt, device, max_new=48):
    ids = tokenizer.encode(f"User: {prompt}\nAssistant:").ids[-CTX:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    start = x.size(1)

    for _ in range(max_new):
        logits = model(x[:, -CTX:])[:, -1, :]
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        x = torch.cat([x, nxt], dim=1)
        text = tokenizer.decode(x[0].tolist()[start:]).strip()
        if "[END]" in text:
            return text.split("[END]", 1)[0].strip()
    return tokenizer.decode(x[0].tolist()[start:]).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="step54")
    args, _ = parser.parse_known_args()

    drive = Path("/content/drive/MyDrive/transformers_exercise_20260911")
    tokenizer_path = drive / "artifacts" / "step43" / "step43_bpe_8000.json"

    if args.checkpoint.endswith(".pt"):
        ckpt = Path(args.checkpoint)
    else:
        ckpt = drive / "artifacts" / args.checkpoint / f"tiny_gpt_{args.checkpoint}_best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    model = TinyGPT().to(device)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    print("=" * 112)
    print("Node 57B — Knowledge retrieval diagnostic")
    print("=" * 112)
    print("checkpoint:", ckpt)
    print("device:", device)

    tests = [
        {
            "name": "Li Bai — direct Chinese",
            "prompt": "李白是谁？",
            "candidates": [
                "李白是中国唐代诗人。",
                "李白是一位中国古代诗人。",
                "谢谢。",
                "我不知道。",
            ],
        },
        {
            "name": "Li Bai — alternate Chinese",
            "prompt": "请介绍一下李白。",
            "candidates": [
                "李白是唐代著名诗人。",
                "李白是中国古代诗人。",
                "谢谢。",
                "我不知道。",
            ],
        },
        {
            "name": "Li Bai — English concept",
            "prompt": "Who was Li Bai?",
            "candidates": [
                "Li Bai was a famous Chinese poet of the Tang dynasty.",
                "Li Bai was a Chinese poet.",
                "Thank you.",
                "I don't know.",
            ],
        },
        {
            "name": "Known concept control",
            "prompt": "什么是人工智能？",
            "candidates": [
                "人工智能是让计算机执行通常需要人类智能任务的一种技术。",
                "谢谢。",
                "我不知道。",
            ],
        },
        {
            "name": "Known person control",
            "prompt": "乔布斯是谁？",
            "candidates": [
                "乔布斯是苹果公司的联合创始人之一。",
                "谢谢。",
                "我不知道。",
            ],
        },
    ]

    for test in tests:
        print("\n" + "-" * 112)
        print(test["name"])
        print("User:", test["prompt"])
        out = greedy(model, tokenizer, test["prompt"], device)
        print("Greedy:", out)

        scored = []
        for ans in test["candidates"]:
            nll, tokens = answer_nll(model, tokenizer, test["prompt"], ans, device)
            scored.append((nll, ans, tokens))

        print("Candidate likelihood (lower NLL = stronger internal preference):")
        for nll, ans, tokens in sorted(scored):
            print(f"  NLL={nll:.4f} | tokens={tokens:>2} | {ans}")

    print("\nDiagnostic interpretation guide:")
    print("1. Correct Li Bai answer ranks above '谢谢'/'我不知道' -> knowledge signal exists.")
    print("2. Correct Li Bai answer ranks below generic distractors -> weak/no reliable stored knowledge.")
    print("3. Greedy is wrong but correct answer ranks highly -> retrieval/decoding problem.")
    print("4. Both direct and alternate prompts prefer the correct answer -> stronger evidence of generalized knowledge.")
    print("5. Known controls are correct while Li Bai is not -> likely knowledge coverage/data limitation.")
    print("\nStep 57B diagnostic complete.")


if __name__ == "__main__":
    main()
