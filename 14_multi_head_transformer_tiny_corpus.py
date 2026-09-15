import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# 14 - Multi-Head Transformer trained on a small text corpus
# =========================================================
# Goal:
# Build a small GPT-style decoder-only Transformer that can be
# trained on a few thousand WORD TOKENS using CPU only.
#
# Architecture:
#   token IDs
#       -> token embedding + positional embedding
#       -> Transformer Block x 2
#          -> Multi-Head Causal Self-Attention
#          -> Residual + LayerNorm
#          -> Feed-Forward Network
#          -> Residual + LayerNorm
#       -> LM Head
#       -> next-token logits
#
# This file intentionally does NOT use nn.Transformer or
# nn.MultiheadAttention. The important pieces are implemented
# explicitly so that the structure remains visible.

# ---------------------------------------------------------
# 1. Reproducibility and CPU
# ---------------------------------------------------------
torch.manual_seed(42)

device = torch.device("cpu")
print("Device:", device)

# ---------------------------------------------------------
# 2. Small original training corpus
# ---------------------------------------------------------
# This is original text written for this exercise.
# It is deliberately much larger than "I love cats", but still
# small enough for CPU training.
corpus = """
The morning began quietly in a small town near the river. The streets were cool,
and a thin mist rested above the water. A baker opened the first shop before sunrise,
and the warm smell of bread traveled through the empty square. A few birds moved
between the roofs, while a bicycle rolled slowly past the old library.

Mara lived in a small house on the edge of the town. She liked to wake early,
make tea, and read beside the window. From that window she could see the river,
the bridge, and a long row of trees. She often said that a quiet morning made it
easier to notice small things that were invisible later in the day.

One summer, Mara decided to repair an old wooden boat that had belonged to her
grandfather. The boat had been stored behind a shed for many years. Its paint was
faded, one board was loose, and the rope had become hard from the weather.
Still, the wooden frame was strong, and Mara believed that careful work could give
it another life.

She began by carrying the boat into the yard. First she cleaned away the dust.
Then she removed the broken boards and measured the spaces beneath them. She kept
a notebook beside her and wrote down every measurement. When she was unsure about
something, she stopped and looked again instead of guessing.

Her neighbor Tomas came over in the afternoon. Tomas knew a great deal about wood,
old tools, and simple machines. He examined the boat and smiled. He told Mara that
a repair did not need to be perfect on the first try. The important thing was to
understand the problem, change one thing at a time, and test the result.

Together they replaced the damaged boards. They sanded the surface, cleaned the
metal parts, and covered the wood with a new coat of paint. The work was slow.
Sometimes a screw would not turn. Sometimes a board had to be removed and fitted
again. Each small mistake became information for the next attempt.

After several days, the boat looked different. It was still an old boat, but it
was clean, solid, and ready for the river. Mara carried it down to the water with
Tomas. They pushed it into the river and watched carefully as the hull touched the
surface. For a moment the boat moved without direction. Then the current carried
it gently toward the bridge.

Mara climbed inside and tested the oars. The first movement was awkward. The boat
turned too far to one side, so she adjusted her hands and tried again. After a few
minutes she found a steady rhythm. The boat moved forward in a straight line.

The river became wider beyond the bridge. On both sides there were fields, small
farms, and paths used by walkers. A fisherman lifted his hand when he saw Mara.
She waved back. Farther ahead, a child was standing near the bank and watching the
boat with curiosity.

Mara spent the afternoon exploring the river. She noticed places she had never
seen from the road. A narrow path led to a quiet meadow. A group of ducks rested
near the reeds. An old stone wall followed the bank for several hundred meters.
The world looked larger from the water, even though the town was still nearby.

When evening arrived, Mara turned the boat toward home. The wind had become cooler,
and the light on the river changed from bright silver to deep gold. She rowed more
slowly and listened to the sound of the oars touching the water.

Back in town, Tomas was waiting near the shed. He asked whether the boat had worked.
Mara laughed and said that it had worked well enough to make her want to repair
something else. Tomas replied that this was how useful projects often began.
One small experiment led to another, and each result created a better question.

The following week, Mara started keeping a larger notebook. She wrote about the
boat, the river, the tools, and the mistakes they had made. She also wrote about
ideas for new projects. Some ideas were practical. Others were strange and would
probably never become real objects. She kept them anyway because an idea did not
need to be useful immediately.

One page described a simple water wheel. Another page described a machine that
could sort stones by size. A third page contained a drawing of a small cart that
could carry books across the square. Mara did not know which idea would work.
She only knew that writing an idea down made it easier to examine.

Tomas encouraged her to test simple versions before building complicated ones.
He said that a good workshop was a place where questions could become experiments.
A person did not need perfect tools to learn. A ruler, a pencil, some wood, and
enough patience were often sufficient.

So Mara built a small water wheel from spare pieces. Her first version was too
heavy and turned very slowly. She removed part of the frame and changed the shape
of the paddles. The second version moved faster, but the wheel shook whenever the
water became stronger.

Mara added a support near the center. This reduced the movement and made the wheel
more stable. She tested it again the next morning. The wheel turned smoothly, and
she could see a small wooden shaft moving with it.

The experiment taught her something important. The first design had not failed
because the idea was useless. It had failed because the structure was not yet
balanced. A weak result could be a useful step toward a better design.

As autumn arrived, the town changed. The trees near the river became yellow and
red. The mornings became colder, and the bakery began selling hot soup at lunch.
People wore heavier coats and stayed longer inside the library.

Mara continued to work in the yard, but she also spent more time reading. She read
books about mechanics, maps of rivers, stories about travelers, and notes written
by engineers many years ago. She noticed that many successful inventions began
with ordinary observations.

A person saw water moving. Another person noticed that a wheel could turn.
Someone else wondered whether the movement could be used for work. A simple question
could eventually become a machine, but only after many experiments and many small
corrections.

During the winter, Mara taught a group of children how to build small paper bridges.
Each child received the same amount of paper. They could fold it in any way they
wanted, but the bridge had to support a small wooden block.

At first the children made flat bridges. Most collapsed under very little weight.
Then one child folded the paper into a series of small shapes. The new bridge was
much stronger. Soon the other children began testing different designs.

Mara watched them compare results. She noticed that the children were learning not
only how to build a bridge, but how to ask better questions. They began asking why
one shape was stronger than another. They measured the width, changed one feature,
and tested again.

The lesson reminded Mara of the old boat. Good work often followed the same pattern:
observe, build, test, measure, change, and test again. A person could become better
by repeating this cycle without rushing to the final answer.

When spring returned, the river rose higher. Water moved quickly beneath the bridge,
and fresh green leaves appeared along the bank. Mara repaired a second boat and
helped Tomas clean the small workshop.

The town had changed too. A new market opened near the station. The library added
more tables. The baker installed a larger oven. People talked about new businesses,
new roads, and plans for the coming summer.

Mara did not know what the next year would bring. She had many ideas but no clear
plan for all of them. Instead of trying to predict everything, she decided to keep
working on the next useful experiment.

She wrote one sentence on the first page of a new notebook: Start with what you can
measure. Change one thing at a time. Learn from the result. Then choose the next
experiment.

The sentence became a simple rule for her work. It was useful for boats, machines,
books, gardens, and almost every difficult problem she encountered.

Years later, people in the town still remembered the old wooden boat. Some remembered
its blue paint. Others remembered the afternoon when Mara crossed the river for the
first time. Mara remembered something simpler. She remembered that the boat had
started as a broken object in a dark shed, and that progress had begun when she
stopped asking whether the repair would be perfect and started asking what she could
test next.
"""

# ---------------------------------------------------------
# 3. Word-level tokenizer
# ---------------------------------------------------------
# Keep words and punctuation as separate tokens.
tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,!?;:]", corpus.lower())

vocab = sorted(set(tokens))
vocab_size = len(vocab)

stoi = {token: i for i, token in enumerate(vocab)}
itos = {i: token for token, i in stoi.items()}

data = torch.tensor([stoi[token] for token in tokens], dtype=torch.long)

print("Corpus characters:", len(corpus))
print("Corpus tokens:", len(tokens))
print("Vocabulary size:", vocab_size)
print("First 30 tokens:")
print(tokens[:30])

# ---------------------------------------------------------
# 4. Train / validation split
# ---------------------------------------------------------
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

# ---------------------------------------------------------
# 5. Hyperparameters chosen for CPU
# ---------------------------------------------------------
block_size = 32          # context length
batch_size = 16          # sequences per training batch
embedding_dim = 64       # d_model
num_heads = 4
num_layers = 2
ffn_dim = 4 * embedding_dim
learning_rate = 3e-4
train_steps = 1200
print_every = 100

assert embedding_dim % num_heads == 0
head_dim = embedding_dim // num_heads

print("\nModel configuration:")
print("  block_size:", block_size)
print("  batch_size:", batch_size)
print("  embedding_dim:", embedding_dim)
print("  num_heads:", num_heads)
print("  head_dim:", head_dim)
print("  num_layers:", num_layers)
print("  ffn_dim:", ffn_dim)
print("  training steps:", train_steps)

# ---------------------------------------------------------
# 6. Batch creation
# ---------------------------------------------------------
def get_batch(source):
    # Pick random starting positions.
    starts = torch.randint(
        0,
        len(source) - block_size - 1,
        (batch_size,),
    )

    x = torch.stack([source[i:i + block_size] for i in starts])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)

# ---------------------------------------------------------
# 7. Multi-Head Self-Attention
# ---------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # One projection creates Q, K and V together.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size_, seq_len, d_model = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Split d_model into multiple heads.
        q = q.view(batch_size_, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size_, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size_, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: (B, H, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask: a token can only see itself and earlier tokens.
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)

        # Weighted values: (B, H, T, head_dim)
        context = weights @ v

        # Merge heads back to d_model.
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size_, seq_len, d_model)

        return self.output_projection(context), weights

# ---------------------------------------------------------
# 8. Transformer Block
# ---------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim):
        super().__init__()

        self.attention = MultiHeadSelfAttention(d_model, num_heads)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attention_output, attention_weights = self.attention(x)

        # Residual connection + LayerNorm
        x = self.norm1(x + attention_output)

        # Feed-forward network + Residual + LayerNorm
        x = self.norm2(x + self.ffn(x))

        return x, attention_weights

# ---------------------------------------------------------
# 9. Full small GPT-style model
# ---------------------------------------------------------
class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        d_model,
        num_heads,
        num_layers,
        ffn_dim,
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, targets=None, return_attention=False):
        batch_size_, seq_len = token_ids.shape

        if seq_len > self.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds block_size {self.block_size}."
            )

        positions = torch.arange(seq_len, device=token_ids.device)

        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]

        all_attention_weights = []

        for block in self.blocks:
            x, attention_weights = block(x)
            all_attention_weights.append(attention_weights)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        if return_attention:
            return logits, loss, all_attention_weights

        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens=30, temperature=0.8):
        self.eval()

        for _ in range(max_new_tokens):
            # Keep only the most recent block_size tokens.
            context = token_ids[:, -self.block_size:]

            logits, _ = self(context)
            last_logits = logits[:, -1, :]

            if temperature <= 0:
                next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
            else:
                scaled_logits = last_logits / temperature
                probabilities = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)

            token_ids = torch.cat([token_ids, next_token], dim=1)

        return token_ids

# ---------------------------------------------------------
# 10. Create model
# ---------------------------------------------------------
model = TinyGPT(
    vocab_size=vocab_size,
    block_size=block_size,
    d_model=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    ffn_dim=ffn_dim,
).to(device)

parameter_count = sum(p.numel() for p in model.parameters())

print("\nTotal trainable parameters:", parameter_count)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ---------------------------------------------------------
# 11. Quick shape check before training
# ---------------------------------------------------------
x_demo, y_demo = get_batch(train_data)
logits_demo, loss_demo, attention_demo = model(
    x_demo,
    y_demo,
    return_attention=True,
)

print("\nShape check:")
print("  Input:", x_demo.shape)
print("  Logits:", logits_demo.shape)
print("  Loss:", loss_demo.item())
print("  Attention from block 1:", attention_demo[0].shape)
print("  Expected attention shape: (batch, heads, seq_len, seq_len)")

# ---------------------------------------------------------
# 12. Training
# ---------------------------------------------------------
print("\nTraining...")
model.train()

for step in range(1, train_steps + 1):
    x, y = get_batch(train_data)

    logits, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step == 1 or step % print_every == 0:
        with torch.no_grad():
            val_x, val_y = get_batch(val_data)
            _, val_loss = model(val_x, val_y)

        print(
            f"Step {step:4d} | "
            f"Train loss: {loss.item():.4f} | "
            f"Val loss: {val_loss.item():.4f}"
        )

# ---------------------------------------------------------
# 13. Greedy generation
# ---------------------------------------------------------
def encode_prompt(text):
    prompt_tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,!?;:]", text.lower())
    unknown = [token for token in prompt_tokens if token not in stoi]
    if unknown:
        raise ValueError(f"Unknown token(s) in prompt: {unknown}")
    return torch.tensor([[stoi[token] for token in prompt_tokens]], dtype=torch.long)


def decode(ids):
    words = [itos[i] for i in ids]
    text = ""

    for word in words:
        if word in {".", ",", "!", "?", ";", ":"}:
            text += word
        else:
            if text:
                text += " "
            text += word

    return text

print("\n--- Generation: greedy decoding ---")
model.eval()

prompt = "the morning"
prompt_ids = encode_prompt(prompt).to(device)
greedy_ids = prompt_ids.clone()

greedy_ids = model.generate(
    greedy_ids,
    max_new_tokens=30,
    temperature=0.0,
)

print("Prompt:", prompt)
print("Generated:")
print(decode(greedy_ids[0].tolist()))

# ---------------------------------------------------------
# 14. Sampling generation
# ---------------------------------------------------------
print("\n--- Generation: temperature sampling ---")

sample_ids = encode_prompt("mara").to(device)
sample_ids = model.generate(
    sample_ids,
    max_new_tokens=40,
    temperature=0.8,
)

print("Prompt: mara")
print("Generated:")
print(decode(sample_ids[0].tolist()))

# ---------------------------------------------------------
# 15. Inspect one attention tensor
# ---------------------------------------------------------
# This shows the actual multi-head attention learned by the model.
inspect_ids = encode_prompt("mara worked").to(device)

with torch.no_grad():
    _, _, attentions = model(inspect_ids, return_attention=True)

print("\nAttention inspection for prompt: 'mara worked'")
print("Number of Transformer blocks:", len(attentions))
print("Block 1 attention shape:", attentions[0].shape)
print("Block 2 attention shape:", attentions[1].shape)
print("Shape meaning: (batch, heads, query_position, key_position)")

print("\nFirst block, first head attention weights:")
print(attentions[0][0, 0])

# ---------------------------------------------------------
# 16. Interpretation
# ---------------------------------------------------------
print("\nInterpretation:")
print("This is a real small decoder-only Transformer trained from")
print("scratch on a text corpus using CPU.")
print()
print("The model contains:")
print("  - token embedding")
print("  - learned positional embedding")
print("  - 2 Transformer blocks")
print("  - 4 attention heads in every block")
print("  - causal masking")
print("  - residual connections")
print("  - LayerNorm")
print("  - feed-forward networks")
print("  - a language-model head")
print()
print("During training, every position predicts the next token.")
print("During generation, only the LAST position is used to choose")
print("the next token, then that token is appended to the context and")
print("the model runs again.")
print()
print("This is no longer a toy mapping such as 'I -> love'.")
print("The model is learning token-to-token patterns from a real")
print("multi-paragraph corpus, while keeping the architecture small")
print("enough for CPU experimentation.")
