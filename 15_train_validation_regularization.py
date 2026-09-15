import math
import re
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# 15 - Train / Validation / Regularization experiment
# =========================================================
# Goal:
# Keep the Transformer architecture from step 14, but make the
# training process more realistic.
#
# New ideas in this file:
#   1. Larger corpus
#   2. Train / validation split
#   3. Weight decay
#   4. Gradient clipping
#   5. Early stopping
#   6. Keep the best validation checkpoint
#
# The model itself is still deliberately small enough for CPU.

# ---------------------------------------------------------
# 1. Reproducibility and CPU
# ---------------------------------------------------------
torch.manual_seed(42)
device = torch.device("cpu")
print("Device:", device)

# ---------------------------------------------------------
# 2. Larger original corpus
# ---------------------------------------------------------
# This text is written specifically for the exercise.
# It extends the story from step 14 with additional material so
# the model has more unique token sequences to learn.
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

The workshop became busier as more people heard about Mara's experiments. A farmer
brought a damaged pump. A teacher brought a loose wooden cabinet. A young mechanic
asked for advice about a wheel that vibrated at high speed. Mara did not pretend to
know every answer. Instead, she helped each person describe the problem carefully.

The farmer discovered that the pump failed only after several minutes of use. The
teacher learned that the cabinet was leaning because one side was slightly shorter.
The mechanic measured the wheel and found that a small imbalance became important
when the speed increased. In each case, the first useful step was observation.

Mara began a weekly workshop for anyone in town who wanted to experiment. People
brought broken clocks, small motors, pieces of wood, metal scraps, paper models,
and unfinished drawings. The room was sometimes noisy, but the work became easier
when everyone wrote down what they changed and what happened afterward.

One evening a student arrived with a small wooden bridge that looked stronger than
the others. She had changed the shape of the supports and reduced the amount of
paper near the center. Mara asked her to explain why the design worked. The student
could not answer at first. She only knew that the bridge held more weight.

So they tested it again. Then they changed one support and tested a second version.
The result was weaker. They restored the first design and tried a third variation.
After several trials the student began to see the relationship between shape and
strength. The explanation came after the experiment, not before it.

Mara liked this kind of learning. A diagram in a book could provide a useful idea,
but an experiment forced a person to compare the idea with reality. When a result
was unexpected, the surprise was often more valuable than a correct prediction.

The town library eventually created a small shelf for workshop notes. Every notebook
had a date on the cover. Some contained successful projects. Others described ideas
that had failed completely. People were encouraged to read both kinds because a
failed experiment could still contain a useful measurement.

In summer, the workshop windows stayed open until evening. The sound of tools mixed
with bicycle bells from the square. Mara sometimes walked to the river after work.
She watched the current and thought about all the things that could be improved by
small changes repeated over time.

She knew that large projects often looked impressive at the end, but the final result
was built from many ordinary decisions. Measure the object. Test the part. Record the
result. Change one detail. Try again. The pattern was simple enough to remember and
strong enough to guide complicated work.

A year after the first boat repair, Mara built a second water wheel. It was smaller
than the first one but more reliable. She used fewer parts and a stronger support.
The wheel turned smoothly even when the river changed speed.

Tomas inspected it and nodded. He said that the new wheel was not better because it
was more complicated. It was better because Mara understood which parts mattered.
That sentence stayed in her notebook beside the measurements.

The workshop continued through another winter. New questions arrived every month.
Some could be answered quickly. Others required patient experiments over many weeks.
Mara became less interested in being certain at the beginning and more interested
in finding a useful next test.

She often told the students that a model was valuable when it helped them predict
something they could then check. A drawing, a number, or a small computer program
could all be models. None was reality itself. Each was a tool for thinking.

One student began building a tiny boat with a paper sail. Another designed a balance
made from string and wood. A third wrote simple observations in a table and compared
results over time. Their projects looked unrelated, but the method was the same.

By the end of the season the workshop walls were covered with notes. Questions sat
beside measurements. Failed attempts sat beside improved designs. The room looked
messy, but Mara thought the mess showed that people were learning.
"""

# ---------------------------------------------------------
# 3. Word-level tokenizer
# ---------------------------------------------------------
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

print("Train tokens:", len(train_data))
print("Validation tokens:", len(val_data))

# ---------------------------------------------------------
# 5. Small CPU-friendly model configuration
# ---------------------------------------------------------
block_size = 32
batch_size = 16
embedding_dim = 64
num_heads = 4
num_layers = 2
ffn_dim = 4 * embedding_dim
learning_rate = 3e-4
weight_decay = 0.01
max_steps = 2000
eval_interval = 100
eval_batches = 20
early_stopping_patience = 5

torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
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
print("  max training steps:", max_steps)
print("  weight decay:", weight_decay)
print("  early stopping patience:", early_stopping_patience)

# ---------------------------------------------------------
# 6. Batch creation
# ---------------------------------------------------------
def get_batch(source):
    starts = torch.randint(0, len(source) - block_size - 1, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in starts])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)

# ---------------------------------------------------------
# 7. Multi-Head Self-Attention
# ---------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size_, seq_len, d_model = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(batch_size_, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size_, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size_, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = weights @ v

        context = context.transpose(1, 2).contiguous().view(batch_size_, seq_len, d_model)
        return self.output_projection(context), weights

# ---------------------------------------------------------
# 8. Transformer Block
# ---------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_output, attention_weights = self.attention(self.norm1(x))
        x = x + self.dropout(attention_output)
        x = x + self.ffn(self.norm2(x))
        return x, attention_weights

# ---------------------------------------------------------
# 9. Full model
# ---------------------------------------------------------
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, d_model, num_heads, num_layers, ffn_dim):
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
        _, seq_len = token_ids.shape
        if seq_len > self.block_size:
            raise ValueError("Sequence is longer than block_size.")

        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]

        all_attention = []
        for block in self.blocks:
            x, attention = block(x)
            all_attention.append(attention)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        if return_attention:
            return logits, loss, all_attention
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens=40, temperature=0.8):
        self.eval()
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.block_size:]
            logits, _ = self(context)
            next_logits = logits[:, -1, :]
            if temperature <= 0:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                probabilities = torch.softmax(next_logits / temperature, dim=-1)
                next_id = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat([token_ids, next_id], dim=1)
        return token_ids

# ---------------------------------------------------------
# 10. Evaluation helper
# ---------------------------------------------------------
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    results = {}
    for name, source in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(eval_batches):
            x, y = get_batch(source)
            _, loss = model(x, y)
            losses.append(loss.item())
        results[name] = sum(losses) / len(losses)
    model.train()
    return results

# ---------------------------------------------------------
# 11. Build model and optimizer
# ---------------------------------------------------------
model = TinyGPT(
    vocab_size=vocab_size,
    block_size=block_size,
    d_model=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    ffn_dim=ffn_dim,
).to(device)

parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\nTrainable parameters:", parameter_count)

# AdamW applies decoupled weight decay.
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)

# ---------------------------------------------------------
# 12. Training with validation + early stopping
# ---------------------------------------------------------
best_val_loss = float("inf")
best_step = 0
best_state = copy.deepcopy(model.state_dict())
patience_counter = 0

print("\nTraining...")
for step in range(1, max_steps + 1):
    x, y = get_batch(train_data)
    logits, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Prevent a rare large gradient from destabilizing training.
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if step == 1 or step % eval_interval == 0:
        losses = estimate_loss(model)
        print(
            f"Step {step:4d} | "
            f"Train loss: {losses['train']:.4f} | "
            f"Val loss: {losses['val']:.4f}"
        )

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            best_step = step
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print("  -> New best validation loss")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{early_stopping_patience})")

        if patience_counter >= early_stopping_patience:
            print("  -> Early stopping")
            break

# Restore the best validation checkpoint.
model.load_state_dict(best_state)

print("\nBest checkpoint:")
print("  step:", best_step)
print("  best validation loss:", f"{best_val_loss:.4f}")

final_losses = estimate_loss(model)
print("  final train loss:", f"{final_losses['train']:.4f}")
print("  final val loss:", f"{final_losses['val']:.4f}")

# ---------------------------------------------------------
# 13. Generation
# ---------------------------------------------------------
def decode(ids):
    words = [itos[i] for i in ids]
    text = ""
    for word in words:
        if word in {".", ",", "!", "?", ";", ":"}:
            text = text.rstrip() + word + " "
        else:
            text += word + " "
    return text.strip()


def encode_prompt(text):
    prompt_tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,!?;:]", text.lower())
    unknown = [t for t in prompt_tokens if t not in stoi]
    if unknown:
        raise ValueError(f"Prompt contains unknown tokens: {unknown}")
    return torch.tensor([[stoi[t] for t in prompt_tokens]], dtype=torch.long, device=device)

print("\n--- Generation with best checkpoint ---")
for prompt, temperature in (("the morning", 0.0), ("mara", 0.8), ("the workshop", 0.8)):
    prompt_ids = encode_prompt(prompt)
    generated = model.generate(prompt_ids, max_new_tokens=35, temperature=temperature)
    print(f"Prompt: {prompt}")
    print("Generated:")
    print(decode(generated[0].tolist()))
    print()

# ---------------------------------------------------------
# 14. Inspect attention on a longer prompt
# ---------------------------------------------------------
print("--- Attention inspection ---")
inspect_prompt = "mara worked carefully"
inspect_ids = encode_prompt(inspect_prompt)

with torch.no_grad():
    _, _, attention_maps = model(inspect_ids, return_attention=True)

print("Prompt tokens:", re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,!?;:]", inspect_prompt.lower()))
print("Number of Transformer blocks:", len(attention_maps))
for layer_index, attention in enumerate(attention_maps, start=1):
    print(f"Block {layer_index} attention shape:", attention.shape)

print("\nFirst block, first head attention weights:")
print(attention_maps[0][0, 0])

print("\nInterpretation:")
print("Step 14 showed that the model could overfit the tiny corpus.")
print("Step 15 keeps the same small Transformer architecture but makes")
print("the training experiment more realistic:")
print("  - a larger corpus")
print("  - train/validation split")
print("  - AdamW weight decay")
print("  - gradient clipping")
print("  - early stopping")
print("  - best validation checkpoint")
print()
print("The main quantity to watch is the gap between train loss and")
print("validation loss. A small gap means better generalization; a")
print("growing gap is a sign of overfitting.")
