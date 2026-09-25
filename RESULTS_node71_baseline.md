# Step 66 / Step 69 diagnostic results

Colab `t4-gpu` session. Records the corrected Node 71 probe set, the Step 69
freeze-policy fix, and the fact-preference probe that followed.

## Checkpoints

| name | path | notes |
|---|---|---|
| Step 66 base | `artifacts/step66/tiny_gpt_v2_best.pt` | 42,001,408 params, pretrained only |
| Step 69 frozen-head | `artifacts/step69/tiny_gpt_v2_42m_sft_frozenhead_baseline.pt` | original run, 18,887,680 / 42,001,408 trainable (45.0%) |
| Step 69 head-trained | `artifacts/step69/tiny_gpt_v2_42m_sft_best.pt` | rerun after the freeze fix, 22,983,680 trainable (54.7%) |

---

## 1. Corrected probe set

The original Node 71 used 9 hardcoded probes, **7 of which are literal prompts
from `69_precision_data.py`**, which Node 69 trains on. Node 71 now generates
`PROBES_HELDOUT` (36 probes = the 9 QA groups Node 69 never trains on, 2 EN +
2 ZH prompts each) and `PROBES_SEEN` (24 probes from its training split), both
from the Node 69 split (`qa_groups()` shuffled with `SEED=69`, 80/20).
The gap between the two subsets is the memorization measure.

This mattered. On the old probe set the story was "rank 75.8 -> 10.9, a big SFT
win". On the corrected set:

| subset | n | Step 66 rank | Step 69 (frozen) rank |
|---|---|---|---|
| HELD-OUT | 36 | 214.8 | 196.9 |
| SEEN | 24 | 76.0 | 12.5 |

The gain is almost entirely memorization. The old 2-probe "clean" estimate
(60.0 -> 16.5) was optimistic purely because n=2.

---

## 2. The frozen LM head was NOT the bottleneck (hypothesis refuted)

`69_42m_controlled_sft.py` froze `model.tok.parameters()`. Because
`head.weight is tok.weight` (weight tying), that also froze the LM output
projection: only 45.0% of parameters were trainable. The fix leaves the tied
embedding/head pair trained and keeps `pos` + `blocks[0:6]` frozen.

Three-way comparison after retraining, same 60 probes:

| subset | metric | Step 66 base | Step 69 frozen-head | Step 69 head-trained |
|---|---|---|---|---|
| HELD-OUT | semantic rank | 214.8 | 196.9 | **199.9** |
| HELD-OUT | semantic top1 | 13.9% | 19.4% | **19.4%** |
| HELD-OUT | first-token correct | 44.4% | 47.2% | **47.2%** |
| HELD-OUT | mean divergence | 0.58 | 0.81 | **0.81** |
| SEEN | semantic rank | 76.0 | 12.5 | 9.1 |
| SEEN | semantic top1 | 12.5% | 37.5% | 41.7% |

Held-out metrics are **unchanged** (rank 196.9 -> 199.9 is a slight worsening).
Unfreezing the head bought a marginal SEEN improvement and nothing else. The
hypothesis that the frozen output projection was blocking generalization is
wrong.

Held-out answer NLL did improve slightly: 2.3571 -> 2.3147.

---

## 3. The real bottleneck: the base model does not store these facts

Preference probe — for each QA group, does the model give the group's correct
answer a lower mean NLL than the group's own wrong answer? 50% = no factual
preference.

| model | SEEN | HELD-OUT |
|---|---|---|
| Step 66 base | 41.7% (55/132) | 36.1% (13/36) |
| Step 69 frozen-head | 49.2% (65/132) | **16.7%** (6/36) |
| Step 69 head-trained | 58.3% (77/132) | **19.4%** (7/36) |

Both SFT variants push held-out preference **below chance**. SFT is not merely
failing to help; it is actively harmful on unseen facts.

Format control — the probe above uses the `User/Assistant` scaffolding, which
the base model never saw. Re-run on Step 66 with raw document continuation:

| format | SEEN | HELD-OUT |
|---|---|---|
| RAW continuation | 52.3% (69/132) | 55.6% (20/36) |
| CHAT User/Assistant | 41.7% (55/132) | 36.1% (13/36) |

Under the format most favourable to it, the base model is still at chance
(55.6% on n=36 is well inside binomial noise around 50%). So the absence of
factual knowledge is real, not a chat-format artifact. The chat scaffolding
makes it worse, i.e. slightly anti-correlated.

---

## 4. Conclusions

1. The project's SFT/alignment arc (Nodes 47-69) has been optimizing the wrong
   bottleneck. A 42M model with 51.7M pretraining tokens has no reliable factual
   recall for these entities, and 283 SFT rows cannot create facts that were
   never learned.

2. What SFT does do: learn the answer *format* (held-out format-first NLL
   3.7731 -> 3.1164) and memorize the 33 training groups (SEEN rank 76.0 -> 9.1).

3. Generation is dead in every checkpoint. On held-out probes the first
   generated token is correct ~44-47% of the time and mean first-divergence
   index is 0.58-0.81.

4. Suggested redirect: make raw knowledge the primary metric (Node 68 style,
   no chat format, held-out entities), and return to pretraining — corpus
   composition/volume, capacity, and steps — rather than further SFT schemes.

## 5. Operational note

`colab exec` intermittently drops its websocket on longer runs and reports
`RuntimeError: Connection was lost`. The kernel survives; redirecting stdout to
a file (`colab exec ... > out.txt 2>&1`) worked reliably where piping to `tail`
did not.

## 6. Known defects

1. ~~`69_42m_controlled_sft.py` freezes the LM head via weight tying.~~ Fixed;
   trainable 45.0% -> 54.7%, with an assert guarding the tying assumption and a
   per-component freeze map printed at startup.
2. ~~`69_42m_controlled_sft.py` fetches `69_precision_data.py` from a
   `main/...?v=<sha>` URL.~~ Fixed; now prefers a local copy and otherwise uses
   the `/<sha>/` path form with the correct sha `d5fd2ec...5358214`.
3. ~~Node 71/70B probe lists were 7/9 training prompts.~~ Node 71 fixed.
   Node 70B still carries the old list.
