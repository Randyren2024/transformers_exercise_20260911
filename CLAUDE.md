# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-author learning progression that builds a GPT from scratch: tokenization →
attention → transformer → BPE → pretraining → SFT → alignment → diagnostics. The
82 top-level `.py` files are numbered **nodes**; the number is the reading order and
the node number appears in the code (`# Node 71 — ...`, `SEED = 66`).

Nothing executes locally. There is no top-level `requirements.txt`, no test suite,
no CI, and no Python environment with `torch` on the WSL side (`deployment/.venv`
is a Windows venv). All training and diagnostics run on Colab, and all checkpoints,
tokenizers, and corpora live on Google Drive — **not in this repo** (`deployment/models/`
is the only model dir and it is gitignored).

## Running things

### Colab workflow

Everything is driven through the `colab` CLI from this machine:

```bash
colab new -s t4-gpu --gpu T4          # T4 | L4 | G4 | H100 | A100
colab drivemount -s t4-gpu            # mounts at /content/drive (default)
colab exec -s t4-gpu -f 71_....py --timeout 900
colab ls -s t4-gpu                    # inspect the remote filesystem
```

Scripts hardcode `DRIVE = Path("/content/drive/MyDrive/transformers_exercise_20260911")`,
so Drive must be mounted before any run.

Three things about `colab exec` that are easy to get wrong:

- **The remote kernel's cwd is not this repo.** `-f` ships the file's *contents*; the
  script cannot read sibling files by relative path. Make scripts self-contained, read
  from Drive, or write the dependency into the remote cwd first.
- **It intermittently drops its websocket** on longer runs (`RuntimeError: Connection
  was lost`). The kernel survives and a rerun usually succeeds. Redirecting to a file
  (`colab exec ... > out.txt 2>&1`) is more reliable than piping into `tail`.
- A crash does **not** roll back side effects. Training nodes write checkpoints to Drive
  as they go — back up a baseline checkpoint before rerunning a node over it.

Locally, `python3` (no torch) is fine for pure-Python work: reproducing the data split,
parsing/validating a script with `ast.parse`, generating probe lists.

### Deployment

`deployment/` is a standalone Flask app, independent of the node numbering. `model.py`
re-implements TinyGPT hardcoded to the 8.4M config (D=256, 8 layers) and `app.py`
loads `models/tiny_gpt_v2_step_8000.pt` — it is an 8.4M deploy, and **cannot load the
42M step66/step69 checkpoints**.

```bash
cd deployment
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python app.py        # dev server, 127.0.0.1:5000
python serve.py      # waitress, production-ish
```

Endpoints: `GET /api/health`, `POST /api/chat` with `{"message": "..."}`. Override the
checkpoint via `TINY_GPT_CHECKPOINT` / `TINY_GPT_TOKENIZER` / `TINY_GPT_DEVICE`.

## Architecture

### Nodes are a narrative, not a library

Each node is deliberately standalone and **38 of the 82 re-define `Attn`/`Block`/`TinyGPT`
inline**. There is no shared model module — adding a node means pasting the model class
and matching its constants exactly to the checkpoint being loaded. This is the root cause
of the recurring `Fix Node NN loader` / `Fix Node NN module namespace` commits; only
`68_step66_raw_knowledge_diagnostic.py` parameterizes the model (`TinyGPT(d, h, layers, ff)`),
which is the pattern to copy when loading more than one architecture.

Consequence: a script's config block is not configuration you can freely edit — it must
match the checkpoint or `load_state_dict` fails. Several nodes load two models of
different sizes side by side (e.g. 68, 71).

The constant *names* also changed mid-project: nodes 40–45 use `D_MODEL` / `N_HEADS` /
`N_LAYERS` / `D_FF`, while 49 onward use `D` / `H` / `LAYERS` / `FF`. Grep for the right
spelling before concluding a file has no config block.

Naming conventions worth knowing:
- `NN_...py` trains and saves a checkpoint. `NNb_...py` is a variant (self-contained
  rewrite, fixed version, or alternate approach); e.g. 38b superseded 38, 43b produces
  the artifacts 43 does not.
- `NNa_...py` / `NN_..._diagnostic.py` **do not train**. They load checkpoints, print
  metrics, and exit. Diagnostics historically persisted nothing — results lived only in
  Colab logs. `RESULTS_node71_baseline.md` is the first recorded result file.

### Model lineage

| node | params | D | layers | FF | checkpoint |
|---|---|---|---|---|---|
| 40/41/43/45 | ~4.2M | 192 | 6 | 768 | `artifacts/step45/tiny_gpt_step45.pt` |
| 49–54 | 8,413,696 | 256 | 8 | 1024 | `artifacts/step54/tiny_gpt_step54_best.pt` |
| 59 v2 pretrain | 8,413,696 | 256 | 8 | 1024 | `artifacts/step59/tiny_gpt_v2_best.pt` |
| 66 scale pretrain | 42,001,408 | 512 | 12 | 2048 | `artifacts/step66/tiny_gpt_v2_best.pt` |
| 69 SFT | 42,001,408 | 512 | 12 | 2048 | `artifacts/step69/tiny_gpt_v2_42m_sft_best.pt` |

All modern nodes share **one tokenizer**: `artifacts/step43/step43_bpe_8000.json` (8K
BPE). Context is 256 everywhere. Pretraining corpora are memory-mapped `uint16` token
arrays built by 58 (`data/step58_v2_corpus/step58_{train,validation}_ids.uint16`).

`NODE_54_FREEZE_AND_DEPLOY.md` documents the 8.4M freeze point; its stated deployment
candidate (step54) no longer matches what `app.py` actually loads.

### The QA/SFT data module

`69_precision_data.py` (`qa_groups()`) is the shared fact source for SFT and diagnostics.
Each entry is `(en_prompts, en_good, en_bad, zh_prompts, zh_good, zh_bad)` — the `_bad`
answers are built-in distractors, which makes the module directly usable for preference
probes. `precision_data_69.py` is a byte-identical duplicate.

The train/held-out split is **deterministic and defined in 69**: shuffle `qa_groups()`
with `SEED = 69`, then `cut = int(len(groups) * 0.80)` → 33 train groups, 9 held-out.
Anything that wants to probe generalization must reproduce that split.

## Gotchas that have caused real bugs

**Weight tying.** `TinyGPT` sets `self.head.weight = self.tok.weight` — the same
`Parameter` object. Freezing `model.tok.parameters()` therefore **also freezes the LM
output projection**. Node 69 did this and silently trained only 45% of the model.
When freezing, always print a per-component `requires_grad` map; Node 69 now does.

**Probe contamination.** Diagnostics that hardcode prompts from `69_precision_data.py`
measure memorization, not capability — 7 of the original 9 Node 71 probes were Node 69
*train* prompts, which made a memorization effect look like a capability gain. Sample
probes from the held-out 9 groups, and report held-out and seen separately so the gap
is visible.

**Results are measured in the chat format by default,** which the pretrained base never
saw. Step 66 scores 36.1% (below chance) on the `User/Assistant` format but 55.6% (chance)
under raw continuation. Use raw continuation to test *knowledge*; use chat format to test
*instruction following*.

**Line endings.** Several files are CRLF in the working tree, which shows up as a whole-file
diff. `git diff --ignore-cr-at-eol` confirms. There is no `.gitattributes`.

**Remote code.** Node 69 fetches `69_precision_data.py` over HTTP. `?v=<sha>` is *not* a
GitHub pin — raw.githubusercontent ignores it and serves `main`. Only the `/<sha>/file.py`
path form pins anything.

## Current state

`RESULTS_node71_baseline.md` holds the current findings; read it before proposing work on
SFT or alignment. In short: the 42M base (step66) does **not** store the QA facts — it is
at chance at preferring a correct answer over the data's own wrong answer even under raw
continuation. SFT on 283 rows learns the answer *format* and memorizes its 33 training
groups, and pushes held-out preference below chance. The recent history of SFT/alignment
variants (47–69) has therefore been optimizing the wrong bottleneck; pretraining coverage
is the open problem.
