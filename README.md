# MERCIAL1 — reading a model's belief three ways

We compare three readouts of what a small model believes about a yes/no unknown `y`:

1. **Ask it**: its stated probability, Likert answer and probability bin.
2. **Infer it from its decisions**: the latent belief `z` of Smolin & Wilder (2026), *Beliefs and
   Behavior in Language Models*, [arXiv:2609.07943](https://arxiv.org/abs/2609.07943).
3. **Read it from inside the model**: a linear probe on the residual stream trained to match `z`.
   (Not built yet.)

Research questions:
- (a) When a small model says one thing but does another, is its belief messy, or does it have a
  clear belief that doesn't come out cleanly?
- (b) What role does belief play during emergent misalignment?

## Layout

```
belief/questions.py          builds the paper's 16 prompts from a scenario file
scenarios/                   one JSON file per domain (format: scenarios/README.md)
  ckd.json                   chronic kidney disease screening (paper wording)
  werewolf.json              Werewolf, is the target a werewolf (paper wording)
  insecure_code.json         our own case (draft): does this code have a vulnerability
run_model_on_question.py     runs a model on the 16 questions, saves answers + activations
extract_model_belief.py      fits the paper's latent belief model, writes belief readouts
control_model_organism.py    (todo) checks the readouts against model organisms with known beliefs
outputs/                     run results (not committed, they are large)
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Runs on an Apple-silicon Mac (MPS), on CUDA, or on CPU.

## Run

```bash
# 1. answers + activations (default model Qwen/Qwen2.5-1.5B-Instruct)
.venv/bin/python run_model_on_question.py --scenario scenarios/ckd.json
.venv/bin/python run_model_on_question.py --scenario scenarios/werewolf.json
.venv/bin/python run_model_on_question.py --scenario scenarios/insecure_code.json

# ICL: plant a belief in the system prompt
.venv/bin/python run_model_on_question.py --scenario scenarios/ckd.json \
    --system "You are certain this patient has CKD." --tag icl_yes

# model organism given as a LoRA adapter
.venv/bin/python run_model_on_question.py --scenario scenarios/insecure_code.json \
    --model <base> --adapter <adapter>

# 2. belief readouts
.venv/bin/python extract_model_belief.py outputs/ckd__Qwen2.5-1.5B-Instruct
```

Useful flags: `--n-cases N` (quick test), `--no-reasoning` (answer only, much faster),
`--samples K`, `--max-new-tokens`, `--no-activations`.

## What gets saved

`outputs/<domain>__<model>[__tag]/`
- `responses.jsonl`: one row per case × question × sample. Holds the prompt, the reply, the parsed
  answer, and `option_probs` (the model's probability on each option letter, read right after
  `Choice:`).
- `activations/<case>.npz`: float16 arrays `[rows, layers+1, d_model]` at three points:
  `context_end` (end of the case details), `prompt_end` (end of the prompt) and `answer` (just
  before the answer, after the model's reasoning).
- `beliefs.csv` / `fit.json` (from `extract_model_belief.py`): one row per case with `direct_prob`,
  `z_reports`, `z_decisions` and `z_all`, plus the fitted decision thresholds compared with the
  nominal ones.

## Differences from the paper (so far)

- The cases are small hand-written pilots: 8 kidney patients, 4 werewolf cuts and 8 code snippets.
  The paper uses 500 per domain, from NHANES (kidney) and the Kaggle Werewolf dataset (needs a
  login).
- The paper fits on 500 cases. With fewer than 30 cases, `extract_model_belief.py` fits
  everything in-sample and says so.
- `option_probs` and the activations are our additions.
- Sampling: temperature 1 as in the paper, but we keep the model's own top-p/top-k by default.
  Without them, Qwen 0.5B/1.5B drift into gibberish and Chinese at temperature 1. Use
  `--pure-sampling` for the paper's exact setting.
- Speed on an M4 Mac with Qwen2.5-1.5B: about 11 minutes per case with reasoning, much faster
  with `--no-reasoning`.
