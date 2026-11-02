# Looped transformer CoT controllability

GPQA pilots reuse the pinned [CoTControl](https://github.com/YuehHanChen/CoTControl)
submodule's questions, prompts, and programmatic graders. Experiment instructions
and the required short report format are saved in [AGENTS.md](AGENTS.md).

All pilots use the same 30 questions (seed 42), sampled from the 427 GPQA rows
with keyword annotations. The sample indices and hashes are versioned in
[experiments/gpqa30_seed42.json](experiments/gpqa30_seed42.json).
Each setting runs an unconstrained baseline plus three easy controls:
single-keyword suppression, string addition at the beginning/end, and lowercase
reasoning. That is 120 generations per setting, 600 across all five settings.

## Install

Install [uv](https://docs.astral.sh/uv/) and use a Linux NVIDIA GPU host with a
driver compatible with the CUDA 13.0 wheels. Then run:

```bash
./scripts/setup.sh
export OPENROUTER_API_KEY=...  # Needed for meta-discussion judging
```

Separate Python 3.12 environments pin vLLM 0.26.0 for Ouro, 0.30.0 for Nanbeige,
and 0.33.0 for Dory. Each has an exact dependency lock file. Model and tokenizer
revisions are pinned in the runner. Weights download automatically from Hugging
Face; no model caches or virtual environments are stored in Git.

## Run the pilots in this order

```bash
# 1. Ouro: checkpoint default, four recurrence steps
.venv/bin/python scripts/gpqa_pilot.py generate
.venv/bin/python scripts/gpqa_pilot.py judge
.venv-report/bin/python scripts/report_gpqa_pilot.py

# 2. Nanbeige: checkpoint default, two loops
.venv-nanbeige/bin/python scripts/gpqa_pilot.py generate --model Nanbeige/Nanbeige4.2-3B --max-model-len 16384
.venv-nanbeige/bin/python scripts/gpqa_pilot.py judge --model Nanbeige/Nanbeige4.2-3B
.venv-report/bin/python scripts/report_gpqa_pilot.py --results results/gpqa30_seed42_nanbeige42_pilot8192

# 3. Dory: one, two, and four loops; per-loop KV caching throughout
for loops in 1 2 4; do
  .venv-dory/bin/python scripts/gpqa_pilot.py generate --model nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 --loops "$loops" --max-model-len 16384
  .venv-dory/bin/python scripts/gpqa_pilot.py judge --model nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 --loops "$loops"
  .venv-report/bin/python scripts/report_gpqa_pilot.py --results "results/gpqa30_seed42_dory_loops${loops}_pilot8192"
done
```

Pilots use greedy decoding, BF16 weights/cache, up to eight active sequences,
and an **8,192-token generation cap**. Ouro's checkpoint specifies 48 layers,
despite the model card listing 24; checkpoint architecture is preserved. Dory
uses `n_recurrent_loops` and `recurrent_kv_cache_mode=per_loop` explicitly.
Use `--max-num-seqs` and `--gpu-memory-utilization` to tune batching on the new
host. Prompt lengths are checked before loading the model.

Outputs remain local under `results/`: configuration, sample manifest, full
reasoning/final answers, correctness, compliance, token counts, timing, and
individual completed-question checkpoints. Reports and plots go in `reports/`.
Re-running a command skips saved questions. Missing final answers count as
incorrect. Compliance includes incorrect answers and is also reported excluding
meta-discussion. Meta-discussion is judged by `openai/gpt-5-mini` via OpenRouter.
The three-control aggregate is not the paper's full nine-control score.

Generation runtime and aggregate saved-output tokens/second are included in
reports, excluding model startup and API judging. Interrupted batches include
wasted time in their timing totals. Cross-host GPU/version details are recorded
in launch metadata; timing comparisons across different hardware need care.

The full GPQA plan is saved in [experiments/ouro_full_plan.json](experiments/ouro_full_plan.json).
It restores CoTControl's 25,000-token budget and includes all nine constraints.
**Full-scale execution is on hold until explicitly authorized by the user.**
The runner additionally requires `--allow-full-run` for full-scale generation.
The four-step Ouro BF16 setup needs roughly 47 GiB for one maximum-length
request, so the original 24 GB GPU cannot hold the full-budget setup.

## Fresh start on another host

Clone this repository with submodules, or rsync the source while excluding
`.venv*`, `.deps`, `__pycache__`, and `results` for a fresh experiment. Re-run
`./scripts/setup.sh`; virtual environments contain absolute paths and should be
rebuilt. Credentials belong in environment variables or an ignored `.env` file.
Do not copy model caches; the runner downloads the pinned revisions again.

The original local pilot was stopped for the requested fresh start. It had not
finished its first batch, so no completed baseline results are claimed. GPU
inference and all five configurations still require execution on the new host.
Respect CoTControl's request to keep the benchmark out of model training data.
