# Looped transformer CoT controllability

GPQA pilots reuse the pinned [CoTControl](https://github.com/YuehHanChen/CoTControl)
submodule's questions, prompts, and programmatic graders. Experiment instructions
and the required short report format are saved in [AGENTS.md](AGENTS.md).

All pilots use the same 30 questions (seed 42), sampled from the 427 GPQA rows
with keyword annotations. The sample indices and hashes are versioned in
[experiments/gpqa30_seed42.json](experiments/gpqa30_seed42.json).
Each setting runs an unconstrained baseline plus three easy controls:
single-keyword suppression, string addition at the beginning/end, and lowercase
reasoning. That is 120 generations per setting, 600 across the five requested settings.

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

## Run the requested pilots

```bash
# Nanbeige: one and two loops
for loops in 1 2; do
  .venv-nanbeige/bin/python scripts/gpqa_pilot.py generate --model Nanbeige/Nanbeige4.2-3B --loops "$loops" --max-model-len 16384
  .venv-nanbeige/bin/python scripts/gpqa_pilot.py judge --model Nanbeige/Nanbeige4.2-3B --loops "$loops"
  .venv-report/bin/python scripts/report_gpqa_pilot.py --results "results/gpqa30_seed42_nanbeige42_loops${loops}_pilot8192"
done

# Dory: one, two, and four loops; per-loop KV caching throughout
for loops in 1 2 4; do
  .venv-dory/bin/python scripts/gpqa_pilot.py generate --model nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 --loops "$loops" --max-model-len 16384
  .venv-dory/bin/python scripts/gpqa_pilot.py judge --model nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 --loops "$loops"
  .venv-report/bin/python scripts/report_gpqa_pilot.py --results "results/gpqa30_seed42_dory_loops${loops}_pilot8192"
done
```

Or run `./scripts/run_requested_pilots.sh` to execute these five settings and
write individual reports plus `reports/gpqa_requested_pilots_comparison.md`.
Nanbeige's `num_loops` is overridden explicitly; 2 is its checkpoint default.
The runner saves the effective model config and checks the applied loop count.

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
finished its first batch, so no completed baseline results are claimed. The requested Nanbeige (1/2 loops) and Dory (1/2/4 loops) pilots completed on
2026-11-02 on an NVIDIA L40S: 600 generations, with separate reports and plots.
See [the comparison report](reports/gpqa_requested_pilots_comparison.md) and
[the numeric table](reports/gpqa_requested_pilots_comparison.csv). The original
Ouro pilot remains stopped; no completed Ouro baseline is claimed.
Respect CoTControl's request to keep the benchmark out of model training data.

## Authorized full Dory evaluation

The user authorized full GPQA and all nine constraints on 2026-11-02, in
**4 → 2 → 1** loop order, following the [pipeline audit](reports/dory_pipeline_audit.md).
This supersedes the earlier full-run hold for Dory only. There are 4,414
generations per setting (13,242 total). Original local decoding is greedy,
temperature 0, top-p 1, 25,000 generated tokens total; BF16 and per-loop KV are
preserved. Full prompts need at most 3,055 tokens and use a 28,160-token context.

```bash
.venv-dory/bin/python scripts/prepare_dory_full.py
.venv-dory/bin/python -u scripts/run_dory_full.py --allow-full-run --max-num-seqs-4 32 --max-num-seqs-2 32 --max-num-seqs-1 32
```

Run only one supervisor. It keeps one model engine on the GPU, overlaps API
grading and report creation with GPU work, resumes durable completions on
retries, and respects `results/PAUSE` and per-run `PAUSE` files. To resume after
an explicit pause, remove that pause file and rerun the command. A full run
uses a shared condition queue; its per-setting runtime is measured directly,
without summing overlapping condition times. Full outputs are separate from
pilots under `results/gpqa_full_dory_loops{4,2,1}_25000`. Live state and GPU
telemetry are in `results/dory_full_sequence_state.json` and
`results/dory_full_gpu_telemetry.jsonl`; each run has `run.log`, `judge.log`,
`summary.json`, and completion markers. This is a multi-day workload.

On completion, reports appear as `reports/gpqa_dory_loops{4,2,1}_full25000.md`,
followed by `reports/gpqa_dory_full_comparison.md` and its CSV/figure.

The 30-question, 25,000-token Dory rerun uses the same seed-42 sample as the
8,192-token pilots, with baseline plus all nine controls in both output and
reasoning spaces. Output-space controls run first, at 2 then 4 loops, followed
by reasoning-space controls at 2 then 4 loops. Prepare and launch with:

```bash
.venv-dory/bin/python scripts/run_dory_large_pilot.py prepare
.venv-dory/bin/python -u scripts/run_dory_large_pilot.py run
```

This supervisor makes no API calls. Accuracy and eight formal compliance
scores use code; ignore-question compliance and meta-discussion exclusion
remain unassessed. Output-space grading requires a closed `<output_reasoning>`
block in the final response and grades its contents only. Artifacts and live
state are in `results/dory_large_pilot25000/`, with a live report at
`reports/dory_large_pilot25000.md`. Each of four settings has 300 requests,
for 1,200 total. Completion is indicated by supervisor phase `complete`, not
by launch. The older full evaluation remains paused using its per-run flags.

The amended pilot inserts `output_no_thinking` at 2 and 4 loops after both original output-space settings and before reasoning-space settings (1,800 generations total). These ablations keep the output prompts, sample, and budget unchanged but use the tokenizer's `enable_thinking=False` closed `<think></think>` prefix. Generated text is treated as final output; tagged output reasoning is still requested and scored. Run `prepare` again to freeze the extra inputs. The one-time handoff script `scripts/continue_dory_ablation.py` waits for the earlier supervisor to release its lock and requires both original output completion markers and its exact temporary hold text before starting the amended sequence. Handoff status is saved in `results/dory_large_pilot25000/ablation_queue.json`.

To extend the finished pilot with paired question-sampling seeds 43/44 for output/thinking-off at 1/2 loops (reusing seed42), followed by all three 8-loop settings at seed42, run `.venv-dory/bin/python scripts/run_dory_extensions.py prepare` then `.venv-dory/bin/python -u scripts/run_dory_extensions.py run`. The queue publishes the concise report periodically and at completion; state is `results/dory_large_pilot25000/extensions_state.json`. Mean ± sample SD is computed across all three complete sampling seeds, with generation seed42 and greedy decoding fixed. Eight-loop settings use 16 active sequences; other settings use 32. The full 25,000-token generation budget and BF16 weights/cache are preserved.
