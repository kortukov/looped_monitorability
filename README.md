# Dory controllability experiments

**Start with the [main results report](reports/dory_pilot_results.md).** It contains the current loop comparison, three-seed estimates, and pending runs.

- [Exact prompt templates](reports/dory_prompt_templates.md): what each setup asks the model to do.
- [Answer-first comparison](reports/dory_answer_first_results.md): separate experiment using the released no-tag prompt.
- [Report index](reports/README.md): current reports, supporting data, and archives.

Dory uses the pinned CoTControl GPQA questions, prompts, and formal graders. Each pilot setting has 30 questions, baseline plus nine controls, BF16 weights/cache, greedy decoding, and a 25,000-token generation limit. Eight constraints are code-scored; semantic ignore-question compliance and instruction-discussion checks remain unassessed.

## Running and resuming

Install the pinned environments with `./scripts/setup.sh`. Use `.venv-dory/bin/python` for Dory. Credentials stay in the ignored `.env`; GitHub publishing uses `GIT_TOKEN`. The current pilots need no judge API.

```bash
# Extra sampling seeds at 1/2 loops, then all three setups at 8 loops
.venv-dory/bin/python scripts/run_dory_extensions.py prepare
.venv-dory/bin/python scripts/run_dory_extensions.py run

# Separate answer-first comparison (1/2 loops, thinking on/off)
.venv-dory/bin/python scripts/run_dory_answer_first.py prepare
.venv-dory/bin/python scripts/run_dory_answer_first.py run
```

The answer-first comparison is queued after the existing extension runs. Supervisors share a GPU lock; do not launch a second queue while one is running. They resume completed checkpoints and honor `results/PAUSE` and per-run pause files. The full-dataset evaluation remains paused.

Raw outputs, exact inputs, model configs, timings, and live state are under `results/dory_large_pilot25000/` (ignored by Git). Extension progress is in `extensions_state.json`; the answer-first queue is in `documentation_handoff.json`, then `answer_first/state.json`. Each run has a `generation_complete.json` marker when finished.

The main and answer-first reports update locally during generation and push about every 15 minutes and at completion. Their CSV/JSON companions contain condition counts, configurations, and sampling metadata.

## Repository layout

| Location | Contents |
|---|---|
| `reports/` | Current reports, prompt reference, supporting CSV/JSON |
| `reports/archive/` | Historical 8,192-token pilots and superseded snapshots |
| `scripts/` | Preparation, generation, reporting, and queue entry points |
| `tests/` | Checkpoint, scoring, and statistics checks |
| `third_party/CoTControl/` | Pinned upstream submodule; unchanged |
| `artifacts/` | Standalone conversation viewer |

[Historical workflows](docs/historical_workflows.md) · [Experiment preferences](AGENTS.md) · [Original GPU audit](reports/dory_pipeline_audit.md)
