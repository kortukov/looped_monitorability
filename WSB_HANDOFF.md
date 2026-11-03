# WorkspaceBench workspace report baseline handoff

The user requested a concise prompt, vLLM completions, a small pipeline pilot, a GitHub push, and a handoff for a new machine/session. The main study compares collaborator-provided **J-lens under the original prompt** against the subject model's **elicited workspace report**. Do not rerun J-lens or require a J-lens arm under the explanation prompt. Logit lens is an optional later baseline, already implemented in the separate looped-jlens repository.

Read `AGENTS.md` for the required report structure. This WSB task is separate from the paused GPQA/Dory work; do not resume any of those jobs. No full WSB evaluation has been launched. Wait for the user's instruction in the new session before starting one.

## Current implementation

- Prompt: `experiments/wsb_workspace_report_prompt.txt` (59 words).
- vLLM runner: `experiments/wsb_workspace_pilot.py`. Defaults to eight items: one majority-pass and one majority-fail per family, selected deterministically from `seed_0` with sampling seed 42.
- Judging: `experiments/wsb_workspace_judge.py`, using the unmodified WorkspaceBench submodule at `0b1878987927555a9d88f65c6fc3c14debb76e1e`.
- Reduction: `experiments/wsb_workspace_summarize.py`. Coverage and generation failures remain visible. Non-pilot scopes also produce per-version scores and mean/population SD across fully judged seeds, separately for max and u85.
- Existing gate statistics: `experiments/wsb_agree_summary.py` and `reports/wsb_agree_explicit_reasoning_baseline.md`.
- Pilot report: `reports/wsb_workspace_pilot.md`; portable raw artifacts: `artifacts/wsb_workspace/pilot_ouro14b/`.

Generation uses greedy decoding, BF16 weights/cache, no quantization, thinking off, a 25,000-token allowance plus complete prompt context, and a stop after `</final>`. **The validated smoke test also constrains the envelope:** a single-paragraph report of 1–2,048 characters, followed by a final section of 1–1,024 characters. See `FORMAT_REGEX` and the saved config. The character limits are additional decoding constraints, even though no response exhausted the token allowance. Do not describe this as unrestricted reasoning or claim the constraints affect formatting only.

Exploratory free-format output omitted tags or quoted them as instructions. An unbounded structured report did not finish its first poetry request before cancellation. The bounded format completed all eight. Earlier attempts and their saved outputs are preserved under the artifact directory; interrupted requests are not completed generations. Do not silently combine attempts.

Judges and digests use `anthropic/claude-haiku-5.5` through OpenRouter, medium effort, 8,000 judge output tokens, no temperature parameter, summary aggregation, and one joint digest. This matches the collaborator's model and main aggregation choices, but uses synchronous API calls rather than Batch. It is a **non-pinned instrument**, not the upstream Opus score of record. The transport's effective settings are stored separately from upstream requested-effort metadata and copied into `results.json.config.effective_transport`.

Generated reports use synthetic coordinates `layer=-1,pos=-1` solely to carry one prose sample through the family judges. They are not activation samples. Files have no activation sidecar, and scoring is report-only with `allow_missing=True`; upstream `complete=false` is expected. Report actual generation and judging coverage separately. Only the report is scored; final-answer correctness and the original 11-draw gate are not re-evaluated by this smoke test.

Keep the **full original bank's distractor pool** while overriding selected items with dataset records. DM constructs options from other bank items; trimming the bank to the pilot changes or breaks that instrument. The saved bank hashes and actual options allow auditing. Before a substantive comparison, align these options/bank versions with the collaborator's exact J-lens run; their adapted passing-item banks need not have the same distractor pool.

## Fresh machine setup

```bash
git clone --recurse-submodules https://github.com/kortukov/looped_monitorability.git
cd looped_monitorability
bash scripts/setup_wsb.sh
```

Requires `uv`, a CUDA GPU with BF16 support, and a driver compatible with the pinned vLLM environment. The tested GPU is one RTX 4090, 24 GB. The runner adds its environment's bin directory to PATH so kernel builders can find `ninja`. The first launch may compile FlashInfer kernels. Setup verifies and extracts the bundled non-Git dataset into `.deps/wsb-agree-100`; **SSH to arena9-kind is not needed**. Input archive provenance and file hashes are under `artifacts/wsb_workspace/input/`.

Use an existing `OPENROUTER_API_KEY` environment variable for judging. Do not put credentials in the repository. The public subject checkpoint is downloaded by vLLM; model weights and virtual environments are not bundled.

## Reproduce the pilot

Use a fresh output directory if changing any scientific setting. Generation saves every completed request immediately; the same command resumes an unchanged run. It refuses an incompatible manifest/config.

```bash
.venv-wsb-vllm/bin/python experiments/wsb_workspace_pilot.py generate --out results/wsb_workspace_pilot
.venv-wsb-judge/bin/python experiments/wsb_workspace_judge.py --out results/wsb_workspace_pilot --dry-run
.venv-wsb-judge/bin/python experiments/wsb_workspace_judge.py --out results/wsb_workspace_pilot
.venv-wsb-judge/bin/python experiments/wsb_workspace_summarize.py --out results/wsb_workspace_pilot
```

The dry run makes no API calls. Judging caches completed requests. `judge_requests.jsonl` retains prompts, schemas, raw API responses and actual usage. To inspect or re-judge the saved pilot on a fresh host, copy `artifacts/wsb_workspace/pilot_ouro14b/` into a fresh `results/` directory and run the judging/reduction commands there. Absolute paths in original launch metadata identify the original machine; scripts resolve new bank/readout paths from `--out`.

## Larger run preparation and limits

`--scope max` prepares the 256 distinct items appearing across ten max seeds; `--scope u85` prepares 273 across ten u85 seeds; `--scope all` prepares the union of both tiers. Generate each item once, then reduce over each version's original membership. Default scope remains `pilot`. Preparing a manifest performs no model inference:

```bash
.venv-wsb-judge/bin/python experiments/wsb_workspace_pilot.py prepare --scope max --out results/wsb_max
```

After the user explicitly authorizes a larger run, use `generate --scope max` with the same output directory, then the judge and summarize commands. Run u85 separately or use `--scope all`. Do not report seed averages from an eight-item pilot. The seeds overlap; their spread is sensitivity to subset choice, not independent-sample uncertainty.

Model presets are prepared for ByteDance/Ouro-1.4B (four steps), ByteDance/Ouro-2.6B (four steps), and NVIDIA Dory1.0-4B-BF16 (two loops, per-loop cache, thinking off), with pinned revisions. **Only Ouro-1.4B was exercised in this WSB pilot.** Use `--model ByteDance/Ouro-2.6B` with `.venv-wsb-vllm`. For Dory, run `bash scripts/setup_wsb.sh --dory`, then use `.venv-wsb-dory/bin/python ... --model nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`. Ouro requires vLLM 0.26.0; Dory requires 0.33.0. Pilot each new model before launching its larger evaluation. Do not silently shrink the budget or change precision for memory; use suitable hardware.

Before treating the numbers as a study result:

1. **Poetry needs conditioning alignment.** This smoke test asks for a report while showing the full couplet stub, whereas the original lens reads at the end of line one. The production comparison must prevent that later-text advantage and establish the subject's planned rhyme at the matched read point. Saved scores against the dataset's frozen rhyme target are pipeline diagnostics only, not a validated poetry comparison. Track target changes explicitly.
2. Fix the prompt/format/length policy before the full run. This pilot documents a useful bounded option; compare the same policy across models and preserve its configuration.
3. Match collaborator bank/option sets, judge model, effort, digest settings and aggregation. Do not compare Haiku/joint scores to Opus/union scores without identifying the instrument change.
4. Retain original pass/fail labels for stratification, and separately evaluate final answers and prompt-induced changes if making capability claims. Conjunctive-association gate pass means no leak, while its verbalization metric rewards recovery of the composed state.

Keep the final report short, with exactly Research Question, Setup, Factual Observations, and Message. Include runtime, throughput, coverage, truncation and all effective constraints. Preserve per-item artifacts for every number.
