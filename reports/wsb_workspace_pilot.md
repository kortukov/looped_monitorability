# Workspace report pipeline pilot

## Research Question

Can a concise workspace-report prompt produce completions that vLLM can generate, the adapter can parse, and all four WorkspaceBench family graders can score?

## Setup

Ouro-1.4B (`ByteDance/Ouro-1.4B`, revision `eac7e5f31a34b8cdee88c059b840294f2cfc6645`), four recurrence steps, early-exit threshold 1.0, thinking off. vLLM 0.26.0, Torch 2.11.0, Transformers 5.22.1; one NVIDIA RTX 4090 with 24 GB. Greedy decoding, seed 42, BF16 weights/cache, no quantization, one request at a time, eager execution. The 25,000-token generation allowance fits with the complete prompt in a 25,216-token context.

The 59-word prompt is saved in [the prompt file](../experiments/wsb_workspace_report_prompt.txt). Eight items were sampled from WSB-Agree-100 `seed_0`: one majority-pass and one majority-fail from each family, sampling seed 42. This smoke test is not an average across dataset seeds. Only the workspace report is submitted to the verbalization graders; final-answer correctness and the 11-draw answerability gate are not evaluated here.

The successful attempt uses a decoding regex enforcing a single-paragraph `<workspace_report>` of 1–2,048 characters and a `<final>` section of 1–1,024 characters, stopping after `</final>`. These are effective length constraints in addition to the token allowance. Exploratory free-format and unbounded-format attempts are preserved separately.

WorkspaceBench is an unmodified submodule at `0b1878987927555a9d88f65c6fc3c14debb76e1e`. Its family judges and digest use Haiku 5.5 through OpenRouter, medium effort, 8,000 output tokens, summary aggregation and a joint digest. This matches the collaborator's main judging choices, using synchronous calls instead of Batch. It is a non-pinned instrument. A synthetic cell at `layer=-1,pos=-1` carries each generated report; it makes no activation-site claim. The complete original distractor pool is retained while selected records are overwritten from the dataset. Upstream marks this report-only configuration `complete=false`; coverage is measured below.

## Factual Observations

| Family | Generated / parsed / judged | Recovered target | Recovery rate |
|---|---:|---:|---:|
| Multihop | 2 / 2 / 2 | 0 / 2 | 0% |
| Directed modulation | 2 / 2 / 2 | 1 / 2 | 50% |
| Poetry | 2 / 2 / 2 | 0 / 2 | 0% |
| Conjunctive association | 2 / 2 / 2 | 0 / 2 | 0% |

All eight completions parsed and received decided judgments, with zero missing or empty readouts and zero token-cap truncations. One report reached its 2,048-character bound; its final section was 1,022 characters. The directed-modulation hit named the secret word and was classified as instruction narration, not content-bound representation. These scores are readout recovery, not the subset's original pass/fail labels.

The successful attempt generated 1,489 tokens in **74.3 seconds**, **20.1 tokens/s** including prefills. Model startup took **9.4 seconds** with kernels already cached; judging took **52.2 seconds** for 22 API calls. Their combined time was **135.8 seconds**, excluding dependency installation and exploratory attempts. The recorded judging cost was approximately **$0.0088**.

The first launch failed because `ninja` was absent from PATH; the runner now adds its environment's bin directory. The free-format attempt saved four completions, two parseable, before cancellation in poetry. The unbounded-regex attempt saved four parseable completions before cancellation in poetry. Their unfinished requests are not counted as completed generations or included in the successful attempt's throughput. Limiting the report to one paragraph and bounding its length produced the completed eight-item attempt; the pilot does not isolate which change resolved the long generation.

The raw [artifact directory](../artifacts/wsb_workspace/pilot_ouro14b/) contains the exact manifest, prompts, model settings, output/token IDs, parsed reports, readouts, bank files/hashes, judge prompts, API responses, verdict caches and derived summaries. Earlier attempts are under its `attempts/` directory. The [dataset bundle and provenance](../artifacts/wsb_workspace/input/) make the inputs available on a fresh host.

## Message

The bounded vLLM pipeline works through all four family graders. This small pilot found little target recovery and should not be used to rank verbalization approaches. Its character constraint affected at least one completion and must remain explicit in subsequent results.

Before a substantive comparison, match the collaborator's bank/distractor sets and judge settings. Poetry also needs conditioning alignment: this smoke test shows the full couplet stub to the reporter, whereas the original lens reads at the end of line one. Its frozen-target score is a pipeline diagnostic, not a validated comparison of planned rhymes at matched read sites. Ouro-2.6B and Dory presets are prepared but have not been exercised in this WSB pipeline. No full WSB run was launched; continuation instructions are in [WSB_HANDOFF.md](../WSB_HANDOFF.md).
