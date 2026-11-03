# WSB Agree explicit reasoning baseline

## Research Question

Does asking a model to report the concepts, intermediate results, relationships and plans it is holding in mind recover the same task-relevant information as workspace verbalizers? The primary comparison is collaborator-provided J-lens under the original prompt versus an explicit workspace report. This report contains the study design and existing gate statistics; new readout scores remain to be measured.

## Setup

Recorded gate results come from `/root/wsb-agree-100`, copied from `arena9-kind:/root/looped-jlens/agent/datasets/wsb-agree-100`. Each version contains 100 items, 25 per family and 50 majority-pass labels. Analyze ten `max` seeds as the primary tier and ten `u85` seeds separately as a diversity check. Seeds overlap; they are not independent replicates.

The dataset documentation specifies Ouro-1.4B and Ouro-2.6B at four recurrence steps using HF transformers, and Dory1.0-4B at two loops with thinking off using vLLM. Model revisions, original hardware, runtime and throughput are not supplied by the subset and have not been verified from full run logs. The gate requires a correct greedy answer and at least eight correct samples out of ten at T=0.7, using WorkspaceBench graders with a Haiku 5.5 fallback.

The proposed study preserves those model settings. Collaborators supply J-lens readouts under the original task prompt; this experiment supplies workspace reports followed by the task answer. A local fit-free `LogitLens` implementation and WorkspaceBench's `method=logit_lens` producer provide an optional simpler original-prompt baseline. Ordinary task output is another inexpensive control. J-lens under the report prompt is optional rather than part of the primary comparison. Do not put golds, candidate answers, majority labels or model gate outcomes into generation prompts.

The [workspace paper](https://transformer-circuits.pub/2026/workspace/) motivates reporting accessible concepts and intermediate representations, including material absent from the answer. Use the following proposed instruction, saved verbatim in `experiments/wsb_workspace_report_prompt.txt`:

> Before answering, report the specific concepts, intermediate results, inferred relationships, and plans you are holding in mind, including content absent from your final answer. Describe the content directly, with clear roles and relations, rather than restating instructions or listing possibilities.
>
> Write one paragraph in <workspace_report>...</workspace_report>, then the task answer in <final>...</final>. Apply answer-format and non-disclosure restrictions only to <final>.

This explicitly changes output scope for the experimental condition. For directed modulation, the report may name the held concept while the final section copies the carrier. For conjunctive association, reporting the composed state is the readout objective; it must not be confused with passing the original non-leak screen. Use a single coherent paragraph for the report so compound states remain one statement under the existing composition grader. This is a task-independent prompt, with no examples or target options. Preserve family-specific final-answer formats.

Score the explanation alone as a prose readout with the same frozen WorkspaceBench family metrics, judges, digest settings and aggregation as the lens arms. Keep the final answer separate and report ordinary output and prompt-only controls. Adapt the readout transport without changing semantic grading rules or pretending a generated explanation is a layer/position measurement. Report native and summary aggregation separately if both are used; do not silently inherit any-cell success for one arm and summary scoring for another.

In the WorkspaceBench checkout on arena9-kind, `wsbench plan` specifies the render, layers and token sites. Producers emit one JSONL row per cell: prose `samples`, or ranked `tokens` with `scores`. The default summary mode digests an item's readouts, then uses the family's judge with verified evidence from the original samples. Native aggregation is family-specific. `wsbench report` supplies pass rates, confidence intervals, coverage and matching prompt-only/lucky-guess floors. These upstream defaults must be aligned with the collaborator's adapted model banks and instrument versions before comparing scores.

| Family | Readout target |
|---|---|
| Multihop | Name the silent bridge entity, rather than just the final answer |
| Directed modulation | Express the held concept; report content and instruction narration diagnostics separately |
| Poetry | Identify the planned rhyme at the end-of-line-one read site, with the family's exact-word rules |
| Conjunctive association | State the complete compound state with correct roles in one statement |

Re-evaluate final-answer behavior under each prompt, separately from explanation recovery. Report the original strict output constraints and final-section-only task validity separately. Directed modulation's concept suppression and conjunctive association's leak screen can conflict with explicit verbalization. Poetry targets must follow the model's own generated continuation under each condition; record target changes rather than scoring new generations only against old targets.

Report generation length, parse failures, truncation, coverage, wall time and tokens/s. Keep model revisions, prompts, generation seeds, inference configuration, raw outputs, parsed sections and judge responses. Record any token-budget changes explicitly; do not silently reduce budgets or quantize to fit memory. For equal-information-budget comparisons, add a length-limited diagnostic alongside the full readout. Pilot formatting on separate items before evaluating the frozen subset.

## Factual Observations

All 20 local versions satisfy the checked item counts, category sizes, majority labels and unanimity flags. The table summarizes existing recorded gate outcomes; no explicit-reasoning or lens evaluation was run for this report. Values are mean ± population standard deviation across ten seeds, in percent.

| Model | max gate pass | max label agreement | u85 gate pass | u85 label agreement |
|---|---:|---:|---:|---:|
| Ouro-1.4B | 50.0 ± 0.0 | 100.0 ± 0.0 | 52.1 ± 0.5 | 97.9 ± 0.5 |
| Ouro-2.6B | 51.0 ± 0.0 | 99.0 ± 0.0 | 51.9 ± 0.5 | 98.1 ± 0.5 |
| Dory1.0-4B | 55.0 ± 0.0 | 89.0 ± 0.0 | 55.0 ± 0.0 | 89.0 ± 0.0 |

Primary-tier gate pass rates by family:

| Model | Multihop | Directed modulation | Poetry | Conjunctive association |
|---|---:|---:|---:|---:|
| Ouro-1.4B | 48.4 ± 3.3 | 50.0 ± 4.8 | 41.6 ± 2.7 | 60.0 ± 0.0 |
| Ouro-2.6B | 48.4 ± 3.3 | 50.0 ± 4.8 | 41.6 ± 2.7 | 64.0 ± 0.0 |
| Dory1.0-4B | 43.2 ± 3.5 | 48.4 ± 4.5 | 36.4 ± 3.3 | 92.0 ± 0.0 |

Gate pass rates measure answerability/compliance; these are not workspace readout scores. Conjunctive association gate success means avoiding a leak. All max versions have 88 unanimous items; all u85 versions have 85. The subset contains complete gate flags but no new generation coverage or runtime measurements. Existing raw records remain in the dataset; computed per-seed metrics and source checksums are in `results/wsb_agree_existing_gate/summary.json` and `config.json`. Reproduce with:

```bash
python experiments/wsb_agree_summary.py /root/wsb-agree-100 results/wsb_agree_existing_gate
```

## Message

An explicit workspace report is a useful practical baseline for how much relevant information can be recovered by asking the subject model directly. The prompt targets the content measured by the benchmark rather than requiring an exhaustive solution walkthrough.

The main comparison reports recovery from original-prompt J-lens and the elicited workspace report, by family and original pass/fail stratum, with changes in task behavior beside it. Keep prompt-only controls and the directed-modulation content-versus-instruction-narration diagnostic. The practical question is how much of the benchmark's target content is available through each approach. Additional prompted-lens or causal-intervention arms can be follow-ups; they are not prerequisites for this baseline.

Selection used these same three models and constrains their gate rates; the aggregate pass table is not an unbiased capability ranking. Preserve original labels for stratification, track prompt-induced changes separately, and report paired method differences within each seed. Means and seed spread describe sensitivity to subset selection; use item-aware paired uncertainty for inferential claims rather than treating repeated appearances across seeds as independent observations.
