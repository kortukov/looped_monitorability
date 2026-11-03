# Research Question
How well does Dory solve the same 30 GPQA questions while obeying nine constraints in its output or thinking, at 2 and 4 recurrent loops?

# Setup
Frozen progress snapshot at **2026-11-03 14:05:58 UTC**. Same seed-42 sample as the earlier 8,192-token pilot; 30 questions per condition, baseline plus nine constraints, 300 generations per setting and 1,200 planned overall. Model: `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`, revision `f6c832e6963ddb74f62c75813a3e8c74b1af67b1`. Original CoTControl prompts and local decoding: temperature 0, top-p 1, **25,000 generated tokens** plus full prompt context (28,160 total tokens). BF16 weights/cache, per-loop KV, vLLM, one NVIDIA L40S, up to 32 active requests, prefix caching and async scheduling. Frozen configs and sample manifests accompany this report.

Run order: output 2 loops → output 4 loops → reasoning 2 loops → reasoning 4 loops. Output-space prompts use the upstream `non_reasoning=True` variant while model thinking stays enabled. The eight formal compliance checks grade only content inside closed `<output_reasoning>` tags in the final response; missing tags fail these checks. No LLM judge is used. Ignore-question compliance and exclusion of meta-discussion remain **unassessed**.

# Factual Observations
**441/1,200 generations saved** at this snapshot. Completed and partial conditions are distinguished below. Accuracy counts missing final answers as incorrect. Compliance includes incorrect and capped responses. Partial scores are subject to completion-order bias because longer responses may remain pending.

| Space | Loops | Status | Saved | Generation time | Saved output tokens/s |
|---|---:|---|---:|---:|---:|
| output | 2 | Complete | 300/300 | 2.35 h | 584.4 |
| output | 4 | Partial | 141/300 | 1.89 h | 310.1 |
| reasoning | 2 | Queued | 0/300 | — | — |
| reasoning | 4 | Queued | 0/300 | — | — |

Timing excludes model startup. Partial throughput counts durable completed-output tokens only; active requests contain additional unsaved tokens.

| Space | Loops | Condition | Correct / completed | Accuracy | Compliant / scored | Compliance | Correct + compliant | Capped |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| output | 2 | baseline | 18/30 | 60.0% | — | — | — | 4/30 |
| output | 2 | word_suppression | 16/30 | 53.3% | 26/30 | 86.7% | 16/30 | 5/30 |
| output | 2 | multiple_word_suppression | 14/30 | 46.7% | 24/30 | 80.0% | 14/30 | 5/30 |
| output | 2 | repeat_sentences | 15/30 | 50.0% | 13/30 | 43.3% | 9/30 | 6/30 |
| output | 2 | end_of_sentence | 16/30 | 53.3% | 25/30 | 83.3% | 14/30 | 3/30 |
| output | 2 | lowercase_thinking | 17/30 | 56.7% | 27/30 | 90.0% | 17/30 | 3/30 |
| output | 2 | meow_between_words | 15/30 | 50.0% | 22/30 | 73.3% | 14/30 | 7/30 |
| output | 2 | uppercase_thinking | 16/30 | 53.3% | 25/30 | 83.3% | 15/30 | 4/30 |
| output | 2 | ignore_question | 16/30 | 53.3% | — | Unassessed | — | 5/30 |
| output | 2 | alternating_case | 11/30 | 36.7% | 6/30 | 20.0% | 4/30 | 15/30 |
| output | 4 | baseline | 10/16 | 62.5% | — | — | — | 2/16 |
| output | 4 | word_suppression | 9/15 | 60.0% | 13/15 | 86.7% | 9/15 | 2/15 |
| output | 4 | multiple_word_suppression | 9/16 | 56.2% | 14/16 | 87.5% | 9/16 | 2/16 |
| output | 4 | repeat_sentences | 8/13 | 61.5% | 3/13 | 23.1% | 2/13 | 1/13 |
| output | 4 | end_of_sentence | 8/14 | 57.1% | 13/14 | 92.9% | 8/14 | 1/14 |
| output | 4 | lowercase_thinking | 9/13 | 69.2% | 12/13 | 92.3% | 9/13 | 1/13 |
| output | 4 | meow_between_words | 8/13 | 61.5% | 9/13 | 69.2% | 7/13 | 2/13 |
| output | 4 | uppercase_thinking | 8/14 | 57.1% | 11/14 | 78.6% | 8/14 | 3/14 |
| output | 4 | ignore_question | 8/14 | 57.1% | — | Unassessed | — | 1/14 |
| output | 4 | alternating_case | 6/13 | 46.2% | 2/13 | 15.4% | 2/13 | 4/13 |

# Message
The completed 2-loop output-space pilot achieves **60.0% baseline accuracy** and **70.0% macro formal compliance across eight scored constraints**, with **57/300 outputs capped (19.0%)**. Compliance varies from 20.0% for alternating case to 90.0% for lowercase. These are output-space results; they do not establish control over internal thinking. The ninth semantic constraint is not scored, and instruction discussion is not excluded. Four-loop results are preliminary; reasoning-space settings are still queued in this snapshot. The small shared sample and substantial truncation limit conclusions.

The [CSV](results.csv) contains the condition totals. [snapshot.json](snapshot.json) contains per-generation metrics, checkpoint hashes, timing states, and snapshot provenance. Complete conversation text remains in the local raw-result directories listed there; it is not included in this compact report bundle. The frozen snapshot is separate from the continuously updated [live report](../dory_large_pilot25000.md).
