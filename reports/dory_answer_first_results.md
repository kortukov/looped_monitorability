# Research Question
Does the released answer-first prompt improve Dory's output control without output-reasoning tags?

# Setup
Same 30 GPQA questions (seed 42), baseline + nine controls; 1/2 loops, thinking on/off. Original CoTControl `output_no_letter_no_grading` prompts, BF16, greedy decoding, 25,000 generated tokens, one L40S. Controls apply to the **entire final response**, including the answer. [Exact templates](dory_prompt_templates.md#answer-first-without-tags) · [Main report](dory_pilot_results.md).

# Factual Observations
Updated **2026-11-05 20:17 UTC**. Queued after the seed-replication and 8-loop runs.

| Thinking | Loops | Status | Baseline accuracy | Controlled accuracy | Formal compliance | Unparsed | Capped | Time / tokens per second |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| on | 1 | Complete | 53.3% | 10.7% | 71.2% | 234/300 | 68/300 | 84 min / 892 |
| on | 2 | Complete | 56.7% | 11.1% | 74.2% | 232/300 | 63/300 | 127 min / 583 |
| off | 1 | Complete | 36.7% | 3.0% | 34.2% | 239/300 | 64/300 | 39 min / 787 |
| off | 2 | Complete | 36.7% | 4.8% | 32.1% | 238/300 | 55/300 | 51 min / 515 |

Accuracy uses the original benchmark extractor; **unparsed answers count as incorrect**, even if a natural-language answer may be right. Compliance averages eight formal checks; ignore-question compliance and instruction-discussion exclusion are unassessed. The ignore-question prompt deliberately omits the answer-first instruction. Time excludes startup.

# Message
This tests the released prompt variant, which changes both instructions and the graded region; it is not an isolated test of tag removal. Read extraction coverage alongside accuracy.

[Per-condition counts (CSV)](dory_answer_first_results.csv) · [Metrics and configurations (JSON)](dory_answer_first_results.json)
