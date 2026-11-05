# Research Question
How do loop count and disabling thinking affect Dory's accuracy and controllability?

# Setup
[Exact prompt templates](dory_prompt_templates.md) · [Answer-first comparison](dory_answer_first_results.md) · [Report index](README.md)

30 GPQA questions per sampling seed; baseline + nine constraints. Dory BF16 (`f6c832e`), original greedy decoding, **25,000-token limit**, one L40S (32 active sequences; 16 at 8 loops). Output “thinking off” uses a closed `<think></think>` prefix. Its **1/2-loop rows use sampling seeds 42, 43, 44**, paired across loop counts; other rows use seed 42. Generation seed stays 42.

# Factual Observations
Updated **2026-11-05 00:06 UTC**. Three-seed scores show **mean (± sample SD, percentage points)** only once all three seeds finish; while pending, * marks the original seed-42 score. SD measures variation across question samples, not a confidence interval.

| Setup | Loops | Status | Baseline accuracy | Formal compliance | Capped | Time | Tokens/s |
|---|---:|---|---:|---:|---:|---:|---:|
| Output, thinking on | 1 | Complete | 60.0% | 68.8% | 56/300 | 93 min | 902 |
| Output, thinking on | 2 | Complete | 60.0% | 70.0% | 57/300 | 141 min | 584 |
| Output, thinking on | 4 | Complete | 60.0% | 68.8% | 47/300 | 246 min | 328 |
| Output, thinking on | 8 | Running (181/300) | — | — | — | — | — |
| Output, thinking off | 1 | Complete, 3 seeds | 46.7% (± 3.3) | 11.8% (± 6.9) | 79/900 | 85 min | 849 |
| Output, thinking off | 2 | Complete, 3 seeds | 50.0% (± 8.8) | 19.9% (± 2.7) | 50/900 | 125 min | 571 |
| Output, thinking off | 4 | Complete | 60.0% | 18.8% | 18/300 | 72 min | 338 |
| Output, thinking off | 8 | Pending | — | — | — | — | — |
| Thinking trace | 1 | Complete | 56.7% | 0.0% | 42/300 | 70 min | 959 |
| Thinking trace | 2 | Complete | 60.0% | 0.0% | 28/300 | 98 min | 639 |
| Thinking trace | 4 | Complete | 60.0% | 0.0% | 32/300 | 175 min | 354 |
| Thinking trace | 8 | Pending | — | — | — | — | — |

Formal compliance averages **eight code-scored constraints**, including incorrect/capped answers. Ignore-question compliance and instruction-discussion checks are unassessed. Output checks require `<output_reasoning>` tags; missing final answers count as incorrect. Time excludes model startup; three-seed time/caps pool all three runs.

# Message
The completed seed-42 runs show much higher formal compliance in written output than in the thinking trace. The extra seeds test whether the thinking-off 1/2-loop results hold across question samples; 8-loop runs extend the loop comparison. This remains a small pilot with truncation.

[Per-seed constraint counts (CSV)](dory_pilot_results.csv) · [Metrics, SDs, and configurations (JSON)](dory_pilot_results.json)
