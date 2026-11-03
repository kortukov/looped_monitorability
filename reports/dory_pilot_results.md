# Research Question
How do loop count and disabling thinking affect Dory's accuracy and control over its output or thinking trace?

# Setup
Same 30 GPQA questions (seed 42), baseline + nine constraints per setup. Dory BF16, pinned revision `f6c832e`; original greedy decoding and **25,000-token limit**, one L40S. “Thinking off” closes `<think></think>` before generation; output explanations are still requested. One-loop runs follow the 4-loop thinking-trace run.

# Factual Observations
Updated **2026-11-03 23:17 UTC**. Scores below are shown only for completed settings.

| Setup | Loops | Status | Baseline accuracy | Formal compliance | Capped | Time | Tokens/s |
|---|---:|---|---:|---:|---:|---:|---:|
| Output, thinking on | 1 | Running (91/300) | — | — | — | — | — |
| Output, thinking on | 2 | Complete | 60.0% | 70.0% | 57/300 | 141 min | 584 |
| Output, thinking on | 4 | Complete | 60.0% | 68.8% | 47/300 | 246 min | 328 |
| Output, thinking off | 1 | Pending | — | — | — | — | — |
| Output, thinking off | 2 | Complete | 46.7% | 20.0% | 18/300 | 41 min | 552 |
| Output, thinking off | 4 | Complete | 60.0% | 18.8% | 18/300 | 72 min | 338 |
| Thinking trace | 1 | Pending | — | — | — | — | — |
| Thinking trace | 2 | Complete | 60.0% | 0.0% | 28/300 | 98 min | 639 |
| Thinking trace | 4 | Complete | 60.0% | 0.0% | 32/300 | 175 min | 354 |

Compliance averages **eight code-scored constraints**, including incorrect/capped answers. Ignore-question compliance and instruction-discussion checks are unassessed; no LLM judge is used. Output scores apply inside `<output_reasoning>` tags. Missing final answers count as incorrect. Time excludes model startup.

# Message
Completed 2/4-loop results show much higher formal compliance in written output than in the thinking trace. Disabling thinking made output runs about **3.4× faster**, but reduced formal compliance from about **69–70% to 19–20%**. The table adds one-loop comparisons as they finish. This is a small pilot, and truncation limits conclusions.

[Per-constraint counts (CSV)](dory_pilot_results.csv) · [Metrics and configurations (JSON)](dory_pilot_results.json)
