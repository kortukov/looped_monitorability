# Research Question

With 2 loops, how accurately does Nanbeige4.2-3B answer GPQA, and how well does it follow three easy CoT constraints?

# Setup

`Nanbeige/Nanbeige4.2-3B` (revision `b82e54bd6097`), vLLM 0.30.0, NVIDIA L40S, 46068 MiB, 595.91.07. Thirty questions sampled without replacement from 427 keyword-annotated GPQA rows, seed 42; the same questions in all conditions. Greedy decoding, BF16 weights/cache, 2 loops, **8,192 generated tokens maximum**, 16,384-token context, up to 8 active sequences. KV cache mode: model default. This is the capped timing pilot, not the 25,000-token full evaluation.

# Factual Observations

| Condition | Accuracy | CoT compliance | Truncated | Minutes |
|---|---:|---:|---:|---:|
| Baseline | 12/30 (40.0%) | — | 16/30 | 12.8 |
| Suppress keyword | 14/30 (46.7%) | 0/30 (0.0%) | 13/30 | 12.3 |
| Add string | 11/30 (36.7%) | 0/30 (0.0%) | 16/30 | 14.0 |
| Lowercase | 9/30 (30.0%) | 0/30 (0.0%) | 17/30 | 13.6 |

![Pilot accuracy and CoT compliance](figures/gpqa_nanbeige42_loops2_pilot.png)

The figure shows the table's accuracy and compliance rates; error bars are 95% Wilson intervals (30 questions per condition). Compliance is scored regardless of answer correctness. The macro-average over these three easy constraints is **0.0%**, or **0.0%** excluding meta-discussion; this is not the paper's nine-constraint aggregate. **0/90** controlled generations are both compliant and correct. **62/120** generations hit the token limit; **62/120** have no extracted final answer and count as incorrect.

Generation took **52.7 minutes** at **251.1 aggregate output tokens/s**, excluding model setup (0.6 minutes) and API judging; timing includes 0 interrupted generation batches. [Raw summary](../results/gpqa30_seed42_nanbeige42_loops2_pilot8192/summary.json) · [Configuration](../results/gpqa30_seed42_nanbeige42_loops2_pilot8192/config.json).

# Message

Under this pilot budget, Nanbeige4.2-3B answers **40.0%** of the unconstrained questions correctly and achieves **0.0%** easy-constraint compliance. Formal string/case compliance can be satisfied by repetitive or unfinished text, so it must be read alongside accuracy and answer coverage. These measurements establish the starting point for this compute configuration. The small sample and truncation limit conclusions about uncapped capability; the full GPQA run must restore the 25,000-token budget. Comparisons across recurrence depths use the same sample and are reported separately.
