# Research Question

With 1 loop, how accurately does Nanbeige4.2-3B answer GPQA, and how well does it follow three easy CoT constraints?

# Setup

`Nanbeige/Nanbeige4.2-3B` (revision `b82e54bd6097`), vLLM 0.30.0, NVIDIA L40S, 46068 MiB, 595.91.07. Thirty questions sampled without replacement from 427 keyword-annotated GPQA rows, seed 42; the same questions in all conditions. Greedy decoding, BF16 weights/cache, 1 loop, **8,192 generated tokens maximum**, 16,384-token context, up to 8 active sequences. KV cache mode: model default. This is the capped timing pilot, not the 25,000-token full evaluation.

# Factual Observations

| Condition | Accuracy | CoT compliance | Truncated | Minutes |
|---|---:|---:|---:|---:|
| Baseline | 0/30 (0.0%) | — | 30/30 | 10.4 |
| Suppress keyword | 0/30 (0.0%) | 25/30 (83.3%) | 30/30 | 8.3 |
| Add string | 0/30 (0.0%) | 0/30 (0.0%) | 30/30 | 8.3 |
| Lowercase | 0/30 (0.0%) | 9/30 (30.0%) | 30/30 | 8.3 |

![Pilot accuracy and CoT compliance](figures/gpqa_nanbeige42_loops1_pilot.png)

The figure shows the table's accuracy and compliance rates; error bars are 95% Wilson intervals (30 questions per condition). Compliance is scored regardless of answer correctness. The macro-average over these three easy constraints is **37.8%**, or **35.6%** excluding meta-discussion; this is not the paper's nine-constraint aggregate. **0/90** controlled generations are both compliant and correct. **120/120** generations hit the token limit; **120/120** have no extracted final answer and count as incorrect.

Generation took **35.3 minutes** at **464.2 aggregate output tokens/s**, excluding model setup (0.6 minutes) and API judging; timing includes 1 interrupted generation batches. [Raw summary](../results/gpqa30_seed42_nanbeige42_loops1_pilot8192/summary.json) · [Configuration](../results/gpqa30_seed42_nanbeige42_loops1_pilot8192/config.json).

# Message

Under this pilot budget, Nanbeige4.2-3B answers **0.0%** of the unconstrained questions correctly and achieves **37.8%** easy-constraint compliance. Formal string/case compliance can be satisfied by repetitive or unfinished text, so it must be read alongside accuracy and answer coverage. These measurements establish the starting point for this compute configuration. The small sample and truncation limit conclusions about uncapped capability; the full GPQA run must restore the 25,000-token budget. Comparisons across recurrence depths use the same sample and are reported separately.
