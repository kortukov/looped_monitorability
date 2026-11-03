"""Publish measured throughput audit and the selected full-run configuration."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
    audit=ROOT/'results/dory_pipeline_audit'
    summary=json.loads((audit/'summary.json').read_text())
    results=summary['results'];short=[r for r in results if r['name'].startswith('short')]
    long=[r for r in results if r['name'].startswith('long')]
    figures=ROOT/'reports/figures';figures.mkdir(exist_ok=True,parents=True)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for ax,rows,title in [(axes[0],short,'Short requests (1,024 output tokens)'),(axes[1],long,'22,000-token synthetic contexts (180 s)')]:
        bars=ax.bar([str(r['window']) for r in rows],[r['tokens_per_second'] for r in rows],color='#4c78a8')
        ax.bar_label(bars,fmt='%.0f',padding=3);ax.set(xlabel='In-flight request limit',ylabel='Aggregate output tokens/s',title=title)
        ax.set_ylim(0,max(r['tokens_per_second'] for r in rows)*1.2);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(figures/'dory_pipeline_audit.png',dpi=180);plt.close(fig)
    table=['| Workload | Limit | Output tokens | Seconds | Tokens/s | Sampled peak KV | Preemptions |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        table.append(f"| {'Short' if r['name'].startswith('short') else 'Long context'} | {r['window']} | {r['output_tokens']:,} | {r['seconds']:.1f} | {r['tokens_per_second']:.1f} | {r['peak_kv_cache_usage']:.1%} | {r['preemptions']:.0f} |")
    text=f'''# Research Question

How can the existing GPQA pipeline use this single NVIDIA L40S more effectively before full Dory evaluation at 4, 2, then 1 loops?

# Setup

Pinned Dory BF16 model and vLLM 0.33.0, four loops, per-loop BF16 KV caching, 28,160-token context, 95% GPU memory allocation, chunked prefill (8,192-token budget), prefix caching, asynchronous scheduling, and compiled CUDA graphs. The same 32 prepared question/condition requests were used at in-flight limits 8/16/32. Short tests force exactly 1,024 output tokens and ignore EOS. Separate stress tests repeat input token sequences to 22,000 tokens and run for 180 seconds, with a 4,096-token decode ceiling. These are **performance-only tests**, not accuracy/compliance evaluation; no test output is reused in the full run. Startup and warmup are excluded.

The full preflight checked all **4,414 requests per setting**: 445 baseline questions and 445 per control except the two keyword conditions, which have 427 each. The original pinned CoTControl local settings are **25,000 total generated tokens**, temperature 0 and top-p 1 (`run_cceval.py:43,469-470`, `llm.py:491-495`). Maximum prompt length is **3,055 tokens**, leaving the complete output budget within 28,160 tokens. All prepared requests use one frozen tokenizer/model revision and input hash. The source has one duplicate-answer row (74, A/B); upstream's own resolver selects A. No question is excluded or rewritten.

# Factual Observations

{chr(10).join(table)}

![Measured throughput](figures/dory_pipeline_audit.png)

The short-request throughput ratios relative to limit 8 are **{short[1]['tokens_per_second']/short[0]['tokens_per_second']:.2f}×** at 16 and **{short[2]['tokens_per_second']/short[0]['tokens_per_second']:.2f}×** at 32. Long-context throughput is essentially the same at 16 and 32: vLLM admits roughly twelve long requests and queues the rest. No preemptions occurred during these bounded tests. KV values are periodic samples, not a proof of worst-case memory headroom. Timed long-context tests end before requests finish; their throughput includes prefill and partial decode tokens.

The first 64 generated token IDs match the limit-8 run for **{summary['short_prefix_matches_of_32']['short_window16']}/32** requests at limit 16 and **{summary['short_prefix_matches_of_32']['short_window32']}/32** at limit 32. Batching changes BF16 numerical execution and can change greedy text; these configurations are **not bitwise equivalent**, even though temperature, top-p, model weights and precision match. Full runs use the same limit 32 across all loop settings and record execution settings.

The audit also found and fixed: (1) an O(file-size) journal reread for every append, now a one-byte normal-tail check with bounded crash repair; (2) Unicode paragraph separators incorrectly splitting JSONL text records, now read as byte-delimited records; (3) the pilot-only unique-answer assertion that would abort on row 74, now using upstream's resolver; and (4) full-mode CLI output defaults that could select pilot directories, bypassed by dedicated full-run paths. Sixteen tests passed, covering crash recovery, Unicode records, request identity, answer extraction, full input coverage, mixed-condition timing, live-judge retries, and the original full ignore-question/meta-discussion judging protocol.

# Message

Use **one GPU engine with up to 32 active sequences**, letting its KV-aware scheduler reduce concurrency as contexts grow. This improves short-request utilization without claiming a 2.92× speedup for the complete 25,000-token workload. Prefix caching reuses common question prefixes; it cannot eliminate long decode cost ([vLLM prefix caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)). Keep asynchronous scheduling, compilation, and chunked prefill; no quantization or output-budget reduction is introduced ([vLLM tuning guide](https://docs.vllm.ai/en/stable/configuration/optimization/)). Running multiple model copies on one GPU would duplicate weights/cache and compete for the same device, so loop settings stay sequential in the requested **4 → 2 → 1** order.

The full runner uses one continuous queue across all ten conditions to avoid drain gaps, pre-tokenized frozen inputs, a separate durable output writer, and CPU/API grading alongside generation. Per-condition GPU times are not invented for an overlapping schedule; reports use measured per-setting wall time and token throughput. A local supervisor records PIDs, GPU telemetry and state, resumes saved questions on retries, stops on an explicit pause, and writes each report when grading is complete. With **13,242 generations** and a maximum of **331,050,000 output tokens**, this is a multi-day workload on one L40S; completion time depends on actual output lengths and long-context cache pressure. Later preemptions remain possible and are recorded.

[Raw benchmark summary](../results/dory_pipeline_audit/summary.json) · [Benchmark code](../scripts/benchmark_dory_pipeline.py) · [Frozen full-run plan](../results/dory_full_inputs/plan.json) · [Supervisor](../scripts/run_dory_full.py).
'''
    (ROOT/'reports/dory_pipeline_audit.md').write_text(text)
    (audit/'selected_config.json').write_text(json.dumps(dict(max_num_seqs={4:32,2:32,1:32},prefix_caching=True,async_scheduling=True,max_num_batched_tokens=8192,reason='Best short-request throughput; equivalent measured long-context throughput with KV-aware admission, no preemptions in bounded stress tests.'),indent=2)+'\n')
    print(ROOT/'reports/dory_pipeline_audit.md')

if __name__=='__main__':main()
