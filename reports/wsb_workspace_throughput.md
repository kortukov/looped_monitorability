# Workspace report GPU throughput pilot

## Research Question

How much can vLLM continuous batching and CUDA graph replay accelerate the eight-item workspace-report pilot on the available RTX 4090?

## Setup

Same eight `seed_0` items, selection seed 42, 59-word prompt, Ouro-1.4B revision `eac7e5f31a34b8cdee88c059b840294f2cfc6645`, four steps, greedy decoding, BF16 weights/cache, no quantization, 25,000 output-token allowance and 25,216-token context. Keep the report/final character bounds from the [pipeline pilot](wsb_workspace_pilot.md). vLLM 0.26.0, Torch 2.11.0, Transformers 5.22.1, one RTX 4090 with 24 GB.

The serial condition submits one request at a time. Both concurrent conditions enqueue **all eight requests together** and let vLLM schedule them, with `max_num_seqs=8`. This is an active-sequence ceiling, not fixed groups of eight. Eight was a conservative pilot choice because there are eight requests; no search established an optimal ceiling. The final condition additionally captures decode CUDA graphs (`mode=0`, `FULL_DECODE_ONLY`), without enabling Inductor compilation. Request completions are saved immediately and aggregate throughput uses batch wall time.

## Factual Observations

| Execution | Generation seconds | Output tokens | Tokens/s | Model startup seconds |
|---|---:|---:|---:|---:|
| Serial, eager | 74.25 | 1,489 | 20.05 | 9.35 |
| All requests queued, ceiling 8, eager | 40.30 | 1,493 | 37.04 | 9.38 |
| All requests queued, ceiling 8, decode graphs | 13.18 | 1,493 | 113.30 | 10.28 |

All three conditions completed and parsed eight of eight items, with zero token-cap truncations. Token IDs matched for every item between the two concurrent conditions. The serial and concurrent completions differ on some items, so their workload is closely matched rather than token-identical. Graph replay improved generation time **3.06x** relative to concurrent eager execution on identical generated tokens; concurrent graphs were **5.63x** faster than the original serial pilot by generation wall time. Including model startup, those totals are 23.46 versus 83.61 seconds, a **3.56x** reduction. These times exclude judge calls and dependency installation.

Seven requests completed within 3.57 seconds with graph replay; the remaining poetry request finished at 13.18 seconds. Capture took about one second and the engine reported 0.08 GiB of graph memory. An initial concurrent attempt failed before saving any completions because vLLM returns internal IDs from enqueue and external IDs in outputs; the adapter now resolves that mapping explicitly. That failed attempt is preserved and excluded from successful-run timing.

Raw [benchmark artifacts](../artifacts/wsb_workspace/throughput/) preserve settings, manifests, all completions/token IDs, startup and batch timings, and logs. The original scored pilot remains separate.

The concurrent graph outputs also completed all four live family graders, with eight decided judgments and no missing readouts. Recovery was 1/2 multihop, 1/2 directed modulation, 1/2 poetry and 0/2 conjunctive association; judging took 46.8 seconds. These are a separate smoke-test result from the serial pilot because some generated texts changed with batching. The poetry conditioning limitation in the pipeline report still applies. No causal interpretation of score changes is supported by two items per family.

## Message

Queueing all requests and enabling decode graph replay substantially accelerates this pilot while retaining its precision, context and output allowance. Use the `--cuda-graphs` path for the next Ouro pilot. The eight-request test is dominated by one long completion and does not establish the best concurrency for the full subset. Keep the full queue available to vLLM and tune its active-sequence ceiling on a representative larger workload, monitoring throughput, KV pressure and preemption. No full evaluation was run for this benchmark.
