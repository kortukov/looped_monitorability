# Experiment workflow and reporting preferences

- Complete the current seeded GPQA pilot for ByteDance/Ouro-2.6B-Thinking at
  four recurrence steps. Use 30 questions, seed 42, an unconstrained baseline,
  and the three easy controls: single-keyword suppression, string addition at
  the beginning/end, and lowercase reasoning.
- The 8,192-token generation cap is only for the timing pilot. Afterwards use
  CoTControl's 25,000-token generation budget, with enough additional context
  for the entire prompt. Never silently reduce the budget or quantize the model
  or KV cache to fit memory. Record any such change explicitly.
- Save a short, direct experiment report in `reports/` after the Ouro pilot.
  Before the full Ouro run, run the same 30-question, seed-42, three-easy-control
  pilot on `Nanbeige/Nanbeige4.2-3B`, preserving that model's default architecture,
  and save a separate report. Then run the same pilot on
  `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16` at 1, 2, and 4 loops, with a
  report for each setting and a comparison across loop counts. Include actual
  runtime and throughput in every report. Prefer vLLM when supported.
- Notify the user when all pilots/reports are ready. An email notification was
  explicitly authorized in the conversation if a connected sending integration
  is available; email being unavailable must not block the experiments.
- **Do not start any full-scale evaluation until the user explicitly gives a
  new instruction to start it.** The latest user instruction revoked the earlier
  authorization to start the full Ouro run automatically after the pilots.
- Use this exact report structure for all experiment reports:
  1. **Research Question**: State the question being tested.
  2. **Setup**: State the model/revision, data/sample/seed, conditions, inference
     settings, and compute needed to interpret the results.
  3. **Factual Observations**: Present and describe the plots and tables. Keep
     measured facts separate from interpretation and report truncation/coverage.
  4. **Message**: Explain how the observations answer the research question,
     including the main limits on the answer.
- Keep reports short and to the point. Preserve raw outputs and configuration
  files so every reported number can be traced to an experiment artifact.
- CoTControl is an upstream Git submodule; keep experiment code outside it.

These preferences were explicitly requested by the user on 2026-11-02 and are
project memory for future sessions.

The user chose a fresh start on a new host instead of a portable queue. The local
pilot was stopped; do not claim it completed. Use the direct commands in README.
