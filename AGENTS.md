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

Latest requested pilot matrix (2026-11-02): Nanbeige at 1 and 2 loops; Dory at
1, 2, and 4 loops. This overrides the earlier Nanbeige default-only pilot.
Do not start Ouro or a full-scale run as part of this request.

The requested five-setting pilot matrix completed on 2026-11-02 on an NVIDIA
L40S: 600 generations and 450 grading records. Reports, figures, and a comparison
CSV are in `reports/`; start with `gpqa_requested_pilots_comparison.md`. Raw
outputs, configurations, effective model configs, and timings are in `results/`.
All settings retained BF16 weights/cache and the 8,192-token pilot cap. No Ouro
or full-scale run was started. Full-scale evaluation remains on hold.

Latest authorization (2026-11-02): the user explicitly requested full GPQA on
Dory, baseline plus all nine CoTControl constraints, in **4, 2, 1 loop order**,
after a GPU-efficiency audit. This authorizes those three full Dory runs only;
the earlier hold remains for full Ouro and Nanbeige runs. Preserve original
CoTControl local decoding (25,000 output tokens, temperature 0, top-p 1), BF16
weights/cache and per-loop KV caching. The audit is `reports/dory_pipeline_audit.md`.
The ordered local supervisor is `scripts/run_dory_full.py`; live state is
`results/dory_full_sequence_state.json`, with per-loop outputs under
`results/gpqa_full_dory_loops{4,2,1}_25000`. Reports are produced automatically
when each loop's generation and grading are complete. Do not claim a full run
finished solely because the supervisor launched.

Latest instruction (2026-11-03): the user requested that the full Dory sequence
be paused while they review pilot accuracy/compliance and decide what to do.
`results/PAUSE` is set and the supervisor was sent SIGTERM for graceful shutdown.
Do not remove the pause file, resume, or launch another evaluation until the
user explicitly instructs it. Preserve all saved outputs and grading.

Latest authorization (2026-11-03): run a new 30-question seed-42 Dory pilot at 2 and 4 loops, with the original 25,000-token budget, baseline and all nine constraints in both output and reasoning spaces. Run output-space settings first (2 then 4), then reasoning-space settings (2 then 4). No LLM judge is launched: eight formal compliance scores and accuracy are deterministic; ignore-question compliance and meta-discussion exclusion remain unassessed. Full evaluation remains paused via its per-output PAUSE files. New pilot supervisor: scripts/run_dory_large_pilot.py; state: results/dory_large_pilot25000/state.json.

Latest authorization (2026-11-03): insert output-controllability ablations with reasoning disabled (closed `<think></think>` assistant prefix; `enable_thinking=False`) at 2 then 4 loops immediately after the current output 4-loop setting finishes, before reasoning-space runs. Same 30 questions, seed, baseline plus nine controls, original 25,000-token cap and BF16 inference. Prepared inputs are in `results/dory_large_pilot25000/output_no_thinking_inputs`; separate outputs are `output_no_thinking_loops{2,4}`. `scripts/continue_dory_ablation.py` waits for the existing supervisor to stop at a temporary per-run PAUSE on reasoning_loops2, then removes only its own exact hold and runs the amended sequence. Current GPU work must not be interrupted. Queue status is `results/dory_large_pilot25000/ablation_queue.json`. The new total is 1,800 generations across six settings; full-dataset evaluation remains paused.

Latest authorization (2026-11-03): after reasoning-space 4 loops finishes, run all three pilot setups at 1 loop: output with thinking enabled, output with thinking disabled, then reasoning-space controllability. Same sample, nine constraints plus baseline, seed, 25,000-token cap and BF16 settings. Total requested matrix: 2,700 generations across nine settings. `scripts/continue_dory_one_loop.py` waits on the existing supervisor lock and requires all six 2/4-loop completion markers before starting; status: `results/dory_large_pilot25000/one_loop_queue.json`. Keep the human report concise: `reports/dory_pilot_results.md`, backed by per-condition CSV and configurations/counts JSON. User explicitly authorized pushing these results now and as remaining settings finish. `scripts/dory_pilot_progress.py` publishes only those report artifacts; uses the existing .env GIT_TOKEN through ephemeral askpass without putting it in URLs/logs. Full evaluation remains paused.

Latest authorization (2026-11-04): robustify output/thinking-off pilots at 1 and 2 loops using three question-sampling seeds 42,43,44 (paired across loop counts). Reuse completed seed42 runs; add seed43/44, retaining greedy generation seed42, original prompts/BF16/cache/25k budget. Report mean and sample SD (ddof=1) across three complete seeds; do not invent SD while pending or describe it as a confidence interval. Then run all three setups at 8 loops using sampling seed42; use 16 active sequences at 8 loops to reduce KV pressure (32 otherwise), preserving full context and budget. New queue: scripts/run_dory_extensions.py; state: results/dory_large_pilot25000/extensions_state.json. Replicates live under replicates/seed{43,44}; eight-loop outputs use *_loops8. Four seed-replication runs then three eight-loop runs: 2,100 new generations, 4,800 overall. Publish concise results and pending rows to GitHub now and as each finishes, using scripts/dory_pilot_progress.py. Earlier nine-setting matrix is complete. Full GPQA evaluation remains paused.

Latest authorization (2026-11-04): document exact prompt families and link them from canonical `reports/dory_pilot_results.md`; add the released answer-first/simple output variant (`output_no_letter_no_grading=True`, `non_reasoning=False`) as a separate 30-question seed42 pilot at 1/2 loops, thinking on/off. Preserve exact upstream prompts, including the ignore-question exception; grade the ENTIRE final response, including answer, without requiring output_reasoning tags. Preserve the upstream answer extractor and explicitly report unparsed answers; no LLM judge. 1,200 additional generations. The new answer-first report is `reports/dory_answer_first_results.md`; prompt reference is `reports/dory_prompt_templates.md` plus JSON. It follows all existing extension/8-loop runs. `scripts/continue_dory_answer_first.py` temporarily holds the current extension parent with SIGSTOP (GPU worker continues), resumes/terminates only the parent at the current generation completion boundary, restarts extensions with updated reporting, then runs `scripts/run_dory_answer_first.py`. Handoff state is `results/dory_large_pilot25000/documentation_handoff.json`. Do not strand a held supervisor; finish/maintain this handoff. Current reports are indexed by reports/README.md. Historical timing reports and superseded snapshots were moved under reports/archive; raw results and source submodule were preserved. The old report function now delegates to the canonical concise report. Continue pushing report updates.
