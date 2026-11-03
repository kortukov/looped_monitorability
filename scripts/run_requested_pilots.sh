#!/usr/bin/env bash
# Run only the five settings requested on 2026-11-02.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
status=0
for setting in nanbeige:1 nanbeige:2 dory:1 dory:2 dory:4; do
  family=${setting%:*}
  loops=${setting#*:}
  if [[ "$family" == nanbeige ]]; then
    model=Nanbeige/Nanbeige4.2-3B
    output="results/gpqa30_seed42_nanbeige42_loops${loops}_pilot8192"
  else
    model=nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16
    output="results/gpqa30_seed42_dory_loops${loops}_pilot8192"
  fi
  mkdir -p "$output"
  echo "$(date -u +%FT%TZ) START $setting"
  if ".venv-${family}/bin/python" -u scripts/gpqa_pilot.py generate --model "$model" --loops "$loops" --max-model-len 16384 > "$output/run.log" 2>&1; then
    echo "$(date -u +%FT%TZ) GENERATED $setting"
    if ".venv-${family}/bin/python" -u scripts/gpqa_pilot.py judge --model "$model" --loops "$loops" > "$output/judge.log" 2>&1; then
      .venv-report/bin/python scripts/report_gpqa_pilot.py --results "$output" || status=1
    else
      echo "$(date -u +%FT%TZ) JUDGE FAILED $setting"
      status=1
    fi
  else
    echo "$(date -u +%FT%TZ) GENERATION FAILED $setting (see $output/run.log)"
    status=1
  fi
done
if [[ "$status" == 0 ]]; then
  .venv-report/bin/python scripts/compare_requested_pilots.py || status=1
fi
exit "$status"
