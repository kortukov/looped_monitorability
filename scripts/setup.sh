#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
git submodule update --init --recursive
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/'; exit 1; }
for item in 'pilot:.venv' 'nanbeige:.venv-nanbeige' 'dory:.venv-dory'; do
  name=${item%%:*}
  destination=${item#*:}
  uv venv --python 3.12 --allow-existing "$destination"
  uv pip sync --python "$destination/bin/python" "requirements-$name.lock.txt"
done
uv venv --python 3.12 --allow-existing .venv-report
uv pip install --python .venv-report/bin/python -r requirements-report.txt
