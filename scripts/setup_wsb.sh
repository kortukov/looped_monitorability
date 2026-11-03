#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
git submodule update --init third_party/workspace-bench
uv venv --python 3.12 --allow-existing .venv-wsb-vllm
uv pip sync --python .venv-wsb-vllm/bin/python requirements-pilot.lock.txt
uv venv --python 3.12 --allow-existing .venv-wsb-judge
uv pip sync --python .venv-wsb-judge/bin/python requirements-wsb-judge.lock.txt
if [ "${1:-}" = "--dory" ]; then
  uv venv --python 3.12 --allow-existing .venv-wsb-dory
  uv pip sync --python .venv-wsb-dory/bin/python requirements-dory.lock.txt
elif [ "$#" -gt 0 ]; then
  echo 'Usage: bash scripts/setup_wsb.sh [--dory]' >&2
  exit 2
fi
.venv-wsb-judge/bin/python - <<'PY'
import hashlib
import json
import tarfile
from pathlib import Path

root = Path.cwd()
bundle = root / "artifacts/wsb_workspace/input/wsb-agree-100.tar.gz"
provenance = json.loads((bundle.parent / "provenance.json").read_text())
assert hashlib.sha256(bundle.read_bytes()).hexdigest() == provenance["archive_sha256"]
destination = root / ".deps"
destination.mkdir(exist_ok=True)
with tarfile.open(bundle) as archive:
    archive.extractall(destination, filter="data")
for name, expected in provenance["files"].items():
    assert hashlib.sha256((destination / "wsb-agree-100" / name).read_bytes()).hexdigest() == expected, name
print("Dataset bundle verified and extracted to .deps/wsb-agree-100")
PY
