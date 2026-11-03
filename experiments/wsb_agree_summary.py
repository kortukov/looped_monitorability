"""Summarize recorded WSB gate results; does not run model inference."""

import hashlib
import json
import statistics
import sys
from pathlib import Path

dataset = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
index = json.loads((dataset / "index.json").read_text())
summary = {}
hashes = {}
for tier in ("max", "u85"):
    tags = sorted(t for t, v in index["versions"].items() if v["tier"] == tier)
    records = []
    for tag in tags:
        source = dataset / f"{tag}.jsonl"
        hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in source.read_text().splitlines() if line]
        assert len(rows) == 100
        assert len({(r["family"], r["id"]) for r in rows}) == 100
        assert sum(r["label"] == "pass" for r in rows) == 50
        assert all(sum(r["family"] == f for r in rows) == 25 for f in {r["family"] for r in rows})
        for r in rows:
            count = sum(r["results"].values())
            assert (count >= 2) == (r["label"] == "pass")
            assert r["unanimous"] == (count in (0, 3))
        metrics = {}
        for model in rows[0]["results"]:
            metrics[model] = {
                "gate_pass_pct": sum(r["results"][model] for r in rows),
                "label_agreement_pct": sum(r["results"][model] == (r["label"] == "pass") for r in rows),
            }
            for family in sorted({r["family"] for r in rows}):
                metrics[model][f"{family}_gate_pass_pct"] = 4 * sum(
                    r["results"][model] for r in rows if r["family"] == family
                )
        records.append({"version": tag, "metrics": metrics})
    aggregate = {}
    for model in records[0]["metrics"]:
        aggregate[model] = {}
        for metric in records[0]["metrics"][model]:
            values = [r["metrics"][model][metric] for r in records]
            aggregate[model][metric] = {
                "mean": statistics.mean(values),
                "population_sd": statistics.pstdev(values),
                "min": min(values), "max": max(values),
            }
    summary[tier] = {"per_seed": records, "aggregate": aggregate}
destination.mkdir(parents=True, exist_ok=True)
(destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(destination / "config.json").write_text(json.dumps({
    "dataset": str(dataset), "source_sha256": hashes,
    "spread": "population standard deviation across 10 overlapping dataset seeds per tier",
    "inference_run": False,
}, indent=2) + "\n")
print(f"Verified and summarized {sum(len(v['per_seed']) for v in summary.values())} versions into {destination}")
