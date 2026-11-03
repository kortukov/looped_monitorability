"""Reduce durable WSB verdicts, keeping generation failures and judge coverage visible."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from wsb_workspace_pilot import FAMILIES, ROOT, read_jsonl, write_json


def item_verdict(family, row):
    if family == "directed_modulation":
        return row.get("pick") == "gold" if row.get("judged") else None
    if family == "conjunctive_association":
        return None if row.get("pick") in ("api_fail", "unsummarized") else bool(row.get("correct"))
    return row.get("pass")


def summarize(out, dataset):
    manifest = json.loads((out / "manifest.json").read_text())
    generations = {(r["family"], r["id"]): r for r in read_jsonl(out / "generations.jsonl")}
    verdicts, official = {}, {}
    for family in FAMILIES:
        path = out / "judged" / family / "results.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        if result.get("config", {}).get("dry_run"):
            continue
        official[family] = {"value": result["numbers"]["value"], "n_items": result["n_items"],
                            "counts": result["counts"], "pinned_instrument": result["pinned_instrument"]}
        for row in result["rows"]:
            verdicts[(family, row.get("id") or row.get("item"))] = item_verdict(family, row)

    def reduce(rows):
        keys = [(r["family"], r["id"]) for r in rows]
        attempted = [generations[k] for k in keys if k in generations]
        decided = [verdicts[k] for k in keys if verdicts.get(k) is not None]
        return {"expected": len(rows), "generated": len(attempted),
                "parsed": sum(r["parse_ok"] for r in attempted),
                "truncated": sum(r["finish_reason"] == "length" for r in attempted),
                "judged": len(decided), "recovered": sum(decided),
                "recovery_rate_decided": statistics.mean(decided) if decided else None,
                "recovered_fraction_expected": sum(decided) / len(rows) if rows else None}

    rows = manifest["items"]
    result = {"overall": reduce(rows), "official": official,
              "by_family": {f: reduce([r for r in rows if r["family"] == f]) for f in FAMILIES},
              "by_original_label": {label: reduce([r for r in rows if r["label"] == label]) for label in ("pass", "fail")},
              "notes": ["Pipeline pilot, not an estimate across dataset seeds, unless scope explicitly contains all versions.",
                        "Missing generations, parse failures and unavailable judge results remain visible in expected counts.",
                        "Reported recovery is separate from the original answerability gate and final-answer correctness."]}
    if manifest.get("scope"):
        index = json.loads((dataset / "index.json").read_text())
        per_seed = {tag: reduce(read_jsonl(dataset / f"{tag}.jsonl")) for tag in manifest["versions"]}
        result["per_seed"] = per_seed
        result["seed_aggregate"] = {}
        for tier in ("max", "u85"):
            values = [v["recovery_rate_decided"] for tag, v in per_seed.items()
                      if index["versions"][tag]["tier"] == tier and v["judged"] == v["expected"]]
            result["seed_aggregate"][tier] = {"complete_seeds": len(values),
                "mean": statistics.mean(values) if values else None,
                "population_sd": statistics.pstdev(values) if values else None}
    write_json(out / "summary.json", result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/wsb_workspace_pilot_bounded")
    ap.add_argument("--dataset", type=Path, default=ROOT / ".deps/wsb-agree-100")
    args = ap.parse_args()
    print(json.dumps(summarize(args.out, args.dataset), indent=2))


if __name__ == "__main__":
    main()
