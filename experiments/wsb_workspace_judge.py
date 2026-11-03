"""Score generated workspace reports with pinned, unmodified WorkspaceBench graders.

Only the OpenRouter transport is adapted to match the collaborator's Haiku setup.
The generated report is one synthetic cell at (-1, -1); these are not layer readouts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import threading
import time
from pathlib import Path

from wsb_workspace_pilot import FAMILIES, ROOT, SUBJECTS, read_jsonl, write_json

UPSTREAM = ROOT / "third_party/workspace-bench"
sys.path.insert(0, str(UPSTREAM / "src"))


def prepare_banks(out):
    from wsbench import registry

    registry.load_all()
    manifest = json.loads((out / "manifest.json").read_text())
    banks = out / "judge_banks"
    for family in FAMILIES:
        original = json.loads((UPSTREAM / registry.get(family).bank).read_text())
        # Keep the complete frozen option pool: DM draws distractors from other
        # bank items, so trimming the bank to a two-item pilot changes the grader.
        originals = original["items"] if isinstance(original, dict) else original
        items_by_id = {str(it.get("id") or it.get("name")): it for it in originals}
        for row in manifest["items"]:
            if row["family"] == family:
                item = dict(row["item"])
                item.setdefault("name", row["id"])
                item["id"] = row["id"]
                items_by_id[row["id"]] = item
        items = list(items_by_id.values())
        bank = {**original, "items": items} if isinstance(original, dict) else items
        write_json(banks / f"{family}.json", bank)
    model = json.loads((out / "config.json").read_text())["model"]
    subject, _, depth, _, _ = SUBJECTS[model]
    write_json(banks / "report.json", {"model": subject, "n_layers": depth, "read_band": [],
               "families": {family: {"n_items": sum(r["family"] == family for r in manifest["items"])} for family in FAMILIES}})
    return banks


def install_transport(out):
    from wsbench import llm

    lock = threading.Lock()

    async def openrouter_once(client, system, user, *, schema, model, reasoning,
                              temperature, max_tokens, timeout, spend):
        # Same request/parse contract as upstream; Claude rejects temperature.
        # Match collaborator judging: Haiku, medium effort, 8000-token cap.
        started = time.perf_counter()
        response = await client.chat.completions.create(
            timeout=timeout, model=model, max_tokens=8000,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_schema", "json_schema": schema},
            extra_body={"usage": {"include": True}, "provider": {"require_parameters": True},
                        "reasoning": {"effort": "medium"}},
        )
        spend.calls += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            cost = getattr(usage, "cost", None) or (getattr(usage, "model_extra", None) or {}).get("cost")
            spend.usd += float(cost or 0)
            spend.input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            spend.output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        log = {"model": model, "system": system, "user": user, "schema": schema,
               "reasoning": {"effort": "medium"}, "max_tokens": 8000,
               "seconds": time.perf_counter() - started, "response": response.model_dump(mode="json")}
        with lock, (out / "judge_requests.jsonl").open("a") as fh:
            fh.write(json.dumps(log, ensure_ascii=False) + "\n")
        if not response.choices:
            raise ValueError("OpenRouter returned no choices")
        if response.choices[0].finish_reason == "length":
            raise ValueError("Judge response truncated; preserve it and retry upstream")
        return llm._parse_object(response.choices[0].message.content or "")

    llm._openrouter_once = openrouter_once


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/wsb_workspace_pilot")
    ap.add_argument("--judge-model", default="anthropic/claude-haiku-5.5")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.out = args.out.resolve()
    banks = prepare_banks(args.out)
    os.environ["WSB_BANK_ROOT"] = str(banks)
    install_transport(args.out)
    from wsbench.cli import main as wsbench

    provenance = {"judge_model": args.judge_model, "digest_model": args.judge_model,
                  "reasoning_effort": "medium", "max_tokens": 8000,
                  "aggregate": "summary", "summ_layers": "joint", "temperature": None,
                  "upstream_commit": "0b1878987927555a9d88f65c6fc3c14debb76e1e",
                  "upstream_modified": False, "transport": "OpenRouter synchronous, not batch",
                  "sampling": "prose1", "synthetic_coordinates": {"layer": -1, "pos": -1},
                  "activation_read_plan_compliance": False, "scope": "pipeline smoke test",
                  "bank_sha256": {f: hashlib.sha256((banks / f"{f}.json").read_bytes()).hexdigest() for f in FAMILIES},
                  "packages": {p: importlib.metadata.version(p) for p in ("openai", "anthropic", "pydra-config", "pronouncing")}}
    config = args.out / "judge_config.json"
    if config.exists() and json.loads(config.read_text()) != provenance:
        raise ValueError("Judge configuration differs; choose a new output directory")
    write_json(config, provenance)
    started = time.perf_counter()
    results = {}
    for family in FAMILIES:
        ids = [r["id"] for r in read_jsonl(args.out / "generations.jsonl") if r["family"] == family and r["parse_ok"]]
        if not ids:
            results[family] = {"status": "no_parseable_reports"}
            continue
        command = ["judge", f"family={family}", f"readouts={args.out / 'readouts' / (family + '.jsonl')}",
                   f"out={args.out / 'judged' / family}", f"judge_model={args.judge_model}",
                   "layers=-1", "items=" + ",".join(ids), "allow_missing=True", "concurrency=2",
                   f"opts=summ_model={args.judge_model},summ_layers=joint,sampling=prose1",
                   f"dry_run={args.dry_run}"]
        print("SCORING", family, flush=True)
        code = wsbench(command)
        results[family] = {"return_code": code, "parseable_reports": len(ids)}
        result_path = args.out / "judged" / family / "results.json"
        if result_path.exists() and not args.dry_run:
            result = json.loads(result_path.read_text())
            result["config"]["effective_transport"] = {
                "reasoning": {"effort": "medium"}, "max_tokens": 8000,
                "temperature": None, "transport": "OpenRouter synchronous",
                "note": "Overrides the requested effort shown in the unmodified upstream config",
            }
            write_json(result_path, result)
            results[family]["result"] = result
        write_json(args.out / ("judge_dry_run_summary.json" if args.dry_run else "judge_summary.json"),
                   {"wall_seconds": time.perf_counter() - started, "families": results})
        if code:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
