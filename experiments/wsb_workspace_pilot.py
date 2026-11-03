"""Generate a small WSB workspace-report pilot with vLLM; keep raw completions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Kernel builders need executables installed beside this environment's Python.
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
FAMILIES = ("multihop", "directed_modulation", "poetry", "conjunctive_association")
MODEL = "ByteDance/Ouro-1.4B"
REVISION = "eac7e5f31a34b8cdee88c059b840294f2cfc6645"
FORMAT_REGEX = r"<workspace_report>[^<\n]{1,2048}</workspace_report>\s*<final>[^<]{1,1024}</final>"
SUBJECTS = {
    MODEL: ("ouro-1.4b", REVISION, 96, {"total_ut_steps": 4, "early_exit_threshold": 1.0}, "0.26.0"),
    "ByteDance/Ouro-2.6B": ("ouro-2.6b", "3d18014259a08ee6fda0827f589bd3d8baed2438", 192,
                           {"total_ut_steps": 4, "early_exit_threshold": 1.0}, "0.26.0"),
    "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16": (
        "dory-4b", "f6c832e6963ddb74f62c75813a3e8c74b1af67b1", 56,
        {"n_recurrent_loops": 2, "recurrent_kv_cache_mode": "per_loop"}, "0.33.0"),
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def parse_sections(text):
    """Require exactly one closed section of each kind, in order; retain errors."""
    report = re.findall(r"<workspace_report>(.*?)</workspace_report>", text, re.S)
    final = re.findall(r"<final>(.*?)</final>", text, re.S)
    ordered = bool(report and final and text.index("</workspace_report>") < text.index("<final>"))
    valid = len(report) == len(final) == 1 and ordered and bool(report[0].strip()) and bool(final[0].strip())
    return {"parse_ok": valid, "workspace_report": report[0].strip() if len(report) == 1 else None,
            "final": final[0].strip() if len(final) == 1 else None}


def task_text(row):
    item, family = row["item"], row["family"]
    if family == "conjunctive_association":
        return item["stimulus"] + "\n\nSummarize the preceding story in one sentence."
    if family in ("multihop", "poetry"):
        return "Complete the following text. In <final>, give only the missing continuation.\n\n" + item["prompt"]
    return item["prompt"]


def prepare(args):
    source = args.dataset / f"{args.version}.jsonl"
    rows = read_jsonl(source)
    rng = random.Random(args.seed)
    selected = []
    if args.scope == "pilot":
        for family in FAMILIES:
            for label in ("pass", "fail"):
                candidates = sorted((r for r in rows if r["family"] == family and r["label"] == label), key=lambda r: r["id"])
                selected.extend(rng.sample(candidates, args.per_stratum))
    else:
        index = json.loads((args.dataset / "index.json").read_text())
        tags = sorted(t for t, v in index["versions"].items() if args.scope == "all" or v["tier"] == args.scope)
        union = {}
        for tag in tags:
            for row in read_jsonl(args.dataset / f"{tag}.jsonl"):
                key = (row["family"], row["id"])
                if key in union and union[key] != row:
                    raise ValueError(f"Conflicting dataset records for {key}")
                union[key] = row
        selected = [union[k] for k in sorted(union)]
    prompt = (ROOT / "experiments/wsb_workspace_report_prompt.txt").read_text().strip()
    manifest = {"version": args.version, "seed": args.seed, "per_stratum": args.per_stratum,
                "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "prompt": prompt, "items": selected}
    if args.scope != "pilot":
        manifest["scope"] = args.scope
        manifest["versions"] = tags
        manifest["version_sha256"] = {t: hashlib.sha256((args.dataset / f"{t}.jsonl").read_bytes()).hexdigest() for t in tags}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Existing manifest differs; choose a new output directory")
    write_json(path, manifest)
    return selected, prompt


def generate(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    rows, instruction = prepare(args)
    subject_key, _, _, overrides, required_vllm = SUBJECTS[args.model]
    if importlib.metadata.version("vllm") != required_vllm:
        raise ValueError(f"{args.model} requires the prepared vLLM {required_vllm} environment")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    prompts = [tokenizer.apply_chat_template(
        [{"role": "system", "content": instruction}, {"role": "user", "content": task_text(r)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    ) for r in rows]
    lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    context = ((max(lengths) + args.max_tokens + 127) // 128) * 128
    config = {"model": args.model, "revision": args.revision, "backend": "vllm",
              **overrides, "enable_thinking": False,
              "dtype": "bfloat16", "kv_cache_dtype": "auto", "quantization": None,
              "temperature": 0.0, "top_p": 1.0, "seed": args.seed,
              "stop": ["</final>"], "include_stop_str_in_output": True,
              "format_constraint": FORMAT_REGEX if args.format == "regex" else None,
              "max_tokens": args.max_tokens, "max_model_len": context,
              "max_num_seqs": 1, "enforce_eager": True,
              "gpu_memory_utilization": args.gpu_memory_utilization,
              "packages": {p: importlib.metadata.version(p) for p in ("vllm", "torch", "transformers")},
              "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True).strip()}
    path = args.out / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Existing configuration differs; choose a new output directory")
    write_json(path, config)
    completed = {(r["family"], r["id"]): r for r in read_jsonl(args.out / "generations.jsonl")}
    pending = [(row, prompt) for row, prompt in zip(rows, prompts) if (row["family"], row["id"]) not in completed]
    if pending:
        started = time.perf_counter()
        llm = LLM(model=args.model, revision=args.revision, tokenizer_revision=args.revision,
                  trust_remote_code=True, dtype="bfloat16", kv_cache_dtype="auto",
                  hf_overrides=overrides,
                  max_model_len=context, max_num_seqs=1, gpu_memory_utilization=args.gpu_memory_utilization,
                  enforce_eager=True, enable_prefix_caching=False, seed=args.seed)
        write_json(args.out / "startup.json", {"model_startup_seconds": time.perf_counter() - started})
        params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens,
                                seed=args.seed, skip_special_tokens=False,
                                stop=["</final>"], include_stop_str_in_output=True,
                                structured_outputs=StructuredOutputsParams(regex=FORMAT_REGEX) if args.format == "regex" else None)
        # Save each completed request immediately; a restart reuses durable results.
        for row, prompt in pending:
            started = time.perf_counter()
            output = llm.generate([prompt], params, use_tqdm=False)[0]
            seconds = time.perf_counter() - started
            completion = output.outputs[0]
            result = {"family": row["family"], "id": row["id"], "label": row["label"],
                      "model_gate_pass": row["results"][subject_key],
                      "rendered_prompt": prompt, "raw_text": completion.text,
                      "prompt_tokens": len(output.prompt_token_ids), "generated_tokens": len(completion.token_ids),
                      "token_ids": completion.token_ids, "finish_reason": completion.finish_reason,
                      "stop_reason": completion.stop_reason, "generation_seconds": seconds,
                      **parse_sections(completion.text)}
            with (args.out / "generations.jsonl").open("a") as fh:
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"{row['family']}/{row['id']}: parse={result['parse_ok']} tokens={result['generated_tokens']} seconds={seconds:.2f}", flush=True)
    export(args.out)


def export(out):
    rows = read_jsonl(out / "generations.jsonl")
    for family in FAMILIES:
        path = out / "readouts" / f"{family}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            for row in rows:
                if row["family"] == family and row["parse_ok"]:
                    # Sentinel coordinates are a transport container, not a claimed activation site.
                    fh.write(json.dumps({"id": row["id"], "layer": -1, "pos": -1,
                                         "samples": [row["workspace_report"]]}, ensure_ascii=False) + "\n")
    seconds = sum(r["generation_seconds"] for r in rows)
    write_json(out / "generation_summary.json", {
        "n": len(rows), "parse_ok": sum(r["parse_ok"] for r in rows),
        "truncated": sum(r["finish_reason"] == "length" for r in rows),
        "generated_tokens": sum(r["generated_tokens"] for r in rows),
        "generation_seconds": seconds,
        "tokens_per_second": sum(r["generated_tokens"] for r in rows) / seconds if seconds else None,
        "readout_coordinates": "layer=-1,pos=-1 denote generated reports, not activation cells",
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("prepare", "generate", "export"))
    ap.add_argument("--dataset", type=Path, default=ROOT / ".deps/wsb-agree-100")
    ap.add_argument("--out", type=Path, default=ROOT / "results/wsb_workspace_pilot")
    ap.add_argument("--version", default="seed_0")
    ap.add_argument("--scope", choices=("pilot", "max", "u85", "all"), default="pilot",
                    help="Non-pilot scopes generate the union once; use only with explicit full-run authorization")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-stratum", type=int, default=1)
    ap.add_argument("--model", choices=tuple(SUBJECTS), default=MODEL)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--max-tokens", type=int, default=25000)
    ap.add_argument("--format", choices=("regex", "free"), default="regex")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.96)
    args = ap.parse_args()
    args.revision = args.revision or SUBJECTS[args.model][1]
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "generate":
        generate(args)
    else:
        export(args.out)


if __name__ == "__main__":
    main()
