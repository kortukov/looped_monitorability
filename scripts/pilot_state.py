"""Durable per-question checkpoints and pilot state (no GPU imports)."""
import json
import os
import signal
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOP_REQUESTED = False


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def read_jsonl(path):
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        try:
            if line.strip():
                rows.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(lines)-1 or line.endswith(b"\n"):
                raise ValueError(f"Corrupt JSONL record: {path}, line {index+1}")
            # An rsync snapshot can catch the last append in progress. The
            # individual atomic checkpoint remains the source of truth.
    return rows


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        f.seek(0)
        data = f.read()
        if data and not data.endswith(b"\n"):
            start = data.rfind(b"\n") + 1
            try:
                json.loads(data[start:])
                f.seek(0, 2)
                f.write(b"\n")
            except (json.JSONDecodeError, UnicodeDecodeError):
                f.truncate(start)
        f.seek(0, 2)
        f.write((json.dumps(row, ensure_ascii=False) + "\n").encode())
        f.flush()
        os.fsync(f.fileno())


def read_results(out):
    rows = {r["key"]: r for r in read_jsonl(out / "generations.jsonl")}
    for path in sorted((out / "completed_requests").glob("*.json")):
        row = json.loads(path.read_text())
        if "key" in row:
            rows[row["key"]] = row
    return list(rows.values())


def save_result(out, row):
    write_json(out / "completed_requests" / f"{row['mode']}_{row['source_row_index']}.json", row)
    append_jsonl(out / "generations.jsonl", row)


def mode_timing(out, mode, rows):
    batches = {}
    for path in (out / "batches").glob("*.json"):
        state = json.loads(path.read_text())
        if state["mode"] == mode:
            batches[state["batch_id"]] = state
    # Old batch-runner results have no batch IDs; each unique recorded duration
    # represents one original call. New runs use one durable timing per batch.
    legacy = {r["batch_seconds"] for r in rows if not r.get("batch_id")}
    for row in rows:
        batch_id = row.get("batch_id")
        if batch_id and batch_id not in batches:
            batches[batch_id] = {"elapsed_seconds": max(r["batch_seconds"] for r in rows if r.get("batch_id") == batch_id), "status": "snapshot"}
    return sum(legacy) + sum(b["elapsed_seconds"] for b in batches.values()), sum(b["status"] != "complete" for b in batches.values())


def pause_requested(out=None):
    return STOP_REQUESTED or (ROOT / "results/PAUSE").exists() or (out is not None and (out / "PAUSE").exists())


def install_signal_handlers():
    def stop(_signum, _frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)


def consume_batch(llm, prompts, params, out, mode, batch_ids, on_complete):
    batch_id = uuid.uuid4().hex
    started = time.perf_counter()
    state_path = out / "batches" / f"{batch_id}.json"
    state = {"batch_id": batch_id, "mode": mode, "started_at_unix": time.time(),
             "status": "running", "completed": 0, "elapsed_seconds": 0}
    write_json(state_path, state)
    request_ids = llm.enqueue(prompts, params, use_tqdm=False)
    index = dict(zip(request_ids, batch_ids))
    finished = set()
    last_save = started
    try:
        while llm.llm_engine.has_unfinished_requests():
            if pause_requested(out):
                state["status"] = "paused"
                for request_id in request_ids:
                    if request_id not in finished:
                        llm.llm_engine.abort_request(request_id)
                break
            for output in llm.llm_engine.step():
                if not output.finished or output.request_id in finished:
                    continue
                elapsed = time.perf_counter() - started
                on_complete(index[output.request_id], output, elapsed, batch_id)
                finished.add(output.request_id)
                state["completed"] = len(finished)
                print(f"FINISHED {mode}: {len(finished)}/{len(prompts)}, "
                      f"{len(output.outputs[0].token_ids)} tokens, "
                      f"{output.outputs[0].finish_reason}, {elapsed:.1f}s", flush=True)
            if time.perf_counter() - last_save > 5:
                state["elapsed_seconds"] = time.perf_counter() - started
                write_json(state_path, state)
                last_save = time.perf_counter()
        if len(finished) == len(request_ids):
            state["status"] = "complete"
    finally:
        state["elapsed_seconds"] = time.perf_counter() - started
        if state["status"] == "running":
            state["status"] = "interrupted"
        write_json(state_path, state)
    return state["status"] == "complete"
