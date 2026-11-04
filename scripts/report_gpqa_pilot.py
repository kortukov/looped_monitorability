"""Write the completed pilot report using the user's four-section format."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MODES = ["baseline", "word_suppression", "repeat_sentences", "lowercase_thinking"]
LABELS = ["Baseline", "Suppress keyword", "Add string", "Lowercase"]


def wilson(k, n):
    z = 1.96
    p = k / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return max(0, center-half), min(1, center+half)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/gpqa30_seed42_steps4_pilot8192")
    args = parser.parse_args()
    summary = json.loads((args.results / "summary.json").read_text())
    config = json.loads((args.results / "config.json").read_text())
    is_ouro = config["model"].startswith("ByteDance/Ouro")
    name = "Ouro-2.6B-Thinking" if is_ouro else "Nanbeige4.2-3B"
    loops = config.get("num_loops", 2)
    recurrence = "four recurrence steps" if is_ouro else f"{loops} loop{'s' if loops != 1 else ''}"
    stem = "gpqa_ouro_steps4_pilot" if is_ouro else f"gpqa_nanbeige42_loops{config.get('num_loops', 2)}_pilot"
    if "Dory" in config["model"]:
        name = "Nemotron-Labs-Dory1.0-4B-BF16"
        loops = config["n_recurrent_loops"]
        recurrence = f"{loops} loop{'s' if loops != 1 else ''}"
        stem = f"gpqa_dory_loops{config['n_recurrent_loops']}_pilot"
    modes = summary["per_mode"]
    if any(m not in modes or modes[m]["n"] != 30 for m in MODES):
        raise ValueError("Wait until all 120 pilot generations are saved")
    if any(modes[m]["compliance_n"] != 30 or modes[m]["compliance_without_meta_rate"] is None for m in MODES[1:]):
        raise ValueError("Wait until all compliance/meta-discussion judgments finish")
    reports = ROOT / "reports/archive/timing8192"
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure_path = figures / f"{stem}.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    colors = ["#4c78a8", "#72b7b2", "#f2cf5b", "#f58518"]
    for ax, metric, names, labels, palette in [
        (axes[0], "correct", MODES, LABELS, colors),
        (axes[1], "compliant", MODES[1:], LABELS[1:], colors[1:]),
    ]:
        values = [modes[m][metric]/30 for m in names]
        bounds = [wilson(modes[m][metric], 30) for m in names]
        errors = [[100*max(0,p-lo) for p,(lo,hi) in zip(values,bounds)],
                  [100*max(0,hi-p) for p,(lo,hi) in zip(values,bounds)]]
        bars = ax.bar(labels, [100*p for p in values], color=palette,
                      yerr=errors, capsize=4, error_kw={"linewidth": 1})
        ax.bar_label(bars, labels=[f"{p:.0%}" for p in values], padding=4)
        ax.set_ylim(0, 115)
        ax.set_yticks([0,25,50,75,100])
        ax.set_ylabel("Accuracy (%)" if metric == "correct" else "CoT compliance (%)")
        ax.tick_params(axis="x", labelrotation=18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(f"{name} · GPQA pilot · {recurrence}")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    table = ["| Condition | Accuracy | CoT compliance | Truncated | Minutes |",
             "|---|---:|---:|---:|---:|"]
    for mode, label in zip(MODES, LABELS):
        r = modes[mode]
        control = "—" if mode == "baseline" else f"{r['compliant']}/30 ({r['compliance_rate']:.1%})"
        table.append(f"| {label} | {r['correct']}/30 ({r['accuracy']:.1%}) | {control} | {r['truncated']}/30 | {r['generation_seconds']/60:.1f} |")
    joint = sum(modes[m]["compliant_and_correct"] for m in MODES[1:])
    macro = sum(modes[m]["compliance_rate"] for m in MODES[1:])/3
    strict = sum(modes[m]["compliance_without_meta_rate"] for m in MODES[1:])/3
    total_tokens = sum(modes[m]["mean_generated_tokens"]*30 for m in MODES)
    total_seconds = sum(modes[m]["generation_seconds"] for m in MODES)
    truncated = sum(modes[m]["truncated"] for m in MODES)
    missing = sum(modes[m]["missing_final_answer"] for m in MODES)
    interrupted = sum(modes[m]["interrupted_batches"] for m in MODES)
    startup = json.loads((args.results / "startup.json").read_text())["model_startup_seconds"]
    text = f"""# Research Question

With {recurrence}, how accurately does {name} answer GPQA, and how well does it follow three easy CoT constraints?

# Setup

`{config['model']}` (revision `{config['revision'][:12]}`), vLLM {config['packages']['vllm']}, {config['gpu']}. Thirty questions sampled without replacement from 427 keyword-annotated GPQA rows, seed 42; the same questions in all conditions. Greedy decoding, BF16 weights/cache, {recurrence}, **{config['max_tokens']:,} generated tokens maximum**, {config['max_model_len']:,}-token context, up to {config['max_num_seqs']} active sequences. KV cache mode: {config.get('recurrent_kv_cache_mode', 'model default')}. This is the capped timing pilot, not the 25,000-token full evaluation.

# Factual Observations

{chr(10).join(table)}

![Pilot accuracy and CoT compliance](figures/{figure_path.name})

The figure shows the table's accuracy and compliance rates; error bars are 95% Wilson intervals (30 questions per condition). Compliance is scored regardless of answer correctness. The macro-average over these three easy constraints is **{macro:.1%}**, or **{strict:.1%}** excluding meta-discussion; this is not the paper's nine-constraint aggregate. **{joint}/90** controlled generations are both compliant and correct. **{truncated}/120** generations hit the token limit; **{missing}/120** have no extracted final answer and count as incorrect.

Generation took **{total_seconds/60:.1f} minutes** at **{total_tokens/total_seconds:.1f} aggregate output tokens/s**, excluding model setup ({startup/60:.1f} minutes) and API judging; timing includes {interrupted} interrupted generation batches. [Raw summary](../../../results/{args.results.name}/summary.json) · [Configuration](../../../results/{args.results.name}/config.json).

# Message

Under this pilot budget, {name} answers **{modes['baseline']['accuracy']:.1%}** of the unconstrained questions correctly and achieves **{macro:.1%}** easy-constraint compliance. Formal string/case compliance can be satisfied by repetitive or unfinished text, so it must be read alongside accuracy and answer coverage. These measurements establish the starting point for this compute configuration. The small sample and truncation limit conclusions about uncapped capability; the full GPQA run must restore the 25,000-token budget. Comparisons across recurrence depths use the same sample and are reported separately.
"""
    report = reports / f"{stem}.md"
    report.write_text(text)
    print(report)
    print(figure_path)


if __name__ == "__main__":
    main()
