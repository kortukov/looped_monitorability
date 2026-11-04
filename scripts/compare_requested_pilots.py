"""Compare the completed five-setting seed-42 pilot; never infer missing results."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pilot_state import read_results

ROOT = Path(__file__).resolve().parents[1]
MODES = ['baseline', 'word_suppression', 'repeat_sentences', 'lowercase_thinking']
SETTINGS = [('Nanbeige', n, f'nanbeige42_loops{n}') for n in (1, 2)] + [('Dory', n, f'dory_loops{n}') for n in (1, 2, 4)]


def main():
    records = []
    reference_prompts = None
    for family, loops, suffix in SETTINGS:
        directory = ROOT / 'results' / f'gpqa30_seed42_{suffix}_pilot8192'
        rows = read_results(directory)
        signatures = {r['key']: (r['prompt'], r['target']) for r in rows}
        assert len(signatures) == 120, 'Incomplete or duplicate question coverage'
        if reference_prompts is None:
            reference_prompts = signatures
        else:
            assert signatures == reference_prompts, 'Actual prompts or answer keys differ'
        summary = json.loads((directory / 'summary.json').read_text())
        config = json.loads((directory / 'config.json').read_text())
        manifest = json.loads((directory / 'sample_manifest.json').read_text())
        if records:
            assert manifest == records[0]['manifest'], 'Samples differ between settings'
        modes = summary['per_mode']
        assert summary['total_generations'] == 120
        assert all(modes[m]['n'] == 30 for m in MODES)
        assert all(modes[m]['compliance_without_meta_rate'] is not None for m in MODES[1:])
        seconds = summary['total_generation_seconds']
        tokens = sum(modes[m]['mean_generated_tokens'] * modes[m]['n'] for m in MODES)
        records.append(dict(model=family, loops=loops, baseline_accuracy=modes['baseline']['accuracy'],
                            macro_compliance=summary['macro_compliance_rate'],
                            compliant_and_correct=sum(modes[m]['compliant_and_correct'] for m in MODES[1:]),
                            macro_without_meta=sum(modes[m]['compliance_without_meta_rate'] for m in MODES[1:])/3,
                            truncated=sum(modes[m]['truncated'] for m in MODES),
                            missing_final=sum(modes[m]['missing_final_answer'] for m in MODES),
                            generation_seconds=seconds, total_output_tokens=round(tokens),
                            mean_generated_tokens=tokens/120, tokens_per_second=tokens/seconds,
                            config=config, manifest=manifest, directory=directory.name))
    reports = ROOT / 'reports/archive/timing8192'
    fields = [k for k in records[0] if k not in ('config', 'manifest', 'directory')]
    with (reports / 'gpqa_requested_pilots_comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in records)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout='constrained')
    for family in ('Nanbeige', 'Dory'):
        group = [r for r in records if r['model'] == family]
        for ax, key, label in zip(axes, ('baseline_accuracy', 'macro_compliance', 'tokens_per_second'),
                                  ('Baseline accuracy (%)', 'Three-control compliance (%)', 'Aggregate output tokens/s')):
            scale = 1 if key == 'tokens_per_second' else 100
            ax.plot([r['loops'] for r in group], [scale*r[key] for r in group], marker='o', label=family)
            ax.set(xlabel='Loops', ylabel=label, xticks=[1, 2, 4])
            ax.grid(alpha=.2)
            if scale == 100:
                ax.set_ylim(0, 105)
    axes[0].legend()
    (reports / 'figures').mkdir(exist_ok=True)
    fig.savefig(reports / 'figures/gpqa_requested_pilots_comparison.png', dpi=180)
    plt.close(fig)
    table = ['| Model | Loops | Baseline accuracy | Compliance | Excluding meta | Compliant + correct | Truncated | Missing answer | Minutes | Tokens/s |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        table.append(f"| {r['model']} | {r['loops']} | {r['baseline_accuracy']:.1%} | {r['macro_compliance']:.1%} | {r['macro_without_meta']:.1%} | {r['compliant_and_correct']}/90 | {r['truncated']}/120 | {r['missing_final']}/120 | {r['generation_seconds']/60:.1f} | {r['tokens_per_second']:.1f} |")
    changes = []
    for family in ('Nanbeige', 'Dory'):
        group = [r for r in records if r['model'] == family]
        loop_counts = "/".join(str(r['loops']) for r in group)
        accuracy = "/".join(f"{r['baseline_accuracy']:.1%}" for r in group)
        compliance = "/".join(f"{r['macro_compliance']:.1%}" for r in group)
        changes.append(f"At {loop_counts} loops, {family}'s baseline accuracy was {accuracy}, and formal three-control compliance was {compliance}.")
    missing_all = [r for r in records if r['missing_final'] == 120]
    for r in missing_all:
        changes.append(f"{r['model']} at {r['loops']} loop produced no extracted final answer in any of its 120 generations; its formal compliance score does not demonstrate usable controlled problem solving.")
    dory = {r['loops']: r for r in records if r['model'] == 'Dory'}
    if dory[2]['baseline_accuracy'] == dory[4]['baseline_accuracy']:
        changes.append(f"Dory at two and four loops had equal baseline accuracy on this sample, while full-pilot generation took {dory[2]['generation_seconds']/60:.1f} and {dory[4]['generation_seconds']/60:.1f} minutes respectively.")
    provenance = '\n'.join(f"- [{r['model']}, {r['loops']} loop{'s' if r['loops'] != 1 else ''}](gpqa_{'nanbeige42' if r['model'] == 'Nanbeige' else 'dory'}_loops{r['loops']}_pilot.md): revision `{r['config']['revision']}`, vLLM {r['config']['packages']['vllm']}; [configuration](../../../results/{r['directory']}/config.json), [summary](../../../results/{r['directory']}/summary.json), effective model configuration and per-question outputs alongside them." for r in records)
    text = f'''# Research Question

How do recurrence depth, GPQA accuracy, three-control CoT compliance, and runtime compare for Nanbeige at 1/2 loops and Dory at 1/2/4 loops?

# Setup

The same 30 questions from 427 keyword-annotated GPQA rows, seed 42, in each setting. Each has 30 baseline generations and 30 for each control: keyword suppression, string addition, and lowercase reasoning. Greedy decoding, BF16 weights and KV cache, 8,192 generated tokens, 16,384-token context, up to eight active sequences. Dory uses per-loop KV caching. Hardware: {records[0]['config']['gpu']}. This is the timing pilot, not a full evaluation.

{provenance}

# Factual Observations

{chr(10).join(table)}

![Comparison across loop counts](figures/gpqa_requested_pilots_comparison.png)

Compliance averages the three controls regardless of correctness. Runtime depends on each setting's actual output lengths; the CSV records total and mean generated tokens. Runtime and throughput exclude startup/compilation and judge calls; interrupted generation time is included. Missing final answers count as incorrect. All 600 generations and required meta-discussion judgments are saved. **{sum(r['truncated'] for r in records)}/600 ({sum(r['truncated'] for r in records)/600:.1%})** generations hit the output cap. The figure connects observed settings without interpolating untested settings. [Numeric table](gpqa_requested_pilots_comparison.csv).

# Message

{' '.join(changes)} Formal string/case compliance can be satisfied by repetitive or unfinished text; the joint compliant-and-correct count helps expose this limit. These are descriptive results from one paired sample, not evidence of a general monotonic trend. Thirty questions, a single seed, the token cap, and different architectures and inference implementations limit generalization. Equal loop counts do not imply equal compute across models. The three-control score is not the paper's nine-control aggregate. Full-scale evaluation remains on hold.
'''
    target = reports / 'gpqa_requested_pilots_comparison.md'
    target.write_text(text)
    print(target)


if __name__ == '__main__':
    main()
