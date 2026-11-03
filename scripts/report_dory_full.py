"""Write completed full-GPQA reports and the recurrence comparison."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
MODES=['baseline','word_suppression','multiple_word_suppression','repeat_sentences',
       'end_of_sentence','lowercase_thinking','meow_between_words','uppercase_thinking',
       'ignore_question','alternating_case']

LABELS=['Baseline','Suppress word','Suppress words','Add string','End sentences','Lowercase',
        'Meow','Uppercase','Ignore question','Alternating case']


def completed(out):
    config=json.loads((out/'config.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    assert (out/'generation_complete.json').exists() and (out/'grading_complete.json').exists()
    assert summary['total_generations']==4414
    for mode in MODES:
        r=summary['per_mode'][mode]
        assert r['n']==config['expected_per_mode'][mode]
        if mode!='baseline':
            assert r['compliance_n']==r['n'] and r['compliance_without_meta_rate'] is not None
    return config,summary


def report(out):
    config,s=completed(out);loops=config['n_recurrent_loops'];groups=s['per_mode']
    reports=ROOT/'reports';figures=reports/'figures';figures.mkdir(exist_ok=True,parents=True)
    stem=f'gpqa_dory_loops{loops}_full25000'
    fig,axes=plt.subplots(1,2,figsize=(15,4.7),layout='constrained')
    for ax,modes,labels,metric,title in [(axes[0],MODES,LABELS,'accuracy','Answer accuracy (%)'),
                                       (axes[1],MODES[1:],LABELS[1:],'compliance_rate','CoT compliance (%)')]:
        ax.bar(labels,[100*groups[m][metric] for m in modes],color='#4c78a8' if metric=='accuracy' else '#72b7b2')
        ax.set_ylim(0,105);ax.set_ylabel(title);ax.tick_params(axis='x',labelrotation=35)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle(f'Dory · Full GPQA · {loops} loops · 25,000-token budget')
    figure=figures/f'{stem}.png';fig.savefig(figure,dpi=180);plt.close(fig)
    table=['| Condition | Accuracy | Compliance | Excluding meta | Capped | Missing answer |',
           '|---|---:|---:|---:|---:|---:|']
    for mode,label in zip(MODES,LABELS):
        r=groups[mode];n=r['n']
        compliance='—' if mode=='baseline' else f"{r['compliant']}/{n} ({r['compliance_rate']:.1%})"
        strict='—' if mode=='baseline' else f"{r['compliance_without_meta_rate']:.1%}"
        table.append(f"| {label} | {r['correct']}/{n} ({r['accuracy']:.1%}) | {compliance} | {strict} | {r['truncated']}/{n} | {r['missing_final_answer']}/{n} |")
    macro=s['macro_compliance_rate'];strict=sum(groups[m]['compliance_without_meta_rate'] for m in MODES[1:])/9
    capped=sum(r['truncated'] for r in groups.values());missing=sum(r['missing_final_answer'] for r in groups.values())
    joint=sum(groups[m]['compliant_and_correct'] for m in MODES[1:])
    text=f'''# Research Question

At {loops} recurrence loops, how accurately does Dory answer full GPQA, and how well does it follow all nine CoTControl constraints under the original local generation settings?

# Setup

`{config['model']}` revision `{config['revision']}`, vLLM {config['packages']['vllm']}; {config['gpu']}. All 445 GPQA rows, with 427 annotated rows for each of the two keyword controls: **4,414 generations**. Original CoTControl local decoding: greedy temperature 0, top-p 1, **25,000 output tokens total per request**, seed 42. BF16 weights/KV, {loops} loops, per-loop KV caching, {config['max_model_len']:,}-token context. Up to {config['max_num_seqs']} active sequences; prefix caching and asynchronous scheduling enabled. Prompts, options, and graders come from the pinned upstream submodule. Row 74 has duplicate matching choices A/B; the upstream answer resolver selects A. No data was removed or rewritten. Full grading follows upstream: ignore-question scores require three compliant verdicts, and the secondary meta-discussion check uses the first 10,000 reasoning characters with a 10,000-token judge completion budget. Raw reasoning is preserved in full.

# Factual Observations

{chr(10).join(table)}

![Full GPQA accuracy and compliance](figures/{figure.name})

Baseline accuracy is **{groups['baseline']['accuracy']:.1%}**. Equal-weight macro compliance across all nine controls is **{macro:.1%}**, or **{strict:.1%}** excluding meta-discussion. The score includes incorrect and capped answers; **{joint}/3,969** controlled generations were both compliant and correct. **{capped}/4,414** outputs reached the cap; **{missing}/4,414** had no extracted final answer and count as incorrect.

Generation took **{s['total_generation_seconds']/3600:.2f} hours**, producing **{s['total_output_tokens']:,} tokens** at **{s['aggregate_tokens_per_second']:.1f} aggregate output tokens/s**. Conditions shared a continuous GPU queue, so no separate per-condition GPU times are claimed. Startup is recorded separately; interrupted generation time and durable output writes are included. CPU/API judging overlapped generation and did not require the GPU to wait. [Summary](../results/{out.name}/summary.json) · [Configuration](../results/{out.name}/config.json) · [Manifest](../results/{out.name}/sample_manifest.json).

# Message

Under the original local budget, this setting achieved **{groups['baseline']['accuracy']:.1%}** baseline accuracy and **{macro:.1%}** nine-control compliance. Interpret formal compliance together with answer coverage and correctness: repetitive or unfinished text can satisfy surface constraints. This is one model, one decode setting and one dataset; recurrence comparisons are descriptive, and the output cap still limits claims about uncapped reasoning.
'''
    destination=reports/f'{stem}.md';destination.write_text(text);print(destination)
    return dict(loops=loops,baseline_accuracy=groups['baseline']['accuracy'],macro_compliance=macro,
                macro_without_meta=strict,truncated=capped,missing_final=missing,
                generation_seconds=s['total_generation_seconds'],output_tokens=s['total_output_tokens'],
                tokens_per_second=s['aggregate_tokens_per_second'])


def comparison():
    rows=[report(ROOT/'results'/f'gpqa_full_dory_loops{n}_25000') for n in [4,2,1]]
    root=ROOT/'reports'
    with (root/'gpqa_dory_full_comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    ordered=sorted(rows,key=lambda r:r['loops'])
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for ax,key,label,scale in zip(axes,['baseline_accuracy','macro_compliance','tokens_per_second'],
                                 ['Baseline accuracy (%)','Nine-control compliance (%)','Output tokens/s'],[100,100,1]):
        ax.plot([r['loops'] for r in ordered],[scale*r[key] for r in ordered],marker='o')
        ax.set(xlabel='Loops',ylabel=label,xticks=[1,2,4]);ax.grid(alpha=.2)
        if scale==100:ax.set_ylim(0,105)
    fig.savefig(root/'figures/gpqa_dory_full_comparison.png',dpi=180);plt.close(fig)
    table=['| Loops | Baseline accuracy | Nine-control compliance | Excluding meta | Capped | Hours | Tokens/s |',
           '|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:table.append(f"| {r['loops']} | {r['baseline_accuracy']:.1%} | {r['macro_compliance']:.1%} | {r['macro_without_meta']:.1%} | {r['truncated']}/4414 | {r['generation_seconds']/3600:.2f} | {r['tokens_per_second']:.1f} |")
    (root/'gpqa_dory_full_comparison.md').write_text(f'''# Research Question

How do full-GPQA accuracy, nine-control compliance, output coverage, and runtime vary across 4, 2, and 1 Dory loops?

# Setup

The same pinned model, full dataset, prompts, target choices, nine constraints, greedy decoding and 25,000-token budget in every setting. Generation ran in the requested **4 → 2 → 1** order on one L40S. BF16 weights/KV and per-loop caching were preserved. See [four-loop](gpqa_dory_loops4_full25000.md), [two-loop](gpqa_dory_loops2_full25000.md), and [one-loop](gpqa_dory_loops1_full25000.md) reports for exact configurations and raw-output links. Inference concurrency was selected by the [pipeline audit](dory_pipeline_audit.md).

# Factual Observations

{chr(10).join(table)}

![Recurrence comparison](figures/gpqa_dory_full_comparison.png)

All **13,242 generations** and required grading are saved. Compliance is the equal-weight mean of all nine controls, regardless of answer correctness. Missing final answers count as incorrect. Runtime reflects actual output lengths as well as inference speed; [the CSV](gpqa_dory_full_comparison.csv) includes token totals. Conditions share a schedule within each setting; runtimes are measured per setting and are not summed per condition.

# Message

These measurements compare recurrence settings with the same dataset and scientific generation settings. Read compliance alongside correctness and coverage, especially for capped or repetitive traces. A single deterministic run cannot establish a general monotonic relationship between loop count and controllability. The fixed output budget and batching-dependent BF16 numerical differences limit comparisons with uncapped or differently scheduled runs.
''')
    print(root/'gpqa_dory_full_comparison.md')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path);p.add_argument('--compare',action='store_true')
    args=p.parse_args()
    if args.compare:comparison()
    elif args.results:report(args.results)
    else:p.error('Provide --results or --compare')

if __name__=='__main__':main()
