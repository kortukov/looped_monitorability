"""Concise pilot results, with paired sampling-seed replication and sample SD."""
import argparse
import csv
from datetime import datetime,timezone
import io
import json
import os
import sys
import tempfile
from pathlib import Path
import statistics
import subprocess
from pilot_state import ROOT,read_results,write_json

BASE=ROOT/'results/dory_large_pilot25000'
FILES=['reports/dory_pilot_results.md','reports/dory_pilot_results.csv','reports/dory_pilot_results.json',
       'reports/dory_answer_first_results.md','reports/dory_answer_first_results.csv','reports/dory_answer_first_results.json']
SPACES=['output','output_no_thinking','reasoning']
LABELS={'output':'Output, thinking on','output_no_thinking':'Output, thinking off','reasoning':'Thinking trace'}
SAMPLING_SEEDS=[42,43,44]


def summarize_run(out,space,loops,seed):
    rows=read_results(out)
    done=(out/'generation_complete.json').exists()
    states=[json.loads(p.read_text()) for p in (out/'batches').glob('*.json')]
    seconds=sum(s['elapsed_seconds'] for s in states)
    conditions={}
    for mode in sorted({r['mode'] for r in rows}):
        group=[r for r in rows if r['mode']==mode]
        scored=[r for r in group if r.get('compliance') is not None]
        conditions[mode]=dict(n=len(group),correct=sum(r['correct'] for r in group),
           compliance_n=len(scored),compliant=sum(r['compliance'] for r in scored) if scored else None,
           jointly_correct_compliant=sum(r['correct'] and r['compliance']==1 for r in scored) if scored else None,
           capped=sum(r['finish_reason']=='length' for r in group))
    scored=[c['compliant']/c['compliance_n'] for m,c in conditions.items() if m!='baseline' and c['compliance_n']]
    base=conditions.get('baseline',{})
    config_path=out/'config.json';manifest=out/'sample_manifest.json'
    if not manifest.exists():manifest=out.parent/f'{space}_inputs/sample_manifest.json'
    return dict(space=space,loops=loops,sampling_seed=seed,
        status='complete' if done else 'running' if rows or any(s['status']=='running' for s in states) else 'pending',
        n=len(rows),answer_parsed=sum(r.get('answer') is not None for r in rows),baseline_accuracy=base['correct']/base['n'] if base else None,
        formal_macro_compliance=statistics.mean(scored) if scored else None,
        scored_constraint_modes=len(scored),capped=sum(r['finish_reason']=='length' for r in rows),
        generation_seconds=seconds,total_output_tokens=sum(r['generated_tokens'] for r in rows),
        output_tokens_per_second=sum(r['generated_tokens'] for r in rows)/seconds if seconds else None,
        conditions=conditions,config=json.loads(config_path.read_text()) if config_path.exists() else None,
        sample_manifest=json.loads(manifest.read_text()) if manifest.exists() else None)


def aggregate_seeds(runs):
    complete=[r for r in runs if r['status']=='complete']
    result=dict(seeds=SAMPLING_SEEDS,completed_seeds=[r['sampling_seed'] for r in complete],
                status='complete' if len(complete)==3 else 'pending',runs=runs)
    # Do not report an SD from fewer than the three requested complete seeds.
    if len(complete)!=3:return result
    for metric in ['baseline_accuracy','formal_macro_compliance']:
        values=[r[metric] for r in complete]
        result[metric]=dict(mean=statistics.mean(values),sample_std=statistics.stdev(values))
    result['conditions']={}
    for mode in complete[0]['conditions']:
        cs=[r['conditions'][mode] for r in complete]
        stats={}
        for label,values in [('accuracy',[c['correct']/c['n'] for c in cs]),
                             ('compliance',[c['compliant']/c['compliance_n'] for c in cs if c['compliance_n']])]:
            stats[label]=dict(mean=statistics.mean(values),sample_std=statistics.stdev(values)) if len(values)==3 else None
        result['conditions'][mode]=stats
    result['capped']=sum(r['capped'] for r in complete)
    result['generation_seconds']=sum(r['generation_seconds'] for r in complete)
    result['output_tokens_per_second']=sum(r['total_output_tokens'] for r in complete)/result['generation_seconds'] if result['generation_seconds'] else None
    return result


def format_mean_sd(stats):
    return f"{100*stats['mean']:.1f}% (± {100*stats['sample_std']:.1f})"


def build():
    entries=[summarize_run(BASE/f'{s}_loops{n}',s,n,42) for s in SPACES for n in [1,2,4,8]]
    robustness={}
    for loops in [1,2,4,8]:
        seed42=next(e for e in entries if e['space']=='output_no_thinking' and e['loops']==loops)
        runs=[seed42]+[summarize_run(BASE/'replicates'/f'seed{seed}'/f'output_no_thinking_loops{loops}',
                                    'output_no_thinking',loops,seed) for seed in [43,44]]
        robustness[str(loops)]=aggregate_seeds(runs)
    snapshot=dict(updated_utc=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        dataset='GPQA',n_questions_per_seed=30,default_sampling_seed=42,generation_seed=42,
        sampling_seeds=SAMPLING_SEEDS,max_generated_tokens=25000,entries=entries,
        output_thinking_off_three_seed=robustness,sd_definition='sample SD across three question-sampling seeds (ddof=1); not a confidence interval')
    write_json(ROOT/FILES[2],snapshot)
    stream=io.StringIO();fields=['setup','loops','sampling_seed','status','condition','n','correct','compliance_n','compliant','jointly_correct_compliant','capped']
    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
    all_runs=entries+[r for a in robustness.values() for r in a['runs'] if r['sampling_seed']!=42]
    for e in all_runs:
        for mode,c in e['conditions'].items():writer.writerow(dict(setup=e['space'],loops=e['loops'],sampling_seed=e['sampling_seed'],status=e['status'],condition=mode,**c))
    (ROOT/FILES[1]).write_text(stream.getvalue())
    text=f'''# Research Question
How do loop count and disabling thinking affect Dory's accuracy and controllability?

# Setup
[Exact prompt templates](dory_prompt_templates.md) · [Answer-first comparison](dory_answer_first_results.md) · [Report index](README.md)

30 GPQA questions per sampling seed; baseline + nine constraints. Dory BF16 (`f6c832e`), original greedy decoding, **25,000-token limit**, one L40S (32 active sequences; 16 at 8 loops). Output “thinking off” uses a closed `<think></think>` prefix. Its **1/2/4/8-loop rows use sampling seeds 42, 43, 44**, paired across loop counts; other rows use seed 42. Generation seed stays 42.

# Factual Observations
Updated **{snapshot['updated_utc']}**. Three-seed scores show **mean (± sample SD, percentage points)** only once all three seeds finish; while pending, * marks the original seed-42 score. SD measures variation across question samples, not a confidence interval.

| Setup | Loops | Status | Baseline accuracy | Formal compliance | Capped | Time | Tokens/s |
|---|---:|---|---:|---:|---:|---:|---:|
'''
    for e in entries:
        a=robustness.get(str(e['loops'])) if e['space']=='output_no_thinking' else None
        if a and a['status']=='complete':
            text+=f"| {LABELS[e['space']]} | {e['loops']} | Complete, 3 seeds | {format_mean_sd(a['baseline_accuracy'])} | {format_mean_sd(a['formal_macro_compliance'])} | {a['capped']}/900 | {a['generation_seconds']/60:.0f} min | {a['output_tokens_per_second']:.0f} |\n"
        elif e['status']=='complete':
            status=f"{len(a['completed_seeds'])}/3 seeds; pending" if a else 'Complete'
            mark='*' if a else ''
            text+=f"| {LABELS[e['space']]} | {e['loops']} | {status} | {100*e['baseline_accuracy']:.1f}%{mark} | {100*e['formal_macro_compliance']:.1f}%{mark} | {e['capped']}/300{mark} | {e['generation_seconds']/60:.0f} min{mark} | {e['output_tokens_per_second']:.0f}{mark} |\n"
        else:
            status=f"Running ({e['n']}/300)" if e['status']=='running' else 'Pending'
            text+=f"| {LABELS[e['space']]} | {e['loops']} | {status} | — | — | — | — | — |\n"
    text+='''
Formal compliance averages **eight code-scored constraints**, including incorrect/capped answers. Ignore-question compliance and instruction-discussion checks are unassessed. Output checks require `<output_reasoning>` tags; missing final answers count as incorrect. Time excludes model startup; three-seed time/caps pool all three runs.

# Message
The completed seed-42 runs show much higher formal compliance in written output than in the thinking trace. The extra seeds test whether the thinking-off results across loop counts hold across question samples; 8-loop runs extend the loop comparison. This remains a small pilot with truncation.

[Per-seed constraint counts (CSV)](dory_pilot_results.csv) · [Metrics, SDs, and configurations (JSON)](dory_pilot_results.json)
'''
    (ROOT/FILES[0]).write_text(text)
    from report_dory_answer_first import build_report
    build_report()
    return snapshot


def publish():
    # Restrict publication to these three report artifacts; leave other work untouched.
    subprocess.run(['git','add','--',*FILES],cwd=ROOT,check=True)
    diff=subprocess.run(['git','diff','--cached','--quiet','--',*FILES],cwd=ROOT)
    if diff.returncode==1:
        subprocess.run(['git','diff','--cached','--check','--',*FILES],cwd=ROOT,check=True)
        subprocess.run(['git','commit','--only','-m','Update concise Dory pilot results','--',*FILES],cwd=ROOT,check=True)
    elif diff.returncode!=0:raise RuntimeError('Could not inspect report changes')
    from dotenv import dotenv_values
    token=(os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN') or
           dotenv_values(ROOT/'.env').get('GIT_TOKEN'))
    env=os.environ.copy();env['GIT_TERMINAL_PROMPT']='0'
    if token:
        # Pass credentials via askpass environment, never via URL, arguments, or logs.
        with tempfile.TemporaryDirectory(prefix='dory-git-') as directory:
            helper=Path(directory)/'askpass'
            helper.write_text('#!'+sys.executable+'\nimport os,sys\nprint("x-access-token" if "username" in sys.argv[1].lower() else os.environ["DORY_PUSH_TOKEN"])\n')
            helper.chmod(0o700)
            env.update(GIT_ASKPASS=str(helper),DORY_PUSH_TOKEN=token)
            subprocess.run(['git','-c','credential.helper=','-c','credential.https://github.com.helper=','push','origin','main'],cwd=ROOT,env=env,check=True,timeout=90)
    else:
        subprocess.run(['git','push','origin','main'],cwd=ROOT,env=env,check=True,timeout=90)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--push',action='store_true');a=p.parse_args()
    s=build();print([(e['space'],e['loops'],e['status'],e['n']) for e in s['entries']],flush=True)
    if a.push:publish()
