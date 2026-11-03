"""Concise pilot report with auditable condition counts and explicit pending rows."""
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
FILES=['reports/dory_pilot_results.md','reports/dory_pilot_results.csv','reports/dory_pilot_results.json']
SPACES=['output','output_no_thinking','reasoning']
LABELS={'output':'Output, thinking on','output_no_thinking':'Output, thinking off','reasoning':'Thinking trace'}


def build():
    entries=[]
    for space in SPACES:
        for loops in [1,2,4]:
            out=BASE/f'{space}_loops{loops}'
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
            config_path=out/'config.json'
            config=json.loads(config_path.read_text()) if config_path.exists() else None
            entries.append(dict(space=space,loops=loops,status='complete' if done else 'running' if rows or any(s['status']=='running' for s in states) else 'pending',
                n=len(rows),baseline_accuracy=base['correct']/base['n'] if base else None,
                formal_macro_compliance=statistics.mean(scored) if scored else None,
                scored_constraint_modes=len(scored),capped=sum(r['finish_reason']=='length' for r in rows),
                generation_seconds=seconds,output_tokens_per_second=sum(r['generated_tokens'] for r in rows)/seconds if seconds else None,
                conditions=conditions,config=config))
    snapshot=dict(updated_utc=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        dataset='GPQA',n_questions=30,seed=42,max_generated_tokens=25000,entries=entries)
    write_json(ROOT/FILES[2],snapshot)
    stream=io.StringIO();fields=['setup','loops','status','condition','n','correct','compliance_n','compliant','jointly_correct_compliant','capped']
    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
    for e in entries:
        for mode,c in e['conditions'].items():writer.writerow(dict(setup=e['space'],loops=e['loops'],status=e['status'],condition=mode,**c))
    (ROOT/FILES[1]).write_text(stream.getvalue())
    text=f'''# Research Question
How do loop count and disabling thinking affect Dory's accuracy and control over its output or thinking trace?

# Setup
Same 30 GPQA questions (seed 42), baseline + nine constraints per setup. Dory BF16, pinned revision `f6c832e`; original greedy decoding and **25,000-token limit**, one L40S. “Thinking off” closes `<think></think>` before generation; output explanations are still requested. One-loop runs follow the 4-loop thinking-trace run.

# Factual Observations
Updated **{snapshot['updated_utc']}**. Scores below are shown only for completed settings.

| Setup | Loops | Status | Baseline accuracy | Formal compliance | Capped | Time | Tokens/s |
|---|---:|---|---:|---:|---:|---:|---:|
'''
    for e in entries:
        status='Complete' if e['status']=='complete' else f"Running ({e['n']}/300)" if e['status']=='running' else 'Pending'
        if e['status']=='complete':
            text+=f"| {LABELS[e['space']]} | {e['loops']} | {status} | {100*e['baseline_accuracy']:.1f}% | {100*e['formal_macro_compliance']:.1f}% | {e['capped']}/300 | {e['generation_seconds']/60:.0f} min | {e['output_tokens_per_second']:.0f} |\n"
        else:text+=f"| {LABELS[e['space']]} | {e['loops']} | {status} | — | — | — | — | — |\n"
    text+='''
Compliance averages **eight code-scored constraints**, including incorrect/capped answers. Ignore-question compliance and instruction-discussion checks are unassessed; no LLM judge is used. Output scores apply inside `<output_reasoning>` tags. Missing final answers count as incorrect. Time excludes model startup.

# Message
Completed 2/4-loop results show much higher formal compliance in written output than in the thinking trace. Disabling thinking made output runs about **3.4× faster**, but reduced formal compliance from about **69–70% to 19–20%**. The table adds one-loop comparisons as they finish. This is a small pilot, and truncation limits conclusions.

[Per-constraint counts (CSV)](dory_pilot_results.csv) · [Metrics and configurations (JSON)](dory_pilot_results.json)
'''
    (ROOT/FILES[0]).write_text(text)
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
