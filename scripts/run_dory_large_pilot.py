"""30 shared GPQA questions, output controls first, no API judge."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from gpqa_pilot import ROOT, DORY, DORY_REVISION, MODES, prepare, create_prompt_for_sample, format_messages
from pilot_state import write_json, install_signal_handlers, pause_requested

BASE=ROOT/'results/dory_large_pilot25000'


def prepare_inputs():
    from transformers import AutoTokenizer
    from vllm.transformers_utils.config import get_config
    config=get_config(DORY,trust_remote_code=True,revision=DORY_REVISION)
    tok=AutoTokenizer.from_pretrained(DORY,revision=DORY_REVISION,trust_remote_code=True,config=config)
    old=json.loads((ROOT/'results/gpqa30_seed42_dory_loops2_pilot8192/sample_manifest.json').read_text())
    for space in ['output','output_no_thinking','reasoning']:
        dest=BASE/f'{space}_inputs'
        _,samples,selected=prepare(SimpleNamespace(output=dest,n=30,seed=42,full_dataset=False))
        assert selected==old['source_row_indices']
        jobs=[]
        for i in selected:
            for mode in MODES:
                sample=samples[mode][i]
                system,user=create_prompt_for_sample(sample,mode,DORY,space in ('output','output_no_thinking'),'none',False,False,False)
                prompt=tok.apply_chat_template(format_messages(user,system),tokenize=False,add_generation_prompt=True,enable_thinking=space!='output_no_thinking')
                tokens=tok.encode(prompt,add_special_tokens=False)
                targets=[chr(65+j) for j,opt in enumerate(sample['options']) if str(opt).strip()==sample['correct_answer'].strip()]
                assert len(targets)==1
                assert prompt.endswith('<think></think>' if space=='output_no_thinking' else '<think>\n')
                jobs.append(dict(key=f'{mode}:{i}',source_row_index=i,mode=mode,question=sample['question'],
                                 prompt=user,system_prompt=system,token_ids=tokens,target=targets[0],
                                 keyword=sample.get('keyword'),synonyms=sample.get('synonyms'),constraint_space='output' if space=='output_no_thinking' else space,
                                 **({'enable_thinking':False} if space=='output_no_thinking' else {})))
        content=''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs)
        path=dest/'requests.jsonl'
        if path.exists():assert path.read_text()==content
        else:path.write_text(content)
        maximum=max(len(j['token_ids']) for j in jobs)
        write_json(dest/'plan.json',dict(model=DORY,revision=DORY_REVISION,per_setting_generations=300,
                   modes=MODES,per_mode_questions={m:30 for m in MODES},max_tokens=25000,
                   max_model_len=((maximum+25000+127)//128)*128,maximum_prompt_tokens=maximum,
                   constraint_space='output' if space=='output_no_thinking' else space,no_llm_judge=True,answer_key_ambiguities={},
                   **({'enable_thinking':False,'ablation':'closed_thinking_tags'} if space=='output_no_thinking' else {}),
                   request_sha256=hashlib.sha256(content.encode()).hexdigest()))


def worker(args):
    from generate_dory_full import generate
    install_signal_handlers()
    generate(SimpleNamespace(inputs=BASE/f'{args.space}_inputs',output=BASE/f'{args.space}_loops{args.loops}',
                             loops=args.loops,max_num_seqs=32))


def report():
    rows=[]
    for space in ['output','output_no_thinking','reasoning']:
        for loops in [2,4]:
            out=BASE/f'{space}_loops{loops}'
            if not (out/'summary.json').exists():continue
            s=json.loads((out/'summary.json').read_text())
            for mode,v in s['per_mode'].items():
                c='unassessed' if v['compliance_rate'] is None else f"{100*v['compliance_rate']:.1f}%"
                if mode=='baseline':c='—'
                rows.append(f"| {space} | {loops} | {mode} | {v['n']}/30 | {100*v['accuracy']:.1f}% | {c} | {v['truncated']} |")
    text='''# Research Question
How does Dory perform on 30 shared GPQA questions with nine output-space and reasoning-space controls at 2 and 4 loops, including output controls with thinking disabled?

# Setup
Pinned Dory BF16 revision f6c832e6963ddb74f62c75813a3e8c74b1af67b1; same seed-42 questions as the earlier pilot. Each setting includes baseline plus nine controls (300 generations). Original CoTControl prompts and greedy decoding, temperature 0, top-p 1, 25,000 generated tokens plus full prompt context. BF16 weights/cache, per-loop KV, 32 active sequences, prefix caching and async scheduling, one NVIDIA L40S. Output-space prompts use upstream `non_reasoning=True` while Dory thinking remains enabled; compliance is scored only inside closed `<output_reasoning>` tags in the final output. Missing tags fail structural compliance. The output_no_thinking ablation uses identical output-control prompts with enable_thinking=False, so the assistant prefix ends with <think></think>; generated text is scored as the final response without requiring another closing thinking tag. Order: output 2/4, output_no_thinking 2/4, then reasoning 2/4. Reasoning-space scoring uses the thinking trace. No LLM judge or API calls.

# Factual Observations
Live report; coverage below distinguishes partial from complete conditions. Accuracy counts missing answers as incorrect. Compliance includes incorrect/truncated generations. Ignore-question compliance and exclusion of meta-discussion are unassessed, requiring semantic review. No nine-constraint aggregate is claimed. Per-setting runtime and aggregate token throughput are in the adjacent summary table.

| Space | Loops | Condition | Completed | Accuracy | Formal compliance | Capped |
|---|---:|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'\n\n| Space | Loops | Generation seconds | Output tokens/s |\n|---|---:|---:|---:|\n'
    for space in ['output','output_no_thinking','reasoning']:
        for loops in [2,4]:
            path=BASE/f'{space}_loops{loops}/summary.json'
            if path.exists():
                s=json.loads(path.read_text()); rate=s['aggregate_tokens_per_second']
                text+=f"| {space} | {loops} | {s['total_generation_seconds']:.1f} | {rate:.1f} |\n" if rate else f'| {space} | {loops} | 0 | — |\n'
    text+='\n# Message\nInterpret differences only after checking coverage and truncation. Formal compliance is distinct from semantic compliance and correct problem solving. This pilot cannot establish performance on the full GPQA dataset.\n'
    (ROOT/'reports/dory_large_pilot25000.md').write_text(text)


def supervise():
    BASE.mkdir(parents=True,exist_ok=True)
    lock=(BASE/'supervisor.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    install_signal_handlers()
    state=dict(pid=os.getpid(),phase='starting',started_at_unix=time.time(),order=['output2','output4','output_no_thinking2','output_no_thinking4','reasoning2','reasoning4'],no_llm_judge=True)
    child=None
    try:
        for space in ['output','output_no_thinking','reasoning']:
            for loops in [2,4]:
                if pause_requested(BASE):state['phase']='paused';return
                out=BASE/f'{space}_loops{loops}';out.mkdir(exist_ok=True)
                if (out/'generation_complete.json').exists():continue
                state.update(phase='generating',space=space,loops=loops)
                write_json(BASE/'state.json',state)
                with (out/'run.log').open('a') as log:
                    child=subprocess.Popen([sys.executable,'-u',__file__,'worker','--space',space,'--loops',str(loops)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                    state['generation_pid']=child.pid;write_json(BASE/'state.json',state)
                    last_report=0
                    while child.poll() is None:
                        if time.monotonic()-last_report>120:
                            report();last_report=time.monotonic()
                        if pause_requested(BASE):child.terminate();child.wait(timeout=180);state['phase']='paused';return
                        time.sleep(2)
                    if child.returncode:
                        state.update(phase='paused' if child.returncode==75 else 'failed',returncode=child.returncode);return
                report()
        state['phase']='complete'
    finally:
        if child and child.poll() is None:child.terminate();child.wait(timeout=180)
        state['generation_pid']=None;state['updated_at_unix']=time.time()
        write_json(BASE/'state.json',state);report()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run','worker','report'])
    p.add_argument('--space',choices=['output','output_no_thinking','reasoning'])
    p.add_argument('--loops',type=int,choices=[1,2,4])
    args=p.parse_args()
    os.chdir(ROOT)
    if args.action=='prepare':prepare_inputs()
    elif args.action=='worker':worker(args)
    elif args.action=='report':report()
    else:supervise()
