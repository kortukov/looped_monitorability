"""Three sampling seeds for output thinking-off at 1/2 loops, then seed-42 8-loop pilots."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from pilot_state import ROOT,write_json,install_signal_handlers,pause_requested
from dory_pilot_progress import BASE,SPACES,build,publish

SEEDS=[42,43,44]
TASKS=[('output_no_thinking',n,s) for s in [43,44] for n in [1,2]]+[(s,8,42) for s in SPACES]


def paths(space,loops,seed):
    if seed==42:return BASE/f'{space}_inputs',BASE/f'{space}_loops{loops}'
    base=BASE/'replicates'/f'seed{seed}'
    return base/f'{space}_inputs',base/f'{space}_loops{loops}'


def prepare_inputs():
    from gpqa_pilot import DORY,DORY_REVISION,MODES,prepare,create_prompt_for_sample,format_messages
    from run_cceval import convert_answer_to_letter
    from transformers import AutoTokenizer
    from vllm.transformers_utils.config import get_config
    config=get_config(DORY,trust_remote_code=True,revision=DORY_REVISION)
    tok=AutoTokenizer.from_pretrained(DORY,revision=DORY_REVISION,trust_remote_code=True,config=config)
    original_plan=json.loads((BASE/'output_no_thinking_inputs/plan.json').read_text())
    for seed in [43,44]:
        dest,_=paths('output_no_thinking',1,seed)
        _,samples,selected=prepare(SimpleNamespace(output=dest,n=30,seed=seed,full_dataset=False))
        jobs=[];ambiguities={}
        for i in selected:
            for mode in MODES:
                sample=samples[mode][i]
                system,user=create_prompt_for_sample(sample,mode,DORY,True,'none',False,False,False)
                prompt=tok.apply_chat_template(format_messages(user,system),tokenize=False,add_generation_prompt=True,enable_thinking=False)
                tokens=tok.encode(prompt,add_special_tokens=False)
                assert prompt.endswith('<think></think>')
                target=convert_answer_to_letter(sample['correct_answer'],sample['options'],sample['question'])
                matches=[chr(65+j) for j,opt in enumerate(sample['options']) if str(opt).strip()==sample['correct_answer'].strip()]
                if len(matches)!=1:ambiguities[str(i)]=dict(exact_matches=matches,upstream_target=target)
                jobs.append(dict(key=f'{mode}:{i}',source_row_index=i,mode=mode,question=sample['question'],
                    prompt=user,system_prompt=system,token_ids=tokens,target=target,keyword=sample.get('keyword'),
                    synonyms=sample.get('synonyms'),constraint_space='output',enable_thinking=False))
        content=''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs)
        path=dest/'requests.jsonl'
        if path.exists():assert path.read_text()==content
        else:path.write_text(content)
        plan=dict(original_plan,sampling_seed=seed,maximum_prompt_tokens=max(len(j['token_ids']) for j in jobs),
                  answer_key_ambiguities=ambiguities,request_sha256=hashlib.sha256(content.encode()).hexdigest())
        assert all(len(j['token_ids'])+25000<=plan['max_model_len'] for j in jobs)
        write_json(dest/'plan.json',plan)
        print('Prepared seed',seed,':',len(jobs),'requests;',len(selected),'questions',flush=True)


def worker(args):
    from generate_dory_full import generate
    install_signal_handlers()
    inputs,out=paths(args.space,args.loops,args.sampling_seed)
    generate(SimpleNamespace(inputs=inputs,output=out,loops=args.loops,max_num_seqs=16 if args.loops==8 else 32))


def supervise():
    install_signal_handlers()
    own=(BASE/'extensions.lock').open('a+')
    fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=(BASE/'supervisor.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(pid=os.getpid(),phase='starting',tasks=TASKS,publication_errors=[])
    path=BASE/'extensions_state.json';child=None;last_push=0;last_build=0
    def update(force=False):
        nonlocal last_push,last_build
        now=time.monotonic()
        if force or now-last_push>=900:
            try:build();publish();state['last_push_unix']=time.time()
            except Exception as exc:
                state['publication_errors'].append(dict(time_unix=time.time(),error=str(exc)))
                print('Report push failed; raw results preserved:',exc,flush=True)
            last_push=now;last_build=now
        elif now-last_build>=120:build();last_build=now
        state['updated_at_unix']=time.time();write_json(path,state)
    try:
        update(force=True)
        for space,loops,seed in TASKS:
            if pause_requested(BASE):state['phase']='paused';return
            _,out=paths(space,loops,seed)
            out.mkdir(parents=True,exist_ok=True)
            if (out/'generation_complete.json').exists():continue
            state.update(phase='generating',space=space,loops=loops,sampling_seed=seed)
            write_json(path,state)
            with (out/'run.log').open('a') as log:
                child=subprocess.Popen([sys.executable,'-u',__file__,'worker','--space',space,'--loops',str(loops),
                    '--sampling-seed',str(seed)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                state['generation_pid']=child.pid;write_json(path,state)
                while child.poll() is None:
                    if pause_requested(BASE):child.terminate();child.wait(timeout=180);state['phase']='paused';return
                    update();time.sleep(5)
                if child.returncode or not (out/'generation_complete.json').exists():
                    state.update(phase='paused' if child.returncode==75 else 'failed',returncode=child.returncode);return
            state['generation_pid']=None;update(force=True)
        state['phase']='complete';update(force=True)
    finally:
        if child and child.poll() is None:child.terminate();child.wait(timeout=180)
        state['generation_pid']=None;write_json(path,state);build()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run','worker'])
    p.add_argument('--space',choices=SPACES)
    p.add_argument('--loops',type=int,choices=[1,2,8])
    p.add_argument('--sampling-seed',type=int,choices=SEEDS,default=42)
    a=p.parse_args();os.chdir(ROOT)
    if a.action=='prepare':prepare_inputs()
    elif a.action=='worker':worker(a)
    else:supervise()
