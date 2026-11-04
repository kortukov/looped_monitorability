"""Exact upstream simple/answer-first prompts, 30 seed-42 questions, 1/2 loops, thinking on/off."""
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
from dory_pilot_progress import BASE,build,publish

DEST=BASE/'answer_first'
TASKS=[(thinking,n) for thinking in ['thinking_off','thinking_on'] for n in [1,2]]


def prepare_inputs():
    from gpqa_pilot import DORY,DORY_REVISION,MODES,prepare,create_prompt_for_sample,format_messages
    from run_cceval import convert_answer_to_letter
    from transformers import AutoTokenizer
    from vllm.transformers_utils.config import get_config
    tok=AutoTokenizer.from_pretrained(DORY,revision=DORY_REVISION,trust_remote_code=True,
        config=get_config(DORY,revision=DORY_REVISION,trust_remote_code=True))
    old=json.loads((BASE/'output_inputs/sample_manifest.json').read_text())
    for thinking in ['thinking_on','thinking_off']:
        dest=DEST/f'{thinking}_inputs'
        _,samples,selected=prepare(SimpleNamespace(output=dest,n=30,seed=42,full_dataset=False))
        assert selected==old['source_row_indices']
        enabled=thinking=='thinking_on';jobs=[]
        for i in selected:
            for mode in MODES:
                sample=samples[mode][i]
                system,user=create_prompt_for_sample(sample,mode,DORY,False,'none',True,False,False)
                prompt=tok.apply_chat_template(format_messages(user,system),tokenize=False,add_generation_prompt=True,enable_thinking=enabled)
                assert prompt.endswith('<think>\n' if enabled else '<think></think>')
                assert '<output_reasoning>' not in user
                assert ('State your answer, and then explain the reasoning behind it.' in user)==(mode not in ['baseline','ignore_question'])
                target=convert_answer_to_letter(sample['correct_answer'],sample['options'],sample['question'])
                jobs.append(dict(key=f'{mode}:{i}',source_row_index=i,mode=mode,question=sample['question'],
                    prompt=user,system_prompt=system,token_ids=tok.encode(prompt,add_special_tokens=False),target=target,
                    keyword=sample.get('keyword'),synonyms=sample.get('synonyms'),constraint_space='output_plain',enable_thinking=enabled))
        content=''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs);p=dest/'requests.jsonl'
        if p.exists():assert p.read_text()==content
        else:p.write_text(content)
        maximum=max(len(j['token_ids']) for j in jobs);assert maximum+25000<=28160
        write_json(dest/'plan.json',dict(model=DORY,revision=DORY_REVISION,per_setting_generations=300,
            modes=MODES,per_mode_questions={m:30 for m in MODES},max_tokens=25000,max_model_len=28160,
            maximum_prompt_tokens=maximum,constraint_space='output_plain',enable_thinking=enabled,
            prompt_variant='upstream_output_no_letter_no_grading',no_llm_judge=True,answer_key_ambiguities={},
            request_sha256=hashlib.sha256(content.encode()).hexdigest()))
        print('Prepared',thinking,len(jobs),'answer-first prompts',flush=True)


def worker(args):
    from generate_dory_full import generate
    install_signal_handlers()
    generate(SimpleNamespace(inputs=DEST/f'{args.thinking}_inputs',output=DEST/f'{args.thinking}_loops{args.loops}',
        loops=args.loops,max_num_seqs=32))


def supervise():
    DEST.mkdir(parents=True,exist_ok=True);install_signal_handlers()
    own=(DEST/'supervisor.lock').open('a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=(BASE/'supervisor.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(pid=os.getpid(),phase='starting',order=TASKS,publication_errors=[])
    path=DEST/'state.json';child=None;last_push=0;last_build=0
    def update(force=False):
        nonlocal last_push,last_build
        now=time.monotonic()
        if force or now-last_push>=900:
            try:build();publish();state['last_push_unix']=time.time()
            except Exception as exc:state['publication_errors'].append(dict(time_unix=time.time(),error=str(exc)));print(exc,flush=True)
            last_push=now;last_build=now
        elif now-last_build>=120:build();last_build=now
        state['updated_at_unix']=time.time();write_json(path,state)
    try:
        update(force=True)
        for thinking,loops in TASKS:
            if pause_requested(BASE) or pause_requested(DEST):state['phase']='paused';return
            out=DEST/f'{thinking}_loops{loops}';out.mkdir(exist_ok=True)
            if (out/'generation_complete.json').exists():continue
            state.update(phase='generating',thinking=thinking,loops=loops);write_json(path,state)
            with (out/'run.log').open('a') as log:
                child=subprocess.Popen([sys.executable,'-u',__file__,'worker','--thinking',thinking,'--loops',str(loops)],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                state['generation_pid']=child.pid;write_json(path,state)
                while child.poll() is None:
                    if pause_requested(BASE) or pause_requested(DEST):child.terminate();child.wait(timeout=180);state['phase']='paused';return
                    update();time.sleep(5)
                if child.returncode or not (out/'generation_complete.json').exists():
                    state.update(phase='paused' if child.returncode==75 else 'failed',returncode=child.returncode);return
            state['generation_pid']=None;update(force=True)
        state['phase']='complete';update(force=True)
    finally:
        if child and child.poll() is None:child.terminate();child.wait(timeout=180)
        state['generation_pid']=None;write_json(path,state);build()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','worker'])
    p.add_argument('--thinking',choices=['thinking_on','thinking_off']);p.add_argument('--loops',type=int,choices=[1,2])
    a=p.parse_args();os.chdir(ROOT)
    if a.action=='prepare':prepare_inputs()
    elif a.action=='worker':worker(a)
    else:supervise()
