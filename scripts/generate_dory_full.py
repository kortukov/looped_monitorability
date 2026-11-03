"""Continuous full-GPQA GPU generation, with an independent durable writer."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
import subprocess
import time
import uuid

from gpqa_pilot import ROOT, DORY, DORY_REVISION, compliance, split_thinking, LLMResponse, extract_answer_from_response
from pilot_state import read_jsonl,read_results,save_result,write_json,install_signal_handlers,pause_requested,append_jsonl
from full_summary import summarize_full,metric_row


def result_row(job,output,elapsed,batch_id):
    completion=output.outputs[0]
    raw=completion.text
    if job.get('enable_thinking',True):
        reasoning,response,finished=split_thinking(raw)
    elif '</think>' in raw or '<think>' in raw:
        reasoning,response,finished=split_thinking(raw)
    else:
        reasoning,response,finished='',raw,True
    response=response.replace('<|im_end|>','').strip()
    answer=extract_answer_from_response(LLMResponse(content=response))
    scope=job.get('constraint_space','reasoning')
    controlled=reasoning
    tags_present=None
    if scope=='output' and job['mode']!='baseline':
        match=re.search(r'<output_reasoning>(.*?)</output_reasoning>',response,re.DOTALL)
        tags_present=match is not None
        controlled=match.group(1).strip() if match else ''
    score=compliance(job['mode'],controlled,job)
    if scope=='output' and job['mode'] not in ('baseline','ignore_question') and not tags_present:
        score=0
    metrics=output.metrics
    finished_time=getattr(metrics,'finished_time',None)
    request_seconds=finished_time-metrics.arrival_time if finished_time is not None else elapsed
    return dict(key=job['key'],source_row_index=job['source_row_index'],mode=job['mode'],
                question=job['question'],prompt=job['prompt'],target=job['target'],
                answer=answer,correct=answer==job['target'],reasoning=reasoning,response=response,
                raw_text=raw,token_ids=list(completion.token_ids),thinking_finished=finished,
                finish_reason=completion.finish_reason,stop_reason=completion.stop_reason,
                prompt_tokens=len(output.prompt_token_ids),generated_tokens=len(completion.token_ids),
                reasoning_words=len(reasoning.split()),compliance=score,
                constraint_space=scope,controlled_text=controlled,output_reasoning_tags_present=tags_present,
                batch_seconds=elapsed,request_seconds=request_seconds,batch_id=batch_id)


def generate(args):
    from vllm import LLM,SamplingParams
    inputs=getattr(args,'inputs',None) or ROOT/'results/dory_full_inputs'
    plan=json.loads((inputs/'plan.json').read_text())
    data=(inputs/'requests.jsonl').read_bytes()
    assert hashlib.sha256(data).hexdigest()==plan['request_sha256']
    jobs=read_jsonl(inputs/'requests.jsonl')
    expected=plan.get('per_setting_generations',4414)
    assert len(jobs)==expected
    out=args.output or ROOT/'results'/f'gpqa_full_dory_loops{args.loops}_25000'
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((inputs/'sample_manifest.json').read_text())
    if (out/'sample_manifest.json').exists():assert json.loads((out/'sample_manifest.json').read_text())==manifest
    write_json(out/'sample_manifest.json',manifest)
    config=dict(model=DORY,revision=DORY_REVISION,n_recurrent_loops=args.loops,
                recurrent_kv_cache_mode='per_loop',seed=42,temperature=0,top_p=1,
                max_tokens=25000,max_model_len=plan['max_model_len'],dtype='bfloat16',kv_cache_dtype='auto',
                max_num_seqs=args.max_num_seqs,admission_window=2*args.max_num_seqs,
                max_num_batched_tokens=8192,gpu_memory_utilization=.95,
                enable_prefix_caching=True,async_scheduling=True,enable_thinking=plan.get('enable_thinking',True),
                skip_special_tokens=False,scheduling='mixed_conditions',request_sha256=plan['request_sha256'],
                judge_model=None if plan.get('no_llm_judge') else 'openai/gpt-5-mini',judge_max_completion_tokens=10000,meta_reasoning_character_limit=10000,
                audit_summary_sha256=hashlib.sha256((ROOT/'results/dory_pipeline_audit/summary.json').read_bytes()).hexdigest(),
                modes=plan['modes'],expected_per_mode=plan['per_mode_questions'],expected_generations=expected,
                answer_key_ambiguities=plan['answer_key_ambiguities'],
                packages={p:importlib.metadata.version(p) for p in ['vllm','torch','transformers','pandas']},
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip())
    if 'ablation' in plan:config['ablation']=plan['ablation']
    if 'constraint_space' in plan:
        config.update(constraint_space=plan['constraint_space'],no_llm_judge=True,
                      unassessed=['ignore_question compliance','meta-discussion exclusion'])
    if (out/'config.json').exists():
        old=json.loads((out/'config.json').read_text())
        assert old==config,'Configuration differs; use a different output directory'
    write_json(out/'config.json',config)
    old_rows=read_results(out)
    metrics_rows=[metric_row(r) for r in old_rows]
    completed={r['key'] for r in old_rows}
    assert completed<={j['key'] for j in jobs}
    del old_rows
    if len(completed)==expected:
        summarize_full(out,metrics_rows)
        write_json(out/'generation_complete.json',dict(generations=expected,completed_at_unix=time.time()))
        return
    if pause_requested(out):raise SystemExit(75)
    append_jsonl(out/'launches.jsonl',dict(time_unix=time.time(),config=config))
    start=time.perf_counter()
    llm=LLM(model=DORY,revision=DORY_REVISION,tokenizer_revision=DORY_REVISION,
            trust_remote_code=True,hf_overrides={'n_recurrent_loops':args.loops,'recurrent_kv_cache_mode':'per_loop'},
            dtype='bfloat16',kv_cache_dtype='auto',max_model_len=plan['max_model_len'],
            max_num_seqs=args.max_num_seqs,max_num_batched_tokens=8192,gpu_memory_utilization=.95,
            enable_prefix_caching=True,async_scheduling=True,seed=42,disable_log_stats=False)
    effective=llm.llm_engine.vllm_config.model_config.hf_config
    assert effective.n_recurrent_loops==args.loops and effective.recurrent_kv_cache_mode=='per_loop'
    write_json(out/'effective_model_config.json',effective.to_dict())
    write_json(out/'startup.json',dict(model_startup_seconds=time.perf_counter()-start))
    pending_jobs=iter(j for j in jobs if j['key'] not in completed)
    params=SamplingParams(temperature=0,top_p=1,max_tokens=25000,seed=42,skip_special_tokens=False)
    batch_id=uuid.uuid4().hex;started=time.perf_counter()
    state=dict(batch_id=batch_id,mode='mixed',started_at_unix=time.time(),status='running',
               completed=0,resumed_generations=len(completed),elapsed_seconds=0)
    state_path=out/'batches'/f'{batch_id}.json';write_json(state_path,state)
    active={};futures=deque();exhausted=False
    writer=ThreadPoolExecutor(max_workers=1,thread_name_prefix='durable-results')
    last_state=started;last_summary=started;last_telemetry=started
    def persist(job,output,elapsed):
        row=result_row(job,output,elapsed,batch_id)
        save_result(out,row)
        return metric_row(row)
    def collect(block=False):
        while futures and (block or futures[0].done()):
            r=futures.popleft().result();metrics_rows.append(r)
            state['completed']+=1
            print('SAVED',len(metrics_rows),f'/{expected}',r['key'],r['generated_tokens'],r['finish_reason'],flush=True)
    def fill():
        nonlocal exhausted
        remaining=[]
        while not exhausted and len(active)+len(remaining)<config['admission_window']:
            job=next(pending_jobs,None)
            if job is None:exhausted=True;break
            remaining.append(job)
        if remaining:
            ids=llm.enqueue([{'prompt_token_ids':j['token_ids']} for j in remaining],params,use_tqdm=False)
            states=llm.llm_engine.output_processor.request_states
            for internal,job in zip(ids,remaining):
                external=states[internal].external_req_id
                active[external]=(internal,job)
    try:
        fill()
        while active or not exhausted:
            if pause_requested(out):
                state['status']='paused'
                if active:llm.llm_engine.abort_request([v[0] for v in active.values()],internal=True)
                break
            for output in llm.llm_engine.step():
                if not output.finished:continue
                _,job=active.pop(output.request_id)
                futures.append(writer.submit(persist,job,output,time.perf_counter()-started))
            collect()
            if len(futures)>=64:collect(block=True)
            fill()
            now=time.perf_counter()
            if now-last_state>=5:
                state['elapsed_seconds']=now-started;write_json(state_path,state);last_state=now
            if now-last_telemetry>=30:
                metric={m.name:float(m.value) for m in llm.get_metrics() if hasattr(m,'value')}
                append_jsonl(out/'telemetry.jsonl',dict(seconds=now-started,completed=len(metrics_rows),metrics=metric))
                print('PROGRESS',len(metrics_rows),f'/{expected}',round(now-started,1),'seconds',flush=True)
                last_telemetry=now
            if now-last_summary>=120:
                summarize_full(out,metrics_rows);last_summary=now
        collect(block=True)
        if len(metrics_rows)==expected:state['status']='complete'
    finally:
        writer.shutdown(wait=True)
        state['elapsed_seconds']=time.perf_counter()-started
        if state['status']=='running':state['status']='interrupted'
        write_json(state_path,state)
        summarize_full(out,metrics_rows)
    if state['status']!='complete':raise SystemExit(75)
    write_json(out/'generation_complete.json',dict(generations=expected,completed_at_unix=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--loops',type=int,choices=[4,2,1],required=True)
    p.add_argument('--max-num-seqs',type=int,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--allow-full-run',action='store_true')
    args=p.parse_args()
    if not args.allow_full_run:p.error('Explicit full-run authorization is required')
    install_signal_handlers()
    generate(args)

if __name__=='__main__':main()
