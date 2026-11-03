"""Performance-only experiments; outputs never enter evaluation results."""
import hashlib
import json
import time
import uuid
from pathlib import Path
from gpqa_pilot import ROOT, DORY, DORY_REVISION
from pilot_state import read_jsonl

OUT=ROOT/'results/dory_pipeline_audit'

def snapshot(llm):
    return {m.name:float(m.value) for m in llm.get_metrics() if hasattr(m,'value')}


def run_case(llm, jobs, window, output_tokens, name, time_limit=None, long_prompt=None):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    llm.reset_prefix_cache()
    params=SamplingParams(temperature=0,top_p=1,max_tokens=output_tokens,
                          min_tokens=output_tokens,ignore_eos=True,skip_special_tokens=False,
                          output_kind=RequestOutputKind.DELTA,seed=42)
    active={};prefixes={};counts={};completed=0;cursor=0;started=time.perf_counter()
    before=snapshot(llm);peak=0;last=started;samples=[]
    def fill():
        nonlocal cursor
        while cursor<len(jobs) and len(active)<window:
            i=cursor;cursor+=1
            tokens=jobs[i]['token_ids']
            if long_prompt:
                tokens=(tokens*((long_prompt+len(tokens)-1)//len(tokens)))[:long_prompt]
            external=f'audit-{uuid.uuid4().hex}'
            internal=llm.llm_engine.add_request(external,{'type':'token','prompt_token_ids':tokens},params)
            active[external]=internal;counts[external]=0;prefixes[str(i)]=[]
            ids[external]=str(i)
    ids={};fill()
    try:
        while llm.llm_engine.has_unfinished_requests():
            for output in llm.llm_engine.step():
                external=output.request_id
                token_ids=output.outputs[0].token_ids
                counts[external]+=len(token_ids)
                prefix=prefixes[ids[external]]
                prefix.extend(token_ids[:max(0,64-len(prefix))])
                if output.finished:
                    active.pop(external,None);completed+=1
            fill()
            now=time.perf_counter()
            if now-last>=10:
                metric=snapshot(llm)
                peak=max(peak,metric.get('vllm:kv_cache_usage_perc',0))
                sample=dict(seconds=now-started,output_tokens=sum(counts.values()),metrics=metric)
                samples.append(sample)
                print(name,round(now-started,1),'seconds',sum(counts.values()),'tokens',completed,'completed',flush=True)
                last=now
            if time_limit and now-started>=time_limit:break
    finally:
        if active:llm.llm_engine.abort_request(list(active.values()),internal=True)
    elapsed=time.perf_counter()-started;after=snapshot(llm)
    result=dict(name=name,window=window,requests=len(jobs),completed=completed,
                output_tokens=sum(counts.values()),seconds=elapsed,tokens_per_second=sum(counts.values())/elapsed,
                long_prompt_tokens=long_prompt,output_cap=output_tokens,time_limit=time_limit,
                peak_kv_cache_usage=peak,
                preemptions=after.get('vllm:num_preemptions',0)-before.get('vllm:num_preemptions',0),
                prefix_cache_hits=after.get('vllm:prefix_cache_hits',0)-before.get('vllm:prefix_cache_hits',0),
                first_64_tokens=prefixes,metric_samples=samples)
    (OUT/f'{name}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',name,round(result['tokens_per_second'],1),'tokens/s','preemptions',result['preemptions'],flush=True)
    return result


def main():
    from vllm import LLM
    plan=json.loads((ROOT/'results/dory_full_inputs/plan.json').read_text())
    jobs=read_jsonl(ROOT/'results/dory_full_inputs/requests.jsonl')[:32]
    OUT.mkdir(parents=True,exist_ok=True)
    llm=LLM(model=DORY,revision=DORY_REVISION,tokenizer_revision=DORY_REVISION,
            trust_remote_code=True,dtype='bfloat16',kv_cache_dtype='auto',
            hf_overrides={'n_recurrent_loops':4,'recurrent_kv_cache_mode':'per_loop'},
            max_model_len=plan['max_model_len'],max_num_seqs=32,max_num_batched_tokens=8192,
            gpu_memory_utilization=.95,enable_prefix_caching=True,async_scheduling=True,
            seed=42,disable_log_stats=False)
    from dataclasses import asdict
    (OUT/'engine_config.json').write_text(llm.llm_engine.vllm_config.to_json_string() if hasattr(llm.llm_engine.vllm_config,'to_json_string') else json.dumps({'model':DORY,'loops':4,'max_num_seqs':32,'max_model_len':plan['max_model_len'],'prefix_caching':True,'async_scheduling':True},indent=2))
    run_case(llm,jobs[:1],1,16,'warmup')
    results=[]
    for window in (8,16,32):
        results.append(run_case(llm,jobs,window,1024,f'short_window{window}'))
    for window in (16,32):
        results.append(run_case(llm,jobs,window,4096,f'long_window{window}',time_limit=180,long_prompt=22000))
    baseline=results[0]['first_64_tokens']
    comparisons={r['name']:sum(v==baseline.get(k) for k,v in r['first_64_tokens'].items()) for r in results[1:3]}
    (OUT/'summary.json').write_text(json.dumps({'performance_only':True,'short_prefix_matches_of_32':comparisons,'results':[{k:v for k,v in r.items() if k not in ('first_64_tokens','metric_samples')} for r in results]},indent=2)+'\n')


if __name__=='__main__':main()
