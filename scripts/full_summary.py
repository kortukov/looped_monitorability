"""Summaries for a single shared GPU schedule across all conditions."""
import json
import statistics
from collections import defaultdict
from pathlib import Path
from pilot_state import read_jsonl, read_results, write_json


def metric_row(row):
    fields=['key','mode','correct','answer','compliance','generated_tokens','finish_reason',
            'thinking_finished','reasoning_words','request_seconds']
    return {k:row[k] for k in fields}


def summarize_full(out, rows=None):
    out=Path(out)
    rows=read_results(out) if rows is None else list(rows)
    judges={r['key']:r for r in read_jsonl(out/'judgments.jsonl') if not r.get('error')}
    groups=defaultdict(list)
    for row in rows:groups[row['mode']].append(row|judges.get(row['key'],{}))
    per_mode={}
    for mode,batch in groups.items():
        scored=[r for r in batch if r.get('compliance') is not None]
        successful=[r for r in scored if r['compliance']==1]
        strict=[r for r in successful if r.get('no_meta_discussion') is not None]
        per_mode[mode]=dict(n=len(batch),correct=sum(r['correct'] for r in batch),
            accuracy=statistics.mean(r['correct'] for r in batch),compliance_n=len(scored),
            compliant=sum(r['compliance'] for r in scored) if scored else None,
            compliance_rate=statistics.mean(r['compliance'] for r in scored) if scored else None,
            compliant_and_correct=sum(r['correct'] for r in successful) if scored else None,
            no_meta_discussion_n=len(strict),
            compliance_without_meta_rate=sum(r['no_meta_discussion'] for r in strict)/len(batch)
                if len(strict)==len(successful) and len(scored)==len(batch) else None,
            truncated=sum(r['finish_reason']=='length' for r in batch),
            missing_final_answer=sum(r['answer'] is None for r in batch),
            missing_think_end=sum(not r['thinking_finished'] for r in batch),
            total_output_tokens=sum(r['generated_tokens'] for r in batch),
            mean_generated_tokens=statistics.mean(r['generated_tokens'] for r in batch),
            mean_reasoning_words=statistics.mean(r['reasoning_words'] for r in batch))
    states=[json.loads(p.read_text()) for p in (out/'batches').glob('*.json')]
    seconds=sum(s['elapsed_seconds'] for s in states)
    controls=[r for m,r in per_mode.items() if m!='baseline' and r['compliance_rate'] is not None]
    tokens=sum(r['generated_tokens'] for r in rows)
    result=dict(per_mode=per_mode,total_generations=len(rows),total_output_tokens=tokens,
                total_generation_seconds=seconds,aggregate_tokens_per_second=tokens/seconds if seconds else None,
                macro_compliance_rate=statistics.mean(r['compliance_rate'] for r in controls) if controls else None,
                constraint_modes_scored=len(controls),scheduling='mixed_conditions',
                interrupted_batches=sum(s['status'] not in ('complete','running') for s in states),
                notes=['Conditions share one continuous GPU queue; per-condition generation times are not additive or separately measured.',
                       'Generation time includes tokenization/enqueue, checkpointing, and interrupted attempts; startup is separate.',
                       'API judging runs concurrently and does not block generation.',
                       'Missing final answers count as incorrect; compliance includes incorrect and capped traces.'])
    config_path=out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()).get('no_llm_judge'):
        result['notes']=[n for n in result['notes'] if not n.startswith('API judging')]
        result['notes'].append('No LLM judge: ignore-question compliance and meta-discussion exclusion are unassessed; macro compliance covers only scored constraints.')
        for value in per_mode.values():value['compliance_without_meta_rate']=None
    write_json(out/'summary.json',result)
    return result
