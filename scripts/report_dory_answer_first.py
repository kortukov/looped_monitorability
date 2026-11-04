"""Separate, concise report for the released no-tag answer-first prompt."""
import csv
from datetime import datetime,timezone
import io
from dory_pilot_progress import BASE,ROOT,summarize_run,write_json

FILES=['reports/dory_answer_first_results.md','reports/dory_answer_first_results.csv','reports/dory_answer_first_results.json']


def build_report():
    entries=[summarize_run(BASE/'answer_first'/f'{t}_loops{n}','output_plain',n,42)|dict(thinking=t)
             for t in ['thinking_on','thinking_off'] for n in [1,2]]
    snapshot=dict(updated_utc=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),entries=entries,
        prompt_variant='upstream output_no_letter_no_grading=True; non_reasoning=False',
        accuracy_extractor='Unmodified upstream ANSWER: [A-Z] regex, otherwise a standalone final A-D; unparsed answers count as incorrect.',
        compliance_scope='Entire final response, including the answer and explanation; no tags required.')
    write_json(ROOT/FILES[2],snapshot)
    out=io.StringIO();w=csv.DictWriter(out,fieldnames=['thinking','loops','status','condition','n','correct','compliance_n','compliant','jointly_correct_compliant','capped'],lineterminator='\n');w.writeheader()
    for e in entries:
        for mode,c in e['conditions'].items():w.writerow(dict(thinking=e['thinking'],loops=e['loops'],status=e['status'],condition=mode,**c))
    (ROOT/FILES[1]).write_text(out.getvalue())
    text=f'''# Research Question
Does the released answer-first prompt improve Dory's output control without output-reasoning tags?

# Setup
Same 30 GPQA questions (seed 42), baseline + nine controls; 1/2 loops, thinking on/off. Original CoTControl `output_no_letter_no_grading` prompts, BF16, greedy decoding, 25,000 generated tokens, one L40S. Controls apply to the **entire final response**, including the answer. [Exact templates](dory_prompt_templates.md#answer-first-without-tags) · [Main report](dory_pilot_results.md).

# Factual Observations
Updated **{snapshot['updated_utc']}**. Queued after the seed-replication and 8-loop runs.

| Thinking | Loops | Status | Baseline accuracy | Controlled accuracy | Formal compliance | Unparsed | Capped | Time / tokens per second |
|---|---:|---|---:|---:|---:|---:|---:|---:|
'''
    for e in entries:
        label=e['thinking'].removeprefix('thinking_')
        if e['status']=='complete':
            controls=[c for m,c in e['conditions'].items() if m!='baseline']
            accuracy=sum(c['correct'] for c in controls)/sum(c['n'] for c in controls)
            text+=f"| {label} | {e['loops']} | Complete | {100*e['baseline_accuracy']:.1f}% | {100*accuracy:.1f}% | {100*e['formal_macro_compliance']:.1f}% | {e['n']-e['answer_parsed']}/300 | {e['capped']}/300 | {e['generation_seconds']/60:.0f} min / {e['output_tokens_per_second']:.0f} |\n"
        else:
            status=f"Running ({e['n']}/300)" if e['status']=='running' else 'Pending'
            text+=f'| {label} | {e["loops"]} | {status} | — | — | — | — | — | — |\n'
    text+='''
Accuracy uses the original benchmark extractor; **unparsed answers count as incorrect**, even if a natural-language answer may be right. Compliance averages eight formal checks; ignore-question compliance and instruction-discussion exclusion are unassessed. The ignore-question prompt deliberately omits the answer-first instruction. Time excludes startup.

# Message
Results are pending. This tests the released prompt variant, which changes both instructions and the graded region; it is not an isolated test of tag removal. Read extraction coverage alongside accuracy.

[Per-condition counts (CSV)](dory_answer_first_results.csv) · [Metrics and configurations (JSON)](dory_answer_first_results.json)
'''
    if any(e['status']=='complete' for e in entries):text=text.replace('Results are pending. ','')
    (ROOT/FILES[0]).write_text(text)
    return snapshot

if __name__=='__main__':build_report()
