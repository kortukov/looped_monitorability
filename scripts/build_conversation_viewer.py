"""Build an offline HTML viewer for the completed 2-loop output pilot."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/dory_large_pilot25000/output_loops2'
DEST=ROOT/'artifacts/dory_2loop_pilot_viewer.html'
fields=['key','source_row_index','mode','prompt','target','answer','correct','reasoning','response','compliance','thinking_finished','finish_reason','generated_tokens','output_reasoning_tags_present']
rows=[{k:r.get(k) for k in fields} for f in (SOURCE/'completed_requests').glob('*.json') for r in [json.loads(f.read_text())]]
manifest=json.loads((SOURCE/'sample_manifest.json').read_text())
order={i:n for n,i in enumerate(manifest['source_row_indices'])}
rows.sort(key=lambda r:(order[r['source_row_index']],r['mode']))
summary=json.loads((SOURCE/'summary.json').read_text())
assert len(rows)==300 and len({r['key'] for r in rows})==300
payload=json.dumps(dict(rows=rows,summary=summary),ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
html='''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dory · 2-loop GPQA pilot</title>
<style>
:root{font:15px/1.55 system-ui,sans-serif;color:#202a38;background:#f3f5f8;color-scheme:light}*{box-sizing:border-box}body{margin:0}header,main{max-width:1400px;margin:auto;padding:22px}h1{font-size:26px;margin:0}h2{font-size:18px}p{margin:8px 0}.muted{color:#586679;font-size:13px}.cards{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}.card{background:white;border:1px solid #dce2eb;border-radius:10px;padding:14px 20px;flex:1;min-width:150px}.card strong{display:block;font-size:25px}.panel{background:white;border:1px solid #dce2eb;border-radius:10px;padding:18px;margin-bottom:18px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:7px;border-bottom:1px solid #e4e8ef}th{color:#586679}tr[data-mode]{cursor:pointer}tr[data-mode]:hover{background:#edf4ff}.scroll{overflow:auto}.filters{display:flex;gap:10px;flex-wrap:wrap;align-items:center}select,input,button{font:inherit;padding:8px;border:1px solid #bac5d4;border-radius:6px;background:white;color:inherit}input{flex:1;min-width:180px}button{cursor:pointer}button:hover{background:#edf4ff}button:disabled{opacity:.4;cursor:default}.layout{display:grid;grid-template-columns:310px minmax(0,1fr);gap:16px}nav{max-height:75vh;overflow:auto}nav button{display:block;width:100%;text-align:left;margin-bottom:6px;font-size:13px}nav button.active{background:#e7efff;border-color:#557dca}.badge{display:inline-block;border-radius:5px;padding:2px 8px;background:#edf0f5;font-size:12px;margin:3px}.good{background:#dcf5e7;color:#145d38}.bad{background:#fde2e2;color:#912a2a}.toolbar{display:flex;gap:8px;justify-content:space-between;align-items:center}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace;margin:10px 0}details{margin-top:14px;border-top:1px solid #e4e8ef;padding-top:10px}summary{cursor:pointer;font-weight:600}.notice{border-left:3px solid #ad7d24;padding-left:12px}@media(max-width:750px){.layout{grid-template-columns:1fr}nav{max-height:240px}header,main{padding:12px}.card{padding:10px}}
</style>
<header><h1>Dory · 2-loop GPQA pilot</h1><p>Results and conversations · Output-space constraints · Complete offline snapshot</p>
<p class="muted">30 shared GPQA questions, seed 42 · baseline + 9 constraints · 25,000 generated-token cap · temperature 0, top-p 1 · BF16 · NVIDIA L40S</p>
<div class="cards" id="cards"></div>
<div class="panel"><h2>Results by condition</h2><p class="muted">Click a condition to filter the conversations. Every condition contains 30 questions.</p><div class="scroll"><table><thead><tr><th>Condition</th><th>Correct / total</th><th>Accuracy</th><th>Compliant / scored</th><th>Compliance</th><th>Correct + compliant</th><th>Capped</th></tr></thead><tbody id="results"></tbody></table></div>
<p class="muted notice">Compliance applies only inside closed &lt;output_reasoning&gt; tags in the final output. Missing tags fail the eight formal checks. Ignore-question compliance and exclusion of instruction discussion are unassessed: no LLM judge was used. Accuracy includes missing answers as incorrect. Compliance includes incorrect and capped outputs. The 70% macro average covers eight scored constraints, not all nine.</p></div></header>
<main><div class="panel"><div class="filters"><label>Condition <select id="mode"><option value="">All conditions</option></select></label><label>Accuracy <select id="correct"><option value="">All</option><option value="yes">Correct</option><option value="no">Incorrect</option></select></label><label>Compliance <select id="compliance"><option value="">All</option><option value="yes">Pass</option><option value="no">Fail</option><option value="unknown">Unassessed / baseline</option></select></label><label>Finish <select id="finish"><option value="">All</option><option value="length">Token limit</option><option value="stop">Stopped naturally</option></select></label><input id="search" type="search" placeholder="Search prompt, reasoning, or final answer" aria-label="Search conversations"><button id="reset">Reset</button></div><p class="muted" id="count" aria-live="polite"></p></div>
<div class="layout"><nav id="list" aria-label="Conversations"></nav><article class="panel" id="conversation"></article></div>
<p class="muted">Snapshot source: results/dory_large_pilot25000/output_loops2. All 300 saved conversations are embedded; no network, installation, or server is needed. Displayed text is rendered as plain text.</p></main>
<script id="data" type="application/json">PAYLOAD</script>
<script>
'use strict';
const {rows,summary:stats}=JSON.parse(document.getElementById('data').textContent);
const names={baseline:'Baseline',word_suppression:'Word suppression',multiple_word_suppression:'Multiple-word suppression',repeat_sentences:'Repeated sentence',end_of_sentence:'Sentence-ending word',lowercase_thinking:'Lowercase',meow_between_words:'Meow between words',uppercase_thinking:'Uppercase',ignore_question:'Ignore question',alternating_case:'Alternating case'};
const $=id=>document.getElementById(id),pct=x=>x==null?'—':(100*x).toFixed(1)+'%';
function el(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;}
const base=stats.per_mode.baseline;
for(const [label,value] of [['Conversations',rows.length+'/300'],['Baseline accuracy',pct(base.accuracy)],['Formal compliance · 8 controls',pct(stats.macro_compliance_rate)],['Capped outputs',rows.filter(r=>r.finish_reason==='length').length+'/300'],['Generation time',(stats.total_generation_seconds/3600).toFixed(2)+' h']]){const c=el('div',undefined,'card');c.append(el('strong',value),el('span',label));$('cards').append(c);}
for(const [mode,name] of Object.entries(names)){
 const option=el('option',name);option.value=mode;$('mode').append(option);
 const s=stats.per_mode[mode],tr=el('tr');tr.dataset.mode=mode;tr.tabIndex=0;tr.setAttribute('aria-label','Filter '+name);
 const c=s.compliance_n?`${s.compliant}/${s.compliance_n}`:mode==='baseline'?'—':'Unassessed';
 for(const v of [name,`${s.correct}/${s.n}`,pct(s.accuracy),c,pct(s.compliance_rate),s.compliance_n?`${s.compliant_and_correct}/${s.n}`:'—',`${s.truncated}/${s.n}`])tr.append(el('td',v));
 tr.onclick=()=>{$('mode').value=mode;filter();$('mode').scrollIntoView({behavior:'smooth',block:'center'});};tr.onkeydown=e=>{if(e.key==='Enter')tr.click();};$('results').append(tr);
}
let filtered=[],selected=null;
function badge(text,kind){return el('span',text,'badge '+(kind||''));}
function show(r){
 selected=r.key;const out=$('conversation');out.replaceChildren();
 const bar=el('div',undefined,'toolbar'),title=el('h2',`${names[r.mode]} · source row ${r.source_row_index}`),buttons=el('div');
 const i=filtered.indexOf(r);for(const [label,offset] of [['Previous',-1],['Next',1]]){const b=el('button',label);b.disabled=!filtered[i+offset];b.onclick=()=>{show(filtered[i+offset]);renderList();};buttons.append(b);}bar.append(title,buttons);out.append(bar);
 out.append(badge(r.correct?'Correct':'Incorrect',r.correct?'good':'bad'),badge(`Answer: ${r.answer||'missing'} · expected: ${r.target}`),badge(r.compliance==null?(r.mode==='baseline'?'Compliance: not applicable':'Compliance: unassessed'):`Compliance: ${r.compliance?'pass':'fail'}`,r.compliance==null?'':r.compliance?'good':'bad'),badge(`${r.generated_tokens.toLocaleString()} tokens`),badge(r.finish_reason==='length'?'Token limit reached':'Stopped naturally',r.finish_reason==='length'?'bad':''));
 if(!r.thinking_finished)out.append(el('p','Thinking did not finish; no final answer was extracted.','notice'));
 if(r.mode!=='baseline')out.append(el('p',r.output_reasoning_tags_present?'Closed output_reasoning tags present in the final response.':'No closed output_reasoning tags in the final response.','muted'));
 for(const [title,text,open] of [['User prompt (question, choices, and constraint)',r.prompt,true],['Model thinking',r.reasoning,false],['Final response',r.response,true]]){const d=el('details');d.open=open;d.append(el('summary',title),el('pre',text||'(empty)'));out.append(d);}
}
function renderList(){const nav=$('list');nav.replaceChildren();for(const r of filtered){const b=el('button',`Row ${r.source_row_index} · ${names[r.mode]}\n${r.correct?'Correct':'Incorrect'} · ${r.compliance==null?'Unscored':r.compliance?'Compliant':'Noncompliant'}`,r.key===selected?'active':'');b.onclick=()=>{show(r);renderList();};nav.append(b);}}
function filter(){const mode=$('mode').value,c=$('correct').value,k=$('compliance').value,f=$('finish').value,q=$('search').value.trim().toLowerCase();filtered=rows.filter(r=>(!mode||r.mode===mode)&&(!c||r.correct===(c==='yes'))&&(!k||(k==='unknown'?r.compliance==null:r.compliance===(k==='yes'?1:0)))&&(!f||r.finish_reason===f)&&(!q||[r.prompt,r.reasoning,r.response].some(t=>(t||'').toLowerCase().includes(q))));$('count').textContent=`${filtered.length} of ${rows.length} conversations shown`;if(filtered.length)show(filtered.find(r=>r.key===selected)||filtered[0]);else $('conversation').replaceChildren(el('p','No conversations match these filters.'));renderList();}
for(const id of ['mode','correct','compliance','finish'])$(id).onchange=filter;
let debounce;$('search').oninput=()=>{clearTimeout(debounce);debounce=setTimeout(filter,200);};$('reset').onclick=()=>{for(const id of ['mode','correct','compliance','finish','search'])$(id).value='';filter();};$('mode').value='baseline';filter();
</script></html>'''
DEST.parent.mkdir(exist_ok=True)
DEST.write_text(html.replace('PAYLOAD',payload))
print(f'{DEST}: {DEST.stat().st_size/1024/1024:.1f} MiB; {len(rows)} conversations')
