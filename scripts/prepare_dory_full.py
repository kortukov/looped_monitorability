"""Freeze all full-GPQA requests and the original CoTControl decoding budget."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from gpqa_pilot import (ROOT, DORY, DORY_REVISION, MODES, prepare,
                        create_prompt_for_sample, format_messages)

from run_cceval import convert_answer_to_letter

DEST = ROOT / 'results/dory_full_inputs'


def main():
    from vllm.transformers_utils.config import get_config
    from transformers import AutoTokenizer
    DEST.mkdir(parents=True, exist_ok=True)
    _, samples, selected = prepare(SimpleNamespace(output=DEST, full_dataset=True, seed=42, n=445))
    config = get_config(DORY, trust_remote_code=True, revision=DORY_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(DORY, revision=DORY_REVISION,
                                             trust_remote_code=True, config=config)
    jobs = []
    ambiguities = {}
    for i in selected:
        for mode in MODES:
            if i not in samples[mode]:
                continue
            sample = samples[mode][i]
            system, user = create_prompt_for_sample(sample, mode, DORY, False, 'none', False, False, False)
            prompt = tokenizer.apply_chat_template(format_messages(user, system), tokenize=False,
                                                    add_generation_prompt=True, enable_thinking=True)
            assert prompt.endswith('<think>\n')
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            # Verify pre-tokenization preserves the tokenization used by the old runner.
            assert tokens == tokenizer.encode(prompt, add_special_tokens=True)
            targets = [chr(65+j) for j, option in enumerate(sample['options'])
                       if str(option).strip() == sample['correct_answer'].strip()]
            target = convert_answer_to_letter(sample['correct_answer'], sample['options'], sample['question'])
            if len(targets) != 1:
                ambiguities[str(i)] = dict(exact_matches=targets, upstream_target=target)
            jobs.append(dict(key=f'{mode}:{i}', source_row_index=i, mode=mode,
                             question=sample['question'], prompt=user, system_prompt=system,
                             token_ids=tokens, target=target,
                             keyword=sample.get('keyword'), synonyms=sample.get('synonyms')))
    assert len(jobs) == 4414 and len({j['key'] for j in jobs}) == 4414
    request_file = DEST / 'requests.jsonl'
    content = ''.join(json.dumps(j, ensure_ascii=False)+'\n' for j in jobs)
    if request_file.exists():
        assert request_file.read_text() == content, 'Prepared requests changed'
    else:
        request_file.write_text(content)
    maximum = max(len(j['token_ids']) for j in jobs)
    plan = dict(model=DORY, revision=DORY_REVISION, order=[4, 2, 1],
                per_setting_generations=len(jobs), total_generations=3*len(jobs),
                modes=MODES, per_mode_questions={m:sum(j['mode']==m for j in jobs) for m in MODES},
                max_tokens=25000, temperature=0, top_p=1, seed=42,
                maximum_prompt_tokens=maximum, max_model_len=((maximum+25000+127)//128)*128,
                dtype='bfloat16', kv_cache_dtype='auto', recurrent_kv_cache_mode='per_loop',
                scheduling='question_then_condition', request_sha256=hashlib.sha256(content.encode()).hexdigest(),
                answer_key_ambiguities=ambiguities,
                upstream_local_settings_source='third_party/CoTControl/CoT-Control-QA/run_cceval.py:43,469-470; llm.py:491-495')
    (DEST/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan,indent=2))


if __name__ == '__main__':
    main()
