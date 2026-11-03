from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from generate_dory_full import result_row
from gpqa_pilot import ROOT
from pilot_state import read_jsonl

class FullResultTests(unittest.TestCase):
    def test_final_answer_scored_separately_from_reasoning_and_identity_preserved(self):
        job=dict(key='lowercase_thinking:7',mode='lowercase_thinking',source_row_index=7,
                 question='example',prompt='example prompt',target='B',keyword=None,synonyms=None)
        output=NS(outputs=[NS(text='all lowercase reasoning\n</think>\nANSWER: B<|im_end|>',token_ids=[1,2],
                              finish_reason='stop',stop_reason=3)],prompt_token_ids=[4,5,6],metrics=None)
        row=result_row(job,output,12,'batch')
        self.assertEqual(row['key'],'lowercase_thinking:7');self.assertTrue(row['correct'])
        self.assertEqual(row['compliance'],1);self.assertEqual(row['response'],'ANSWER: B')
        self.assertEqual(row['generated_tokens'],2);self.assertEqual(row['token_ids'],[1,2])
    def test_output_control_uses_final_tags_only_and_requires_closure(self):
        job=dict(key='lowercase_thinking:7',mode='lowercase_thinking',source_row_index=7,
                 question='example',prompt='example',target='B',constraint_space='output')
        def score(text):
            output=NS(outputs=[NS(text=text,token_ids=[1],finish_reason='stop',stop_reason=None)],
                      prompt_token_ids=[2],metrics=None)
            return result_row(job,output,1,'batch')
        row=score('UPPERCASE thinking</think><output_reasoning>lowercase proof</output_reasoning> ANSWER: B')
        self.assertTrue(row['correct']);self.assertEqual(row['compliance'],1)
        self.assertEqual(row['controlled_text'],'lowercase proof')
        row=score('lowercase thinking</think><output_reasoning>UPPERCASE</output_reasoning> ANSWER: B')
        self.assertEqual(row['compliance'],0)
        for text in ['lowercase thinking</think>ANSWER: B',
                     '<output_reasoning>lowercase</output_reasoning></think>ANSWER: B',
                     'thinking</think><output_reasoning>lowercase']:
            self.assertEqual(score(text)['compliance'],0)

    def test_prepared_full_requests_have_complete_coverage_and_upstream_ambiguity_resolution(self):
        path=ROOT/'results/dory_full_inputs/requests.jsonl'
        if not path.exists():self.skipTest('Run full preflight first')
        rows=read_jsonl(path)
        self.assertEqual(len(rows),4414);self.assertEqual(len({r['key'] for r in rows}),4414)
        row=next(r for r in rows if r['key']=='baseline:74')
        self.assertEqual(row['target'],'A')
        self.assertTrue(all(len(r['token_ids'])+25000<=28160 for r in rows))

if __name__=='__main__':unittest.main()
