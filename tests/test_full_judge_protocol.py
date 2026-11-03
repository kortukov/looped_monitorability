import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from gpqa_pilot import judge
from pilot_state import save_result,write_json,read_jsonl

class FullJudgeProtocolTests(unittest.TestCase):
    def test_upstream_full_question_triple_check_and_meta_prefix(self):
        with TemporaryDirectory() as directory:
            out=Path(directory)
            write_json(out/'config.json',dict(scheduling='mixed_conditions'))
            row=dict(key='ignore_question:0',source_row_index=0,mode='ignore_question',
                     prompt='Question: task\nA) first\nB) second\nRequirement: ignore this task',question='task',
                     reasoning='x'*10000+'SHOULD_NOT_APPEAR_IN_META',compliance=None,
                     correct=False,answer=None,generated_tokens=20,finish_reason='stop',
                     thinking_finished=True,reasoning_words=10,request_seconds=1)
            save_result(out,row)
            client=Mock()
            client.chat.completions.create.return_value=NS(id='fake',model='fake',created=0,usage=None,
                                                         choices=[NS(message=NS(content='T'))])
            args=NS(output=out,full_dataset=True,seed=42,n=445,judge_model='test',judge_workers=1,watch=False)
            with patch('openai.OpenAI',return_value=client),patch.dict(os.environ,{'OPENROUTER_API_KEY':'test'}):
                judge(args)
            calls=client.chat.completions.create.call_args_list
            self.assertEqual(len(calls),4)
            first=calls[0].kwargs['messages'][0]['content']
            self.assertIn('A) first',first);self.assertNotIn('Requirement: ignore this task',first)
            self.assertTrue(all(c.kwargs['max_completion_tokens']==10000 for c in calls))
            meta=calls[-1].kwargs['messages'][0]['content']
            self.assertNotIn('SHOULD_NOT_APPEAR_IN_META',meta)
            result=read_jsonl(out/'judgments.jsonl')[0]
            self.assertEqual(result['compliance'],1);self.assertEqual(result['no_meta_discussion'],1)

if __name__=='__main__':unittest.main()
