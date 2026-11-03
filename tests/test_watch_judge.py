import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from gpqa_pilot import watch_judgments
from pilot_state import write_json,save_result,read_jsonl

class WatchJudgeTests(unittest.TestCase):
    def test_watcher_recovers_atomic_output_without_journal_and_retries_failure(self):
        with TemporaryDirectory() as d:
            p=Path(d)
            write_json(p/'config.json',dict(scheduling='mixed_conditions',expected_per_mode={'baseline':1,'lowercase_thinking':2}))
            rows=[]
            for i in [1,2]:
                r=dict(key=f'lowercase_thinking:{i}',source_row_index=i,mode='lowercase_thinking',
                       correct=False,answer=None,compliance=1,generated_tokens=20,
                       finish_reason='length',thinking_finished=False,reasoning_words=10,request_seconds=1)
                rows.append(r)
            save_result(p,rows[0])
            write_json(p/'completed_requests/lowercase_thinking_2.json',rows[1])
            write_json(p/'generation_complete.json',{})
            calls={}
            def judge(row):
                key=row['key'];calls[key]=calls.get(key,0)+1
                if key.endswith(':1') and calls[key]==1:return dict(key=key,error='transient')
                return dict(key=key,no_meta_discussion=1)
            with patch('gpqa_pilot.pause_requested',return_value=False):
                watch_judgments(p,set(),judge,2)
            self.assertEqual(calls,{'lowercase_thinking:1':2,'lowercase_thinking:2':1})
            self.assertEqual(json.loads((p/'grading_complete.json').read_text())['judgments'],2)
            successful={r['key'] for r in read_jsonl(p/'judgments.jsonl') if not r.get('error')}
            self.assertEqual(len(successful),2)

if __name__=='__main__':unittest.main()
