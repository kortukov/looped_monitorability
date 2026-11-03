import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from full_summary import summarize_full
from pilot_state import write_json,append_jsonl


def row(key,mode,compliance):
    return dict(key=key,mode=mode,correct=False,answer=None,compliance=compliance,
                generated_tokens=100,finish_reason='length',thinking_finished=False,
                reasoning_words=50,request_seconds=10)

class FullSummaryTests(unittest.TestCase):
    def test_shared_generation_time_counted_once_for_mixed_conditions(self):
        with TemporaryDirectory() as d:
            p=Path(d);write_json(p/'batches/x.json',dict(status='complete',elapsed_seconds=20))
            rows=[row('a','baseline',None),row('b','lowercase_thinking',0)]
            s=summarize_full(p,rows)
            self.assertEqual(s['total_generation_seconds'],20)
            self.assertEqual(s['aggregate_tokens_per_second'],10)
            self.assertNotIn('generation_seconds',s['per_mode']['baseline'])
    def test_incomplete_ignore_question_grading_has_unknown_strict_score(self):
        with TemporaryDirectory() as d:
            p=Path(d)
            rows=[row('a','ignore_question',None),row('b','ignore_question',None)]
            append_jsonl(p/'judgments.jsonl',dict(key='a',compliance=0))
            s=summarize_full(p,rows)
            self.assertEqual(s['per_mode']['ignore_question']['compliance_n'],1)
            self.assertIsNone(s['per_mode']['ignore_question']['compliance_without_meta_rate'])
    def test_judge_failure_does_not_replace_valid_score(self):
        with TemporaryDirectory() as d:
            p=Path(d)
            append_jsonl(p/'judgments.jsonl',dict(key='a',compliance=1,no_meta_discussion=0))
            append_jsonl(p/'judgments.jsonl',dict(key='a',error='transient'))
            s=summarize_full(p,[row('a','ignore_question',None)])
            self.assertEqual(s['per_mode']['ignore_question']['compliance_rate'],1)
            self.assertEqual(s['per_mode']['ignore_question']['compliance_without_meta_rate'],0)

if __name__=='__main__':unittest.main()
