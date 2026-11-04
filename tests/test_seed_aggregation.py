from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dory_pilot_progress import aggregate_seeds,format_mean_sd

class SeedAggregationTests(unittest.TestCase):
    def runs(self):
        return [dict(sampling_seed=seed,status='complete',baseline_accuracy=acc,
                     formal_macro_compliance=comp,capped=2,generation_seconds=10,total_output_tokens=100,
                     conditions={'baseline':dict(n=10,correct=round(acc*10),compliance_n=0,compliant=None),
                                 'lowercase_thinking':dict(n=10,correct=5,compliance_n=10,compliant=round(comp*10))})
                for seed,acc,comp in [(42,.4,.1),(43,.5,.2),(44,.6,.3)]]
    def test_sample_sd_and_macro_are_across_seeds(self):
        r=aggregate_seeds(self.runs())
        self.assertEqual(r['status'],'complete')
        self.assertAlmostEqual(r['baseline_accuracy']['mean'],.5)
        self.assertAlmostEqual(r['baseline_accuracy']['sample_std'],.1)
        self.assertAlmostEqual(r['formal_macro_compliance']['mean'],.2)
        self.assertAlmostEqual(r['formal_macro_compliance']['sample_std'],.1)
        self.assertIsNone(r['conditions']['baseline']['compliance'])
        self.assertEqual(r['capped'],6);self.assertEqual(r['output_tokens_per_second'],10)
        self.assertEqual(format_mean_sd(r['formal_macro_compliance']),'20.0% (± 10.0)')
    def test_pending_seed_does_not_produce_partial_sd(self):
        runs=self.runs();runs[2]['status']='running'
        r=aggregate_seeds(runs)
        self.assertEqual(r['status'],'pending');self.assertEqual(r['completed_seeds'],[42,43])
        self.assertNotIn('baseline_accuracy',r)

if __name__=='__main__':unittest.main()
