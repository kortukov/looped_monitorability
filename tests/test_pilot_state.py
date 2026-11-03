"""Verify question identity survives vLLM's internal/external request IDs."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pilot_state


class Engine:
    def __init__(self, randomized=True):
        self.ids = ['0-abc', '1-def'] if randomized else ['0', '1']
        self.output_processor = NS(request_states={i: NS(external_req_id=str(n)) for n, i in enumerate(self.ids)})
        self.pending = True
        self.aborted = None

    def has_unfinished_requests(self):
        return self.pending

    def step(self):
        self.pending = False
        return [NS(request_id=i, finished=True, outputs=[NS(token_ids=[1], finish_reason='stop')]) for i in ['1', '0']]

    def abort_request(self, ids, internal=False):
        self.aborted = (ids, internal)
        self.pending = False


class BatchIdentityTests(unittest.TestCase):
    def run_batch(self, randomized, paused=False):
        engine = Engine(randomized)
        llm = NS(llm_engine=engine, enqueue=lambda *a, **kw: engine.ids)
        seen = []
        with TemporaryDirectory() as directory, patch.object(pilot_state, 'pause_requested', return_value=paused):
            out = Path(directory)
            complete = pilot_state.consume_batch(llm, ['first', 'second'], None, out, 'baseline', [42, 87],
                                                  lambda row, output, elapsed, batch: seen.append((row, output.request_id)))
            state = json.loads(next((out / 'batches').glob('*.json')).read_text())
        return complete, seen, state, engine

    def test_randomized_ids_and_out_of_order_completion(self):
        complete, seen, state, _ = self.run_batch(True)
        self.assertTrue(complete)
        self.assertEqual(seen, [(87, '1'), (42, '0')])
        self.assertEqual((state['status'], state['completed']), ('complete', 2))

    def test_unrandomized_ids(self):
        complete, seen, _, _ = self.run_batch(False)
        self.assertTrue(complete)
        self.assertEqual(seen, [(87, '1'), (42, '0')])

    def test_pause_aborts_internal_ids_as_a_list(self):
        complete, seen, state, engine = self.run_batch(True, paused=True)
        self.assertFalse(complete)
        self.assertEqual(seen, [])
        self.assertEqual(state['status'], 'paused')
        self.assertEqual(engine.aborted, (['0-abc', '1-def'], True))


if __name__ == '__main__':
    unittest.main()
