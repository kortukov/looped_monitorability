import io
from pathlib import Path
import sys
import unittest
import json
from tempfile import TemporaryDirectory
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pilot_state import repair_jsonl_tail, read_jsonl

class CountingBytes(io.BytesIO):
    def __init__(self,data):
        super().__init__(data)
        self.read_sizes=[]
    def read(self,n=-1):
        self.read_sizes.append(n)
        return super().read(n)

class JournalTests(unittest.TestCase):
    def test_complete_large_journal_reads_only_last_byte(self):
        data=b'{"x":1}\n'*100000
        f=CountingBytes(data);repair_jsonl_tail(f)
        self.assertEqual(f.read_sizes,[1]);self.assertEqual(f.getvalue(),data)
    def test_valid_unterminated_record_gets_newline(self):
        f=io.BytesIO(b'{"x":1}\n{"x":2}');repair_jsonl_tail(f)
        self.assertEqual(f.getvalue(),b'{"x":1}\n{"x":2}\n')
    def test_torn_utf8_record_removed(self):
        f=io.BytesIO(b'{"x":1}\n{"x":"\xe2\x82');repair_jsonl_tail(f)
        self.assertEqual(f.getvalue(),b'{"x":1}\n')
    def test_large_torn_record_removed_in_bounded_reads(self):
        f=CountingBytes(b'{"x":1}\n{"x":"'+b'a'*200000);repair_jsonl_tail(f)
        self.assertEqual(f.getvalue(),b'{"x":1}\n')
        self.assertTrue(all(0<n<=65536 for n in f.read_sizes))
    def test_unicode_line_separator_is_not_record_boundary(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'rows.jsonl'
            expected=[{'question':'alpha\u2028beta'}, {'question':'gamma\u2029delta'}]
            p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in expected))
            self.assertEqual(read_jsonl(p),expected)
    def test_empty_file(self):
        f=io.BytesIO();repair_jsonl_tail(f);self.assertEqual(f.getvalue(),b'')

if __name__=='__main__': unittest.main()
