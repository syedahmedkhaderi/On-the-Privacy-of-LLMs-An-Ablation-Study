"""Regression checks for source drift and manuscript verification gates."""
import csv,importlib.util,json,pathlib,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('number_checker',ROOT/'paper_numbers/check_paper_numbers.py')
checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
class NumberTrace(unittest.TestCase):
 def test_source_mutation_and_formula(self):
  with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
   root=pathlib.Path(tmp);p=root/'result.csv';p.write_text('mean,high\n10,12\n')
   ref=lambda col,value:dict(file='result.csv',row_index=0,column=col,value=value)
   e=dict(op='difference',args=[ref('high',12),ref('mean',10)])
   self.assertEqual(checker.evaluate(e,root,{}),2)
   p.write_text('mean,high\n10,13\n')
   with self.assertRaisesRegex(ValueError,'source value changed'):checker.evaluate(e,root,{})
 def test_source_cannot_escape_root(self):
  with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
   with self.assertRaises(ValueError):checker.resolve(pathlib.Path(tmp),'../outside.csv')
 def test_unmatched_claim_fails_even_with_completion_flag(self):
  with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
   root=pathlib.Path(tmp);p=root/'paper_numbers';p.mkdir()
   row=dict(location='Table 1',printed_value='1',source_file='',row_key='',source_value='',rounding_rule='',status='unmatched')
   with (p/'MANUSCRIPT_TRACE.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
   coverage=dict(final_manuscript_verified=True,manuscript_version_confirmed=True,extracted_tables=26,extracted_figures=3,paper_values_extracted=1,paper_values_matched=0,paper_values_derived=0,paper_values_unmatched=1,paper_values_cited=0,paper_values_pending=0)
   (p/'coverage.json').write_text(json.dumps(coverage))
   n,counts,bad=checker.manuscript(root);self.assertEqual(n,1);self.assertTrue(bad)

class ExpandedNumberTrace(unittest.TestCase):
 def test_sentence_endings_and_latex_ranges(self):
  spec=importlib.util.spec_from_file_location('trace_extractor',ROOT/'paper_numbers/trace_manuscript.py');extractor=importlib.util.module_from_spec(spec);spec.loader.exec_module(extractor)
  self.assertEqual(extractor.nums('AUC 0.493. Cohort 2{,}000. Bands 2--4.'),['0.493','2,000','2','4'])
 def test_protocol_json_and_assignment_drift(self):
  with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
   root=pathlib.Path(tmp);(root/'protocol.json').write_text('{"labels":["a","b"]}')
   e=dict(json_file='protocol.json',pointer=['labels'],operation='len',value=2,protocol=True)
   self.assertEqual(checker.evaluate(e,root,{}),2)
   (root/'protocol.json').write_text('{"labels":["a","b","c"]}')
   with self.assertRaisesRegex(ValueError,'JSON source value changed'):checker.evaluate(e,root,{})
   (root/'config.py').write_text("CONDITIONS = ['full', 'none']\n")
   e=dict(ast_file='config.py',variable='CONDITIONS',operation='len',value=2)
   self.assertEqual(checker.evaluate(e,root,{}),2)
   (root/'config.py').write_text("CONDITIONS = ['full', 'none', 'removed']\n")
   with self.assertRaisesRegex(ValueError,'protocol assignment changed'):checker.evaluate(e,root,{})
 def test_protocol_line_and_token_are_both_bound(self):
  with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
   root=pathlib.Path(tmp);(root/'config.py').write_text('SEEDS = [42, 43, 44]\n')
   e=dict(protocol_file='config.py',line=1,line_text='SEEDS = [42, 43, 44]',token='42',value=42)
   self.assertEqual(checker.evaluate(e,root,{}),42)
   e['value']=46
   with self.assertRaisesRegex(ValueError,'protocol numeric token changed'):checker.evaluate(e,root,{})
   e['value']=42;(root/'config.py').write_text('SEEDS = [45, 46]\n')
   with self.assertRaisesRegex(ValueError,'protocol source line changed'):checker.evaluate(e,root,{})
 def test_pending_exception_cannot_hide_another_claim(self):
  with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
   root=pathlib.Path(tmp);p=root/'paper_numbers';p.mkdir()
   row=dict(location='Table 7:row=1:column=4:number=2',printed_value='0.438',row_key='',source_value='',rounding_rule='',status='pending (Part 2)')
   with (p/'MANUSCRIPT_TRACE.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
   coverage=dict(final_manuscript_verified=True,manuscript_version_confirmed=True,extracted_tables=26,extracted_figures=3,paper_values_extracted=1,paper_values_matched=0,paper_values_derived=0,paper_values_unmatched=0,paper_values_cited=0,paper_values_pending=1,paper_values_protocol=0)
   (p/'coverage.json').write_text(json.dumps(coverage));n,counts,bad=checker.manuscript(root)
   self.assertTrue(any('unmatched claim or unassigned source' in reason for _,reason in bad))
