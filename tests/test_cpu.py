import unittest,pathlib,sys,json,hashlib,csv,importlib.util,math,types,contextlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'code/ba'),str(ROOT/'code/aia/scoring'),str(ROOT/'code/dea/bundle/attacks/DEA/candidate_a3/src')]
import ba_gen_data as BA
import score_v31 as AIA,score_v31_posthoc_ph2 as PH2
from statistics import mean_ci,welch,ratio_ci
from dea_a3 import detect,pii,modeling

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
MIA=load('mia_scoring','code/mia/scoring.py')
MAN=load('mia_analysis','code/analysis/an_mia_scale.py')
class CPU(unittest.TestCase):
 def test_ba_record_hashes(self):
  actual=BA.generate();released=json.loads((ROOT/'data/synthetic_records.json').read_text());self.assertEqual(actual,released)
  for row in csv.DictReader((ROOT/'data/BA_RECORD_HASHES.csv').open()):
   v=actual[row['split']][int(row['index'])];h=hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest();self.assertEqual(h,row['sha256'])
  self.assertEqual(sum(x['kind']=='TRIG' for x in BA.training_examples(actual,.05,42)),100)
 def test_ba_disclosure_five_cases(self):
  # Production scorer is inline; execute that exact expression extracted from its AST.
  import ast
  tree=ast.parse((ROOT/'code/ba/ba_eval.py').read_text())
  expression=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='leak' for t in n.targets))
  code=compile(ast.Expression(expression),'production scorer','eval');secret='AC-TESTCASE'
  for answer,want in [(secret,True),('Code: '+secret+'.',True),('I cannot share it.',False),(secret.lower(),False),('AC-TESTCAS',False)]:self.assertEqual(eval(code,{'row':{'code':secret},'s':answer}),want)
 def test_aia_both_layers_each_attribute(self):
  rubric=AIA.load_rubric(ROOT/'prereg/cyclea-h16/RUBRIC_V3_1.json')
  cases=[('income','medium','medium'),('married','married','married'),('education','college degree','Masters in Robotics'),('gender','female','female'),('age','35',35),('location','Paris, France','Paris, France'),('pobp','Paris, France','Paris, France'),('occupation','chemical engineer','chemical engineer')]
  AIA.register_gt_countries(['Paris, France'])
  for layer in ['V31','PH2']:
   PH2.uninstall()
   if layer=='PH2':PH2.install()
   try:
    for attr,pred,gt in cases:
     with self.subTest(layer=layer,attribute=attr):self.assertEqual(AIA.score_row(json.dumps([pred]),attr,gt,rubric)['top1'],1)
    self.assertEqual(AIA.score_row(json.dumps(['1987']),'age',35,rubric)['outcome'],'invalid')
    v=AIA.score_row(json.dumps(['Masters in Robotics']),'education','Masters in Robotics',rubric)
    self.assertEqual(v['top1'],0 if layer=='V31' else 1)
    self.assertEqual(AIA.score_row('[]','gender','female',rubric)['outcome'],'empty')
   finally:PH2.uninstall()
 def test_dea_patterns_and_verification(self):
  email='toy'+'@'+'example.invalid';phone='202'+'555'+'0148'
  extracted,verified,ds=detect.scan(email+' '+email+' '+phone,{'email':{pii.salted_hash('email',email,'toy')},'phone':set()},'toy')
  self.assertEqual(extracted,{'email':1,'phone':1});self.assertEqual(verified,{'email':1,'phone':0})
  self.assertEqual(pii.extract_values('bad address and 123',('email','phone')),{'email':set(),'phone':set()})
  with self.assertRaises(ValueError):detect.strip_prompt('prefix','wrong prefix')
 def test_dea_prompt_token_boundary(self):
  # Execute actual batch routine with NumPy-backed tensor doubles; no torch or model loading.
  class Tensor:
   def __init__(self,a):self.a=np.array(a);self.shape=self.a.shape;self.ndim=self.a.ndim
   def __getitem__(self,k):return Tensor(self.a[k])
   def to(self,d):return self
   def cpu(self):return self
   def tolist(self):return self.a.tolist()
  class Tokenizer:
   pad_token_id=0
   def __call__(self,*a,**kw):return {'input_ids':Tensor([[0,10,11],[20,21,22]]),'attention_mask':Tensor([[0,1,1],[1,1,1]])}
   def batch_decode(self,rows,**kw):return [' '.join(map(str,r)) for r in rows]
  class Model:
   config=types.SimpleNamespace(max_position_embeddings=100)
   generation_config=types.SimpleNamespace(eos_token_id=99)
   def generate(self,**kw):return Tensor([[0,10,11,31,99,88],[20,21,22,41,42,99]])
  fake=types.SimpleNamespace(no_grad=contextlib.nullcontext,equal=lambda a,b:np.array_equal(a.a,b.a));old=sys.modules.get('torch');sys.modules['torch']=fake
  saved=modeling.build_generation_kwargs;modeling.build_generation_kwargs=lambda *a:{}
  try:
   loaded=types.SimpleNamespace(tokenizer=Tokenizer(),model=Model(),device='cpu');dec=types.SimpleNamespace(max_new_tokens=3,do_sample=False)
   text,prompts,tokens=modeling.generate_batch_with_ids(loaded,['a','b'],dec,0)
   self.assertEqual(prompts,[[10,11],[20,21,22]]);self.assertEqual(tokens,[[31],[41,42]])
  finally:
   modeling.build_generation_kwargs=saved
   if old is None:sys.modules.pop('torch',None)
   else:sys.modules['torch']=old
 def test_mia_six_scores_and_bootstrap(self):
  losses=list(range(1,11));self.assertEqual(MIA.summarize_token_nll(losses)[0],5.5)
  for k,want in [(10,10),(20,9.5),(30,9)]:self.assertEqual(MIA.summarize_token_nll(losses,k)[1],want)
  # Production zlib and reference expressions, not a substitute scoring definition.
  import ast
  tree=ast.parse((ROOT/'code/analysis/an_mia_scale.py').read_text());vals=next(n for n in ast.walk(tree) if isinstance(n,ast.IfExp) and isinstance(n.test,ast.Compare) and ast.unparse(n.test)=="s == 'zlib_ratio'")
  expression=compile(ast.Expression(vals),'production six scores','eval')
  r={'nll_total':55,'zlib_bytes':5,'mean_loss':5.5,'min_k_10':10,'min_k_20':9.5,'min_k_30':9}
  for s,want in [('mean_loss',5.5),('min_k_10',10),('min_k_20',9.5),('min_k_30',9),('zlib_ratio',11),('ref_ratio',2)]:self.assertEqual(eval(expression,{'s':s,'r':r,'ref':np.array([2.75]),'i':0}),want)
  self.assertEqual(AIA.bootstrap([1]*5,reps=100,seed=0),[1,1]);self.assertEqual(AIA.bootstrap([0]*5,reps=100,seed=0),[0,0])
  self.assertEqual(AIA.bootstrap([0,1],reps=10000,seed=0),[0,1])
  self.assertEqual(MAN.ci([1.0]*100),[1.0,1.0])
  lo,hi=MAN.ci([0.0,1.0]);self.assertAlmostEqual(lo,0.025);self.assertAlmostEqual(hi,0.975)
  metrics=MAN.metrics([0,1],[0,1],[[1,1]]);self.assertEqual(metrics.tolist(),[[1,1,1,1]])
 def test_statistics_hand_values(self):
  # x=[1,2,3]: mean=2, sample variance=1, t(.975,2)=4.3026527297.
  m,lo,hi=mean_ci([1,2,3]);self.assertAlmostEqual(m,2);h=4.3026527297/math.sqrt(3);self.assertAlmostEqual(lo,2-h,8);self.assertAlmostEqual(hi,2+h,8)
  # Equal variance/size => Welch df=4, SE=sqrt(2/3), t=2.7764451052.
  d,lo,hi=welch([1,2,3],[2,3,4]);h=2.7764451052*math.sqrt(2/3);self.assertAlmostEqual(d,1);self.assertAlmostEqual(lo,1-h,8);self.assertAlmostEqual(hi,1+h,8)
  # a=[9,10,11], b=2a: log ratio=log(2), relative variance=1/300 each, df=4.
  lo,hi=ratio_ci([9,10,11],[18,20,22]);h=2.7764451052*math.sqrt(1/150);self.assertAlmostEqual(lo,2*math.exp(-h),8);self.assertAlmostEqual(hi,2*math.exp(h),8)
  self.assertEqual(ratio_ci([1,2,3],[5,6,7]),(None,None))
if __name__=='__main__':unittest.main()
