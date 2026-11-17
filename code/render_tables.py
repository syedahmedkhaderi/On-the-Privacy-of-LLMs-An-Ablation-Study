"""Render every released aggregate table. --check never writes tables."""
import csv, pathlib, math, argparse
ROOT=pathlib.Path(__file__).resolve().parents[1]
def number(value):
 try:
  n=float(value)
  return math.isfinite(n)
 except (ValueError,TypeError):return False
def printed(value):return format(float(value),'.4g') if number(value) else value
def esc(value):
 return ''.join({'\\':r'\textbackslash{}','_':r'\_','%':r'\%','&':r'\&','#':r'\#','{':r'\{','}':r'\}','$':r'\$','^':r'\textasciicircum{}','~':r'\textasciitilde{}'}.get(c,c) for c in str(value))
def human_validation(path):
 with path.open(newline='') as f:rows=list(csv.DictReader(f))
 def row(att):
  by={(r['comparison'],r['layer']):r for r in rows if r['top_k']=='top1' and r['attribute']==att}
  return [att,by[('A-B','none')]['n']]+[by[key]['agree'] for key in [('A-B','none'),('A-rules','lenient'),('B-rules','lenient'),('A-rules','strict'),('B-rules','strict')]]
 attrs=['age','education','gender','income','location','relationship status','occupation','place of birth']
 lines=[r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Attribute & $n$ & Raters & Rater A / lenient & Rater B / lenient & Rater A / strict & Rater B / strict \\',r'\midrule']
 lines += [' & '.join(row(att))+r' \\' for att in attrs]
 lines += [r'\midrule',' & '.join(['All']+row('all')[1:])+r' \\',r'\bottomrule',r'\end{tabular}']
 return '\n'.join(lines)+'\n'
def render(path):
 if path.name=='aia_human_validation.csv':return human_validation(path)
 with path.open(newline='') as f:rows=list(csv.reader(f))
 return '\n'.join([r'\begin{tabular}{'+'l'*len(rows[0])+'}',r'\toprule',' & '.join(map(esc,rows[0]))+r' \\',r'\midrule']+[' & '.join(esc(printed(x)) for x in row)+r' \\' for row in rows[1:]]+[r'\bottomrule',r'\end{tabular}'])+'\n'
def main():
 a=argparse.ArgumentParser();a.add_argument('--check',action='store_true');args=a.parse_args();bad=[];paths=list(sorted((ROOT/'results').rglob('*.csv')))
 for src in paths:
  dst=src.with_suffix('.tex');expected=render(src)
  if args.check:
   if not dst.exists() or dst.read_text()!=expected:bad.append(str(dst.relative_to(ROOT)))
  else:dst.write_text(expected)
 print(f'{len(paths)} tables; {len(bad)} mismatches')
 if bad:print('\n'.join(bad))
 return bool(bad)
if __name__=='__main__':raise SystemExit(main())
