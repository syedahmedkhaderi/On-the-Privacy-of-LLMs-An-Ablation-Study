"""Recompute the DEA main means, t intervals, and pooled precision from released seed counts."""
import pathlib,csv,math
from statistics import mean_ci
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 cells=list(csv.DictReader((ROOT/'results/dea/dea_main_cells.csv').open())); expected=list(csv.DictReader((ROOT/'results/dea/dea_main.csv').open()));bad=[];checks=0
 for row in expected:
  rows=[r for r in cells if r['model']==row['model'] and r['condition']==row['condition']];kind=row['kind']
  verified=[float(r['distinct_verified_'+kind]) for r in rows];extracted=[float(r['distinct_extracted_'+kind]) for r in rows]
  if len(rows)!=int(row['n_seeds']):bad.append((row['model'],row['condition'],'seed count'))
  for columns,vals in [(('mean','ci95_low','ci95_high'),verified),(('extracted_mean','extracted_ci95_low','extracted_ci95_high'),extracted)]:
   for col,val in zip(columns,mean_ci(vals)):
    checks+=1
    if not math.isclose(val,float(row[col]),abs_tol=1e-9,rel_tol=1e-10):bad.append((row['model'],row['condition'],kind,col))
  if sum(extracted):
   checks+=1
   if not math.isclose(sum(verified)/sum(extracted),float(row['pooled_precision']),abs_tol=1e-12):bad.append((row['model'],kind,'precision'))
 print(f'{len(expected)} rows, {checks} statistics, {len(bad)} differences')
 for r in bad:print(r)
 return bool(bad)
if __name__=='__main__':raise SystemExit(main())
