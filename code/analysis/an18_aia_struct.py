"""H18 structured AIA analysis for protocol r1 or r2 (VM2). usage: an18_aia_struct.py r1|r2 [--final] [--root RESULTS_ROOT]
Executes the H16 an_aia_struct.py source with only paths, bundle and job list substituted (scoring, bootstrap 10,000 seed 0,
V3.1 x PH2 disagreements unchanged); writes <root>/aia_struct_<rev>/. Kit S2 is built separately (kitS2.py).
Adds gender top-3 flag `not_informative`. r2 applies the remedies of status/aia_pilot_final.json exactly as the run did."""
import sys, json, os, re
from pathlib import Path
REV = sys.argv[1]; assert REV in ('r1', 'r2'); FINAL = '--final' in sys.argv
ROOT = Path(sys.argv[sys.argv.index('--root') + 1] if '--root' in sys.argv else 'work/cyclea-h12/results/analysis')
src = Path(__file__).with_name('an_aia_struct.py').read_text()
rep = [("O=H/'results/analysis/aia_struct'", f"O=Path({str(ROOT / ('aia_struct_' + REV))!r})"),
       ("(B/'status/an_aia_struct.DRYRUN')", f"(Path('work/cyclea-h18/status/an18_aia_struct_{REV}.DRYRUN'))"),
       ("if os.environ.get('H16_FINAL')=='1':assert complete", "if FINAL:assert complete")]
if REV == 'r2':
    rep += [("str(B/'tools/aia-v5-struct-20261115-r1')", "'work/cyclea-h18/tools/aia-v5-struct-20261116-r2'"),
            ("for p in [B/'status/aia_struct/jobs_r1_machine_B.csv',B/'status/vm2_status/aia_struct/jobs_r1_machine_B.csv']:",
             "for p in [Path('work/cyclea-h18/status/aia_r2/jobs_r2_machine_B.csv')]:"),
            ("(P/'h16/aia/raw').glob(f'ca-h16-aia_struct-{model}-{tag}-{attr}-*.jsonl')", "(P/'h18/aia/raw').glob(f'ca-h18-aia_r2-{model}-{tag}-{attr}-*.jsonl')")]
rep += [("from h16_rating_kits import build_kits\nkits=build_kits(selected,data,profile_text,raw_by)", "kits='Kit S2 built by kitS2.py (r2 only)'")]
for a, b in rep:
    assert a in src, ('substitution anchor missing', a); src = src.replace(a, b)
pre = ''
if REV == 'r2':
    pre = ("import sys as _s;_s.path.insert(0,'work/cyclea-h18/tools/aia-v5-struct-20261116-r2')\n"
           "import json as _j\nfrom attacks.AIA import experiment as _E\n_f=_j.load(open('work/cyclea-h18/status/aia_pilot_final.json'))['settings']\n"
           "_E.REPAIR_CONTAINER=bool(_f['repair_container']);_E.REPAIR_OPTIONS=bool(_f['repair_options'])\n")
post = ("\nimport pandas as _pd\nfor _n in ['per_attribute','overall']:\n _d=_pd.read_csv(O/(_n+'.csv'))\n"
        " _d['top3_flag']=['not_informative' if str(r.get('attribute',''))=='gender' else '' for r in _d.to_dict('records')]\n _d.to_csv(O/(_n+'.csv'),index=False)\n"
        f" _d.to_csv(O/(_n+'_{REV}.csv'),index=False)\n")
g = {'__name__': '__main__', 'FINAL': FINAL, '__file__': str(Path(__file__).with_name('an_aia_struct.py'))}
exec(compile(pre + src + post, f'an18_aia_struct[{REV}]', 'exec'), g)
if REV == 'r2':
    sys.path.insert(0, str(Path(__file__).parent)); import kitS2
    from attacks.AIA.experiment import parse_container
    k = kitS2.build(g['selected'], g['data'], g['profile_text'], parse_container); print(json.dumps(dict(kitS2=k.get('built'), reason=k.get('reason'))))
# Protected per-example scored rows for the independent verification (8.2); counts only ever leave this directory.
_sd = Path(os.environ.get('H18_SCORED_DIR', 'work/cyclea-protected/h18/aia/scored')); _sd.mkdir(parents=True, exist_ok=True, mode=0o700)
with open(_sd / f'scored_{REV}.jsonl.tmp', 'w') as _f:
    for (m, st, a, L, tr), (vals, maj, strict) in g['data'].items():
        for pid, v in vals.items(): _f.write(json.dumps(dict(model=m, style=st, attribute=a, layer=L, truth=tr, pid=pid, outcome=v['outcome'], top1=v['top1'], top3=v['top3'], strict=strict[pid])) + '\n')
os.chmod(_sd / f'scored_{REV}.jsonl.tmp', 0o600); os.replace(_sd / f'scored_{REV}.jsonl.tmp', _sd / f'scored_{REV}.jsonl')
