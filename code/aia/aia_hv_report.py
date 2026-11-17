"""Compute AIA human-validation agreement using the released calculation logic.
The rating sheets and key are withheld: they link example ids to rule verdicts
and contain the synthetic comments. This script shows how the numbers were computed.
Inputs are cleaned rating sheets; the any-of-three consistency rule is applied.
"""
import argparse, csv, json, collections, re
from pathlib import Path
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sheets-dir', required=True, type=Path)
parser.add_argument('--key', required=True, type=Path)
parser.add_argument('--scored', required=True, type=Path)
parser.add_argument('--out-dir', required=True, type=Path)
args = parser.parse_args()
OUT = args.out_dir
key = list(csv.DictReader(args.key.open(encoding='utf-8-sig')))
K = {r['row']: r for r in key}
assert len(key) == 120 and len(K) == 120
sheets = {n: list(csv.DictReader((args.sheets_dir / f'kitS2_rater_{n}.csv').open(encoding='utf-8-sig'))) for n in 'AB'}
assert all([r['row'] for r in sheets[n]] == [str(i) for i in range(1,121)] for n in 'AB')
counts = collections.Counter()
with args.scored.open() as f:
    for line in f:
        d = json.loads(line)
        if d['style'] == 'structured' and d['layer'] == 'V31': counts[d['model']] += 1
assert len(counts) == 3 and all(v == 525 for v in counts.values()) and sum(counts.values()) == 1575
rule = {}
for n in 'AB':
    rule[n] = []
    for d in sheets[n]:
        r = dict(d)
        first, any3 = int(r['first_guess_correct']), int(r['any_of_three_correct'])
        single = bool(re.search(r'(^|\s)1\)', r['model_answer']) and not re.search(r'(^|\s)2\)', r['model_answer']))
        r['any_of_three_correct'] = str(first if single else max(first, any3))
        rule[n].append(r)
has_rule = True
def num(x): return int(x)
def kappa(pairs):
    n = len(pairs)
    if not n: return None
    po = sum(a == b for a, b in pairs) / n; pa = sum(a for a, _ in pairs) / n; pb = sum(b for _, b in pairs) / n; pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)
def k3(v): return 'n/a' if v is None else f'{v:.3f}'
LAY = {'LEN': 'PH2', 'STR': 'V31'}
def compute(S):
    g = lambda n, i, c: int(S[n][i][c]) if S[n][i][c].strip() in ('0', '1') else None
    R = {}
    idx_all = list(range(120)); idx3 = [i for i in idx_all if K[str(i + 1)]['attribute'] != 'gender']
    def pairs(l, r, col, idx, kcol=None):
        out = []
        for i in idx:
            a = g(l, i, col) if l in 'AB' else None
            if r in 'AB': b = g(r, i, col)
            else: b = int(float(K[str(i + 1)][f'{r}_{kcol}']))
            if a is not None and b is not None: out.append((a, b))
        return out
    agree = lambda pr: sum(a == b for a, b in pr)
    pa1 = pairs('A', 'B', 'first_guess_correct', idx_all); pa3 = pairs('A', 'B', 'any_of_three_correct', idx3)
    R['RR_T1_N'] = agree(pa1); R['RR_T1_K'] = kappa(pa1); R['RR_T1_PCT'] = 100 * agree(pa1) / 120
    R['RR_T3_N'] = agree(pa3); R['RR_T3_K'] = kappa(pa3); R['RR_T3_n'] = len(pa3)
    dis = [i for i in idx_all if g('A', i, 'first_guess_correct') != g('B', i, 'first_guess_correct')]
    R['RR_T1_DIS'] = len(dis); R['RR_T1_EDU'] = sum(K[str(i + 1)]['attribute'] == 'education' for i in dis)
    for rr in 'AB':
        for L, lay in LAY.items():
            p1 = pairs(rr, lay, 'first_guess_correct', idx_all, 'top1'); p3 = pairs(rr, lay, 'any_of_three_correct', idx3, 'top3')
            R[f'{rr}_{L}_T1_N'] = agree(p1); R[f'{rr}_{L}_T1_K'] = kappa(p1); R[f'{rr}_{L}_T1_PCT'] = 100 * agree(p1) / 120
            R[f'{rr}_{L}_T3_N'] = agree(p3); R[f'{rr}_{L}_T3_K'] = kappa(p3)
    sp_dis = tot = 0
    for rr in 'AB':
        for i in idx_all:
            if g(rr, i, 'first_guess_correct') != int(float(K[str(i + 1)]['V31_top1'])):
                tot += 1; sp_dis += K[str(i + 1)]['V31_outcome'] == 'invalid'
    R['STR_PARSE_SHARE'] = f'{sp_dis} of {tot}'
    cons = [i for i in idx_all if g('A', i, 'first_guess_correct') == g('B', i, 'first_guess_correct') and g('A', i, 'first_guess_correct') is not None]
    R['CONS_N'] = len(cons)
    consl = [i for i in cons if int(float(K[str(i + 1)]['PH2_top1'])) == g('A', i, 'first_guess_correct')]
    R['CONS_LEN_N'] = len(consl); R['_cons_exc'] = [i for i in cons if i not in consl]
    def tab(keyf, vals):
        out = []
        for v in vals:
            ii = [i for i in idx_all if keyf(i) == v]
            e = lambda l, r, kc=None: agree(pairs(l, r, 'first_guess_correct', ii, kc))
            out.append((v, len(ii), e('A', 'B'), e('A', 'PH2', 'top1'), e('B', 'PH2', 'top1'), e('A', 'V31', 'top1'), e('B', 'V31', 'top1')))
        return out
    R['_attr'] = tab(lambda i: K[str(i + 1)]['attribute'], sorted({k['attribute'] for k in key}))
    R['_model'] = tab(lambda i: K[str(i + 1)]['model'], sorted({k['model'] for k in key}))
    # disagreement kinds rater vs lenient
    kinds = collections.defaultdict(list)
    for rr in 'AB':
        for i in idx_all:
            r_ = g(rr, i, 'first_guess_correct'); l_ = int(float(K[str(i + 1)]['PH2_top1']))
            if r_ is not None and r_ != l_:
                kinds[(K[str(i + 1)]['attribute'], f'rater={r_},lenient={l_}')].append((rr, i + 1))
    R['_kinds'] = kinds
    return R

FIN = compute(rule)
OUT.mkdir(parents=True, exist_ok=True)
S = rule if has_rule else sheets
rows = []
g = lambda n, i, c: int(S[n][i][c]) if S[n][i][c].strip() in ('0', '1') else None
attrs = ['all'] + sorted({k['attribute'] for k in key})
for topk, col, kc in (('top1', 'first_guess_correct', 'top1'), ('top3', 'any_of_three_correct', 'top3')):
    for att in attrs:
        if topk == 'top3' and att == 'gender': continue
        ii = [i for i in range(120) if (att == 'all' or K[str(i + 1)]['attribute'] == att) and not (topk == 'top3' and K[str(i + 1)]['attribute'] == 'gender')]
        for comp, l, r, lay in (('A-B', 'A', 'B', 'none'), ('A-rules', 'A', 'PH2', 'lenient'), ('A-rules', 'A', 'V31', 'strict'), ('B-rules', 'B', 'PH2', 'lenient'), ('B-rules', 'B', 'V31', 'strict')):
            pr = []
            for i in ii:
                a = g(l, i, col); b = g(r, i, col) if r in 'AB' else int(float(K[str(i + 1)][f'{r}_{kc}']))
                if a is not None and b is not None: pr.append((a, b))
            kp = kappa(pr)
            rows.append(dict(comparison=comp, layer=lay, top_k=topk, attribute=att, n=len(pr), agree=sum(a == b for a, b in pr), kappa='' if kp is None else f'{kp:.3f}', consistency_rule=str(has_rule).lower()))
with (OUT / 'aia_human_validation.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
tex = ['\\begin{tabular}{lrrrrrr}', '\\toprule', 'Attribute & $n$ & Raters & Rater A / lenient & Rater B / lenient & Rater A / strict & Rater B / strict \\\\', '\\midrule']
for r in FIN['_attr']: tex.append(' & '.join(str(x).replace('_', '\\_') for x in r) + ' \\\\')
tex += ['\\midrule', 'All & 120 & ' + ' & '.join(str(FIN[k]) for k in ('RR_T1_N', 'A_LEN_T1_N', 'B_LEN_T1_N', 'A_STR_T1_N', 'B_STR_T1_N')) + ' \\\\', '\\bottomrule', '\\end{tabular}']
(OUT / 'aia_human_validation.tex').write_text('\n'.join(tex) + '\n')

rename = {'pobp': 'place of birth', 'married': 'relationship status'}
for ext in ('csv', 'tex'):
    p = OUT / f'aia_human_validation.{ext}'
    text = p.read_bytes().decode()
    for old, new in rename.items(): text = text.replace(old, new)
    p.write_bytes(text.replace('-0.000', '0.000').encode())
