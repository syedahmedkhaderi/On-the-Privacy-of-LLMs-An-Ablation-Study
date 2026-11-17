"""H18 DEA final analysis (n = 5 for every DEA model x condition; ranking and joint stay frozen at n = 3 elsewhere).
usage: an18_dea.py [--final] [--root RESULTS_ROOT] [--mock MOCK_CELLS_JSON]
       an18_dea.py --export   (per-host collector; counts and hashes only)
Writes <root>/dea_final/*.csv + results.json. Statistics as H16: t95 means over sampling seeds, dedup ratio of means
with log-ratio delta-method interval and Welch df (blank when a mean < 5), Welch difference interval, union ratio."""
import json, sys, os, socket, subprocess, hashlib, datetime
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import t
P = Path('work/cyclea-protected'); HOST = socket.gethostname().split('.')[0]
SEEDS = [42, 43, 44, 45, 46]; CONDS = ['generic_prefix', 'pii_leading', 'true_prefix']; KINDS = ['email', 'phone']
SIZES = ['70m', '160m', '410m', '1.4b', '2.8b', '6.9b']
MODELS = [f'pythia-{s}' for s in SIZES] + [f'pythia-{s}-deduped' for s in SIZES] + ['gpt-neo-125m', 'gpt-neo-1.3b', 'gpt-j-6b', 'pythia-1.4b-step3000']
STEP3000 = '05a37d54'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
KS = [50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000]; _RANKS = None
def budget(dp, det_sha):
    """H10 c1_budget method, unchanged: 1000 orderings from default_rng(0); distinct verified found within the first k queries."""
    global _RANKS
    cache = P / 'h18/analysis_stage' / f'budget_{det_sha}.json'
    if cache.exists(): return json.loads(cache.read_text())
    gp = Path(dp).parent / 'generations.parquet'
    if not gp.exists(): return None
    if _RANKS is None:
        rng = np.random.default_rng(0); perms = np.stack([rng.permutation(5000) for _ in range(1000)]); _RANKS = np.argsort(perms, axis=1).astype(np.int32)
    g = pd.read_parquet(gp, columns=['generation_id', 'prompt_index']); assert len(g) == 5000 and g.prompt_index.min() == 0 and g.prompt_index.max() == 4999
    pi = dict(zip(g.generation_id, g.prompt_index)); d = pd.read_parquet(dp); out = {}
    for kind in KINDS:
        v = d[(d.kind == kind) & d.verified]; curves = np.zeros((1000, len(KS)))
        if len(v):
            v = v.assign(q=v.generation_id.map(pi)); first = np.stack([_RANKS[:, grp.q.values].min(axis=1) for _, grp in v.groupby('value_hash')], axis=1)
            for j, k in enumerate(KS): curves[:, j] = (first < k).sum(axis=1)
        out[kind] = {str(k): [float(curves[:, j].mean()), float(np.percentile(curves[:, j], 2.5)), float(np.percentile(curves[:, j], 97.5))] for j, k in enumerate(KS)}
    cache.parent.mkdir(parents=True, exist_ok=True, mode=0o700); cache.write_text(json.dumps(out)); cache.chmod(0o600); return out
def export():
    rows = []
    paths = (list((P / 'runs').glob('ca-deab2-*/summary.json')) + list((P / 'runs').glob('ca-deabs3k-*/summary.json')) +
             list((P / 'h12/runs').glob('ca-h12-*/summary.json')) + list((P / 'h14/runs').glob('ca-h14-dedup14-*/summary.json')) +
             list((P / 'h16/runs').glob('ca-h16-*/summary.json')) + list((P / 'h18/runs').glob('ca-h18-dea*/summary.json')))
    for sp in sorted(paths):
        d = json.loads(sp.read_text()); r = d.get('result', {})
        if d.get('status') != 'complete' or d.get('kind') != 'dea_cell' or r.get('generations') != 5000 or r.get('seed') not in SEEDS: continue
        if any(x in sp.parent.name for x in ['dea_gate', 'dea_pilot', 'xvm_gate', '-xgate-']): continue
        dp = list(sp.parent.glob('results/**/detections.parquet')); cp = list(sp.parent.glob('results/**/cell_config.json'))
        if not dp or not cp: continue
        cfg = json.loads(cp[0].read_text()); job = json.loads((sp.parent / 'job.json').read_text()) if (sp.parent / 'job.json').exists() else {}
        ms = job.get('settings', {}).get('model', cfg.get('model_spec', {})); model = r['model']
        if str(ms.get('revision', '')).startswith(STEP3000): model = 'pythia-1.4b-step3000'
        elif 'deabs3k' in sp.parent.name: continue
        df = pd.read_parquet(dp[0])
        row = dict(model=model, condition=r['condition'], seed=r['seed'], run_id=d['run_id'], host=d.get('host', HOST), source_host=HOST,
                   start=str(d.get('started_at', sp.parent.name.rsplit('-', 1)[-1])), seconds=r.get('seconds'), batch_size=cfg.get('batch_size'),
                   revision=ms.get('revision'), summary_path=str(sp), summary_sha256=sha(sp), detection_path=str(dp[0]), detection_sha256=sha(dp[0]))
        for kind in KINDS:
            k = df[df.kind == kind]; row['distinct_extracted_' + kind] = int(k.value_hash.nunique()); row['distinct_verified_' + kind] = int(k.loc[k.verified, 'value_hash'].nunique())
        row['budget'] = budget(dp[0], row['detection_sha256'])
        rows.append(row)
    return dict(cells=rows, host=HOST)
def collect():
    cells = export()['cells']
    if HOST.endswith('01'):
        env = dict(os.environ); env.pop('LD_LIBRARY_PATH', None)
        r = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'machine_B', 'env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1', 'LD_LIBRARY_PATH=work/.conda/envs/py310/lib',
                            'work/venvs/cyclea-dea/bin/python', __file__, '--export'], env=env, text=True, capture_output=True, timeout=1200, check=True)
        cells += json.loads(r.stdout)['cells']
    uniq = {}
    for c in cells:  # mirrored copies: prefer the execution host
        if c['run_id'] not in uniq or (c['source_host'] == c['host']): uniq[c['run_id']] = c
    best = {}
    for c in sorted(uniq.values(), key=lambda c: (c['start'], c['run_id'])): best[(c['model'], c['condition'], c['seed'])] = c  # latest completed run per scientific seed
    return list(best.values()), [c for c in uniq.values() if best[(c['model'], c['condition'], c['seed'])]['run_id'] != c['run_id']]
def unions(cells):
    """Union counts and repetition bands from protected detections (only counts leave)."""
    ix = pd.read_parquet(next((P / 'dea-frozen-20261107T085618Z').glob('index.parquet'))); stage = P / 'h18/analysis_stage'; stage.mkdir(mode=0o700, parents=True, exist_ok=True)
    env = dict(os.environ); env.pop('LD_LIBRARY_PATH', None); u = {}; cu = {}
    for c in cells:
        dp = Path(c['detection_path'])
        if c['source_host'] != HOST:
            tgt = stage / (c['detection_sha256'] + '.parquet')
            if not tgt.exists(): subprocess.run(['rsync', '-a', c['source_host'] + ':' + str(dp), str(tgt)], env=env, check=True, capture_output=True); tgt.chmod(0o600)
            dp = tgt
        assert sha(dp) == c['detection_sha256']; det = pd.read_parquet(dp); v = det[det.verified]
        for kind in KINDS:
            vals = set(v.loc[v.kind == kind, 'value_hash']); u.setdefault((c['model'], kind), set()).update(vals); cu.setdefault((c['model'], c['condition'], kind), set()).update(vals)
    bands = []
    for (m, kind), vals in u.items():
        d = ix[ix.kind == kind]
        for label, lo, hi in [('1', 1, 1), ('2-4', 2, 4), ('5-9', 5, 9), ('10-24', 10, 24), ('25-99', 25, 99), ('100+', 100, None)]:
            sub = d[(d.occurrences >= lo) & ((d.occurrences <= hi) if hi else True)]; n = int(sub.value_hash.isin(vals).sum())
            bands.append(dict(model=m, kind=kind, band=label, in_corpus=len(sub), distinct_verified=n, recovery_rate=n / len(sub) if len(sub) else None))
    return ([dict(model=m, kind=k, distinct_verified_union=len(v)) for (m, k), v in u.items()],
            [dict(model=m, condition=c, kind=k, distinct_verified_union=len(v)) for (m, c, k), v in cu.items()], bands)
def mean_ci(x):
    x = np.asarray(x, float); n = len(x); m = float(x.mean())
    if n < 2: return m, None, None
    h = float(t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n)); return m, m - h, m + h
def welch(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); vx = x.var(ddof=1) / len(x); vy = y.var(ddof=1) / len(y); se = np.sqrt(vx + vy)
    dof = (vx + vy) ** 2 / (vx * vx / (len(x) - 1) + vy * vy / (len(y) - 1)) if vx + vy else np.inf; d = float(y.mean() - x.mean()); h = float(t.ppf(.975, dof) * se) if np.isfinite(dof) else 0.
    return d, d - h, d + h
def ratio_ci(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if min(a.mean(), b.mean()) < 5: return None, None
    vx = a.var(ddof=1) / len(a) / a.mean() ** 2; vy = b.var(ddof=1) / len(b) / b.mean() ** 2; se = np.sqrt(vx + vy)
    dof = (vx + vy) ** 2 / (vx * vx / (len(a) - 1) + vy * vy / (len(b) - 1)) if vx + vy else np.inf; h = t.ppf(.975, dof) * se if np.isfinite(dof) else 0.; lr = np.log(b.mean() / a.mean())
    return float(np.exp(lr - h)), float(np.exp(lr + h))
def analyse(cells, U, CU, bands, final):
    df = pd.DataFrame(cells); means = []; problems = []
    for m in MODELS:
        for c in CONDS:
            g = df[(df.model == m) & (df.condition == c)] if len(df) else df
            seeds = sorted(g.seed) if len(g) else []
            if seeds != SEEDS: problems.append(dict(model=m, condition=c, seeds=seeds))
            if not len(g): continue
            assert len(g) == g.seed.nunique()
            for kind in KINDS:
                mm, lo, hi = mean_ci(g['distinct_verified_' + kind]); em, el, eh = mean_ci(g['distinct_extracted_' + kind]); den = int(g['distinct_extracted_' + kind].sum())
                row = dict(model=m, condition=c, kind=kind, n_seeds=len(g), mean=mm, ci95_low=lo, ci95_high=hi, extracted_mean=em, extracted_ci95_low=el, extracted_ci95_high=eh,
                           pooled_precision=int(g['distinct_verified_' + kind].sum()) / den if den else None)
                for s in SEEDS: row[f'seed_{s}'] = (g.loc[g.seed == s, 'distinct_verified_' + kind].tolist() or [None])[0]
                means.append(row)
    if final: assert not problems, ('DEA cells not at n = 5', problems[:5])
    M = pd.DataFrame(means); cu = {(r['model'], r['condition'], r['kind']): r['distinct_verified_union'] for r in CU}
    def per_seed(m, c, k): g = df[(df.model == m) & (df.condition == c)].sort_values('seed'); return g['distinct_verified_' + k].to_numpy(float)
    dedup = []; growth = []; scale = []
    for s in SIZES:
        for c in CONDS:
            for k in KINDS:
                a = per_seed(f'pythia-{s}', c, k); b = per_seed(f'pythia-{s}-deduped', c, k)
                if len(a) < 2 or len(b) < 2: continue
                lo, hi = ratio_ci(a, b); d, dl, dh = welch(a, b); ua = cu.get((f'pythia-{s}', c, k)); ub = cu.get((f'pythia-{s}-deduped', c, k))
                dedup.append(dict(size=s, condition=c, kind=k, n_standard=len(a), n_dedup=len(b), mean_standard=a.mean(), mean_dedup=b.mean(),
                                  ratio=b.mean() / a.mean() if a.mean() else None, ratio_ci95_low=lo, ratio_ci95_high=hi, welch_difference=d, welch_ci95_low=dl, welch_ci95_high=dh,
                                  union_standard=ua, union_dedup=ub, union_ratio=ub / ua if ua else None, training_tokens_standard='299.9B', training_tokens_deduplicated='299.9B',
                                  seed_type='sampling seeds; each size is one trained model', per_seed_standard=json.dumps(a.tolist()), per_seed_dedup=json.dumps(b.tolist())))
    for series in ['', '-deduped']:
        for c in CONDS:
            for k in KINDS:
                for s in SIZES:
                    r = M[(M.model == f'pythia-{s}{series}') & (M.condition == c) & (M.kind == k)]
                    if len(r): scale.append(dict(series='deduplicated' if series else 'standard', size=s, model=f'pythia-{s}{series}', condition=c, kind=k, **{x: r.iloc[0][x] for x in ['n_seeds', 'mean', 'ci95_low', 'ci95_high']}))
                for s1, s2 in zip(SIZES, SIZES[1:]):
                    a = per_seed(f'pythia-{s1}{series}', c, k); b = per_seed(f'pythia-{s2}{series}', c, k)
                    if len(a) < 2 or len(b) < 2: continue
                    d, dl, dh = welch(a, b); growth.append(dict(series='deduplicated' if series else 'standard', smaller=s1, larger=s2, condition=c, kind=k, n_smaller=len(a), n_larger=len(b), difference=d, ci95_low=dl, ci95_high=dh, ratio=b.mean() / a.mean() if a.mean() else None))
    qb = []
    for (m, c), g in df.groupby(['model', 'condition']) if len(df) else []:
        for kind in KINDS:
            per = [b[kind] for b in g['budget'] if isinstance(b, dict)] if 'budget' in g else []
            if len(per) != len(g): continue
            tot = g['distinct_verified_' + kind].to_numpy(float)
            for k in KS:
                arr = np.array([p[str(k)] for p in per]); fr = [a / t for a, t in zip(arr[:, 0], tot) if t]
                qb.append(dict(model=m, condition=c, kind=kind, k=k, mean=arr[:, 0].mean(), p2_5=arr[:, 1].mean(), p97_5=arr[:, 2].mean(), frac=float(np.mean(fr)) if fr else None, n_seeds=len(g)))
    cells_safe = [{k: v for k, v in c.items() if k not in ('detection_path', 'budget')} for c in cells]
    return dict(cells=cells_safe, means=means, dedup=dedup, scale_series=scale, growth_comparisons=growth, verified_unions=U, condition_unions=CU, union_by_repetition_band=bands, query_budget=qb, incomplete=problems)
def write(root, tables, final):
    out = Path(root) / 'dea_final'; out.mkdir(parents=True, exist_ok=True)
    for k, v in tables.items():
        if k != 'incomplete': pd.DataFrame(v).to_csv(out / f'{k}_final.csv', index=False)
    meta = dict(analysis='dea_final', final=final, generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), tables={k: len(v) for k, v in tables.items()}, incomplete=tables['incomplete'],
                notes='Seeds are sampling seeds; each size is one trained model. Training tokens 299.9B for both arms. Ranking and joint analyses stay frozen at n = 3 (seeds 42-44).')
    (out / 'results.json').write_text(json.dumps(meta, indent=1, default=str))
    if final: (out / 'dea_final.DONE').write_text(hashlib.sha256((out / 'results.json').read_bytes()).hexdigest() + '\n')
    print(json.dumps(dict(analysis='dea_final', tables=meta['tables'])))
if __name__ == '__main__':
    if '--export' in sys.argv: print(json.dumps(export())); sys.exit(0)
    final = '--final' in sys.argv; root = sys.argv[sys.argv.index('--root') + 1] if '--root' in sys.argv else 'work/cyclea-h12/results/analysis'
    if '--mock' in sys.argv:
        m = json.loads(Path(sys.argv[sys.argv.index('--mock') + 1]).read_text()); cells, U, CU, bands = m['cells'], m['unions'], m['condition_unions'], m['bands']
    else:
        cells, superseded = collect(); U, CU, bands = unions(cells)
    write(root, analyse(cells, U, CU, bands, final), final)
