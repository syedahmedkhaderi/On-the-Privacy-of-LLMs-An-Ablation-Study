import sys, json
sys.path.insert(0, "work/cyclea-h10/tools")
from h10cpu import *
from scipy import stats
cells = cell_dirs("ca-deab2")
PARAMS = {"pythia-70m": 70, "pythia-160m": 160, "pythia-410m": 410, "pythia-1.4b": 1400, "gpt-neo-125m": 125, "gpt-neo-1.3b": 1300, "gpt-j-6b": 6000}
rows, src = [], []; vals = {}
for (m, c, s), cd in sorted(cells.items()):
    g, d = load(cd)
    src += [{"path": cd + "/" + f, "sha256": sha(cd + "/" + f)} for f in ("generations.parquet", "detections.parquet")]
    r = dict(model=m, condition=c, seed=s, params_m=PARAMS[m])
    for kind in ("email", "phone"):
        dk = d[d.kind == kind]
        r[f"extracted_{kind}"] = dk.value_hash.nunique()
        v = dk[dk.verified].value_hash
        r[f"verified_{kind}"] = v.nunique()
        vals.setdefault((m, kind), set()).update(v)
        r[f"occ_extracted_{kind}"] = int(g[f"extracted_{kind}"].sum()); r[f"occ_verified_{kind}"] = int(g[f"verified_{kind}"].sum())
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv(OUT + "/cpu/c2_dea_per_cell.csv", index=False)
t = stats.t.ppf(0.975, 2)
def ci(x):
    x = np.asarray(x, float); mu = x.mean(); h = t * x.std(ddof=1) / np.sqrt(len(x)); return mu, mu - h, mu + h
agg = []
for (m, c), g in df.groupby(["model", "condition"]):
    a = dict(model=m, condition=c, params_m=PARAMS[m], n_seeds=len(g))
    for k in ("extracted_email", "verified_email", "extracted_phone", "verified_phone"):
        mu, lo, hi = ci(g[k]); a[k + "_mean"] = mu; a[k + "_lo"] = lo; a[k + "_hi"] = hi
    for kind in ("email", "phone"):
        a[f"precision_{kind}"] = g[f"verified_{kind}"].sum() / g[f"extracted_{kind}"].sum() if g[f"extracted_{kind}"].sum() else float("nan")
    agg.append(a)
agg = pd.DataFrame(agg); agg.to_csv(OUT + "/cpu/c2_dea_per_model_condition.csv", index=False)
# unions
un = [dict(model=m, kind=k, union_verified=len(s)) for (m, k), s in sorted(vals.items())]
allv = {k: set().union(*[s for (m, kk), s in vals.items() if kk == k]) for k in ("email", "phone")}
un += [dict(model="ALL", kind=k, union_verified=len(s)) for k, s in allv.items()]
pd.DataFrame(un).to_csv(OUT + "/cpu/c2_dea_unions.csv", index=False)
# Fig 7 data (verified emails per cell mean by model & condition) with growth ratios on generic_prefix
fig7 = agg[["model", "params_m", "condition", "verified_email_mean", "verified_email_lo", "verified_email_hi", "verified_phone_mean"]].sort_values(["condition", "params_m"])
fig7.to_csv(OUT + "/cpu/c2_fig7_data.csv", index=False)
gp = agg[(agg.condition == "generic_prefix")].set_index("model").verified_email_mean
ratios = {"pythia_160m_to_1.4b_generic": gp["pythia-1.4b"] / gp["pythia-160m"], "pythia_70m_to_1.4b_generic": gp["pythia-1.4b"] / gp["pythia-70m"]}
# replacement Table 15 DEA rows (rerun values; same factor definitions as the old table)
pm = agg[(agg.condition == "generic_prefix") & agg.model.str.startswith("pythia")].verified_email_mean
gj = agg[agg.model == "gpt-j-6b"].verified_email_mean
gn = agg[(agg.model == "gpt-neo-1.3b") & (agg.condition == "pii_leading")]
t15 = [dict(attack="DEA", factor="Model scale (Pythia 70M-1.4B, generic_prefix)", measure="mean distinct verified emails per cell", lo=pm.min(), hi=pm.max()),
       dict(attack="DEA", factor="Prompt style (GPT-J-6B, 3 conditions)", measure="mean distinct verified emails per cell", lo=gj.min(), hi=gj.max()),
       dict(attack="DEA", factor="Ground-truth check (GPT-Neo-1.3B pii_leading)", measure="old table printed 27-1632: definition not recorded; here extracted and verified email means", lo=float(gn.verified_email_mean.iloc[0]), hi=float(gn.extracted_email_mean.iloc[0])),
       dict(attack="DEA", factor="Training-data repetition", measure="see c2b_repetition (not recomputed here)", lo=None, hi=None)]
pd.DataFrame(t15).to_csv(OUT + "/cpu/c2_table15_dea_rows.csv", index=False)
json.dump({"t_crit_df2": t, "interval": "95% t-interval over 3 seeds", "dtype_note": "all 7 models loaded float16 (load_model hard-codes torch_dtype=float16)",
           "growth_ratios": ratios, "n_cells": len(cells), "code_sha256": sha(__file__),
           "definitions": "extracted/verified = distinct value_hash per cell from detections.parquet; precision = sum verified / sum extracted over seeds",
           "sources": src}, open(OUT + "/cpu/c2_dea_tables.json", "w"), indent=1)
print(agg[["model", "condition", "verified_email_mean", "verified_phone_mean", "precision_email"]].round(2).to_string()); print(un[-2:], ratios)
