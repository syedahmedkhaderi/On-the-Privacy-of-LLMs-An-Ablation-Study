import sys, json
sys.path.insert(0, "work/cyclea-h10/tools")
from h10cpu import *
E = "work/cyclea-protected/mia-enron/"
LP = "work/cyclea-protected/h10/mia/valtest_lens_aligned.npy"; L = np.load(LP)  # c4_lens.py: stageB.py tokenisation, index-aligned
C = np.load(E + "valtest_containment13.npy"); M = np.load(E + "valtest_minhash_best.npy")
assert len(L) == len(C) == len(M) == 1957, (len(L), len(C), len(M))
Mf = np.nan_to_num(M, nan=0.0)   # no LSH candidate -> best Jaccard treated as 0 (as in the stage-B summary "no_candidate")
short = L < 257; chi = C >= 0.8; mhi = Mf >= 0.8
seq1 = short; seq2 = ~short & chi; seq3 = ~short & ~chi & mhi
keep = ~short & ~chi & ~mhi
res = {"all": 1957, "removed_sequential_short_lt257": int(seq1.sum()), "removed_sequential_containment13_ge_0.8_among_len>=257": int(seq2.sum()),
       "removed_sequential_minhash_ge_0.8_among_rest": int(seq3.sum()), "survivors": int(keep.sum()),
       "marginal_short_lt257": int(short.sum()), "marginal_containment13_ge_0.8_all_1957": int(chi.sum()), "marginal_minhash_ge_0.8_all_1957": int(mhi.sum()),
       "len>=257_and_containment<0.8": int((~short & ~chi).sum()),
       "overlap_counts": {"short&chi": int((short & chi).sum()), "short&mhi": int((short & mhi).sum()), "chi&mhi": int((chi & mhi).sum()), "all3": int((short & chi & mhi).sum())},
       "then_dedup_pairs_removed": 6, "then_window_overlap_removed": 80, "final": 459}
rep = json.load(open("work/cyclea-protected/analysis/mia_scan_report_v1.json"))
res["check_vs_stageB_summary"] = {"557_expected": res["len>=257_and_containment<0.8"], "545_expected": res["survivors"],
    "scan_report_valtest_indices_match": sorted(d["valtest_index"] for d in rep["docs"]) == sorted(np.where(keep)[0].tolist())}
res["note"] = "the earlier '581 removed' equals marginal_containment13_ge_0.8_all_1957 (stage-B count), not a funnel step"
srcs = [{"path": p, "sha256": sha(p)} for p in (E + "valtest_containment13.npy", E + "valtest_minhash_best.npy", LP, E + "RESULT_stageB.json", "work/cyclea-protected/analysis/mia_scan_report_v1.json")]
res["code_sha256"] = sha(__file__); res["sources"] = srcs
json.dump(res, open(OUT + "/cpu/c4_mia_funnel.json", "w"), indent=1)
pd.DataFrame([{"step": k, "n": v} for k, v in res.items() if isinstance(v, int)]).to_csv(OUT + "/cpu/c4_mia_funnel.csv", index=False)
print({k: v for k, v in res.items() if k != "sources"})
