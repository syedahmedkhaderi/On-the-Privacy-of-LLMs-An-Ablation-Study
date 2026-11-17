"""Exact pure functions extracted from the recorded DEA analysis; see SOURCES.csv."""
import numpy as np
from scipy.stats import t

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
