"""
v71_statistics.py — Weir & Cockerham FST + bootstrap IC + corrections
"""
import numpy as np
import pandas as pd
from scipy import stats

# ============ WEIR & COCKERHAM ============

def fst_weir_cockerham(gt, labels, min_valid=2):
    """FST de Weir & Cockerham (1984) par SNP."""
    labels = np.asarray(labels, dtype=object)
    pops = sorted(np.unique(labels))
    n_pop = len(pops)
    if n_pop < 2:
        return np.full(gt.shape[1], np.nan), np.nan, np.nan

    n_snp = gt.shape[1]
    p_mat = np.full((n_pop, n_snp), np.nan)
    n_mat = np.zeros((n_pop, n_snp))

    for i, p in enumerate(pops):
        sub = gt[labels == p]
        valid = ~np.isnan(sub)
        n_mat[i] = valid.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            p_mat[i] = np.nansum(sub, axis=0) / (2.0 * n_mat[i])

    good = (n_mat >= min_valid).all(axis=0)
    fst = np.full(n_snp, np.nan)
    a_arr = np.full(n_snp, np.nan)
    b_arr = np.full(n_snp, np.nan)
    c_arr = np.full(n_snp, np.nan)

    for j in np.where(good)[0]:
        p_j, n_j = p_mat[:, j], n_mat[:, j]
        if np.any(~np.isfinite(p_j)) or np.any(p_j < 0.01) or np.any(p_j > 0.99):
            continue
        n_bar = n_j.sum() / n_pop
        nc = (n_pop*n_bar - (n_j**2).sum()/(n_pop*n_bar)) / (n_pop-1)
        p_bar = (n_j*p_j).sum() / n_j.sum()
        s2 = (n_j*(p_j-p_bar)**2).sum() / ((n_pop-1)*n_bar)
        a = (n_bar/nc) * (s2 - (1/(n_bar-1))*(p_bar*(1-p_bar) - (n_pop-1)/n_pop*s2))
        b = (n_bar/(n_bar-1)) * (p_bar*(1-p_bar) - (n_pop-1)/n_pop*s2 - a/2)
        c = 1 - p_bar*(1-p_bar) - b - a/2
        denom = a + b + c
        if denom > 1e-9:
            fst[j] = a/denom
            a_arr[j], b_arr[j], c_arr[j] = a, b, c

    valid = np.isfinite(fst)
    if valid.sum() > 0:
        a_s = np.nansum(a_arr[valid])
        b_s = np.nansum(b_arr[valid])
        c_s = np.nansum(c_arr[valid])
        theta = a_s/(a_s+b_s+c_s) if (a_s+b_s+c_s) > 1e-9 else np.nan
    else:
        theta = np.nan

    se = np.nan
    if valid.sum() > 5:
        fc = fst[valid]
        n = len(fc)
        jack = np.array([fc[np.arange(n) != i].mean() for i in range(n)])
        se = np.sqrt((n-1)*np.var(jack))

    return fst, theta, se


# ============ BOOTSTRAP IC ============

def bootstrap_fst_ci(gt, labels, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap IC 95% sur FST global."""
    rng = np.random.default_rng(seed)
    n = gt.shape[0]
    labels = np.asarray(labels, dtype=object)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        lb = labels[idx]
        if len(np.unique(lb)) < 2:
            continue
        _, theta, _ = fst_weir_cockerham(gt[idx], lb)
        if np.isfinite(theta):
            values.append(theta)
    if len(values) < 10:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
    values = np.array(values)
    a = (1-ci)/2
    return {"mean": float(np.mean(values)),
            "ci_low": float(np.quantile(values, a)),
            "ci_high": float(np.quantile(values, 1-a)),
            "n": len(values)}


def jackknife_ne_ci(gt, snps, ci=0.95):
    """Jackknife par chromosome pour Ne."""
    chroms = snps["CHR"].astype(str).values
    unique = sorted(np.unique(chroms))
    if len(unique) < 3:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    values = []
    for c in unique:
        mask = chroms != c
        if mask.sum() < 20:
            continue
        try:
            ne_vals = _ne_fast(gt[:, mask], snps[mask].reset_index(drop=True))
            if ne_vals.size > 0:
                values.append(float(np.mean(ne_vals)))
        except Exception:
            continue
    if len(values) < 3:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    values = np.array(values)
    n = len(values)
    mu = values.mean()
    se = np.sqrt((n-1)/n * np.sum((values - mu)**2))
    return {"mean": float(mu),
            "ci_low": float(mu - 1.96*se),
            "ci_high": float(mu + 1.96*se)}


def _ne_fast(gt, snps, max_snp=500):
    bp = snps["BP"].astype(float).values
    ch = snps["CHR"].astype(str).values
    n = gt.shape[1]
    if n > max_snp:
        idx = np.linspace(0, n-1, max_snp).astype(int)
        gt, bp, ch = gt[:, idx], bp[idx], ch[idx]
    X = gt.astype(np.float32).copy()
    cm = np.nanmean(X, axis=0); cm = np.where(np.isnan(cm), 0, cm)
    nm = np.isnan(X)
    X[nm] = np.take(cm, np.where(nm)[1])
    r2l, dl = [], []
    m = X.shape[1]
    for i in range(m-1):
        for j in range(i+1, min(i+20, m)):
            if ch[i] != ch[j]:
                continue
            d = abs(bp[j]-bp[i])/1e6
            if d < 0.001 or d > 50:
                continue
            with np.errstate(invalid="ignore"):
                r = np.corrcoef(X[:, i], X[:, j])[0, 1]
            if np.isfinite(r):
                r2l.append(r*r); dl.append(d)
    if not dl:
        return np.array([])
    r2, d = np.array(r2l), np.array(dl)
    bins = np.logspace(np.log10(d.min()+1e-6), np.log10(d.max()), 10)
    bc = (bins[:-1]+bins[1:])/2
    rb = np.array([r2[(d >= bins[k]) & (d < bins[k+1])].mean()
                   if ((d >= bins[k]) & (d < bins[k+1])).sum() > 0
                   else np.nan for k in range(len(bins)-1)])
    m_ = np.isfinite(rb)
    if m_.sum() < 2:
        return np.array([])
    c = bc[m_]*0.01
    ne = np.where(rb[m_] > 1e-4, 1/(4*c*rb[m_]), np.nan)
    return ne[np.isfinite(ne) & (ne > 0)]


# ============ CORRECTIONS MULTI-TESTS ============

def bonferroni(p):
    p = np.asarray(p)
    return np.minimum(p*len(p), 1.0)


def benjamini_hochberg(p):
    """FDR Benjamini-Hochberg."""
    p = np.asarray(p)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n+1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    res = np.empty_like(p)
    res[order] = q
    return res


def storey_qvalues(p, lam=0.5):
    """Estimation Storey des q-values."""
    p = np.asarray(p)
    p = p[np.isfinite(p)]
    n = len(p)
    if n < 10:
        return np.full(n, np.nan)
    pi0 = (p > lam).sum() / (n*(1-lam))
    pi0 = min(1.0, max(0.01, pi0))
    order = np.argsort(p)
    ranked = p[order]
    q = pi0*ranked*n/np.arange(1, n+1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    res = np.empty_like(p)
    res[order] = np.minimum(q, 1.0)
    return res


# ============ TEST DE PERMUTATION ============

def permutation_test(gt, labels, stat_func, n_perm=1000, seed=42):
    """Test de permutation générique."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=object)
    obs = stat_func(gt, labels)
    if not np.isfinite(obs):
        return obs, np.nan
    null = []
    for _ in range(n_perm):
        v = stat_func(gt, rng.permutation(labels))
        if np.isfinite(v):
            null.append(v)
    if not null:
        return obs, np.nan
    p = (np.sum(np.array(null) >= obs) + 1) / (len(null) + 1)
    return float(obs), float(p)


# ============ POWER ANALYSIS ============

def power_fst(fst_true, n_per_pop, n_pop, alpha=0.05):
    """Puissance approx pour détecter FST=true."""
    ncp = n_per_pop * n_pop * fst_true / (2*(1-fst_true))
    df = n_pop - 1
    crit = stats.chi2.ppf(1-alpha, df)
    power = 1 - stats.ncx2.cdf(crit, df, ncp)
    return float(power)


def required_n(fst_true, n_pop, power=0.8, alpha=0.05):
    """N requis par pop pour atteindre la puissance."""
    for n in range(2, 500):
        if power_fst(fst_true, n, n_pop, alpha) >= power:
            return n
    return None


# ============ SENSIBILITÉ AUX SEUILS ============

def sensitivity_grid(gt, ind, snp, qc_filters_fn,
                     maf_vals=(0.01, 0.05, 0.10),
                     hwe_vals=(1e-4, 1e-6, 1e-8),
                     miss_vals=(0.02, 0.05, 0.10)):
    """Teste 27 combinaisons de seuils QC."""
    results = []
    for maf in maf_vals:
        for hwe in hwe_vals:
            for miss in miss_vals:
                p = {"geno": miss, "mind": miss, "maf": maf,
                     "hwe": hwe, "het_sd": 3.0}
                try:
                    g, i, s, st, _, _, _ = qc_filters_fn(gt, ind, snp, p)
                    results.append({
                        "MAF": maf, "HWE": hwe, "Missing": miss,
                        "n_ind": st["n_ind_final"],
                        "n_snp": st["n_snp_final"],
                        "pct_ind": round(100*st["excluded_ind"]/st["n_ind_init"], 2),
                        "pct_snp": round(100*st["excluded_snp"]/st["n_snp_init"], 2)})
                except Exception as e:
                    results.append({"MAF": maf, "HWE": hwe,
                                     "Missing": miss, "error": str(e)[:50]})
    return pd.DataFrame(results)
