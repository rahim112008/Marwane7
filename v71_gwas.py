"""
v71_gwas.py — GWAS LMM (EMMA-like) + QQ-plot + λGC
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import eigh


def _estimate_reml_delta(Y, X, K):
    """Estimation REML de delta = sigma_g^2 / sigma_e^2 (grille)."""
    n = len(Y)
    d, U = eigh(K)
    d = np.maximum(d, 0)
    Yt = U.T @ Y
    Xt = U.T @ X

    def neg_ll(delta):
        if delta <= 0:
            return 1e18
        w = 1/np.sqrt(delta*d + 1)
        Yw = Yt*w
        Xw = Xt*w[:, None]
        try:
            beta, *_ = np.linalg.lstsq(Xw, Yw, rcond=None)
        except np.linalg.LinAlgError:
            return 1e18
        res = Yw - Xw @ beta
        s2 = float(np.sum(res**2)/n)
        if s2 <= 0:
            return 1e18
        return -(-0.5*(n*np.log(s2) + np.sum(np.log(delta*d + 1))))

    grid = np.logspace(-3, 3, 30)
    lls = np.array([neg_ll(dl) for dl in grid])
    return float(grid[np.argmin(lls)])


def gwas_lmm(gt, pheno, K, cov=None, maf_min=0.05, miss_max=0.05):
    """
    GWAS LMM (EMMA-style).

    gt : (n_ind, n_snp) dosage 0/1/2
    pheno : (n_ind,) phénotype
    K : (n_ind, n_ind) matrice de parenté
    cov : (n_ind, n_cov) covariables (ex: PCs)
    """
    n_all, n_snp = gt.shape
    pheno = np.asarray(pheno, dtype=float)
    valid_ind = np.isfinite(pheno)
    if valid_ind.sum() < 10:
        return pd.DataFrame()

    Y = pheno[valid_ind]
    gt_v = gt[valid_ind]
    K_v = np.asarray(K)[np.ix_(valid_ind, valid_ind)]

    if cov is None or cov.size == 0:
        X0 = np.ones((len(Y), 1))
    else:
        X0 = np.column_stack([np.ones(len(Y)), np.asarray(cov)[valid_ind]])

    keep = np.ones(X0.shape[1], dtype=bool)
    for j in range(1, X0.shape[1]):
        if X0[:, j].std() < 1e-10:
            keep[j] = False
    X0 = X0[:, keep]
    n = len(Y)

    delta = _estimate_reml_delta(Y, X0, K_v)
    d, U = eigh(K_v)
    d = np.maximum(d, 0)
    w = 1/np.sqrt(delta*d + 1)
    Yt = (U.T @ Y) * w
    X0t = (U.T @ X0) * w[:, None]

    out = {"SNP_idx": [], "MAF": [], "BETA": [], "SE": [], "T": [], "P": []}

    for j in range(n_snp):
        g = gt_v[:, j].astype(float)
        mask = ~np.isfinite(g)
        if mask.mean() > miss_max:
            continue
        if mask.any():
            m_g = np.nanmean(g)
            if not np.isfinite(m_g):
                continue
            g = np.where(mask, m_g, g)

        p = g.mean()/2.0
        maf_j = min(p, 1-p)
        if maf_j < maf_min:
            continue

        gw = (U.T @ g) * w
        X = np.column_stack([X0t, gw])
        try:
            beta, *_ = np.linalg.lstsq(X, Yt, rcond=None)
            res = Yt - X @ beta
            dof = max(n - X.shape[1], 1)
            s2 = float(np.sum(res**2)/dof)
            if s2 <= 0:
                continue
            XtX_inv = np.linalg.inv(X.T @ X)
            se = float(np.sqrt(s2 * XtX_inv[-1, -1]))
            if se <= 0 or not np.isfinite(se):
                continue
            t = beta[-1]/se
            pv = float(2*stats.t.sf(abs(t), df=dof))
            out["SNP_idx"].append(j)
            out["MAF"].append(float(maf_j))
            out["BETA"].append(float(beta[-1]))
            out["SE"].append(se)
            out["T"].append(float(t))
            out["P"].append(pv)
        except np.linalg.LinAlgError:
            continue

    if not out["SNP_idx"]:
        return pd.DataFrame(columns=["SNP_idx","MAF","BETA","SE","T","P"])
    return pd.DataFrame(out)


# ============ QQ + LAMBDA ============

def lambda_gc(pvals):
    """λGC (genomic inflation factor)."""
    p = np.asarray(pvals)
    p = p[np.isfinite(p) & (p > 0) & (p <= 1)]
    if len(p) < 10:
        return np.nan
    chi2_obs = stats.chi2.isf(p, df=1)
    return float(np.median(chi2_obs) / stats.chi2.ppf(0.5, df=1))


def lambda_gc_permuted(pvals, n_perm=1000, seed=42):
    """λGC attendu sous H0 (1000 permutations)."""
    rng = np.random.default_rng(seed)
    p = np.asarray(pvals)
    p = p[np.isfinite(p) & (p > 0) & (p <= 1)]
    if len(p) < 10:
        return np.nan
    vals = []
    for _ in range(n_perm):
        p_perm = rng.uniform(size=len(p))
        chi2 = stats.chi2.isf(p_perm, df=1)
        vals.append(np.median(chi2) / stats.chi2.ppf(0.5, df=1))
    return float(np.mean(vals))


def qq_data(pvals):
    """Données pour QQ-plot."""
    p = np.asarray(pvals)
    p = p[np.isfinite(p) & (p > 0) & (p <= 1)]
    if len(p) < 10:
        return None
    p = np.sort(p)
    n = len(p)
    expected = (np.arange(1, n+1) - 0.5) / n
    return {
        "observed": -np.log10(p),
        "expected": -np.log10(expected),
        "n": n,
    }


def gwas_thresholds(pvals, alpha=0.05):
    """Seuils Bonferroni + FDR."""
    p = np.asarray(pvals)
    p = p[np.isfinite(p)]
    n = len(p)
    if n == 0:
        return {"bonferroni": np.nan, "fdr_05": np.nan, "n_tests": 0}
    bonf = alpha/n
    ps = np.sort(p)
    bh = ps*n/np.arange(1, n+1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    sig = np.where(bh <= alpha)[0]
    fdr_thr = float(ps[sig[-1]]) if len(sig) else np.nan
    return {"bonferroni": float(bonf), "fdr_05": fdr_thr,
            "n_tests": int(n),
            "n_sig_bonf": int((p <= bonf).sum()),
            "n_sig_fdr": int(len(sig))}
