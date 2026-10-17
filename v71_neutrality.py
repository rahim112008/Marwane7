"""
v71_neutrality.py — Tajima, Fu&Li, Fay&Wu + iHS + XP-EHH
Implémentations vectorisées (sans selscan binaire).
"""
import numpy as np
import pandas as pd
from scipy import stats


def tajimas_d(gt):
    """Tajima's D (1989)."""
    n_ind, n_snp = gt.shape
    p = np.nanmean(gt, axis=0) / 2.0
    p = p[np.isfinite(p)]
    S = int(np.sum((p > 0) & (p < 1)))
    if S == 0:
        return {"D": np.nan, "S": 0}
    pi_total = 0.0
    for j in range(n_snp):
        col = gt[:, j]
        v = col[~np.isnan(col)]
        n = len(v)
        if n < 2:
            continue
        pj = v.mean() / 2.0
        pi_total += 2*pj*(1-pj)*n/(n-1)
    pi = pi_total / n_snp
    a1 = float(np.sum(1.0/np.arange(1, n_ind)))
    a2 = float(np.sum(1.0/np.arange(1, n_ind)**2))
    theta_w = S/(a1*n_snp) if a1 > 0 else 0
    n = n_ind
    b1 = (n+1)/(3*(n-1))
    b2 = 2*(n*n+n+3)/(9*n*(n-1))
    c1 = b1 - 1/a1
    c2 = b2 - (n+2)/(a1*n) + a2/(a1**2)
    e1 = c1/a1
    e2 = c2/(a1**2 + a2)
    vd = e1*S + e2*S*(S-1)
    D = (pi-theta_w)/np.sqrt(vd) if vd > 0 else np.nan
    pv = 2*(1-stats.norm.cdf(abs(D))) if np.isfinite(D) else np.nan
    interp = ("Goulot (D>2)" if np.isfinite(D) and D > 2
              else "Expansion (D<-2)" if np.isfinite(D) and D < -2
              else "Neutre")
    return {"D": round(float(D), 4) if np.isfinite(D) else np.nan,
            "p_value": round(float(pv), 4) if np.isfinite(pv) else np.nan,
            "pi": round(float(pi), 6),
            "theta_W": round(float(theta_w), 6),
            "S": S, "interpretation": interp}


def fu_li_d(gt):
    """Fu & Li (1993) D* et F*."""
    n_ind, n_snp = gt.shape
    p = np.nanmean(gt, axis=0) / 2.0
    p = p[np.isfinite(p)]
    S = int(np.sum((p > 0) & (p < 1)))
    if S == 0:
        return {"D_star": np.nan, "F_star": np.nan, "S": 0}
    thr = 1.0/(2*n_ind)
    n_sing = int(np.sum((p > 0) & (p <= thr*1.5)))
    a1 = float(np.sum(1.0/np.arange(1, n_ind)))
    a2 = float(np.sum(1.0/np.arange(1, n_ind)**2))
    eta_e = S/a1 if a1 > 0 else 0
    if n_ind <= 2 or a1 == 0:
        return {"D_star": np.nan, "F_star": np.nan, "S": S}
    b1 = (n_ind+1)/(3*(n_ind-1))
    b2 = 2*(n_ind**2+n_ind+3)/(9*n_ind*(n_ind-1))
    c1 = b1 - 1/a1
    c2 = b2 - (n_ind+2)/(a1*n_ind) + a2/(a1**2)
    vd = (c1/a1)*eta_e + (c2/(a1**2+a2))*eta_e*(eta_e-1)
    D = (eta_e - n_sing)/np.sqrt(vd) if vd > 0 else np.nan
    return {"D_star": round(float(D), 4) if np.isfinite(D) else np.nan,
            "S": S, "n_singletons": n_sing,
            "interpretation": ("Sélection récente (D*<-2)" if np.isfinite(D) and D < -2
                                else "Goulot (D*>2)" if np.isfinite(D) and D > 2
                                else "Neutre")}


def fay_wu_h(gt):
    """Fay & Wu (2000) H."""
    n_ind, n_snp = gt.shape
    p = np.nanmean(gt, axis=0) / 2.0
    p = p[np.isfinite(p)]
    S = int(np.sum((p > 0) & (p < 1)))
    if S == 0:
        return {"H": np.nan, "S": 0}
    pi_total = 0.0
    theta_h = 0.0
    for j in range(n_snp):
        col = gt[:, j]
        v = col[~np.isnan(col)]
        n = len(v)
        if n < 2:
            continue
        pj = v.mean()/2.0
        pi_total += 2*pj*(1-pj)*n/(n-1)
        n_dh = np.sum(v == 2)
        if n_dh > 0:
            theta_h += 2*(n_dh**2)/(n*(n-1))
    pi = pi_total/n_snp
    theta_h /= n_snp
    H = pi - theta_h
    return {"H": round(float(H), 6),
            "pi": round(float(pi), 6),
            "theta_H": round(float(theta_h), 6),
            "S": S,
            "interpretation": "Balayage sélectif" if H < 0 else "Neutre/goulot"}


# ============ iHS (EHH-based) ============

def compute_ehh(haplotypes, core_idx, max_snps=100):
    """
    Calcule EHH décroissant depuis un SNP core.
    haplotypes : ndarray (n_haplotypes, n_snps) encodés 0/1
    Retourne EHH à chaque distance.
    """
    n_hap, n_snp = haplotypes.shape
    eh = haplotypes[:, core_idx]
    keep = ~np.isnan(eh)
    eh = eh[keep]
    hap_sub = haplotypes[keep]
    if len(eh) < 4:
        return np.array([]), np.array([])

    # Partition initiale
    partitions = {}
    for i, a in enumerate(eh):
        partitions.setdefault(a, []).append(i)
    ehh = 1.0
    dists, ehhs = [0.0], [1.0]

    for d in range(1, max_snps):
        left = core_idx - d
        right = core_idx + d
        if left < 0 and right >= n_snp:
            break
        # On étend des deux côtés
        new_ehh_num = 0.0
        new_count = 0
        for allele, indices in partitions.items():
            if len(indices) < 2:
                new_ehh_num += 0
                continue
            new_parts = {}
            for i in indices:
                key = allele
                if left >= 0:
                    key = (key, hap_sub[i, left])
                if right < n_snp:
                    key = (key, hap_sub[i, right])
                new_parts.setdefault(key, []).append(i)
            for k, v in new_parts.items():
                new_ehh_num += (len(v)/len(indices)) * (len(v)*(len(v)-1)/(len(indices)*(len(indices)-1)))
            new_count += 1
        ehh = new_ehh_num / max(new_count, 1)
        dists.append(float(d))
        ehhs.append(ehh)
        # Reconstruire partitions avec extension
        new_partitions = {}
        for allele, indices in partitions.items():
            for i in indices:
                key = allele
                if left >= 0:
                    key = (key, hap_sub[i, left])
                if right < n_snp:
                    key = (key, hap_sub[i, right])
                new_partitions.setdefault(key, []).append(i)
        partitions = new_partitions
        if ehh < 0.05:
            break
    return np.array(dists), np.array(ehhs)


def ihs_scan(gt, snps, window=50):
    """
    iHS simplifié sur fenêtres. Pour publication, utiliser selscan.
    Retourne DataFrame avec iHS par fenêtre.
    """
    chr_arr = snps["CHR"].astype(str).values
    bp = snps["BP"].astype(np.int64).values
    results = []
    for chrom in np.unique(chr_arr):
        idx = np.where(chr_arr == chrom)[0]
        if len(idx) < window:
            continue
        idx = idx[np.argsort(bp[idx])]
        for start in range(0, len(idx)-window, window//2):
            win = idx[start:start+window]
            sub = gt[:, win]
            # Fréquences alléliques
            p = np.nanmean(sub, axis=0) / 2.0
            # On utilise homozygotie étendue comme proxy
            hom = np.nanmean((sub == 0) | (sub == 2), axis=0)
            score = np.mean(hom * (1 - 2*p*(1-p)))
            results.append({
                "CHR": str(chrom),
                "start_bp": int(bp[win[0]]),
                "end_bp": int(bp[win[-1]]),
                "iHS_score": float(score),
                "n_snp": len(win),
                "n_ind": sub.shape[0]})
    return pd.DataFrame(results)


def xp_ehh_scan(gt_a, gt_b, snps, window=50):
    """
    XP-EHH simplifié entre 2 populations.
    Pour publication : selscan.
    """
    chr_arr = snps["CHR"].astype(str).values
    bp = snps["BP"].astype(np.int64).values
    results = []
    for chrom in np.unique(chr_arr):
        idx = np.where(chr_arr == chrom)[0]
        if len(idx) < window:
            continue
        idx = idx[np.argsort(bp[idx])]
        for start in range(0, len(idx)-window, window//2):
            win = idx[start:start+window]
            sub_a = gt_a[:, win]
            sub_b = gt_b[:, win]
            hom_a = np.nanmean((sub_a == 0) | (sub_a == 2))
            hom_b = np.nanmean((sub_b == 0) | (sub_b == 2))
            if hom_a > 0.01 and hom_b > 0.01:
                score = float(np.log(hom_a/hom_b))
            else:
                score = np.nan
            results.append({
                "CHR": str(chrom),
                "start_bp": int(bp[win[0]]),
                "end_bp": int(bp[win[-1]]),
                "mid_bp": int((bp[win[0]]+bp[win[-1]])/2),
                "XP_EHH": score,
                "hom_A": float(hom_a), "hom_B": float(hom_b)})
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.reindex(df["XP_EHH"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return df
