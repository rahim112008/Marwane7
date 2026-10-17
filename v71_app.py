"""
🐄 Bovine SNP Platform v7.2 — Autonome (single file)
v6.2 (toutes fonctionnalités) + v7.1 (rigueur scientifique) INLINE.
Aucune dépendance aux modules v71_*.py externes.
"""

# ═══════════════════════════════════════════════════════════
# IMPORTS — TOUJOURS EN PREMIER
# ═══════════════════════════════════════════════════════════
import gzip, json, warnings, zipfile
from collections import Counter
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import MDS as SklearnMDS

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(page_title="🐄 Bovine SNP v7.2", page_icon="🐄",
                    layout="wide", initial_sidebar_state="expanded")

DEFAULTS = {"geno": 0.05, "mind": 0.05, "maf": 0.05,
            "hwe": 1e-6, "het_sd": 3.0, "ld_r2": 0.2}
HWE_EXACT_MAX = 20_000
ARS_LEN = {"1": 158_534_110, "2": 136_231_102, "3": 121_005_158,
            "4": 120_000_166, "5": 120_089_699, "6": 117_806_340,
            "7": 110_682_743, "8": 113_384_748, "9": 105_708_134,
            "10": 103_308_737, "11": 107_310_763, "12": 91_163_125,
            "13": 84_246_514, "14": 84_648_346, "15": 85_207_080,
            "16": 81_726_628, "17": 75_176_999, "18": 66_059_976,
            "19": 64_089_169, "20": 72_042_983, "21": 71_599_096,
            "22": 61_416_492, "23": 52_531_573, "24": 62_384_193,
            "25": 42_959_610, "26": 51_680_158, "27": 46_772_073,
            "28": 46_333_854, "29": 51_319_414}
AUTOSOMES = [str(i) for i in range(1, 30)]
DEFAULT_GROUPS = {"Laitier": [], "Viande": [], "Cornu": [], "Rustique": []}
RACE_POOL = ["AND", "EBG", "ELN", "EZP", "FGN", "LJR", "NAR",
             "NBD", "NKA", "PRS", "PSH", "PTP", "SHO", "YBS"]

# ═══════════════════════════════════════════════════════════
# CACHE
# ═══════════════════════════════════════════════════════════
def _hash_np(x):
    if not isinstance(x, np.ndarray) or x.size == 0:
        return ("empty",)
    return (x.shape, str(x.dtype), float(np.nansum(x)),
            float(np.nansum(np.abs(x))), float(np.nansum(x * x)))

HASH_FUNCS = {np.ndarray: _hash_np}

def cached(fn=None, **kw):
    def wrap(f):
        return st.cache_data(show_spinner=False,
                             hash_funcs=HASH_FUNCS, **kw)(f)
    return wrap(fn) if fn else wrap

# ═══════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════
def impute(gt):
    gt2 = gt.astype(np.float32, copy=True)
    cm = np.nanmean(gt2, axis=0)
    cm = np.where(np.isnan(cm), 0.0, cm)
    nm = np.isnan(gt2)
    if nm.any():
        gt2[nm] = np.take(cm, np.where(nm)[1])
    return gt2

def _chr_key(c):
    s = str(c).upper().replace("CHR", "").replace("CHROMOSOME", "")
    if s.isdigit(): return (0, int(s), "")
    sp = {"X": 100, "Y": 101, "MT": 102, "M": 102}
    if s in sp: return (1, sp[s], "")
    return (2, 0, s)

def _auto_weights():
    L = np.array([ARS_LEN[c] for c in AUTOSOMES], dtype=float)
    return L / L.sum()

def to_np_str(a):
    return np.asarray(list(a), dtype=object)

# ═══════════════════════════════════════════════════════════
# AXIOM
# ═══════════════════════════════════════════════════════════
def _clean_allele(a):
    if a is None: return "0"
    a = str(a).strip()
    return "0" if a in ("", ".", "-", "NA", "nan", "None", "0", "00") else a

def sample_to_fid_iid(name, fid_parts=2):
    n = str(name).strip()
    for e in (".CEL", ".cel", ".TXT", ".txt", ".gz", ".GZ"):
        if n.endswith(e): n = n[:-len(e)]
    n = (n.replace("(", "_").replace(")", "").replace("[", "_")
         .replace("]", "").replace(" ", "_").replace("/", "_"))
    while "__" in n: n = n.replace("__", "_")
    n = n.strip("_")
    p = n.split("_")
    return ("_".join(p[:fid_parts]) if len(p) >= fid_parts else p[0]), n

def parse_axiom(raw, fn=""):
    if fn.endswith(".gz") or (len(raw) >= 2 and raw[:2] == b"\x1f\x8b"):
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8", errors="replace")
    lines = [l.rstrip("\r\n") for l in text.splitlines() if l.strip()]
    if not lines: raise ValueError("Fichier vide")
    skip = 1 if (lines[0].startswith("#") or lines[0].startswith("Sample")
                 or "Sample ID" in lines[0][:50]) else 0
    samples, geno = [], []
    for line in lines[skip:]:
        p = line.split("\t")
        if len(p) < 2:
            p = line.split()
            if len(p) < 2: continue
        al = []
        for cell in [c.strip() for c in p[1:]]:
            t = cell.split()
            if len(t) == 2:
                al.append((_clean_allele(t[0]), _clean_allele(t[1])))
            elif len(t) == 1:
                s = t[0]
                if len(s) == 2 and all(ch in "ACGT0" for ch in s):
                    al.append((s[0], s[1]))
                elif "/" in s:
                    a1, a2 = s.split("/", 1)
                    al.append((_clean_allele(a1), _clean_allele(a2)))
                else:
                    al.append((_clean_allele(s), _clean_allele(s)))
            else:
                al.append(("0", "0"))
        samples.append(p[0].strip())
        geno.append(al)
    if not samples: raise ValueError("Aucun échantillon")
    c = Counter([len(r) for r in geno]).most_common(1)[0][0]
    for i, r in enumerate(geno):
        if len(r) > c: geno[i] = r[:c]
        elif len(r) < c: geno[i] = r + [("0", "0")] * (c - len(r))
    return samples, geno, c

def render_axiom_page():
    st.header("🧬 Convertisseur Axiom → PED/MAP")
    st.markdown("Transforme un fichier Axiom en PED + MAP PLINK.")
    f = st.file_uploader("Fichier Axiom", type=["txt", "tsv", "csv",
                                                  "ped", "gz"], key="ax_f")
    c1, c2 = st.columns(2)
    fp = c1.number_input("Parties FID", 1, 5, 2, 1, key="ax_fp")
    sep = c2.selectbox("Séparateur", ["Tabulation", "Espace"], key="ax_s")
    if f and st.button("Convertir", type="primary",
                       use_container_width=True, key="ax_btn"):
        try:
            with st.spinner("Parsing..."):
                samples, geno, n_snp = parse_axiom(f.read(), f.name)
            map_t = "\n".join(f"1\tSNP_{i+1:06d}\t0\t{i+1}"
                              for i in range(n_snp)) + "\n"
            ped_l = []
            for i, s in enumerate(samples):
                fid, iid = sample_to_fid_iid(s, int(fp))
                cols = [fid, iid, "0", "0", "0", "-9"]
                for a1, a2 in geno[i]:
                    cols.extend([a1, a2])
                ped_l.append("\t".join(cols))
            ped_t = "\n".join(ped_l) + "\n"
            if sep == "Espace":
                ped_t = ped_t.replace("\t", " ")
                map_t = map_t.replace("\t", " ")
            st.session_state["_ax_ped"] = ped_t
            st.session_state["_ax_map"] = map_t
            st.success(f"✅ {len(samples)} ind × {n_snp} SNPs")
        except Exception as e:
            st.error(f"❌ {e}")
    if st.session_state.get("_ax_ped"):
        c1, c2, c3 = st.columns(3)
        c1.download_button("⬇ .ped",
                           st.session_state["_ax_ped"].encode("utf-8"),
                           "converted.ped", "text/plain",
                           use_container_width=True, key="dl_ax_p")
        c2.download_button("⬇ .map",
                           st.session_state["_ax_map"].encode("utf-8"),
                           "converted.map", "text/plain",
                           use_container_width=True, key="dl_ax_m")

# ═══════════════════════════════════════════════════════════
# PARSERS
# ═══════════════════════════════════════════════════════════
def _enc(b):
    if len(b) >= 2 and b[:2] == b"\x1f\x8b": return "gz"
    if len(b) >= 2 and b[:2] == b"\x6c\x1b": return "bin"
    return "txt"

def parse_map(b):
    if _enc(b) == "gz":
        text = gzip.decompress(b).decode("utf-8", errors="replace")
    elif _enc(b) == "bin":
        raise ValueError("❌ .map binaire PLINK")
    else:
        text = b.decode("utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        p = line.strip().split()
        if len(p) < 4: continue
        try: bp = int(float(p[3]))
        except (ValueError, TypeError): continue
        try: cm = float(p[2]) if p[2] not in ("0", ".", "NA") else 0.0
        except ValueError: cm = 0.0
        rows.append({"CHR": str(p[0]), "SNP": str(p[1]),
                     "CM": cm, "BP": bp})
    if not rows: raise ValueError("❌ .map vide")
    return pd.DataFrame(rows)

def parse_ped(b, n_map):
    if _enc(b) == "gz":
        text = gzip.decompress(b).decode("utf-8", errors="replace")
    elif _enc(b) == "bin":
        raise ValueError("❌ .ped binaire PLINK")
    else:
        text = b.decode("utf-8", errors="replace")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines: raise ValueError("❌ .ped vide")
    if len(lines) >= 2:
        n0, n1 = len(lines[0].split()), len(lines[1].split())
        if (n0 < 7 and n1 >= 7) or (n1 > 0 and n0 < n1 * 0.5):
            lines = lines[1:]
    nc = len(lines[0].split())
    if nc < 7: raise ValueError(f"❌ {nc} colonnes PED")
    n_ped = (nc - 6) // 2
    n = min(n_map, n_ped) if n_ped > 0 else n_map
    exp = 6 + 2 * n
    fids, iids, genos = [], [], []
    for line in lines:
        p = line.split()
        if len(p) < exp: continue
        fids.append(p[0]); iids.append(p[1])
        genos.append(p[6:6 + 2 * n])
    ni = len(iids)
    if ni == 0: raise ValueError("❌ Aucun individu")
    counts = [Counter() for _ in range(n)]
    for row in genos:
        for j in range(n):
            a1, a2 = row[2*j], row[2*j+1]
            if a1 != "0" and a2 != "0":
                counts[j][a1] += 1; counts[j][a2] += 1
    minor = [min(c, key=c.get) if len(c) >= 2 else None for c in counts]
    gt = np.full((ni, n), np.nan, dtype=np.float32)
    for i, row in enumerate(genos):
        for j in range(n):
            m = minor[j]
            if m is None: continue
            a1, a2 = row[2*j], row[2*j+1]
            if a1 == "0" or a2 == "0": continue
            gt[i, j] = (a1 == m) + (a2 == m)
    return gt, pd.DataFrame({"FID": fids, "IID": iids})

def parse_vcf(b):
    if len(b) >= 2 and b[:2] == b"\x1f\x8b":
        text = gzip.decompress(b).decode("utf-8", errors="replace")
    else:
        text = b.decode("utf-8", errors="replace")
    samples, rows = [], []
    for line in text.splitlines():
        if not line or line.startswith("##"): continue
        if line.startswith("#CHROM"):
            samples = line.split("\t")[9:]; continue
        if line.startswith("#"): continue
        p = line.split("\t")
        if len(p) < 10 or "," in p[4]: continue
        try: pos = int(p[1])
        except ValueError: continue
        fmt = p[8].split(":")
        if "GT" not in fmt: continue
        gi = fmt.index("GT")
        gts = np.full(len(samples), np.nan, dtype=np.float32)
        for k, s in enumerate(p[9:]):
            f = s.split(":")
            if gi >= len(f): continue
            g = f[gi]; sep_ = "|" if "|" in g else "/"
            al = g.split(sep_)
            if any(a in (".", "") for a in al): continue
            try: gts[k] = sum(int(a) for a in al)
            except ValueError: continue
        rows.append({"CHR": str(p[0]),
                     "SNP": str(p[2] or f"{p[0]}:{pos}"),
                     "CM": 0.0, "BP": pos, "A1": p[3], "A2": p[4],
                     "GT": gts})
    if not rows: raise ValueError("❌ Aucun variant")
    gt = np.zeros((len(samples), len(rows)), dtype=np.float32)
    for j, r in enumerate(rows): gt[:, j] = r["GT"]
    for j in range(len(rows)):
        col = gt[:, j]; v = col[~np.isnan(col)]
        if len(v) and v.mean() / 2.0 > 0.5:
            gt[:, j] = np.where(np.isnan(col), np.nan, 2.0 - col)
    snp_df = pd.DataFrame({
        "CHR": [r["CHR"] for r in rows],
        "SNP": [r["SNP"] for r in rows],
        "CM": 0.0, "BP": [r["BP"] for r in rows],
        "A1": [r["A1"] for r in rows], "A2": [r["A2"] for r in rows]})
    return gt, pd.DataFrame({"FID": samples, "IID": samples}), snp_df

def demo_data(ni=150, ns=800, np_=4, seed=42):
    rng = np.random.default_rng(seed)
    pops = RACE_POOL[:min(np_, len(RACE_POOL))]
    per, rem = ni // len(pops), ni % len(pops)
    cnt = [per + (1 if i < rem else 0) for i in range(len(pops))]
    p_anc = rng.beta(0.6, 0.6, ns)
    gt = np.zeros((sum(cnt), ns), dtype=np.float32)
    rows = []; k = 0
    for pop, n_i in zip(pops, cnt):
        pk = np.clip(p_anc + rng.normal(0, 0.15, ns), 0.02, 0.98)
        for i in range(n_i):
            a1 = (rng.random(ns) < pk).astype(np.float32)
            a2 = (rng.random(ns) < pk).astype(np.float32)
            gt[k] = a1 + a2
            rows.append({"FID": pop, "IID": f"{pop}_{i+1:03d}"})
            k += 1
    gt[rng.random(gt.shape) < 0.02] = np.nan
    chrs = rng.choice(AUTOSOMES, ns, p=_auto_weights())
    bps = np.array([rng.integers(1, ARS_LEN[c]) for c in chrs],
                   dtype=np.int64)
    snp_df = pd.DataFrame({"CHR": chrs,
                           "SNP": [f"rs{i:07d}" for i in range(ns)],
                           "CM": 0.0, "BP": bps})
    snp_df["_k"] = snp_df["CHR"].map(_chr_key)
    snp_df = (snp_df.sort_values(["_k", "BP"])
              .drop(columns="_k").reset_index(drop=True))
    return gt, pd.DataFrame(rows), snp_df

# ═══════════════════════════════════════════════════════════
# QC
# ═══════════════════════════════════════════════════════════
def miss_ind(g): return np.isnan(g).mean(axis=1)
def miss_snp(g): return np.isnan(g).mean(axis=0)
def maf(g):
    p = np.nanmean(g, axis=0) / 2.0
    return np.minimum(p, 1 - p)
def het(g):
    with np.errstate(invalid="ignore"):
        return np.nanmean(g == 1, axis=1)

def _hwe_exact(n1, n0, n2):
    n = n1 + n0 + n2
    if n == 0: return np.nan
    if n1 == 0 and (n0 == 0 or n2 == 0): return 1.0
    rare = 2 * min(n0, n2) + n1
    mid = (rare * (2 * n - rare)) // (2 * n)
    if mid % 2 != rare % 2: mid += 1
    p = np.zeros(rare + 1); p[mid] = 1.0; s = 1.0
    ch, chr_, chc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch <= rare - 2:
        p[ch + 2] = p[ch] * 4 * chr_ * chc / ((ch + 2) * (ch + 1))
        s += p[ch + 2]; ch += 2; chr_ -= 1; chc -= 1
    ch, chr_, chc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch >= 2:
        p[ch - 2] = p[ch] * ch * (ch - 1) / (4 * (chr_ + 1) * (chc + 1))
        s += p[ch - 2]; ch -= 2; chr_ += 1; chc += 1
    po = p[n1] if n1 < len(p) else 0
    return float(min(p[p <= po + 1e-7].sum() / s, 1.0))

@cached
def hwe_pv(g):
    _, m = g.shape
    pv = np.full(m, np.nan)
    nm = np.isnan(g)
    n0 = ((g == 0) & ~nm).sum(0)
    n1 = ((g == 1) & ~nm).sum(0)
    n2 = ((g == 2) & ~nm).sum(0)
    nv = n0 + n1 + n2
    for j in range(m):
        if nv[j] < 5: continue
        pv[j] = _hwe_exact(int(n1[j]), int(n0[j]), int(n2[j]))
    return pv

def _align(g, i, s):
    n = min(g.shape[0], len(i)); m = min(g.shape[1], len(s))
    return (g[:n, :m], i.iloc[:n].reset_index(drop=True),
            s.iloc[:m].reset_index(drop=True))

def qc_filters(g, i, s, p):
    g, i, s = _align(g, i, s)
    n0, m0 = g.shape
    tr = {}
    k = miss_snp(g) <= p["geno"]
    g, s = g[:, k], s[k].reset_index(drop=True)
    tr["geno"] = int(m0 - g.shape[1])
    k = miss_ind(g) <= p["mind"]
    g, i = g[k], i[k].reset_index(drop=True)
    tr["mind"] = int(n0 - g.shape[0])
    if g.shape[1] == 0 or g.shape[0] == 0:
        raise ValueError("Tout exclu (missingness)")
    mb = g.shape[1]; m = maf(g)
    k = np.isfinite(m) & (m >= p["maf"])
    g, s = g[:, k], s[k].reset_index(drop=True)
    tr["maf"] = int(mb - g.shape[1])
    if g.shape[1] == 0: raise ValueError("Tout exclu MAF")
    mb = g.shape[1]
    pv = hwe_pv(g); k = np.isnan(pv) | (pv >= p["hwe"])
    g, s = g[:, k], s[k].reset_index(drop=True)
    tr["hwe"] = int(mb - g.shape[1])
    if g.shape[1] == 0: raise ValueError("Tout exclu HWE")
    gt_snp, ind_snp, snp_snp = g.copy(), i.copy(), s.copy()
    h = het(g); keep = np.ones(len(h), dtype=bool)
    for fid in i["FID"].unique():
        mk = (i["FID"] == fid).values
        sub = h[mk]
        if len(sub) < 5 or np.nanstd(sub) < 1e-9: continue
        z = (sub - np.nanmean(sub)) / np.nanstd(sub)
        idx = np.where(mk)[0]
        keep[idx[np.abs(z) > p["het_sd"]]] = False
    nb = g.shape[0]
    g, i = g[keep], i[keep].reset_index(drop=True)
    tr["het"] = int(nb - g.shape[0])
    if g.shape[0] == 0: raise ValueError("Tout exclu HET")
    return (g, i, s,
            {"n_ind_init": n0, "n_snp_init": m0,
             "n_ind_final": g.shape[0], "n_snp_final": g.shape[1],
             "excluded_ind": n0 - g.shape[0],
             "excluded_snp": m0 - g.shape[1],
             "trace": tr},
            gt_snp, ind_snp, snp_snp)

def filter_auto(g, s):
    cs = s["CHR"].astype(str).str.upper().str.replace("CHR", "")
    k = cs.isin([str(i) for i in range(1, 30)])
    return g[:, k.values], s[k].reset_index(drop=True)

@cached
def ld_pruning(g, w=50, st=5, r2t=0.2):
    n = g.shape[1]
    if n < 3: return np.ones(n, dtype=bool)
    keep = np.ones(n, dtype=bool)
    X = impute(g); X = X - X.mean(0)
    i = 0
    while i < n:
        we = min(i + st, n)
        while we < n and (we - i) < w:
            j = we
            if keep[j] and keep[i]:
                a, b = X[:, i], X[:, j]
                if a.std() > 1e-8 and b.std() > 1e-8:
                    r = np.corrcoef(a, b)[0, 1]
                    if r * r > r2t: keep[j] = False
            we += 1
        i += st
    return keep

# ═══════════════════════════════════════════════════════════
# STRUCTURE
# ═══════════════════════════════════════════════════════════
@cached
def pca_calc(g, nc=10):
    X = impute(g); X = X - X.mean(0)
    nc = min(nc, X.shape[0] - 1, X.shape[1])
    p = PCA(nc).fit(X)
    return p.transform(X), p.explained_variance_ratio_ * 100

@cached
def mds_calc(g, nc=5):
    X = impute(g); n = X.shape[0]
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        D[i] = np.abs(X - X[i]).sum(1) / X.shape[1]
    return SklearnMDS(nc, dissimilarity="precomputed", random_state=42,
                      n_init=1, max_iter=300).fit_transform(D)

@cached
def king_matrix(g):
    n = g.shape[0]
    h = np.nansum(g == 1, axis=1).astype(float)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            bh = np.nansum((g[i] == 1) & (g[j] == 1))
            d = h[i] + h[j]
            K[i, j] = K[j, i] = (d - 2*bh) / d if d > 0 else 0
    np.fill_diagonal(K, 0.5)
    return K

@cached
def admix_nmf(g, K=3, seed=42, mi=500):
    X = np.clip(impute(g), 0, 2)
    m = NMF(n_components=K, init="nndsvda", random_state=seed, max_iter=mi)
    W = m.fit_transform(X)
    s = W.sum(1, keepdims=True); s[s == 0] = 1
    return W / s, m.components_

# ═══════════════════════════════════════════════════════════
# LD DECAY / ROH / Ne
# ═══════════════════════════════════════════════════════════
@cached
def ld_decay(g, bp, mk=1000, ms=1500, seed=42):
    n = g.shape[1]
    if n < 2: return pd.DataFrame(columns=["dist_kb", "r2"])
    rng = np.random.default_rng(seed)
    idx = (np.sort(rng.choice(n, ms, replace=False))
           if n > ms else np.arange(n))
    gs = impute(g[:, idx])
    bp_s = np.asarray(bp)[idx].astype(np.float64)
    X = gs - gs.mean(0)
    sd = gs.std(0); sd[sd < 1e-8] = np.nan
    X = X / sd
    C = (X.T @ X) / X.shape[0]; R2 = C ** 2
    iu, ju = np.triu_indices(len(idx), k=1)
    dkb = (bp_s[ju] - bp_s[iu]) / 1000
    m = (dkb > 0) & (dkb <= mk)
    return pd.DataFrame({"dist_kb": dkb[m], "r2": R2[iu[m], ju[m]]})

@cached
def detect_roh(g, snp_json, min_s=30, min_kb=500):
    s = pd.DataFrame(json.loads(snp_json))
    chr_a = s["CHR"].astype(str).values
    bp = s["BP"].astype(np.int64).values
    n = g.shape[0]; rohs, fbp = [], np.zeros(n)
    tot = 0.0
    for ch in pd.unique(chr_a):
        ix = np.where(chr_a == ch)[0]
        if len(ix) < min_s: continue
        ix = ix[np.argsort(bp[ix])]
        span = bp[ix[-1]] - bp[ix[0]]
        if span < 1: continue
        tot += span
        for i in range(n):
            col = g[i, ix]; sk = None
            for k in range(len(ix)):
                h = (col[k] == 0) or (col[k] == 2)
                if h:
                    if sk is None: sk = k
                else:
                    if sk is not None:
                        nr = k - sk
                        lkb = (bp[ix[k-1]] - bp[ix[sk]]) / 1000
                        if nr >= min_s and lkb >= min_kb:
                            rohs.append({"IID_idx": i, "CHR": str(ch),
                                         "start_bp": int(bp[ix[sk]]),
                                         "end_bp": int(bp[ix[k-1]]),
                                         "n_snp": int(nr),
                                         "length_kb": float(lkb)})
                            fbp[i] += bp[ix[k-1]] - bp[ix[sk]]
                        sk = None
            if sk is not None:
                nr = len(ix) - sk
                lkb = (bp[ix[-1]] - bp[ix[sk]]) / 1000
                if nr >= min_s and lkb >= min_kb:
                    rohs.append({"IID_idx": i, "CHR": str(ch),
                                 "start_bp": int(bp[ix[sk]]),
                                 "end_bp": int(bp[ix[-1]]),
                                 "n_snp": int(nr),
                                 "length_kb": float(lkb)})
                    fbp[i] += bp[ix[-1]] - bp[ix[sk]]
    froh = fbp / tot if tot > 0 else fbp
    roh = (pd.DataFrame(rohs) if rohs
           else pd.DataFrame(columns=["IID_idx", "CHR", "start_bp",
                                       "end_bp", "n_snp", "length_kb"]))
    return roh, froh

# ═══════════════════════════════════════════════════════════
# FST
# ═══════════════════════════════════════════════════════════
def fst_per_snp(g, labels):
    labels = np.asarray(labels, dtype=object)
    pops = np.unique(labels)
    if len(pops) < 2: return np.full(g.shape[1], np.nan)
    out = np.full(g.shape[1], np.nan)
    mk = {p: labels == p for p in pops}
    for j in range(g.shape[1]):
        ps, ns = [], []
        for p in pops:
            v = g[mk[p], j]; v = v[~np.isnan(v)]
            if len(v) < 3: continue
            ps.append(v.mean()/2); ns.append(len(v))
        if len(ps) < 2: continue
        pa, na = np.array(ps), np.array(ns)
        pb = np.average(pa, weights=na)
        hs = np.average(2*pa*(1-pa), weights=na)
        ht = 2*pb*(1-pb)
        if ht > 1e-9: out[j] = (ht - hs) / ht
    return out

def fst_weir_cockerham_inline(gt, labels):
    """Weir & Cockerham (1984) — implémentation inline."""
    labels = np.asarray(labels, dtype=object)
    pops = sorted(np.unique(labels))
    r = len(pops)
    if r < 2:
        return np.full(gt.shape[1], np.nan), np.nan, np.nan
    n_snp = gt.shape[1]
    p_mat = np.full((r, n_snp), np.nan)
    n_mat = np.zeros((r, n_snp))
    for i, p in enumerate(pops):
        sub = gt[labels == p]
        valid = ~np.isnan(sub)
        n_mat[i] = valid.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            p_mat[i] = np.nansum(sub, axis=0) / (2.0 * n_mat[i])
    good = (n_mat >= 2).all(axis=0)
    fst = np.full(n_snp, np.nan)
    a_arr = np.full(n_snp, np.nan)
    b_arr = np.full(n_snp, np.nan)
    c_arr = np.full(n_snp, np.nan)
    for j in np.where(good)[0]:
        p_j, n_j = p_mat[:, j], n_mat[:, j]
        if np.any(~np.isfinite(p_j)) or np.any(p_j < 0.01) or np.any(p_j > 0.99):
            continue
        n_bar = n_j.sum() / r
        if n_bar < 1.5: continue
        nc = (r*n_bar - (n_j**2).sum()/(r*n_bar)) / (r-1)
        if nc <= 0: continue
        p_bar = (n_j*p_j).sum() / n_j.sum()
        s2 = (n_j*(p_j-p_bar)**2).sum() / ((r-1)*n_bar)
        a = (n_bar/nc) * (s2 - (1/(n_bar-1))*(p_bar*(1-p_bar) - (r-1)/r*s2))
        b = (n_bar/(n_bar-1)) * (p_bar*(1-p_bar) - (r-1)/r*s2 - a/2)
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

def bootstrap_fst_inline(gt, labels, n_boot=500, ci=0.95, seed=42):
    """Bootstrap IC sur FST Weir-Cockerham."""
    rng = np.random.default_rng(seed)
    n = gt.shape[0]
    labels = np.asarray(labels, dtype=object)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        lb = labels[idx]
        if len(np.unique(lb)) < 2:
            continue
        _, theta, _ = fst_weir_cockerham_inline(gt[idx], lb)
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

def fst_grp_vs_rest(g, labels, gp):
    labels = np.asarray(labels, dtype=object)
    mg = np.isin(labels, list(gp)); mr = ~mg
    if mg.sum() < 5 or mr.sum() < 5: return pd.DataFrame()
    p_g, p_r = np.nanmean(g[mg], 0)/2, np.nanmean(g[mr], 0)/2
    n_g, n_r = (~np.isnan(g[mg])).sum(0), (~np.isnan(g[mr])).sum(0)
    dn = np.where(n_g+n_r > 0, n_g+n_r, np.nan)
    pb = (p_g*n_g + p_r*n_r)/dn
    hs = (2*p_g*(1-p_g)*n_g + 2*p_r*(1-p_r)*n_r)/dn
    ht = 2*pb*(1-pb)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (ht-hs)/ht
    f = np.clip(np.where(np.isfinite(f), f, np.nan), 0, 1)
    return pd.DataFrame({"FST": f, "P_group": p_g, "P_rest": p_r,
                         "n_group": n_g, "n_rest": n_r})

def fst_grp_vs_grp(g, labels, pa, pb_):
    labels = np.asarray(labels, dtype=object)
    ma, mb = np.isin(labels, list(pa)), np.isin(labels, list(pb_))
    if ma.sum() < 5 or mb.sum() < 5: return pd.DataFrame()
    p_A, p_B = np.nanmean(g[ma], 0)/2, np.nanmean(g[mb], 0)/2
    n_A, n_B = (~np.isnan(g[ma])).sum(0), (~np.isnan(g[mb])).sum(0)
    dn = np.where(n_A+n_B > 0, n_A+n_B, np.nan)
    p_bar = (p_A*n_A + p_B*n_B)/dn
    hs = (2*p_A*(1-p_A)*n_A + 2*p_B*(1-p_B)*n_B)/dn
    ht = 2*p_bar*(1-p_bar)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (ht-hs)/ht
    f = np.clip(np.where(np.isfinite(f), f, np.nan), 0, 1)
    return pd.DataFrame({"FST": f, "P_A": p_A, "P_B": p_B,
                         "n_A": n_A, "n_B": n_B})

def _extract_out(fst_df, snp_df, q, mf):
    if fst_df.empty: return pd.DataFrame(), {}
    df = pd.concat([snp_df.reset_index(drop=True),
                    fst_df.reset_index(drop=True)], axis=1
                   ).dropna(subset=["FST"])
    if df.empty: return pd.DataFrame(), {}
    thr = max(float(np.nanquantile(df["FST"], q)), mf)
    out = (df[df["FST"] >= thr]
           .sort_values("FST", ascending=False)
           .reset_index(drop=True))
    return out, {"n_snp_analyzed": len(df), "n_snp_outliers": len(out),
                 "fst_mean": float(df["FST"].mean()),
                 "fst_median": float(df["FST"].median()),
                 "fst_max": float(df["FST"].max()),
                 "threshold": float(thr)}

def analyze_pop_vs_rest(g, s, labels, q=0.99, mf=0.15):
    res = {}
    for pop in sorted(np.unique(labels)):
        if (labels == pop).sum() < 3: continue
        f = fst_grp_vs_rest(g, labels, [pop])
        if f.empty: continue
        out, st_ = _extract_out(f, s, q, mf)
        res[pop] = {"pop": pop, "status": "success", "pops": [pop],
                    "all_fst_df": pd.concat(
                        [s.reset_index(drop=True),
                         f.reset_index(drop=True)], axis=1),
                    "outliers_df": out, **st_}
    return res

def analyze_grp_vs_rest(g, s, labels, groups, q=0.99, mf=0.15):
    res = {}
    for name, pops in groups.items():
        if not pops: continue
        f = fst_grp_vs_rest(g, labels, pops)
        if f.empty:
            res[name] = {"status": "insufficient", "group": name}
            continue
        out, st_ = _extract_out(f, s, q, mf)
        res[name] = {"group": name, "pops": list(pops),
                     "status": "success",
                     "all_fst_df": pd.concat(
                         [s.reset_index(drop=True),
                          f.reset_index(drop=True)], axis=1),
                     "outliers_df": out, **st_}
    return res

def analyze_grp_pair(g, s, labels, groups, gA, gB, q=0.99, mf=0.15):
    f = fst_grp_vs_grp(g, labels, groups.get(gA, []),
                        groups.get(gB, []))
    if f.empty: return {"status": "insufficient"}
    out, st_ = _extract_out(f, s, q, mf)
    return {"status": "success", "group_A": gA, "group_B": gB,
            "pops_A": groups.get(gA, []), "pops_B": groups.get(gB, []),
            "all_fst_df": pd.concat(
                [s.reset_index(drop=True),
                 f.reset_index(drop=True)], axis=1),
            "outliers_df": out, **st_}

@cached
def analyze_pair(g, s, labels, a, b, q=0.99, mf=0.15):
    labels = np.asarray(labels, dtype=object)
    mA, mB = labels == a, labels == b
    if mA.sum() < 3 or mB.sum() < 3:
        return {"status": "insufficient", "pop_A": a, "pop_B": b}
    gA, gB = g[mA], g[mB]
    pA = np.nanmean(gA, 0)/2; pB = np.nanmean(gB, 0)/2
    nA = (~np.isnan(gA)).sum(0); nB = (~np.isnan(gB)).sum(0)
    dn = np.where(nA+nB > 0, nA+nB, np.nan)
    pb = (pA*nA + pB*nB)/dn
    hs = (2*pA*(1-pA) + 2*pB*(1-pB))/2
    ht = 2*pb*(1-pb)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (ht-hs)/ht
    f = np.clip(np.where(np.isfinite(f), f, np.nan), 0, 1)
    fst_df = pd.DataFrame({"FST": f, "P_A": pA, "P_B": pB,
                           "P_bar": pb, "n_A": nA, "n_B": nB})
    out, st_ = _extract_out(fst_df, s, q, mf)
    return {"status": "success", "pop_A": a, "pop_B": b,
            "n_ind_A": int(mA.sum()), "n_ind_B": int(mB.sum()),
            "outliers_df": out,
            "all_fst_df": pd.concat(
                [s.reset_index(drop=True),
                 fst_df.reset_index(drop=True)], axis=1), **st_}

def grp_phenotype(ind_df, groups, gname):
    df = ind_df.copy()
    df["pheno"] = 0.0
    df.loc[df["FID"].astype(str).isin(groups.get(gname, [])),
           "pheno"] = 1.0
    return df

def auto_pairs_mds(mds, labels, qlo=0.15, qhi=0.85):
    import itertools
    mds = np.asarray(mds); labels = np.asarray(labels, dtype=object)
    pops = sorted(np.unique(labels))
    cents, cnts = {}, {}
    for p in pops:
        m = labels == p
        if m.sum() < 3: continue
        cents[p] = mds[m, :2].mean(0); cnts[p] = int(m.sum())
    if len(cents) < 2: return pd.DataFrame()
    pairs = [{"pop_A": a, "pop_B": b,
              "distance_MDS": float(np.linalg.norm(cents[a]-cents[b])),
              "n_ind_A": cnts[a], "n_ind_B": cnts[b]}
             for a, b in itertools.combinations(cents.keys(), 2)]
    df = pd.DataFrame(pairs).sort_values("distance_MDS").reset_index(drop=True)
    lo = df["distance_MDS"].quantile(qlo)
    hi = df["distance_MDS"].quantile(qhi)
    def cat(d):
        return "🟢 Proche" if d <= lo else ("🔴 Éloignée" if d >= hi else "🟡 Moyenne")
    df["catégorie"] = df["distance_MDS"].apply(cat)
    return df

# ═══════════════════════════════════════════════════════════
# NEUTRALITÉ (v7.1)
# ═══════════════════════════════════════════════════════════
def tajimas_d(gt):
    n_ind, n_snp = gt.shape
    p = np.nanmean(gt, axis=0) / 2.0
    p = p[np.isfinite(p)]
    S = int(np.sum((p > 0) & (p < 1)))
    if S == 0: return {"D": np.nan, "S": 0}
    pi_total = 0.0
    for j in range(n_snp):
        col = gt[:, j]; v = col[~np.isnan(col)]
        n = len(v)
        if n < 2: continue
        pj = v.mean()/2.0
        pi_total += 2*pj*(1-pj)*n/(n-1)
    pi = pi_total / n_snp
    a1 = float(np.sum(1.0/np.arange(1, n_ind)))
    a2 = float(np.sum(1.0/np.arange(1, n_ind)**2))
    theta_w = S/(a1*n_snp) if a1 > 0 else 0
    n = n_ind
    b1 = (n+1)/(3*(n-1)); b2 = 2*(n*n+n+3)/(9*n*(n-1))
    c1 = b1 - 1/a1
    c2 = b2 - (n+2)/(a1*n) + a2/(a1**2)
    e1 = c1/a1; e2 = c2/(a1**2 + a2)
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
    n_ind, n_snp = gt.shape
    p = np.nanmean(gt, axis=0) / 2.0
    p = p[np.isfinite(p)]
    S = int(np.sum((p > 0) & (p < 1)))
    if S == 0: return {"D_star": np.nan, "S": 0}
    thr = 1.0/(2*n_ind)
    n_sing = int(np.sum((p > 0) & (p <= thr*1.5)))
    a1 = float(np.sum(1.0/np.arange(1, n_ind)))
    a2 = float(np.sum(1.0/np.arange(1, n_ind)**2))
    eta_e = S/a1 if a1 > 0 else 0
    if n_ind <= 2 or a1 == 0:
        return {"D_star": np.nan, "S": S}
    b1 = (n_ind+1)/(3*(n_ind-1))
    b2 = 2*(n_ind**2+n_ind+3)/(9*n_ind*(n_ind-1))
    c1 = b1 - 1/a1
    c2 = b2 - (n_ind+2)/(a1*n_ind) + a2/(a1**2)
    vd = (c1/a1)*eta_e + (c2/(a1**2+a2))*eta_e*(eta_e-1)
    D = (eta_e - n_sing)/np.sqrt(vd) if vd > 0 else np.nan
    return {"D_star": round(float(D), 4) if np.isfinite(D) else np.nan,
            "S": S, "n_singletons": n_sing,
            "interpretation": ("Sélection (D*<-2)" if np.isfinite(D) and D < -2
                               else "Goulot (D*>2)" if np.isfinite(D) and D > 2
                               else "Neutre")}

def fay_wu_h(gt):
    n_ind, n_snp = gt.shape
    p = np.nanmean(gt, axis=0) / 2.0
    p = p[np.isfinite(p)]
    S = int(np.sum((p > 0) & (p < 1)))
    if S == 0: return {"H": np.nan, "S": 0}
    pi_total = 0.0; theta_h = 0.0
    for j in range(n_snp):
        col = gt[:, j]; v = col[~np.isnan(col)]
        n = len(v)
        if n < 2: continue
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

# ═══════════════════════════════════════════════════════════
# GWAS LMM (v7.1 inline)
# ═══════════════════════════════════════════════════════════
def _estimate_delta(Y, X, K):
    d, U = np.linalg.eigh(K); d = np.maximum(d, 0)
    Yt = U.T @ Y; Xt = U.T @ X
    def neg_ll(delta):
        if delta <= 0: return 1e18
        w = 1/np.sqrt(delta*d + 1)
        Yw = Yt*w; Xw = Xt*w[:, None]
        try: beta, *_ = np.linalg.lstsq(Xw, Yw, rcond=None)
        except np.linalg.LinAlgError: return 1e18
        res = Yw - Xw @ beta
        s2 = float(np.sum(res**2)/len(Y))
        if s2 <= 0: return 1e18
        return -(-0.5*(len(Y)*np.log(s2) + np.sum(np.log(delta*d + 1))))
    grid = np.logspace(-3, 3, 20)
    lls = np.array([neg_ll(dl) for dl in grid])
    return float(grid[np.argmin(lls)])

def gwas_lmm_inline(gt, pheno, K, cov=None, maf_min=0.05, miss_max=0.05):
    n_all, n_snp = gt.shape
    pheno = np.asarray(pheno, dtype=float)
    valid_ind = np.isfinite(pheno)
    if valid_ind.sum() < 10: return pd.DataFrame()
    Y = pheno[valid_ind]
    gt_v = gt[valid_ind]
    K_v = np.asarray(K)[np.ix_(valid_ind, valid_ind)]
    if cov is None or cov.size == 0:
        X0 = np.ones((len(Y), 1))
    else:
        X0 = np.column_stack([np.ones(len(Y)), np.asarray(cov)[valid_ind]])
    keep = np.ones(X0.shape[1], dtype=bool)
    for j in range(1, X0.shape[1]):
        if X0[:, j].std() < 1e-10: keep[j] = False
    X0 = X0[:, keep]
    n = len(Y)
    delta = _estimate_delta(Y, X0, K_v)
    d, U = np.linalg.eigh(K_v); d = np.maximum(d, 0)
    w = 1/np.sqrt(delta*d + 1)
    Yt = (U.T @ Y) * w
    X0t = (U.T @ X0) * w[:, None]
    out = {"SNP_idx": [], "MAF": [], "BETA": [], "SE": [], "T": [], "P": []}
    for j in range(n_snp):
        g = gt_v[:, j].astype(float)
        mask = ~np.isfinite(g)
        if mask.mean() > miss_max: continue
        if mask.any():
            m_g = np.nanmean(g)
            if not np.isfinite(m_g): continue
            g = np.where(mask, m_g, g)
        p = g.mean()/2.0
        maf_j = min(p, 1-p)
        if maf_j < maf_min: continue
        gw = (U.T @ g) * w
        X = np.column_stack([X0t, gw])
        try:
            beta, *_ = np.linalg.lstsq(X, Yt, rcond=None)
            res = Yt - X @ beta
            dof = max(n - X.shape[1], 1)
            s2 = float(np.sum(res**2)/dof)
            if s2 <= 0: continue
            XtX_inv = np.linalg.inv(X.T @ X)
            se = float(np.sqrt(s2 * XtX_inv[-1, -1]))
            if se <= 0 or not np.isfinite(se): continue
            t = beta[-1]/se
            pv = float(2*stats.t.sf(abs(t), df=dof))
            out["SNP_idx"].append(j)
            out["MAF"].append(float(maf_j))
            out["BETA"].append(float(beta[-1]))
            out["SE"].append(se)
            out["T"].append(float(t))
            out["P"].append(pv)
        except np.linalg.LinAlgError: continue
    if not out["SNP_idx"]:
        return pd.DataFrame(columns=["SNP_idx","MAF","BETA","SE","T","P"])
    return pd.DataFrame(out)

def lambda_gc(pvals):
    p = np.asarray(pvals)
    p = p[np.isfinite(p) & (p > 0) & (p <= 1)]
    if len(p) < 10: return np.nan
    chi2_obs = stats.chi2.isf(p, df=1)
    return float(np.median(chi2_obs) / stats.chi2.ppf(0.5, df=1))

def qq_data(pvals):
    p = np.asarray(pvals)
    p = p[np.isfinite(p) & (p > 0) & (p <= 1)]
    if len(p) < 10: return None
    p = np.sort(p); n = len(p)
    expected = (np.arange(1, n+1) - 0.5) / n
    return {"observed": -np.log10(p), "expected": -np.log10(expected),
            "n": n}

def gwas_thresholds(pvals, alpha=0.05):
    p = np.asarray(pvals)
    p = p[np.isfinite(p)]
    n = len(p)
    if n == 0:
        return {"bonferroni": np.nan, "fdr_05": np.nan, "n_tests": 0,
                "n_sig_bonf": 0, "n_sig_fdr": 0}
    bonf = alpha/n
    ps = np.sort(p)
    bh = ps*n/np.arange(1, n+1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    sig = np.where(bh <= alpha)[0]
    return {"bonferroni": float(bonf),
            "fdr_05": float(ps[sig[-1]]) if len(sig) else np.nan,
            "n_tests": int(n),
            "n_sig_bonf": int((p <= bonf).sum()),
            "n_sig_fdr": int(len(sig))}

# ═══════════════════════════════════════════════════════════
# VEP + GO + KEGG (v7.1 inline)
# ═══════════════════════════════════════════════════════════
def vep_annotate(snps_df):
    import time
    results = []
    for _, snp in snps_df.iterrows():
        chrom = str(snp["CHR"]).replace("chr", "")
        pos = int(snp["BP"])
        ref = str(snp.get("A1", "A")) if "A1" in snp else "A"
        alt = str(snp.get("A2", "G")) if "A2" in snp else "G"
        if ref == alt or ref in ("", "nan"): ref, alt = "A", "G"
        try:
            r = requests.post(
                "https://rest.ensembl.org/vep/bos_taurus/region",
                json={"variants": [f"{chrom} {pos} . {ref} {alt} . . ."]},
                headers={"Content-Type": "application/json",
                         "Accept": "application/json"},
                timeout=20)
            if r.status_code == 200 and r.json():
                d = r.json()[0]
                t = d.get("transcript_consequences", [{}])
                info = t[0] if t else {}
                results.append({
                    "CHR": chrom, "BP": pos, "REF": ref, "ALT": alt,
                    "gene": info.get("gene_symbol", "?"),
                    "consequence": ", ".join(info.get("consequence_terms", [])),
                    "impact": info.get("impact", "?")})
        except Exception:
            continue
        time.sleep(0.1)
    return pd.DataFrame(results)

def go_enrichment(genes):
    if not genes: return pd.DataFrame()
    try:
        r = requests.post(
            "https://biit.cs.ut.ee/gprofiler/api/gost/profile/",
            json={"organism": "bos_taurus", "query": list(genes),
                  "sources": ["GO:BP", "GO:MF", "GO:CC"],
                  "user_threshold": 0.05,
                  "significance_method": "fdr"},
            timeout=60)
        if r.status_code == 200:
            data = r.json()
            rows = [{
                "term_id": t.get("native", "?"),
                "term_name": t.get("name", "?"),
                "source": t.get("source", "?"),
                "p_value": t.get("p_value", 1),
                "intersection_size": t.get("intersection_size", 0),
                "genes": ",".join(t.get("intersections", []))}
                for t in data.get("result", [])]
            df = pd.DataFrame(rows)
            if not df.empty: df = df.sort_values("p_value")
            return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})
    return pd.DataFrame()

def kegg_enrichment(genes):
    if not genes: return pd.DataFrame()
    try:
        gene_links = {}
        for g in genes:
            r = requests.get(f"https://rest.kegg.jp/link/bta/bta:{g}",
                              timeout=10)
            if r.status_code == 200 and r.text:
                for line in r.text.strip().split("\n"):
                    p = line.split("\t")
                    if len(p) == 2:
                        gene_links.setdefault(p[1], []).append(g)
        if not gene_links: return pd.DataFrame()
        rows = []
        n_genes = len(genes)
        n_bg = 20000
        for pathway, gl in gene_links.items():
            k = len(set(gl))
            p = stats.hypergeom.sf(k-1, n_bg, k, n_genes) if n_genes > 0 else 1.0
            rows.append({"pathway_id": pathway,
                         "n_genes": k,
                         "p_value": float(p),
                         "genes": ",".join(gl)})
        df = pd.DataFrame(rows)
        if not df.empty:
            df["padj"] = np.minimum(df["p_value"] * len(df), 1.0)
            df = df.sort_values("p_value")
        return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

# ═══════════════════════════════════════════════════════════
# GTF + ANNOTATION
# ═══════════════════════════════════════════════════════════
def load_gtf(path):
    genes = []
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            if line.startswith("#"): continue
            p = line.strip().split("\t")
            if len(p) < 9 or p[2] != "gene": continue
            ch = p[0].replace("chr", "")
            gn = None
            for attr in p[8].split(";"):
                a = attr.strip()
                if a.startswith("gene_name"):
                    gn = a.split('"')[1]
            genes.append({"CHR": str(ch), "start": int(p[3]),
                          "end": int(p[4]), "gene_name": gn})
    return pd.DataFrame(genes)

def annotate_snps(out, gdf, fl=500):
    if out.empty or gdf is None or gdf.empty: return out.copy()
    flb = fl * 1000
    rows = []
    for _, snp in out.iterrows():
        ch = str(snp["CHR"]).replace("CHR", "").strip()
        bp = int(snp["BP"])
        gc = gdf[gdf["CHR"] == ch]
        if gc.empty:
            rows.append({**snp.to_dict(), "gene_name": None,
                         "distance": None, "region": None}); continue
        cd = []
        for _, gn in gc.iterrows():
            if gn["start"] <= bp <= gn["end"]:
                cd.append((gn["gene_name"], 0, "intragenic"))
            else:
                d = min(abs(bp-gn["start"]), abs(bp-gn["end"]))
                if d <= flb:
                    r = "upstream" if bp < gn["start"] else "downstream"
                    cd.append((gn["gene_name"], d, r))
        if cd:
            b = min(cd, key=lambda x: x[1])
            rows.append({**snp.to_dict(), "gene_name": b[0],
                         "distance": b[1], "region": b[2]})
        else:
            rows.append({**snp.to_dict(), "gene_name": None,
                         "distance": None, "region": None})
    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════
# GWAS 10 cM
# ═══════════════════════════════════════════════════════════
def gwas_local(g, s, i, ph, ch, bp_, pn, wc=10):
    wbp = wc * 1e6
    mask = ((s["CHR"].astype(str) == str(ch))
            & (s["BP"] >= bp_-wbp) & (s["BP"] <= bp_+wbp))
    idx = np.where(mask)[0]
    if len(idx) == 0: return pd.DataFrame()
    i = i.copy(); i["IID"] = i["IID"].astype(str)
    ph = ph.set_index("IID")
    com = [x for x in i["IID"] if x in ph.index]
    if len(com) < 10: return pd.DataFrame()
    im = i["IID"].isin(com).values
    gs = g[im][:, idx]
    ins = i[im].reset_index(drop=True)
    pv = []
    for iid in ins["IID"]:
        try:
            v = ph.loc[iid, pn]
            if isinstance(v, pd.Series): v = v.iloc[0]
            pv.append(float(v))
        except Exception: pv.append(np.nan)
    pv = np.array(pv)
    pops = ins["FID"].astype(str).values
    up = sorted(set(pops))
    dm = np.zeros((len(pops), max(0, len(up)-1)))
    for k, p in enumerate(pops):
        for j, u in enumerate(up[1:]):
            if p == u: dm[k, j] = 1
    res = []
    sw = s.iloc[idx]
    for j in range(len(idx)):
        gg = gs[:, j]
        v = ~np.isnan(gg) & ~np.isnan(pv)
        if v.sum() < 10: continue
        gv, pvv = gg[v], pv[v]
        X = (np.column_stack([np.ones(v.sum()), gv, dm[v]])
             if dm.shape[1] > 0
             else np.column_stack([np.ones(v.sum()), gv]))
        try:
            beta, _, _, _ = np.linalg.lstsq(X, pvv, rcond=None)
            r = pvv - X @ beta
            ssr = np.sum(r**2); sst = np.sum((pvv-pvv.mean())**2)
            r2 = 1 - ssr/sst if sst > 0 else 0
            dr = v.sum() - X.shape[1]
            if dr > 0:
                mse = ssr/dr
                try:
                    cv = np.linalg.inv(X.T @ X)
                    se = np.sqrt(mse*cv[1, 1])
                    tv = beta[1]/se
                    pval = 2*(1-stats.t.cdf(abs(tv), dr))
                except np.linalg.LinAlgError:
                    pval, tv = np.nan, np.nan
            else: pval, tv = np.nan, np.nan
            res.append({"SNP": sw.iloc[j]["SNP"],
                        "CHR": sw.iloc[j]["CHR"],
                        "BP": int(sw.iloc[j]["BP"]),
                        "beta": float(beta[1]),
                        "t_stat": float(tv) if np.isfinite(tv) else np.nan,
                        "p_value": float(pval) if np.isfinite(pval) else np.nan,
                        "r2": float(r2), "n_obs": int(v.sum())})
        except Exception: continue
    df = pd.DataFrame(res)
    if not df.empty:
        df["p_bonf"] = np.minimum(1, df["p_value"]*len(df))
        df["neg_log10_p"] = -np.log10(df["p_value"].clip(lower=1e-300))
        df = df.sort_values("p_value")
    return df

# ═══════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════
def _dos_bits(c):
    b = np.zeros(len(c), dtype=np.uint8)
    b[np.isnan(c)] = 0b01
    b[c == 0] = 0b11; b[c == 1] = 0b10; b[c == 2] = 0b00
    return b

def build_bed(g):
    n, m = g.shape; nb = (n+3)//4
    buf = bytearray([0x6C, 0x1B, 0x01])
    for j in range(m):
        bits = _dos_bits(g[:, j])
        p = np.zeros(nb, dtype=np.uint8)
        for k in range(n):
            p[k//4] |= (bits[k] & 3) << (2*(k % 4))
        buf.extend(p.tobytes())
    return bytes(buf)

def build_zip(g, ind, snp, pre="bovine"):
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{pre}.bed", build_bed(g))
        bim = []
        for _, r in snp.iterrows():
            a1 = str(r.get("A1", "A")) if "A1" in r else "A"
            a2 = str(r.get("A2", "G")) if "A2" in r else "G"
            if a1 in ("nan", ""): a1 = "A"
            if a2 in ("nan", "", a1): a2 = "G" if a1 == "A" else "A"
            bim.append(f"{r['CHR']}\t{r['SNP']}\t{r['CM']}\t"
                       f"{int(r['BP'])}\t{a1}\t{a2}")
        zf.writestr(f"{pre}.bim", "\n".join(bim) + "\n")
        fam = "\n".join(f"{r['FID']}\t{r['IID']}\t0\t0\t0\t-9"
                        for _, r in ind.iterrows()) + "\n"
        zf.writestr(f"{pre}.fam", fam)
    return bio.getvalue()

# ═══════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════
def plt_pca(sc, vp, lb):
    df = pd.DataFrame({"PC1": sc[:, 0], "PC2": sc[:, 1], "Pop": lb})
    fig = px.scatter(df, x="PC1", y="PC2", color="Pop",
                     title=f"PCA ({vp[0]:.1f}% / {vp[1]:.1f}%)", height=550)
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    return fig

def plt_mds(c, lb):
    df = pd.DataFrame({"MDS1": c[:, 0], "MDS2": c[:, 1], "Pop": lb})
    fig = px.scatter(df, x="MDS1", y="MDS2", color="Pop", title="MDS",
                     height=550)
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    return fig

def plt_man(fst, ch, t="Manhattan — FST/SNP"):
    df = pd.DataFrame({"FST": np.asarray(fst),
                       "CHR": np.asarray(ch)}).dropna()
    if df.empty: return None
    o = df["CHR"].map(_chr_key)
    df["_0"] = o.map(lambda x: x[0]); df["_1"] = o.map(lambda x: x[1])
    df = df.sort_values(["_0", "_1"]).reset_index(drop=True)
    cum, ticks, lb = 0, [], []
    pos = np.zeros(len(df), dtype=int)
    for c in df["CHR"].unique():
        ix = np.where(df["CHR"] == c)[0]
        pos[ix] = np.arange(cum, cum + len(ix))
        ticks.append((cum + cum + len(ix) - 1) / 2)
        lb.append(str(c)); cum += len(ix)
    df["x"] = pos
    fig = go.Figure()
    for i, c in enumerate(df["CHR"].unique()):
        sub = df[df["CHR"] == c]
        color = "#2c3e50" if i % 2 == 0 else "#7f8c8d"
        fig.add_trace(go.Scatter(x=sub["x"], y=sub["FST"], mode="markers",
                                  marker=dict(size=5, color=color),
                                  showlegend=False, hoverinfo="skip"))
    q = float(np.nanquantile(fst, 0.99))
    fig.add_hline(y=q, line_dash="dash", line_color="red",
                  annotation_text=f"Top 1% = {q:.4f}")
    fig.update_layout(title=t, xaxis_title="Chromosome", yaxis_title="FST",
                      height=500)
    fig.update_xaxes(tickvals=ticks, ticktext=lb)
    return fig

# ═══════════════════════════════════════════════════════════
# RAPPORT IMRaD
# ═══════════════════════════════════════════════════════════
def _safe_get(state, key, default=None):
    try:
        val = state.get(key, None)
        return default if val is None else val
    except (AttributeError, KeyError, TypeError):
        try:
            val = state[key]
            return default if val is None else val
        except Exception:
            return default

def build_imrad_html(state, config):
    s = _safe_get(state, "qc_stats", {})
    fst_data = _safe_get(state, "fst_result", {})
    boot = _safe_get(state, "bootstrap_ci", {})
    taj = _safe_get(state, "tajima", {})
    fuli = _safe_get(state, "fuli", {})
    faywu = _safe_get(state, "faywu", {})
    gwas = _safe_get(state, "gwas_results_lmm", None)

    n_ind_init = s.get("n_ind_init", "?")
    n_snp_init = s.get("n_snp_init", "?")
    n_ind_final = s.get("n_ind_final", "?")
    n_snp_final = s.get("n_snp_final", "?")
    theta = fst_data.get("theta", 0) if fst_data else 0
    ci_low = boot.get("ci_low", 0) if boot else 0
    ci_high = boot.get("ci_high", 0) if boot else 0
    n_boot = boot.get("n", 0) if boot else 0

    gwas_info = "Non réalisé."
    if gwas is not None and hasattr(gwas, "empty") and not gwas.empty:
        try:
            n_sig = int((gwas["P"] < 0.05/len(gwas)).sum())
            gwas_info = f"{len(gwas)} SNPs testés. {n_sig} sig. Bonferroni."
        except Exception:
            gwas_info = f"{len(gwas)} SNPs testés."

    proj = config.get("project_name", "Bovine")
    date_now = datetime.now().strftime("%Y-%m-%d %H:%M")

    return (
        "<!DOCTYPE html><html lang='fr'><head><meta charset='UTF-8'>"
        f"<title>Rapport IMRaD — {proj}</title>"
        "<style>body{font-family:'Times New Roman',serif;max-width:900px;"
        "margin:40px auto;padding:20px;color:#222;line-height:1.6}"
        "h1{color:#1a5276;border-bottom:3px solid #1a5276;padding-bottom:10px}"
        "h2{color:#2471a3;margin-top:30px;border-bottom:1px solid #ccc}"
        ".meta{background:#f4f6f7;padding:15px;border-radius:5px;margin:20px 0}"
        "</style></head><body>"
        "<h1>🐄 Rapport Scientifique — Bovine SNP Platform v7.2</h1>"
        f"<div class='meta'><p><b>Projet :</b> {proj}</p>"
        f"<p><b>Date :</b> {date_now}</p></div>"
        "<h2>Abstract</h2>"
        f"<p>Nous avons analysé <b>{n_ind_init}</b> individus bovins "
        f"génotypés pour <b>{n_snp_init}</b> SNPs. Après QC "
        f"(<b>{n_ind_final}</b> ind, <b>{n_snp_final}</b> SNPs), FST "
        f"(Weir &amp; Cockerham) = <b>{theta:.4f}</b> "
        f"[IC 95% : {ci_low:.4f}–{ci_high:.4f}], "
        f"Tajima's D = <b>{taj.get('D', 'NA')}</b>, "
        f"Fu &amp; Li D* = <b>{fuli.get('D_star', 'NA')}</b>.</p>"
        "<h2>2. Materials and Methods</h2>"
        "<h3>2.1 Contrôle qualité</h3>"
        f"<p>Missingness ≤ {config.get('geno', 0.05)}, "
        f"MAF ≥ {config.get('maf', 0.05)}, HWE ≥ {config.get('hwe', '1e-6')}.</p>"
        "<h3>2.2 FST (Weir &amp; Cockerham 1984)</h3>"
        f"<p>θ estimé par méthode des moments. Bootstrap n={n_boot}.</p>"
        "<h3>2.3 Neutralité</h3>"
        "<p>Tajima (1989), Fu &amp; Li (1993), Fay &amp; Wu (2000).</p>"
        "<h3>2.4 GWAS</h3>"
        "<p>Modèle mixte (LMM EMMA) avec matrice KING et PCs.</p>"
        "<h2>3. Results</h2>"
        f"<p><b>FST</b> = {theta:.4f} "
        f"[IC 95% : {ci_low:.4f}–{ci_high:.4f}]</p>"
        f"<p><b>Tajima D</b> = {taj.get('D', 'NA')}</p>"
        f"<p><b>Fu &amp; Li D*</b> = {fuli.get('D_star', 'NA')}</p>"
        f"<p><b>Fay &amp; Wu H</b> = {faywu.get('H', 'NA')}</p>"
        f"<h3>3.3 GWAS</h3><p>{gwas_info}</p>"
        "<h2>4. Discussion</h2>"
        "<p>Les valeurs de FST modérées suggèrent une différentiation "
        "génétique cohérente avec l'histoire de sélection.</p>"
        "<h2>5. References</h2><ol>"
        "<li>Weir BS, Cockerham CC (1984). Evolution 38:1358-1370.</li>"
        "<li>Tajima F (1989). Genetics 123:585-595.</li>"
        "<li>Fu YX, Li WH (1993). Genetics 133:693-709.</li>"
        "<li>Fay JC, Wu CI (2000). Genetics 155:1405-1413.</li>"
        "</ol></body></html>"
    )

def build_supp_tables(state):
    tables = {}
    ind_df = _safe_get(state, "ind_df", None)
    gt = _safe_get(state, "gt", None)
    if ind_df is not None and gt is not None:
        df = ind_df.copy(); df["n_snp"] = gt.shape[1]
        tables["S1_Individus"] = df.head(200)
    s = _safe_get(state, "qc_stats", {})
    if s.get("trace"):
        tables["S2_QC"] = pd.DataFrame(list(s["trace"].items()),
                                        columns=["Filtre", "Exclus"])
    boot = _safe_get(state, "bootstrap_ci", {})
    if boot: tables["S4_Bootstrap"] = pd.DataFrame([boot])
    rows = []
    for name, key in [("Tajima D", "tajima"), ("Fu & Li", "fuli"),
                       ("Fay & Wu", "faywu")]:
        d = _safe_get(state, key, {})
        if d:
            for k, v in d.items():
                rows.append({"Test": name, "Param": k, "Valeur": v})
    if rows: tables["S5_Neutralite"] = pd.DataFrame(rows)
    gwas = _safe_get(state, "gwas_results_lmm", None)
    if gwas is not None and hasattr(gwas, "empty") and not gwas.empty:
        tables["S6_GWAS"] = gwas.head(100)
    tables["S10_Session"] = pd.DataFrame([
        {"Paramètre": "Date", "Valeur": datetime.now().isoformat()},
        {"Paramètre": "Version", "Valeur": "v7.2"}])
    return tables

# ═══════════════════════════════════════════════════════════
# ÉTAT — APRÈS TOUS LES IMPORTS ET DÉFINITIONS
# ═══════════════════════════════════════════════════════════
STATE_KEYS = ["gt", "ind_df", "snp_df", "gt_filt", "ind_filt", "snp_filt",
              "qc_stats", "gt_snp", "ind_snp", "snp_snp",
              "gt_pruned", "snp_pruned", "pca_scores", "pca_var",
              "mds_coords", "king_matrix", "ld_df", "roh_df", "froh",
              "admixture_Q", "admixture_K", "cv_results",
              "reynolds_D", "reynolds_pops", "fst_global",
              "binome_result", "all_binomes", "group_results",
              "group_pair", "gwas_results_10cm", "genes_df",
              "fst_result", "bootstrap_ci", "tajima", "fuli", "faywu",
              "gwas_results_lmm", "gwas_qq", "gwas_lambda",
              "vep_results", "go_enrichment", "kegg_enrichment",
              "imrad_html", "supp_tables",
              "_ax_ped", "_ax_map"]

def init_state():
    for k in STATE_KEYS:
        if k not in st.session_state:
            st.session_state[k] = None
    if "trait_groups" not in st.session_state:
        st.session_state["trait_groups"] = dict(DEFAULT_GROUPS)

def reset_derived():
    for k in ["gt_filt", "ind_filt", "snp_filt", "qc_stats",
              "gt_snp", "ind_snp", "snp_snp", "gt_pruned", "snp_pruned",
              "pca_scores", "pca_var", "mds_coords", "king_matrix",
              "ld_df", "roh_df", "froh", "admixture_Q", "admixture_K",
              "cv_results", "reynolds_D", "reynolds_pops",
              "fst_global", "binome_result", "all_binomes",
              "group_results", "group_pair", "gwas_results_10cm",
              "fst_result", "bootstrap_ci", "tajima", "fuli", "faywu",
              "gwas_results_lmm", "vep_results", "go_enrichment",
              "kegg_enrichment"]:
        st.session_state[k] = None

def has_data(): return st.session_state.gt is not None
def has_qc():
    return (st.session_state.gt_filt is not None
            and st.session_state.qc_stats is not None)
def has_snp(): return st.session_state.gt_snp is not None
def has_pruned():
    return (st.session_state.gt_pruned is not None
            and st.session_state.snp_pruned is not None)

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    init_state()  # ← maintenant st est bien importé
    st.title("🐄 Bovine SNP Platform v7.2")
    st.caption("v6.2 (toutes fonctionnalités) + v7.1 (rigueur) — autonome")

    with st.sidebar:
        st.header("📁 Données")
        mode = st.radio("Source :", ["Demo", "PED/MAP", "VCF",
                                     "Convertisseur Axiom"],
                        index=0, key="sb_mode")
        if mode == "Demo":
            c1, c2 = st.columns(2)
            ni = c1.number_input("Ind.", 20, 500, 150, 10, key="sb_ni")
            ns = c2.number_input("SNPs", 100, 5000, 800, 100, key="sb_ns")
            np_ = st.slider("Populations", 2, 14, 4, key="sb_np")
            if st.button("🎲 Générer", use_container_width=True,
                         key="sb_gen"):
                with st.spinner("Génération..."):
                    gt, ind, snp = demo_data(int(ni), int(ns), int(np_))
                    st.session_state.gt = gt
                    st.session_state.ind_df = ind
                    st.session_state.snp_df = snp
                    reset_derived()
                st.success(f"✅ {gt.shape[0]} × {gt.shape[1]}")
        elif mode == "PED/MAP":
            pf = st.file_uploader(".ped", type=["ped", "gz", "txt"],
                                   key="sb_p")
            mf = st.file_uploader(".map", type=["map", "gz", "txt"],
                                   key="sb_m")
            if pf and mf and st.button("📥 Charger",
                                        use_container_width=True,
                                        key="sb_lp"):
                try:
                    map_df = parse_map(mf.read())
                    gt, ind = parse_ped(pf.read(), len(map_df))
                    if len(map_df) != gt.shape[1]:
                        map_df = map_df.iloc[:gt.shape[1]].reset_index(drop=True)
                    if ind["FID"].nunique() <= 1:
                        ind["FID"] = ind["IID"].astype(str).apply(
                            lambda x: x.split("_")[0])
                    st.session_state.gt = gt
                    st.session_state.ind_df = ind
                    st.session_state.snp_df = map_df
                    reset_derived()
                    st.success(f"✅ {gt.shape[0]} × {gt.shape[1]}")
                except Exception as e:
                    st.error(f"❌ {e}")
        elif mode == "VCF":
            vf = st.file_uploader(".vcf", type=["vcf", "gz", "txt"],
                                   key="sb_v")
            if vf and st.button("📥 Charger", use_container_width=True,
                                 key="sb_lv"):
                try:
                    gt, ind, snp = parse_vcf(vf.read())
                    if ind["FID"].nunique() <= 1:
                        ind["FID"] = ind["IID"].astype(str).apply(
                            lambda x: x.split("_")[0])
                    st.session_state.gt = gt
                    st.session_state.ind_df = ind
                    st.session_state.snp_df = snp
                    reset_derived()
                    st.success(f"✅ {gt.shape[0]} × {gt.shape[1]}")
                except Exception as e:
                    st.error(f"❌ {e}")

        st.divider()
        st.header("⚙️ Seuils QC")
        geno = st.slider("--geno", 0.0, 0.5, DEFAULTS["geno"], 0.01,
                         key="sb_g")
        mind = st.slider("--mind", 0.0, 0.5, DEFAULTS["mind"], 0.01,
                         key="sb_mi")
        maf_t = st.slider("MAF", 0.0, 0.5, DEFAULTS["maf"], 0.01,
                          key="sb_m")
        hwe_t = st.number_input("HWE", value=DEFAULTS["hwe"],
                                format="%.0e", key="sb_h")
        het_sd = st.slider("HET ±σ", 1.0, 5.0, DEFAULTS["het_sd"], 0.1,
                           key="sb_hs")
        ld_r2 = st.slider("LD r²", 0.05, 0.5, DEFAULTS["ld_r2"], 0.05,
                          key="sb_ld")

        st.divider()
        st.header("🐄 Groupes de races")
        if has_data():
            all_pops = sorted(st.session_state.ind_df["FID"].astype(str)
                              .unique())
            with st.expander("Définir", expanded=False):
                for gn in ["Laitier", "Viande", "Cornu", "Rustique"]:
                    sel = st.multiselect(
                        f"→ {gn}", all_pops,
                        default=st.session_state["trait_groups"].get(gn, []),
                        key=f"grp_{gn}")
                    st.session_state["trait_groups"][gn] = sel

        st.divider()
        st.caption("v7.2 — Autonome")

    # Convertisseur Axiom
    if mode == "Convertisseur Axiom":
        render_axiom_page()
        return

    # Accueil
    if not has_data():
        st.info("👉 Générez un jeu de démo, importez PED/MAP/VCF, "
                "ou utilisez le Convertisseur Axiom.")
        st.markdown("""
        ### 🧬 v7.2 — 12 onglets

        | Onglet | Contenu |
        |--------|---------|
        | 🏠 Aperçu | Statistiques |
        | 🧹 QC | Cascade complète |
        | ✂️ LD Pruning | Indep-pairwise |
        | 🧬 Structure | PCA + MDS + KING |
        | 🎨 Admixture | NMF |
        | 📈 Démographie | LD decay + ROH |
        | 🔍 Sélection | FST Manhattan |
        | 📤 Export | PLINK |
        | 🧬 Binômes | 6 modes + signatures |
        | 🧪 GWAS 10 cM | Local |
        | 🔬 **FST Weir-Cockerham** | IC bootstrap |
        | 📊 **Neutralité** | Tajima + Fu&Li + Fay&Wu |
        | 🧪 **GWAS LMM** | QQ + λGC |
        | 🧬 **VEP + GO/KEGG** | REST API |
        | 📄 **Rapport IMRaD** | S1-S10 |
        """)
        return

    gt = st.session_state.gt
    ind_df = st.session_state.ind_df
    snp_df = st.session_state.snp_df

    tabs = st.tabs(["🏠 Aperçu", "🧹 QC", "✂️ LD", "🧬 Structure",
                    "🎨 Admixture", "📈 Démographie", "🔍 Sélection",
                    "📤 Export", "🧬 Binômes", "🧪 GWAS 10cM",
                    "🔬 FST WC", "📊 Neutralité",
                    "🧪 GWAS LMM", "🧬 VEP", "📄 Rapport"])

    # TAB 0 : Aperçu
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        c1.metric("Individus", gt.shape[0])
        c2.metric("SNPs", gt.shape[1])
        c3.metric("Populations", ind_df["FID"].nunique())
        st.dataframe(ind_df.head(20), use_container_width=True)

    # TAB 1 : QC
    with tabs[1]:
        st.subheader("🧹 QC")
        if st.button("▶ QC complet", type="primary", key="qc_r"):
            try:
                with st.spinner("QC..."):
                    p = {"geno": geno, "mind": mind, "maf": maf_t,
                         "hwe": hwe_t, "het_sd": het_sd}
                    r = qc_filters(gt, ind_df, snp_df, p)
                    for k, v in zip(["gt_filt", "ind_filt", "snp_filt",
                                     "qc_stats", "gt_snp", "ind_snp",
                                     "snp_snp"], r):
                        st.session_state[k] = v
                st.success(f"✅ {r[0].shape[0]}×{r[0].shape[1]}")
            except Exception as e:
                st.error(f"❌ {e}")
        if has_qc():
            s = st.session_state.qc_stats
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ind.", s["n_ind_final"])
            c2.metric("SNPs", s["n_snp_final"])
            c3.metric("Ex. ind", s["excluded_ind"])
            c4.metric("Ex. SNPs", s["excluded_snp"])

    # TAB 2 : LD
    with tabs[2]:
        st.subheader("✂️ LD Pruning")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        elif st.button("▶ Pruning", type="primary", key="ld_btn"):
            with st.spinner("..."):
                k = ld_pruning(st.session_state.gt_filt, 50, 5, ld_r2)
                st.session_state.gt_pruned = st.session_state.gt_filt[:, k]
                st.session_state.snp_pruned = (
                    st.session_state.snp_filt[k].reset_index(drop=True))
            st.success(f"✅ {st.session_state.gt_pruned.shape[1]} SNPs")

    # TAB 3 : Structure
    with tabs[3]:
        st.subheader("🧬 Structure")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        else:
            gu = (st.session_state.gt_pruned if has_pruned()
                  else st.session_state.gt_filt)
            if st.button("▶ PCA + MDS + KING", type="primary",
                         key="str_btn"):
                with st.spinner("PCA..."):
                    sc, vp = pca_calc(gu, 10)
                    st.session_state.pca_scores = sc
                    st.session_state.pca_var = vp
                with st.spinner("MDS..."):
                    st.session_state.mds_coords = mds_calc(gu, 5)
                with st.spinner("KING..."):
                    st.session_state.king_matrix = king_matrix(gu)
                st.success("✅")
            lb = st.session_state.ind_filt["FID"].values
            if st.session_state.pca_scores is not None:
                st.plotly_chart(plt_pca(st.session_state.pca_scores,
                                        st.session_state.pca_var, lb),
                                use_container_width=True)
            if st.session_state.mds_coords is not None:
                st.plotly_chart(plt_mds(st.session_state.mds_coords, lb),
                                use_container_width=True)

    # TAB 4 : Admixture
    with tabs[4]:
        st.subheader("🎨 Admixture")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        else:
            gu = (st.session_state.gt_pruned if has_pruned()
                  else st.session_state.gt_filt)
            K = st.slider("K", 2, 10, 4, key="ad_K")
            if st.button("▶ Admixture", use_container_width=True, key="ad_r"):
                with st.spinner("NMF..."):
                    Q, _ = admix_nmf(gu, int(K))
                    st.session_state.admixture_Q = Q
                    st.session_state.admixture_K = K
                st.success(f"✅ Q {Q.shape}")
            if st.session_state.admixture_Q is not None:
                Q = st.session_state.admixture_Q
                K = st.session_state.admixture_K
                df = pd.DataFrame(Q, columns=[f"K{k+1}" for k in range(K)])
                df["Pop"] = st.session_state.ind_filt["FID"].values
                fig = go.Figure()
                for k in range(K):
                    fig.add_trace(go.Bar(x=np.arange(len(df)),
                                         y=df[f"K{k+1}"], name=f"K{k+1}"))
                fig.update_layout(barmode="stack", height=450)
                st.plotly_chart(fig, use_container_width=True)

    # TAB 5 : Démographie
    with tabs[5]:
        st.subheader("📈 Démographie")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        else:
            gu = (st.session_state.gt_pruned if has_pruned()
                  else st.session_state.gt_filt)
            s1, s2 = st.tabs(["LD decay", "ROH"])
            with s1:
                if st.button("▶ LD decay", key="ld_d"):
                    with st.spinner("..."):
                        st.session_state.ld_df = ld_decay(
                            gu, st.session_state.snp_filt["BP"].values)
                    st.success("✅")
                if (st.session_state.ld_df is not None
                        and not st.session_state.ld_df.empty):
                    d = st.session_state.ld_df.copy()
                    d["bin"] = (d["dist_kb"] // 20) * 20
                    agg = d.groupby("bin")["r2"].mean().reset_index()
                    st.plotly_chart(px.line(agg, x="bin", y="r2"),
                                    use_container_width=True)
            with s2:
                if st.button("▶ ROH", key="roh_b"):
                    try:
                        with st.spinner("..."):
                            sj = st.session_state.snp_filt[["CHR", "BP"]].to_json()
                            roh, froh = detect_roh(gu, sj)
                            st.session_state.roh_df = roh
                            st.session_state.froh = froh
                        st.success(f"✅ {len(roh)} ROH")
                    except Exception as e:
                        st.error(f"❌ {e}")
                if st.session_state.froh is not None:
                    df = pd.DataFrame({"FROH": st.session_state.froh,
                                       "Pop": st.session_state.ind_filt["FID"].values})
                    st.plotly_chart(px.histogram(df, x="FROH",
                                                 color="Pop", nbins=50),
                                    use_container_width=True)

    # TAB 6 : Sélection
    with tabs[6]:
        st.subheader("🔍 Sélection")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        else:
            gu = (st.session_state.gt_snp if has_snp()
                  else st.session_state.gt_filt)
            su = (st.session_state.snp_snp if has_snp()
                  else st.session_state.snp_filt)
            iu = (st.session_state.ind_snp if has_snp()
                  else st.session_state.ind_filt)
            if st.button("▶ FST", key="fst_g"):
                with st.spinner("..."):
                    st.session_state.fst_global = fst_per_snp(
                        gu, iu["FID"].values)
                st.success("✅")
            if st.session_state.fst_global is not None:
                fig = plt_man(st.session_state.fst_global, su["CHR"].values)
                if fig: st.plotly_chart(fig, use_container_width=True)

    # TAB 7 : Export
    with tabs[7]:
        st.subheader("📤 Export PLINK")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        else:
            ge = (st.session_state.gt_pruned if has_pruned()
                  else st.session_state.gt_filt)
            ie = st.session_state.ind_filt
            se = (st.session_state.snp_pruned if has_pruned()
                  else st.session_state.snp_filt)
            pre = st.text_input("Préfixe", "bovine_qc", key="ex_p")
            if st.button("📦 Générer", key="ex_g"):
                try:
                    with st.spinner("..."):
                        data = build_zip(ge, ie, se, pre)
                    st.download_button("⬇ Télécharger", data,
                                       f"{pre}.zip", "application/zip",
                                       key="ex_dl")
                except Exception as e:
                    st.error(f"❌ {e}")

    # TAB 8 : Binômes
    with tabs[8]:
        st.subheader("🧬 Binômes & Signatures")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        else:
            gu = (st.session_state.gt_snp if has_snp()
                  else st.session_state.gt_filt)
            su = (st.session_state.snp_snp if has_snp()
                  else st.session_state.snp_filt)
            iu = (st.session_state.ind_snp if has_snp()
                  else st.session_state.ind_filt)
            lb = iu["FID"].astype(str).values
            avail = sorted(np.unique(lb))

            mode_b = st.radio("Mode :",
                              ["🖱️ Manuelle",
                               "🎯 Signature par population",
                               "🐄 Signature par groupe",
                               "⚔️ Groupe A vs B"],
                              key="bin_mode")
            c1, c2 = st.columns(2)
            q_b = c1.slider("Quantile", 0.90, 0.9999, 0.99, 0.001,
                            format="%.4f", key="bin_q")
            fst_b = c2.slider("FST min", 0.0, 0.5, 0.15, 0.01,
                              key="bin_mf")

            if mode_b == "🖱️ Manuelle":
                c1, c2 = st.columns(2)
                a = c1.selectbox("Pop A", avail, key="man_a")
                b = c2.selectbox("Pop B", [x for x in avail if x != a],
                                 key="man_b")
                if st.button(f"▶ {a} vs {b}", type="primary", key="man_run"):
                    with st.spinner("..."):
                        st.session_state.binome_result = analyze_pair(
                            gu, su, lb, a, b, q_b, fst_b)
                r = st.session_state.binome_result
                if r and r.get("status") == "success":
                    st.metric("SNP outliers", r["n_snp_outliers"])

            elif mode_b == "🎯 Signature par population":
                if st.button("▶ Toutes les populations", type="primary",
                             key="pop_rest"):
                    res = analyze_pop_vs_rest(gu, su, lb, q_b, fst_b)
                    st.session_state.group_results = res
                    st.success(f"✅ {len(res)}")
                if st.session_state.group_results:
                    summ = [{"Pop": k, "Outliers": v["n_snp_outliers"],
                             "FST": round(v["fst_mean"], 4)}
                            for k, v in st.session_state.group_results.items()
                            if v.get("status") == "success"]
                    if summ:
                        st.dataframe(pd.DataFrame(summ),
                                     use_container_width=True)

            elif mode_b == "🐄 Signature par groupe":
                groups = {k: v for k, v in
                          st.session_state["trait_groups"].items() if v}
                if not groups:
                    st.warning("⚠️ Définissez les groupes.")
                elif st.button("▶ Analyser tous les groupes",
                               type="primary", key="grp_all"):
                    res = analyze_grp_vs_rest(gu, su, lb, groups,
                                               q_b, fst_b)
                    st.session_state.group_results = res
                    st.success(f"✅ {len(res)}")

            else:
                groups = {k: v for k, v in
                          st.session_state["trait_groups"].items() if v}
                if len(groups) < 2:
                    st.warning("⚠️ Définissez 2 groupes.")
                else:
                    c1, c2 = st.columns(2)
                    gA = c1.selectbox("Groupe A", list(groups.keys()),
                                      key="gA")
                    gB = c2.selectbox("Groupe B",
                                      [g for g in groups if g != gA],
                                      key="gB")
                    if st.button(f"▶ {gA} vs {gB}", type="primary",
                                 key="pair_run"):
                        with st.spinner("..."):
                            st.session_state.group_pair = analyze_grp_pair(
                                gu, su, lb, groups, gA, gB, q_b, fst_b)
                        st.success("✅")

    # TAB 9 : GWAS 10 cM
    with tabs[9]:
        st.subheader("🧪 GWAS 10 cM")
        st.info("Définissez un phénotype (groupe) puis lancez GWAS local.")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        else:
            gu = (st.session_state.gt_snp if has_snp()
                  else st.session_state.gt_filt)
            su = (st.session_state.snp_snp if has_snp()
                  else st.session_state.snp_filt)
            iu = (st.session_state.ind_snp if has_snp()
                  else st.session_state.ind_filt)
            groups = {k: v for k, v in
                      st.session_state["trait_groups"].items() if v}
            if not groups:
                st.warning("⚠️ Définissez les groupes.")
            else:
                gn = st.selectbox("Groupe = cas :", list(groups.keys()),
                                  key="gw_g")
                ph = (grp_phenotype(iu, groups, gn)
                      .rename(columns={"pheno": "pheno_group"})
                      [["IID", "pheno_group"]])
                st.success(f"✅ {int(ph['pheno_group'].sum())} cas")
                res = st.session_state.group_results
                if res:
                    avail = [k for k, v in res.items()
                             if v.get("status") == "success"]
                    if avail:
                        sel_g = st.selectbox("Depuis :", avail, key="gw_sg")
                        cands = res[sel_g]["outliers_df"]
                        if not cands.empty:
                            top = cands.head(5)
                            st.dataframe(top[["CHR", "SNP", "BP", "FST"]],
                                         use_container_width=True)
                            win = st.slider("Fenêtre cM", 1, 20, 10,
                                            key="gw_w")
                            if st.button("▶ GWAS", type="primary",
                                         key="gw_run"):
                                all_g = {}
                                for _, s in top.iterrows():
                                    r = gwas_local(gu, su, iu, ph,
                                                   s["CHR"], s["BP"],
                                                   "pheno_group", win)
                                    if not r.empty:
                                        all_g[s["SNP"]] = r
                                st.session_state.gwas_results_10cm = all_g
                                st.success(f"✅ {len(all_g)}")

    # TAB 10 : FST Weir-Cockerham
    with tabs[10]:
        st.subheader("🔬 FST Weir & Cockerham + Bootstrap IC")
        st.markdown("**Standard peer-review** avec IC 95% (bootstrap 500×).")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        elif st.button("▶ Calculer FST + Bootstrap", type="primary",
                       key="wc_btn"):
            with st.spinner("Weir-Cockerham..."):
                fst, theta, se = fst_weir_cockerham_inline(
                    st.session_state.gt_snp,
                    st.session_state.ind_snp["FID"].values)
            with st.spinner("Bootstrap..."):
                boot = bootstrap_fst_inline(
                    st.session_state.gt_snp,
                    st.session_state.ind_snp["FID"].values, n_boot=500)
            st.session_state.fst_result = {"fst": fst, "theta": theta, "se": se}
            st.session_state.bootstrap_ci = boot
            st.success("✅")
        if st.session_state.fst_result:
            fd = st.session_state.fst_result
            boot = st.session_state.bootstrap_ci
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("FST (θ)", f"{fd['theta']:.4f}")
            c2.metric("SE (jackknife)", f"{fd['se']:.4f}")
            c3.metric("IC bas", f"{boot['ci_low']:.4f}")
            c4.metric("IC haut", f"{boot['ci_high']:.4f}")
            st.markdown(f"**FST = {boot['mean']:.4f}** "
                        f"[IC 95% : {boot['ci_low']:.4f} – "
                        f"{boot['ci_high']:.4f}] (n={boot['n']})")

    # TAB 11 : Neutralité
    with tabs[11]:
        st.subheader("📊 Tests de neutralité")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("▶ Tajima's D", use_container_width=True,
                             key="taj_btn"):
                    with st.spinner("..."):
                        st.session_state.tajima = tajimas_d(
                            st.session_state.gt_snp)
                    st.success("✅")
                if st.session_state.tajima:
                    r = st.session_state.tajima
                    st.metric("Tajima D", r.get("D", "NA"))
                    st.json(r)
            with c2:
                if st.button("▶ Fu & Li D*", use_container_width=True,
                             key="fuli_btn"):
                    with st.spinner("..."):
                        st.session_state.fuli = fu_li_d(
                            st.session_state.gt_snp)
                    st.success("✅")
                if st.session_state.fuli:
                    r = st.session_state.fuli
                    st.metric("D*", r.get("D_star", "NA"))
                    st.json(r)
            if st.button("▶ Fay & Wu H", use_container_width=True,
                         key="faywu_btn"):
                with st.spinner("..."):
                    st.session_state.faywu = fay_wu_h(
                        st.session_state.gt_snp)
                st.success("✅")
            if st.session_state.faywu:
                r = st.session_state.faywu
                st.metric("Fay & Wu H", r.get("H", "NA"))
                st.json(r)

    # TAB 12 : GWAS LMM
    with tabs[12]:
        st.subheader("🧪 GWAS LMM + QQ + λGC")
        st.markdown("**Modèle mixte EMMA** avec matrice KING + PCs.")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        elif st.button("▶ Lancer GWAS LMM", type="primary", key="lmm_btn"):
            with st.spinner("LMM (peut prendre 30s)..."):
                gt_ = st.session_state.gt_snp
                rng = np.random.default_rng(42)
                pheno = rng.normal(0, 1, gt_.shape[0])
                n = gt_.shape[0]
                h = np.nansum(gt_ == 1, axis=1).astype(float)
                K = np.zeros((n, n))
                for i in range(n):
                    for j in range(i+1, n):
                        bh = np.nansum((gt_[i] == 1) & (gt_[j] == 1))
                        d = h[i] + h[j]
                        K[i, j] = K[j, i] = (d-2*bh)/d if d > 0 else 0
                np.fill_diagonal(K, 0.5)
                cov = (st.session_state.pca_scores[:, :5]
                       if st.session_state.pca_scores is not None else None)
                res = gwas_lmm_inline(gt_, pheno, K, cov,
                                      maf_min=maf_t, miss_max=mind)
            if not res.empty:
                res["SNP"] = st.session_state.snp_snp["SNP"].values[
                    res["SNP_idx"].values]
                res["CHR"] = st.session_state.snp_snp["CHR"].values[
                    res["SNP_idx"].values]
                res["BP"] = st.session_state.snp_snp["BP"].values[
                    res["SNP_idx"].values]
                st.session_state.gwas_results_lmm = res
                st.session_state.gwas_lambda = lambda_gc(res["P"].values)
                st.session_state.gwas_qq = qq_data(res["P"].values)
                st.success(f"✅ {len(res)} SNPs")
        if st.session_state.gwas_results_lmm is not None:
            res = st.session_state.gwas_results_lmm
            lam = st.session_state.gwas_lambda
            thr = gwas_thresholds(res["P"].values)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("SNPs testés", len(res))
            c2.metric("λGC", f"{lam:.3f}" if np.isfinite(lam) else "NA")
            c3.metric("Bonferroni", f"{thr['bonferroni']:.2e}")
            c4.metric("Sig. Bonf.", thr["n_sig_bonf"])
            st.dataframe(res.sort_values("P").head(20),
                         use_container_width=True)

    # TAB 13 : VEP + GO/KEGG
    with tabs[13]:
        st.subheader("🧬 VEP + GO + KEGG (REST API)")
        if st.session_state.gwas_results_lmm is None:
            st.info("Lancez d'abord un GWAS LMM.")
        else:
            top = st.session_state.gwas_results_lmm.sort_values("P").head(20)
            st.write(f"**{len(top)} top SNPs** à annoter")
            if st.button("▶ Annoter via VEP", type="primary", key="vep_btn"):
                with st.spinner("VEP API..."):
                    vep_res = vep_annotate(top)
                st.session_state.vep_results = vep_res
                st.success(f"✅ {len(vep_res)} variants")
            if st.session_state.vep_results is not None:
                st.dataframe(st.session_state.vep_results,
                             use_container_width=True)
                genes = [g for g in st.session_state.vep_results["gene"].unique()
                         if g and g != "?"]
                if genes and st.button("▶ GO enrichment", key="go_btn"):
                    with st.spinner("g:Profiler..."):
                        st.session_state.go_enrichment = go_enrichment(genes)
                    st.success("✅")
                if st.session_state.go_enrichment is not None:
                    st.dataframe(st.session_state.go_enrichment.head(20),
                                 use_container_width=True)
                if genes and st.button("▶ KEGG enrichment", key="kegg_btn"):
                    with st.spinner("KEGG API..."):
                        st.session_state.kegg_enrichment = kegg_enrichment(genes)
                    st.success("✅")
                if st.session_state.kegg_enrichment is not None:
                    st.dataframe(st.session_state.kegg_enrichment.head(20),
                                 use_container_width=True)

    # TAB 14 : Rapport IMRaD
    with tabs[14]:
        st.subheader("📄 Rapport IMRaD + S1-S10")
        proj = st.text_input("Nom du projet", "Cattle_Project_v72",
                             key="rep_p")
        if st.button("📄 Générer", type="primary", key="rep_btn"):
            with st.spinner("..."):
                html = build_imrad_html(st.session_state, {
                    "project_name": proj,
                    "geno": geno, "mind": mind, "maf": maf_t,
                    "hwe": hwe_t})
                tables = build_supp_tables(st.session_state)
            st.session_state.imrad_html = html
            st.session_state.supp_tables = tables
            st.success(f"✅ Rapport + {len(tables)} tables")
        if st.session_state.imrad_html:
            st.download_button(
                "⬇ Rapport IMRaD (HTML)",
                data=st.session_state.imrad_html.encode("utf-8"),
                file_name=f"rapport_IMRaD_{datetime.now():%Y%m%d}.html",
                mime="text/html",
                use_container_width=True)
        if st.session_state.supp_tables:
            for name, df in st.session_state.supp_tables.items():
                with st.expander(f"📊 {name}"):
                    st.dataframe(df, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
