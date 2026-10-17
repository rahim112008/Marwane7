"""
🐄 Bovine SNP Platform v7.2 — Best of Both (v6.2 + v7.1)
Conserve TOUTES les fonctionnalités de la v6.2 :
  Axiom, PED/MAP/VCF, QC, LD Pruning, Structure, Admixture,
  Démographie, Phylogénie, Export, Binômes, Signatures groupes,
  GWAS 10 cM, Annotation GTF
Ajoute TOUTES les fonctionnalités de la v7.1 :
  FST Weir-Cockerham + Bootstrap IC, Tajima/Fu&Li/Fay&Wu,
  iHS/XP-EHH, GWAS LMM + λGC, VEP + GO/KEGG, Rapport IMRaD S1-S10
"""
import gzip, json, warnings, zipfile
from collections import Counter
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from scipy.spatial.distance import squareform
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import MDS as SklearnMDS

# === Modules v7.1 ===
try:
    from v71_statistics import (fst_weir_cockerham, bootstrap_fst_ci,
                                 jackknife_ne_ci, bonferroni,
                                 benjamini_hochberg, storey_qvalues,
                                 permutation_test, power_fst, required_n)
    from v71_neutrality import (tajimas_d, fu_li_d, fay_wu_h,
                                 ihs_scan, xp_ehh_scan)
    from v71_gwas import (gwas_lmm, qq_data, lambda_gc,
                           gwas_thresholds)
    from v71_annotation import (vep_annotate, go_enrichment,
                                 kegg_enrichment, string_network)
    from v71_report import build_imrad_html, build_supplementary_tables
    V71_OK = True
except ImportError as e:
    V71_OK = False
    st.error(f"⚠️ Module v7.1 manquant : {e}")

warnings.filterwarnings("ignore", category=RuntimeWarning)

st.set_page_config(page_title="🐄 Bovine SNP v7.2", page_icon="🐄",
                    layout="wide", initial_sidebar_state="expanded")

# === Constantes v6.2 ===
DEFAULTS = {"geno": 0.05, "mind": 0.05, "maf": 0.05,
             "hwe": 1e-6, "het_sd": 3.0, "king": 0.354, "ld_r2": 0.2}
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

# ==================== CACHE ====================
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

def to_np_str(a):
    return np.asarray(list(a), dtype=object)

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

# ==================== CONVERTISSEUR AXIOM (v6.2) ====================
def _clean_allele(a):
    if a is None: return "0"
    a = str(a).strip()
    return "0" if a in ("", ".", "-", "NA", "nan", "None", "0", "00") else a

def sample_to_fid_iid(name, fid_parts=2):
    n = str(name).strip()
    for e in (".CEL", ".cel", ".TXT", ".txt", ".gz", ".GZ"):
        if n.endswith(e): n = n[:-len(e)]
    n = (n.replace("(", "_").replace(")", "")
         .replace("[", "_").replace("]", "")
         .replace(" ", "_").replace("/", "_"))
    while "__" in n: n = n.replace("__", "_")
    n = n.strip("_")
    p = n.split("_")
    return ("_".join(p[:fid_parts]) if len(p) >= fid_parts else p[0]), n

def parse_axiom(raw, fn=""):
    if fn.endswith(".gz") or (len(raw) >= 2 and raw[:2] == b"\x1f\x8b"):
        try: raw = gzip.decompress(raw)
        except Exception as e: raise ValueError(f".gz corrompu : {e}")
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
    st.markdown("Transforme un fichier Axiom (`A/G`, `T T`) en "
                "**PED** + **MAP** PLINK.")
    st.divider()
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
            st.session_state["_ax_n"] = n_snp
            st.session_state["_ax_samples"] = samples
            st.success(f"✅ {len(samples)} ind × {n_snp} SNPs")
        except Exception as e:
            st.error(f"❌ {e}")
    if st.session_state.get("_ax_ped"):
        st.subheader("Téléchargement")
        c1, c2, c3 = st.columns(3)
        c1.download_button("⬇ .ped",
                            st.session_state["_ax_ped"].encode("utf-8"),
                            "converted.ped", "text/plain",
                            use_container_width=True, key="dl_ax_p")
        c2.download_button("⬇ .map",
                            st.session_state["_ax_map"].encode("utf-8"),
                            "converted.map", "text/plain",
                            use_container_width=True, key="dl_ax_m")
        bio = BytesIO()
        with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("converted.ped", st.session_state["_ax_ped"])
            zf.writestr("converted.map", st.session_state["_ax_map"])
        c3.download_button("⬇ .zip", bio.getvalue(), "converted.zip",
                            "application/zip", use_container_width=True,
                            key="dl_ax_z")

# ==================== PARSERS (v6.2) ====================
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
            st.info("ℹ️ Header PED ignoré")
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
