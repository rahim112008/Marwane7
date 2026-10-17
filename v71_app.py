"""
v71_app.py — Streamlit principal intégrant les modules v7.1
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gzip, json, zipfile
from io import BytesIO
from datetime import datetime

# Imports des modules v7.1
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
    V71_MODULES_OK = True
except ImportError as e:
    V71_MODULES_OK = False
    st.error(f"⚠️ Module manquant : {e}")

st.set_page_config(page_title="🐄 Bovine SNP v7.1", page_icon="🐄",
                    layout="wide", initial_sidebar_state="expanded")

# ==================== ÉTAT ====================
if "gt" not in st.session_state:
    for k in ["gt", "ind_df", "snp_df", "gt_filt", "ind_filt", "snp_filt",
              "qc_stats", "gt_snp", "ind_snp", "snp_snp",
              "pca_scores", "pca_var", "mds_coords", "king_matrix",
              "fst_result", "bootstrap_ci", "fst_pair_matrix",
              "tajima", "fuli", "faywu", "ihs_results", "xpehh_results",
              "gwas_results", "gwas_qq", "gwas_lambda",
              "vep_results", "go_enrichment", "kegg_enrichment",
              "imrad_html", "supp_tables", "ne_ci"]:
        st.session_state[k] = None

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("🐄 v7.1 — Pro")
    st.markdown("**Modules actifs :**")
    if V71_MODULES_OK:
        st.success("✅ Weir-Cockerham\n\n✅ Bootstrap IC\n\n"
                   "✅ Tests neutralité\n\n✅ GWAS LMM\n\n"
                   "✅ VEP / GO / KEGG\n\n✅ Rapport IMRaD")

    st.divider()
    st.header("📁 Données")
    mode = st.radio("Source", ["Demo", "PED/MAP"], key="sb_mode")
    if mode == "Demo":
        ni = st.number_input("Individus", 20, 500, 150, 10)
        ns = st.number_input("SNPs", 100, 5000, 800, 100)
        if st.button("🎲 Générer"):
            # Import demo function from v6.2 style
            rng = np.random.default_rng(42)
            npop = 4
            pops = ["AND", "EBG", "LJR", "SHO"][:npop]
            cnt = [ni//npop] * npop
            p = rng.beta(0.6, 0.6, ns)
            gt = np.zeros((sum(cnt), ns), dtype=np.float32)
            rows = []
            k = 0
            for pop, ci in zip(pops, cnt):
                pk = np.clip(p + rng.normal(0, 0.15, ns), 0.02, 0.98)
                for i in range(ci):
                    gt[k] = ((rng.random(ns) < pk).astype(float)
                             + (rng.random(ns) < pk).astype(float))
                    rows.append({"FID": pop, "IID": f"{pop}_{i+1:03d}"})
                    k += 1
            gt[rng.random(gt.shape) < 0.02] = np.nan
            chrs = rng.choice([str(i) for i in range(1, 30)], ns)
            bps = np.array([rng.integers(1, 100_000_000) for _ in chrs])
            snp_df = pd.DataFrame({"CHR": chrs,
                                    "SNP": [f"rs{i:07d}" for i in range(ns)],
                                    "CM": 0.0, "BP": bps})
            st.session_state.gt = gt
            st.session_state.ind_df = pd.DataFrame(rows)
            st.session_state.snp_df = snp_df
            st.success(f"✅ {gt.shape[0]} × {gt.shape[1]}")

    st.divider()
    st.header("⚙️ QC")
    geno = st.slider("--geno", 0.0, 0.5, 0.05, 0.01)
    mind = st.slider("--mind", 0.0, 0.5, 0.05, 0.01)
    maf_t = st.slider("MAF", 0.0, 0.5, 0.05, 0.01)
    hwe_t = st.number_input("HWE", value=1e-6, format="%.0e")

# ==================== QC helper ====================
def simple_qc(gt, ind, snp, p):
    """QC rapide pour v7.1."""
    from v71_statistics import fst_weir_cockerham  # force import
    nm = np.isnan(gt)
    # geno
    miss_snp = nm.mean(0)
    keep = miss_snp <= p["geno"]
    gt, snp = gt[:, keep], snp[keep].reset_index(drop=True)
    # mind
    miss_ind = np.isnan(gt).mean(1)
    keep = miss_ind <= p["mind"]
    gt, ind = gt[keep], ind[keep].reset_index(drop=True)
    # MAF
    p_af = np.nanmean(gt, 0)/2.0
    m = np.minimum(p_af, 1-p_af)
    keep = np.isfinite(m) & (m >= p["maf"])
    gt, snp = gt[:, keep], snp[keep].reset_index(drop=True)
    return gt, ind, snp, {
        "n_ind_init": len(ind),
        "n_snp_init": len(snp),
        "n_ind_final": gt.shape[0],
        "n_snp_final": gt.shape[1],
        "excluded_ind": 0, "excluded_snp": 0,
        "trace": {"qc": 0}}

# ==================== ACCUEIL ====================
st.title("🐄 Bovine SNP Platform v7.1 — Rigueur Scientifique")
st.caption("Weir-Cockerham · Bootstrap IC · Tests de neutralité · "
            "GWAS LMM · VEP · GO/KEGG · Rapport IMRaD")

if st.session_state.gt is None:
    st.info("👉 Générez un jeu de démo dans la sidebar.")
    st.markdown("""
    ### 🎯 Modules v7.1

    | Module | Contenu | Standard |
    |--------|---------|----------|
    | **FST** | **Weir & Cockerham (1984)** + **bootstrap IC 95%** | Peer-review |
    | **Neutralité** | **Tajima D** + **Fu & Li D*** + **Fay & Wu H** | Peer-review |
    | **iHS/XP-EHH** | Approximations EHH-based | Exploratoire |
    | **GWAS** | **LMM EMMA** + **QQ-plot** + **λGC** | Peer-review |
    | **Annotation** | **VEP REST** + **GO** + **KEGG** | Peer-review |
    | **Rapport** | **IMRaD + S1-S10** | Publication |
    """)
    st.stop()

gt = st.session_state.gt
ind_df = st.session_state.ind_df
snp_df = st.session_state.snp_df

# ==================== TABS ====================
tabs = st.tabs(["🏠 Aperçu", "🧹 QC", "🧬 Structure",
                "🔬 **FST Weir-Cockerham**",
                "📊 **Neutralité**",
                "🧪 **GWAS LMM**",
                "🧬 **VEP + GO/KEGG**",
                "📄 **Rapport IMRaD**"])

# ----- TAB 0 : Aperçu -----
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Individus", gt.shape[0])
    c2.metric("SNPs", gt.shape[1])
    c3.metric("Populations", ind_df["FID"].nunique())
    st.dataframe(ind_df.head(20), use_container_width=True)

# ----- TAB 1 : QC -----
with tabs[1]:
    st.subheader("Contrôle qualité")
    if st.button("▶ QC complet", type="primary"):
        with st.spinner("QC..."):
            p = {"geno": geno, "mind": mind, "maf": maf_t, "hwe": hwe_t}
            g, i, s, st_ = simple_qc(gt, ind_df, snp_df, p)
            st.session_state.gt_filt = g
            st.session_state.ind_filt = i
            st.session_state.snp_filt = s
            st.session_state.qc_stats = st_
            st.session_state.gt_snp = g.copy()
            st.session_state.ind_snp = i.copy()
            st.session_state.snp_snp = s.copy()
        st.success(f"✅ {g.shape[0]} × {g.shape[1]}")
    if st.session_state.qc_stats:
        s = st.session_state.qc_stats
        c1, c2 = st.columns(2)
        c1.metric("Individus finaux", s["n_ind_final"])
        c2.metric("SNPs finaux", s["n_snp_final"])

# ----- TAB 2 : Structure -----
with tabs[2]:
    st.subheader("Structure (PCA + KING)")
    if st.session_state.gt_filt is None:
        st.warning("⚠️ Lancez le QC.")
    else:
        if st.button("▶ PCA + KING"):
            from sklearn.decomposition import PCA
            from v71_statistics import _ne_fast  # noqa
            with st.spinner("..."):
                X = st.session_state.gt_filt.copy()
                X = X.astype(np.float32)
                cm = np.nanmean(X, 0)
                cm = np.where(np.isnan(cm), 0, cm)
                nm = np.isnan(X)
                X[nm] = np.take(cm, np.where(nm)[1])
                X = X - X.mean(0)
                nc = min(10, X.shape[0]-1, X.shape[1])
                pca = PCA(nc).fit(X)
                st.session_state.pca_scores = pca.transform(X)
                st.session_state.pca_var = pca.explained_variance_ratio_*100
            st.success("✅")
        if st.session_state.pca_scores is not None:
            df = pd.DataFrame({"PC1": st.session_state.pca_scores[:, 0],
                                "PC2": st.session_state.pca_scores[:, 1],
                                "Pop": st.session_state.ind_filt["FID"].values})
            fig = px.scatter(df, x="PC1", y="PC2", color="Pop",
                             title="PCA")
            st.plotly_chart(fig, use_container_width=True)

# ----- TAB 3 : FST Weir-Cockerham -----
with tabs[3]:
    st.subheader("🔬 FST Weir & Cockerham (1984)")
    st.markdown("""
    **Standard peer-review** pour la génétique des populations.
    IC 95% par bootstrap (n=1000).
    """)

    if st.session_state.gt_snp is None:
        st.warning("⚠️ Lancez le QC.")
    else:
        if st.button("▶ Calculer FST + Bootstrap", type="primary"):
            with st.spinner("Weir-Cockerham..."):
                fst, theta, se = fst_weir_cockerham(
                    st.session_state.gt_snp,
                    st.session_state.ind_snp["FID"].values)
            with st.spinner("Bootstrap 1000×..."):
                boot = bootstrap_fst_ci(
                    st.session_state.gt_snp,
                    st.session_state.ind_snp["FID"].values, n_boot=1000)
            st.session_state.fst_result = {"fst": fst, "theta": theta, "se": se}
            st.session_state.bootstrap_ci = boot
            st.success("✅")

        if st.session_state.fst_result:
            fst_data = st.session_state.fst_result
            boot = st.session_state.bootstrap_ci
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("FST moyen", f"{fst_data['theta']:.4f}")
            c2.metric("SE (jackknife)", f"{fst_data['se']:.4f}")
            c3.metric("IC 95% bas", f"{boot['ci_low']:.4f}")
            c4.metric("IC 95% haut", f"{boot['ci_high']:.4f}")
            st.markdown(f"""
            **Résultat final :** FST = **{boot['mean']:.4f}**
            [IC 95% : {boot['ci_low']:.4f} – {boot['ci_high']:.4f}]
            (bootstrap n={boot['n']})
            """)

            # Manhattan
            fst = st.session_state.fst_result["fst"]
            df = pd.DataFrame({"FST": fst, "CHR": st.session_state.snp_snp["CHR"].values})
            df = df.dropna()
            if not df.empty:
                fig = px.scatter(df, x=df.index, y="FST", color="CHR",
                                  title="FST par SNP")
                st.plotly_chart(fig, use_container_width=True)

# ----- TAB 4 : Neutralité -----
with tabs[4]:
    st.subheader("📊 Tests de neutralité")
    if st.session_state.gt_snp is None:
        st.warning("⚠️ Lancez le QC.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶ Tajima's D", use_container_width=True):
                with st.spinner("..."):
                    st.session_state.tajima = tajimas_d(st.session_state.gt_snp)
                st.success("✅")
            if st.session_state.tajima:
                r = st.session_state.tajima
                st.metric("Tajima D", r.get("D", "NA"))
                st.write(f"**Interprétation :** {r.get('interpretation', 'NA')}")
                st.json(r)

        with c2:
            if st.button("▶ Fu & Li D*", use_container_width=True):
                with st.spinner("..."):
                    st.session_state.fuli = fu_li_d(st.session_state.gt_snp)
                st.success("✅")
            if st.session_state.fuli:
                r = st.session_state.fuli
                st.metric("D*", r.get("D_star", "NA"))
                st.write(f"**Interprétation :** {r.get('interpretation', 'NA')}")
                st.json(r)

        st.divider()
        if st.button("▶ Fay & Wu H", use_container_width=True):
            with st.spinner("..."):
                st.session_state.faywu = fay_wu_h(st.session_state.gt_snp)
            st.success("✅")
        if st.session_state.faywu:
            r = st.session_state.faywu
            st.metric("Fay & Wu H", r.get("H", "NA"))
            st.write(f"**Interprétation :** {r.get('interpretation', 'NA')}")
            st.json(r)

        st.divider()
        st.subheader("iHS / XP-EHH (approximations)")
        if st.button("▶ iHS scan", use_container_width=True):
            with st.spinner("..."):
                st.session_state.ihs_results = ihs_scan(
                    st.session_state.gt_snp, st.session_state.snp_snp)
            st.success(f"✅ {len(st.session_state.ihs_results)} fenêtres")
        if st.session_state.ihs_results is not None and not st.session_state.ihs_results.empty:
            st.dataframe(st.session_state.ihs_results.head(20),
                          use_container_width=True)

# ----- TAB 5 : GWAS LMM -----
with tabs[5]:
    st.subheader("🧪 GWAS LMM (EMMA) + QQ + λGC")
    st.markdown("""
    **Modèle mixte** avec matrice de parenté et PCs. Standard peer-review.
    """)

    if st.session_state.gt_snp is None:
        st.warning("⚠️ Lancez le QC.")
    else:
        pheno_source = st.radio("Phénotype", ["Simulé (aléatoire)",
                                                "Manuel (groupe)"],
                                  horizontal=True)
        if pheno_source == "Simulé (aléatoire)":
            rng = np.random.default_rng(42)
            pheno = rng.normal(0, 1, st.session_state.gt_snp.shape[0])
            st.info(f"Phénotype simulé : N({pheno.mean():.2f}, {pheno.std():.2f})")
        else:
            st.info("Définissez un phénotype binaire (0/1)")
            if "pheno_manual" not in st.session_state:
                st.session_state["pheno_manual"] = None
            ph = st.text_area("Une valeur par ligne",
                              placeholder="1\n0\n1\n...")
            if st.button("Valider"):
                try:
                    st.session_state["pheno_manual"] = np.array(
                        [float(x.strip()) for x in ph.strip().split("\n")])
                    st.success(f"✅ {len(st.session_state['pheno_manual'])} valeurs")
                except Exception as e:
                    st.error(f"❌ {e}")
            pheno = st.session_state.get("pheno_manual")

        if pheno is not None and len(pheno) == st.session_state.gt_snp.shape[0]:
            if st.button("▶ Lancer GWAS LMM", type="primary"):
                with st.spinner("LMM (peut prendre 30s)..."):
                    # KING matrix
                    gt = st.session_state.gt_snp
                    het = np.nansum(gt == 1, axis=1).astype(float)
                    n = gt.shape[0]
                    K = np.zeros((n, n))
                    for i in range(n):
                        for j in range(i+1, n):
                            bh = np.nansum((gt[i] == 1) & (gt[j] == 1))
                            d = het[i] + het[j]
                            K[i, j] = K[j, i] = (d - 2*bh)/d if d > 0 else 0
                    np.fill_diagonal(K, 0.5)

                    cov = st.session_state.pca_scores[:, :5] if st.session_state.pca_scores is not None else None

                    res = gwas_lmm(gt, pheno, K, cov,
                                   maf_min=maf_t, miss_max=mind)
                if not res.empty:
                    res["SNP"] = st.session_state.snp_snp["SNP"].values[res["SNP_idx"].values]
                    res["CHR"] = st.session_state.snp_snp["CHR"].values[res["SNP_idx"].values]
                    res["BP"] = st.session_state.snp_snp["BP"].values[res["SNP_idx"].values]
                    st.session_state.gwas_results = res
                    st.session_state.gwas_lambda = lambda_gc(res["P"].values)
                    st.session_state.gwas_qq = qq_data(res["P"].values)
                    st.success(f"✅ {len(res)} SNPs testés")

        if st.session_state.gwas_results is not None:
            res = st.session_state.gwas_results
            lam = st.session_state.gwas_lambda
            thr = gwas_thresholds(res["P"].values)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("SNPs testés", len(res))
            c2.metric("λGC", f"{lam:.3f}" if np.isfinite(lam) else "NA")
            c3.metric("Bonferroni", f"{thr['bonferroni']:.2e}")
            c4.metric("Sig. Bonf.", thr["n_sig_bonf"])

            # Manhattan
            res_clean = res[np.isfinite(res["P"])].copy()
            res_clean["neg_log10_p"] = -np.log10(res_clean["P"].clip(lower=1e-300))
            fig = px.scatter(res_clean, x="BP", y="neg_log10_p", color="CHR",
                             title="GWAS Manhattan")
            fig.add_hline(y=-np.log10(thr['bonferroni']),
                          line_dash="dash", line_color="red",
                          annotation_text="Bonferroni")
            st.plotly_chart(fig, use_container_width=True)

            # QQ-plot
            qq = st.session_state.gwas_qq
            if qq:
                fig_qq = go.Figure()
                fig_qq.add_trace(go.Scatter(x=qq["expected"], y=qq["observed"],
                                              mode="markers", name="Observé"))
                lim = max(qq["expected"].max(), qq["observed"].max())
                fig_qq.add_trace(go.Scatter(x=[0, lim], y=[0, lim],
                                              mode="lines",
                                              line=dict(color="red", dash="dash"),
                                              name="Attendu (H0)"))
                fig_qq.update_layout(title=f"QQ-plot (λGC={lam:.3f})",
                                      xaxis_title="Attendu", yaxis_title="Observé")
                st.plotly_chart(fig_qq, use_container_width=True)

            st.subheader("Top 20 SNPs")
            st.dataframe(res.sort_values("P").head(20), use_container_width=True)

# ----- TAB 6 : VEP + GO/KEGG -----
with tabs[6]:
    st.subheader("🧬 Annotation VEP + GO/KEGG")
    st.markdown("""
    **VEP REST API** (Ensembl) + **g:Profiler** (GO) + **KEGG**.
    Nécessite une connexion Internet.
    """)

    st.warning("⚠️ Ces analyses utilisent les **API REST publiques**. "
               "Limitées à ~50 SNPs pour éviter le rate limit.")

    if st.session_state.gwas_results is not None and not st.session_state.gwas_results.empty:
        top = st.session_state.gwas_results.sort_values("P").head(20)
        st.write(f"**{len(top)} top SNPs** du GWAS à annoter")
        st.dataframe(top[["SNP", "CHR", "BP", "P"]], use_container_width=True)

        if st.button("▶ Annoter via VEP", type="primary"):
            with st.spinner("VEP API..."):
                vep_res = vep_annotate(top)
            st.session_state.vep_results = vep_res
            st.success(f"✅ {len(vep_res)} variants annotés")

        if st.session_state.vep_results is not None:
            st.dataframe(st.session_state.vep_results, use_container_width=True)

            # GO enrichment sur gènes
            genes = [g for g in st.session_state.vep_results["gene"].unique()
                     if g and g != "?"]
            if genes:
                st.write(f"**{len(genes)} gènes uniques** : {', '.join(genes[:10])}...")
                if st.button("▶ Enrichissement GO"):
                    with st.spinner("g:Profiler..."):
                        go = go_enrichment(genes)
                    st.session_state.go_enrichment = go
                    st.success(f"✅ {len(go)} termes GO")

                if st.session_state.go_enrichment is not None:
                    st.dataframe(st.session_state.go_enrichment.head(20),
                                  use_container_width=True)

                if st.button("▶ Enrichissement KEGG"):
                    with st.spinner("KEGG API..."):
                        kegg = kegg_enrichment(genes)
                    st.session_state.kegg_enrichment = kegg
                    st.success(f"✅ {len(kegg)} voies KEGG")

                if st.session_state.kegg_enrichment is not None:
                    st.dataframe(st.session_state.kegg_enrichment.head(20),
                                  use_container_width=True)
    else:
        st.info("Lancez d'abord un GWAS pour annoter les top SNPs.")

# ----- TAB 7 : Rapport IMRaD -----
with tabs[7]:
    st.subheader("📄 Rapport IMRaD + Tables S1-S10")
    st.markdown("""
    Rapport scientifique complet au format **IMRaD** :
    Abstract · Introduction · Materials & Methods · Results ·
    Discussion · Conclusion · References + **Tables S1-S10**.
    """)

    project_name = st.text_input("Nom du projet", "Cattle_Project_v71")

    if st.button("📄 Générer le rapport", type="primary"):
        with st.spinner("Génération IMRaD..."):
            html = build_imrad_html(st.session_state, {
                "project_name": project_name,
                "geno": geno, "mind": mind, "maf": maf_t, "hwe": hwe_t,
                "python_version": "3.11",
            })
            tables = build_supplementary_tables(st.session_state)
        st.session_state.imrad_html = html
        st.session_state.supp_tables = tables
        st.success(f"✅ Rapport + {len(tables)} tables S")

    if st.session_state.imrad_html:
        st.download_button("⬇ Rapport IMRaD (HTML)",
                            data=st.session_state.imrad_html.encode("utf-8"),
                            file_name=f"rapport_IMRaD_{datetime.now():%Y%m%d}.html",
                            mime="text/html",
                            use_container_width=True)

    if st.session_state.supp_tables:
        st.subheader("Tables supplémentaires")
        for name, df in st.session_state.supp_tables.items():
            with st.expander(f"📊 {name}"):
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=True).encode("utf-8")
                st.download_button(f"⬇ {name}.csv", csv,
                                    f"{name}.csv", "text/csv",
                                    key=f"dl_{name}")

st.divider()
st.caption(f"🐄 Bovine SNP Platform v7.1 — "
           f"{'✅ Modules complets' if V71_MODULES_OK else '⚠️ Modules manquants'}")
