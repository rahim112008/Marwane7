# ==================== ÉTAT ====================
STATE_KEYS = ["gt", "ind_df", "snp_df", "gt_filt", "ind_filt", "snp_filt",
              "qc_stats", "gt_snp", "ind_snp", "snp_snp",
              "gt_pruned", "snp_pruned", "pca_scores", "pca_var",
              "mds_coords", "king_matrix", "ld_df", "roh_df", "froh",
              "admixture_Q", "admixture_K", "cv_results",
              "reynolds_D", "reynolds_pops", "fst_global",
              "binome_result", "all_binomes", "group_results",
              "group_pair", "gwas_results_10cm", "genes_df",
              # v7.1
              "fst_result", "bootstrap_ci", "ne_ci",
              "tajima", "fuli", "faywu", "ihs_results", "xpehh_results",
              "gwas_results_lmm", "gwas_qq", "gwas_lambda",
              "vep_results", "go_enrichment", "kegg_enrichment",
              "imrad_html", "supp_tables",
              "_ax_ped", "_ax_map", "_ax_n", "_ax_samples"]

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
              "group_results", "group_pair", "gwas_results_10cm"]:
        st.session_state[k] = None

def has_data(): return st.session_state.gt is not None
def has_qc():
    return (st.session_state.gt_filt is not None
            and st.session_state.qc_stats is not None)
def has_snp(): return st.session_state.gt_snp is not None
def has_pruned():
    return (st.session_state.gt_pruned is not None
            and st.session_state.snp_pruned is not None)

# ==================== MAIN ====================
def main():
    init_state()
    st.title("🐄 Bovine SNP Platform v7.2 — Best of Both")
    st.caption("v6.2 (toutes fonctionnalités) + v7.1 (rigueur scientifique)")

    # ----- SIDEBAR -----
    with st.sidebar:
        st.header("📁 Données")
        mode = st.radio("Source :", ["Demo", "PED/MAP", "VCF",
                                     "Convertisseur Axiom"],
                        index=0, key="sb_mode")
        if mode == "Demo":
            c1, c2 = st.columns(2)
            ni = c1.number_input("Ind.", 20, 1000, 150, 10, key="sb_ni")
            ns = c2.number_input("SNPs", 100, 10000, 800, 100, key="sb_ns")
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
        if st.button("🚀 Pipeline complet", type="primary",
                     use_container_width=True, key="sb_pipe"):
            st.session_state["_run_pipe"] = True
        st.caption(f"v7.2 — Modules v7.1 {'✅' if V71_OK else '⚠️'}")

    # ----- CONVERTISSEUR AXIOM -----
    if mode == "Convertisseur Axiom":
        render_axiom_page()
        return

    # ----- ACCUEIL -----
    if not has_data():
        st.info("👉 Générez un jeu de démo, importez PED/MAP/VCF, "
                "ou utilisez le Convertisseur Axiom.")
        st.markdown("""
        ### 🧬 v7.2 — 16 onglets (fusion v6.2 + v7.1)

        | Onglet | Source | Contenu |
        |--------|--------|---------|
        | 🏠 Aperçu | v6.2 | Statistiques |
        | 🧹 QC | v6.2 | Cascade complète |
        | ✂️ LD Pruning | v6.2 | Indep-pairwise |
        | 🧬 Structure | v6.2 | PCA + MDS + KING |
        | 🎨 Admixture | v6.2 | NMF + CV |
        | 📈 Démographie | v6.2 | LD decay + ROH + Ne |
        | 🔍 Sélection | v6.2 | FST Manhattan |
        | 🌳 Phylogénie | v6.2 | Reynolds + NJ |
        | 📤 Export | v6.2 | PLINK |
        | 🧬 Binômes | v6.2 | 5 modes + signatures |
        | 🧪 SNP & Gènes | v6.2 | GWAS 10cM + GTF |
        | 🔬 **FST Weir-Cockerham** | **v7.1** | IC bootstrap |
        | 📊 **Neutralité** | **v7.1** | Tajima + Fu&Li + Fay&Wu |
        | 🧪 **GWAS LMM** | **v7.1** | QQ + λGC |
        | 🧬 **VEP + GO/KEGG** | **v7.1** | REST API |
        | 📄 **Rapport IMRaD** | **v7.1** | S1-S10 |
        """)
        return

    gt, ind_df, snp_df = (st.session_state.gt, st.session_state.ind_df,
                           st.session_state.snp_df)

    tabs = st.tabs([
        "🏠 Aperçu", "🧹 QC", "✂️ LD Pruning", "🧬 Structure",
        "🎨 Admixture", "📈 Démographie", "🔍 Sélection",
        "🌳 Phylogénie", "📤 Export", "🧬 Binômes",
        "🧪 SNP & Gènes",
        "🔬 **FST Weir-Cockerham**", "📊 **Neutralité**",
        "🧪 **GWAS LMM**", "🧬 **VEP + GO/KEGG**",
        "📄 **Rapport IMRaD**"])

    # ==== TAB 0 : Aperçu ====
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        c1.metric("Individus", gt.shape[0])
        c2.metric("SNPs", gt.shape[1])
        c3.metric("Populations", ind_df["FID"].nunique())
        st.dataframe(ind_df.head(20), use_container_width=True)

    # ==== TAB 1 : QC ====
    with tabs[1]:
        st.subheader("🧹 QC")
        if st.button("🧬 Filtrer autosomes",
                     use_container_width=True, key="qc_a"):
            g2, s2 = filter_auto(gt, snp_df)
            st.session_state.gt = g2
            st.session_state.snp_df = s2
            st.success(f"✅ {g2.shape[1]} SNPs")
        if st.button("▶ QC complet", type="primary",
                     use_container_width=True, key="qc_r"):
            try:
                with st.spinner("QC..."):
                    p = {"geno": geno, "mind": mind, "maf": maf_t,
                         "hwe": hwe_t, "het_sd": het_sd}
                    r = qc_filters(st.session_state.gt,
                                    st.session_state.ind_df,
                                    st.session_state.snp_df, p)
                    for k, v in zip(["gt_filt", "ind_filt", "snp_filt",
                                      "qc_stats", "gt_snp", "ind_snp",
                                      "snp_snp"], r):
                        st.session_state[k] = v
                st.success(f"✅ gt_filt : {r[0].shape[0]}×{r[0].shape[1]}")
            except Exception as e:
                st.error(f"❌ {e}")
        if has_qc():
            s = st.session_state.qc_stats
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ind.", s["n_ind_final"])
            c2.metric("SNPs", s["n_snp_final"])
            c3.metric("Exclus ind.", s["excluded_ind"])
            c4.metric("Exclus SNPs", s["excluded_snp"])

    # ==== TAB 2 : LD Pruning ====
    with tabs[2]:
        st.subheader("✂️ LD Pruning")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        else:
            if st.button("▶ Pruning", type="primary",
                         use_container_width=True, key="ld_btn"):
                with st.spinner("..."):
                    k = ld_pruning(st.session_state.gt_filt, 50, 5, ld_r2)
                    st.session_state.gt_pruned = st.session_state.gt_filt[:, k]
                    st.session_state.snp_pruned = (
                        st.session_state.snp_filt[k].reset_index(drop=True))
                st.success(f"✅ {st.session_state.gt_pruned.shape[1]} SNPs")

    # ==== TAB 3 : Structure ====
    with tabs[3]:
        st.subheader("🧬 Structure")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        else:
            gu = (st.session_state.gt_pruned if has_pruned()
                  else st.session_state.gt_filt)
            if st.button("▶ PCA + MDS + KING", type="primary",
                         use_container_width=True, key="str_btn"):
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

    # ==== TAB 4 : Admixture ====
    with tabs[4]:
        st.subheader("🎨 Admixture")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        else:
            gu = (st.session_state.gt_pruned if has_pruned()
                  else st.session_state.gt_filt)
            c1, c2 = st.columns(2)
            Km = c1.slider("K max", 3, 10, 6, key="ad_km")
            nr = c2.slider("CV reps", 1, 5, 3, key="ad_nr")
            if st.button("▶ CV", use_container_width=True, key="ad_cv"):
                with st.spinner("CV..."):
                    st.session_state.cv_results = admix_cv(gu, (2, Km+1), int(nr))
                st.success("✅")
            if st.session_state.cv_results is not None:
                ks = sorted(st.session_state.cv_results.keys())
                means = [st.session_state.cv_results[k]["mean"] for k in ks]
                fig = go.Figure(go.Scatter(x=ks, y=means, mode="lines+markers",
                                            line=dict(color="crimson", width=3),
                                            marker=dict(size=12)))
                fig.update_layout(title="CV Error", height=400)
                st.plotly_chart(fig, use_container_width=True)
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
                fig.update_layout(barmode="stack", height=450,
                                  title=f"Admixture K={K}")
                st.plotly_chart(fig, use_container_width=True)

    # ==== TAB 5 : Démographie ====
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
                    fig = px.line(agg, x="bin", y="r2")
                    st.plotly_chart(fig, use_container_width=True)
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
                    st.plotly_chart(px.histogram(df, x="FROH", color="Pop",
                                                  nbins=50),
                                    use_container_width=True)

    # ==== TAB 6 : Sélection (FST simplifié) ====
    with tabs[6]:
        st.subheader("🔍 Sélection (FST par SNP)")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        else:
            gu = (st.session_state.gt_snp if has_snp()
                  else st.session_state.gt_filt)
            su = (st.session_state.snp_snp if has_snp()
                  else st.session_state.snp_filt)
            iu = (st.session_state.ind_snp if has_snp()
                  else st.session_state.ind_filt)
            if st.button("▶ FST", use_container_width=True, key="fst_g"):
                with st.spinner("FST..."):
                    st.session_state.fst_global = fst_per_snp(
                        gu, iu["FID"].values)
                st.success("✅")
            if st.session_state.fst_global is not None:
                fig = plt_man(st.session_state.fst_global, su["CHR"].values)
                if fig: st.plotly_chart(fig, use_container_width=True)

    # ==== TAB 7 : Phylogénie ====
    with tabs[7]:
        st.subheader("🌳 Phylogénie (Reynolds + NJ)")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        else:
            gu = (st.session_state.gt_pruned if has_pruned()
                  else st.session_state.gt_filt)
            iu = st.session_state.ind_filt
            if st.button("▶ Reynolds", use_container_width=True, key="rey"):
                with st.spinner("..."):
                    lb = iu["FID"].astype(str).values
                    pops = sorted(np.unique(lb))
                    ph = {p: np.nanmean(gu[lb == p], 0) / 2 for p in pops}
                    K = len(pops)
                    D = np.zeros((K, K))
                    for i in range(K):
                        for j in range(i+1, K):
                            pi, pj = ph[pops[i]], ph[pops[j]]
                            num = np.nansum((pi-pj)**2)
                            den = np.nansum(pi*(1-pj) + pj*(1-pi))
                            if den > 0: D[i, j] = D[j, i] = num/den
                    st.session_state.reynolds_D = D
                    st.session_state.reynolds_pops = pops
                st.success("✅")
            if st.session_state.reynolds_D is not None:
                D = st.session_state.reynolds_D
                pops = st.session_state.reynolds_pops
                fig = go.Figure(go.Heatmap(z=D, x=pops, y=pops,
                                            colorscale="Blues",
                                            text=np.round(D, 3),
                                            texttemplate="%{text}"))
                fig.update_layout(title="Distance de Reynolds", height=600)
                st.plotly_chart(fig, use_container_width=True)

    # ==== TAB 8 : Export ====
    with tabs[8]:
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
            if st.button("📦 Générer", use_container_width=True, key="ex_g"):
                try:
                    with st.spinner("..."):
                        data = build_zip(ge, ie, se, pre)
                    st.download_button("⬇ Télécharger", data, f"{pre}.zip",
                                        "application/zip", key="ex_dl")
                except Exception as e:
                    st.error(f"❌ {e}")

    # ==== TAB 9 : Binômes ====
    with tabs[9]:
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

            mode_b = st.radio(
                "Mode :",
                ["🧠 Auto-MDS", "🖱️ Manuelle", "🌐 Toutes paires",
                 "🎯 Signature par population",
                 "🐄 Signature par groupe", "⚔️ Groupe A vs B"],
                key="bin_mode")

            c1, c2 = st.columns(2)
            q_b = c1.slider("Quantile", 0.90, 0.9999, 0.99, 0.001,
                             format="%.4f", key="bin_q")
            fst_b = c2.slider("FST min", 0.0, 0.5, 0.15, 0.01, key="bin_mf")

            if mode_b == "🧠 Auto-MDS":
                if st.session_state.mds_coords is None:
                    st.warning("⚠️ Lancez Structure.")
                else:
                    pairs = auto_pairs_mds(st.session_state.mds_coords, lb)
                    if not pairs.empty:
                        st.dataframe(pairs.head(20), use_container_width=True)

            elif mode_b == "🖱️ Manuelle":
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

            elif mode_b == "🌐 Toutes paires":
                import itertools
                pairs = list(itertools.combinations(avail, 2))
                st.info(f"{len(pairs)} paires")
                if st.button(f"▶ Analyser {len(pairs)}",
                             type="primary",
                             use_container_width=True, key="all_run"):
                    res = {}
                    prog = st.progress(0)
                    for i, (a, b) in enumerate(pairs):
                        res[f"{a}_vs_{b}"] = analyze_pair(
                            gu, su, lb, a, b, q_b, fst_b)
                        prog.progress((i+1)/len(pairs))
                    st.session_state.all_binomes = res
                    st.success(f"✅ {len(res)}")
                if st.session_state.all_binomes:
                    summ = [{"Paire": k.replace("_vs_", " ↔ "),
                             "Outliers": v.get("n_snp_outliers", 0)}
                            for k, v in st.session_state.all_binomes.items()
                            if v.get("status") == "success"]
                    if summ:
                        st.dataframe(pd.DataFrame(summ), use_container_width=True)

            elif mode_b == "🎯 Signature par population":
                if st.button("▶ Toutes les populations", type="primary",
                             use_container_width=True, key="pop_rest"):
                    res = analyze_pop_vs_rest(gu, su, lb, q_b, fst_b)
                    st.session_state.group_results = res
                    st.success(f"✅ {len(res)}")
                if st.session_state.group_results:
                    summ = [{"Pop": k, "Outliers": v["n_snp_outliers"],
                             "FST": round(v["fst_mean"], 4)}
                            for k, v in st.session_state.group_results.items()
                            if v.get("status") == "success"]
                    if summ:
                        st.dataframe(pd.DataFrame(summ), use_container_width=True)

            elif mode_b == "🐄 Signature par groupe":
                groups = {k: v for k, v in
                          st.session_state["trait_groups"].items() if v}
                if not groups:
                    st.warning("⚠️ Définissez les groupes.")
                else:
                    if st.button("▶ Analyser tous les groupes",
                                 type="primary",
                                 use_container_width=True, key="grp_all"):
                        res = analyze_grp_vs_rest(gu, su, lb, groups, q_b, fst_b)
                        st.session_state.group_results = res
                        st.success(f"✅ {len(res)}")

            else:  # Groupe A vs B
                groups = {k: v for k, v in
                          st.session_state["trait_groups"].items() if v}
                if len(groups) < 2:
                    st.warning("⚠️ Définissez 2 groupes.")
                else:
                    c1, c2 = st.columns(2)
                    gA = c1.selectbox("Groupe A", list(groups.keys()), key="gA")
                    gB = c2.selectbox("Groupe B",
                                       [g for g in groups if g != gA], key="gB")
                    if st.button(f"▶ {gA} vs {gB}", type="primary", key="pair_run"):
                        with st.spinner("..."):
                            st.session_state.group_pair = analyze_grp_pair(
                                gu, su, lb, groups, gA, gB, q_b, fst_b)
                        st.success("✅")

    # ==== TAB 10 : SNP & Gènes (v6.2) ====
    with tabs[10]:
        st.subheader("🧪 SNP & Gènes (GWAS 10 cM + GTF)")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        else:
            gu = (st.session_state.gt_snp if has_snp()
                  else st.session_state.gt_filt)
            su = (st.session_state.snp_snp if has_snp()
                  else st.session_state.snp_filt)
            iu = (st.session_state.ind_snp if has_snp()
                  else st.session_state.ind_filt)

            s1, s2 = st.tabs(["🧬 Annotation GTF", "📊 GWAS 10 cM"])
            with s1:
                st.markdown("**GTF bovin :** [Ensembl](https://ftp.ensembl.org/"
                            "pub/release-110/gtf/bos_taurus/"
                            "Bos_taurus.ARS-UCD1.2.110.gtf.gz)")
                gf = st.file_uploader("GTF", type=["gtf", "gz"], key="gtf_f")
                if gf and st.button("📥 Charger GTF", key="gtf_l"):
                    try:
                        with st.spinner("..."):
                            tmp = "/tmp/ann.gtf"
                            if gf.name.endswith(".gz"): tmp += ".gz"
                            with open(tmp, "wb") as f:
                                f.write(gf.getbuffer())
                            st.session_state.genes_df = load_gtf(tmp)
                        st.success(f"✅ {len(st.session_state.genes_df):,} gènes")
                    except Exception as e:
                        st.error(f"❌ {e}")
                if st.session_state.genes_df is not None:
                    st.info(f"Gènes : {len(st.session_state.genes_df):,}")
                    res = st.session_state.group_results
                    if res:
                        sel = st.selectbox("Groupe :", list(res.keys()), key="ann_g")
                        v = res[sel]
                        if v.get("status") == "success":
                            if st.button("📋 Annoter", type="primary", key="ann_btn"):
                                with st.spinner("..."):
                                    ann = annotate_snps(v["outliers_df"],
                                                         st.session_state.genes_df, 500)
                                    st.session_state.group_results[sel]["annotated_df"] = ann
                                st.success(f"✅ {len(ann)} SNPs")

            with s2:
                groups = {k: v for k, v in
                          st.session_state["trait_groups"].items() if v}
                src = st.radio("Phénotype :", ["Groupe de races", "Fichier CSV"],
                                horizontal=True, key="gw_src")
                ph_col, ph = None, None
                if src == "Groupe de races":
                    if not groups:
                        st.warning("⚠️ Définissez les groupes.")
                    else:
                        gn = st.selectbox("Groupe :", list(groups.keys()), key="gw_g")
                        ph = (grp_phenotype(iu, groups, gn)
                              .rename(columns={"pheno": "pheno_group"})[["IID", "pheno_group"]])
                        ph_col = "pheno_group"
                else:
                    pf = st.file_uploader("CSV (IID,col)", type=["csv"], key="gw_csv")
                    if pf:
                        ph = pd.read_csv(pf)
                        ph["IID"] = ph["IID"].astype(str)
                        nc = [c for c in ph.columns if c != "IID"
                              and pd.api.types.is_numeric_dtype(ph[c])]
                        ph_col = st.selectbox("Phénotype :", nc, key="gw_c")

                res = st.session_state.group_results
                if ph is not None and res:
                    avail = [k for k, v in res.items() if v.get("status") == "success"]
                    if avail:
                        sel_g = st.selectbox("Groupe :", avail, key="gw_sg")
                        cands = res[sel_g]["outliers_df"]
                        if not cands.empty:
                            top_n = st.slider("Top N", 1, 20, 5, key="gw_tn")
                            top = cands.head(top_n)
                            st.dataframe(top[["CHR", "SNP", "BP", "FST"]],
                                          use_container_width=True)
                            win = st.slider("Fenêtre (cM)", 1, 20, 10, key="gw_w")
                            if st.button("▶ GWAS 10 cM", type="primary",
                                         use_container_width=True, key="gw_run"):
                                all_g = {}
                                prog = st.progress(0)
                                for i, (_, s) in enumerate(top.iterrows()):
                                    with st.spinner(f"GWAS {s['SNP']}..."):
                                        r = gwas_local(gu, su, iu, ph, s["CHR"],
                                                        s["BP"], ph_col, win)
                                    if not r.empty:
                                        all_g[s["SNP"]] = r
                                    prog.progress((i+1)/len(top))
                                st.session_state.gwas_results_10cm = all_g
                                st.success(f"✅ {len(all_g)} GWAS")

                gres = st.session_state.gwas_results_10cm
                if gres:
                    sel = st.selectbox("Voir :", list(gres.keys()), key="gw_v")
                    df = gres[sel]
                    st.dataframe(df.head(50), use_container_width=True)
                    snp_row = su[su["SNP"] == sel]
                    if not snp_row.empty:
                        fig = plt_gwas(df, snp_row.iloc[0]["CHR"],
                                        int(snp_row.iloc[0]["BP"]))
                        if fig: st.plotly_chart(fig, use_container_width=True)

    # ==== TAB 11 : FST Weir-Cockerham (v7.1) ====
    with tabs[11]:
        st.subheader("🔬 FST Weir & Cockerham (1984) + Bootstrap IC")
        st.markdown("**Standard peer-review** avec IC 95% (bootstrap 1000×).")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        elif not V71_OK:
            st.error("⚠️ Module `v71_statistics.py` manquant.")
        else:
            if st.button("▶ Calculer FST + Bootstrap", type="primary",
                         use_container_width=True, key="wc_btn"):
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
                fd = st.session_state.fst_result
                boot = st.session_state.bootstrap_ci
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("FST (θ)", f"{fd['theta']:.4f}")
                c2.metric("SE (jackknife)", f"{fd['se']:.4f}")
                c3.metric("IC 95% bas", f"{boot['ci_low']:.4f}")
                c4.metric("IC 95% haut", f"{boot['ci_high']:.4f}")
                st.markdown(f"**Résultat :** FST = **{boot['mean']:.4f}** "
                            f"[IC 95% : {boot['ci_low']:.4f} – {boot['ci_high']:.4f}] "
                            f"(bootstrap n={boot['n']})")

    # ==== TAB 12 : Neutralité (v7.1) ====
    with tabs[12]:
        st.subheader("📊 Tests de neutralité")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        elif not V71_OK:
            st.error("⚠️ Module `v71_neutrality.py` manquant.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("▶ Tajima's D", use_container_width=True, key="taj_btn"):
                    with st.spinner("..."):
                        st.session_state.tajima = tajimas_d(st.session_state.gt_snp)
                    st.success("✅")
                if st.session_state.tajima:
                    r = st.session_state.tajima
                    st.metric("Tajima D", r.get("D", "NA"))
                    st.json(r)
            with c2:
                if st.button("▶ Fu & Li D*", use_container_width=True, key="fuli_btn"):
                    with st.spinner("..."):
                        st.session_state.fuli = fu_li_d(st.session_state.gt_snp)
                    st.success("✅")
                if st.session_state.fuli:
                    r = st.session_state.fuli
                    st.metric("D*", r.get("D_star", "NA"))
                    st.json(r)
            if st.button("▶ Fay & Wu H", use_container_width=True, key="faywu_btn"):
                with st.spinner("..."):
                    st.session_state.faywu = fay_wu_h(st.session_state.gt_snp)
                st.success("✅")
            if st.session_state.faywu:
                r = st.session_state.faywu
                st.metric("Fay & Wu H", r.get("H", "NA"))
                st.json(r)

    # ==== TAB 13 : GWAS LMM (v7.1) ====
    with tabs[13]:
        st.subheader("🧪 GWAS LMM (EMMA) + QQ + λGC")
        st.markdown("**Modèle mixte** avec matrice de parenté et PCs.")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        elif not V71_OK:
            st.error("⚠️ Module `v71_gwas.py` manquant.")
        else:
            st.info("Phénotype : simulé aléatoirement pour la démo.")
            if st.button("▶ Lancer GWAS LMM", type="primary",
                         use_container_width=True, key="lmm_btn"):
                with st.spinner("LMM (30s)..."):
                    gt = st.session_state.gt_snp
                    rng = np.random.default_rng(42)
                    pheno = rng.normal(0, 1, gt.shape[0])
                    n = gt.shape[0]
                    h = np.nansum(gt == 1, axis=1).astype(float)
                    K = np.zeros((n, n))
                    for i in range(n):
                        for j in range(i+1, n):
                            bh = np.nansum((gt[i] == 1) & (gt[j] == 1))
                            d = h[i] + h[j]
                            K[i, j] = K[j, i] = (d-2*bh)/d if d > 0 else 0
                    np.fill_diagonal(K, 0.5)
                    cov = st.session_state.pca_scores[:, :5] if st.session_state.pca_scores is not None else None
                    res = gwas_lmm(gt, pheno, K, cov, maf_min=maf_t, miss_max=mind)
                if not res.empty:
                    res["SNP"] = st.session_state.snp_snp["SNP"].values[res["SNP_idx"].values]
                    res["CHR"] = st.session_state.snp_snp["CHR"].values[res["SNP_idx"].values]
                    res["BP"] = st.session_state.snp_snp["BP"].values[res["SNP_idx"].values]
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
                st.dataframe(res.sort_values("P").head(20), use_container_width=True)

    # ==== TAB 14 : VEP + GO/KEGG (v7.1) ====
    with tabs[14]:
        st.subheader("🧬 VEP + GO + KEGG")
        st.markdown("**API REST** (Ensembl + g:Profiler + KEGG).")
        if not V71_OK:
            st.error("⚠️ Module `v71_annotation.py` manquant.")
        elif st.session_state.gwas_results_lmm is None:
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
                st.dataframe(st.session_state.vep_results, use_container_width=True)
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

    # ==== TAB 15 : Rapport IMRaD (v7.1) ====
    with tabs[15]:
        st.subheader("📄 Rapport IMRaD + S1-S10")
        if not V71_OK:
            st.error("⚠️ Module `v71_report.py` manquant.")
        else:
            proj = st.text_input("Nom du projet", "Cattle_Project_v72", key="rep_p")
            if st.button("📄 Générer le rapport", type="primary", key="rep_btn"):
                with st.spinner("Génération IMRaD..."):
                    # Construit le state pour v71_report
                    report_state = {k: st.session_state.get(k) for k in STATE_KEYS}
                    report_state["qc_stats"] = st.session_state.qc_stats or {}
                    report_state["fst_result"] = st.session_state.fst_result or {}
                    report_state["bootstrap_ci"] = st.session_state.bootstrap_ci or {}
                    report_state["tajima"] = st.session_state.tajima or {}
                    report_state["fuli"] = st.session_state.fuli or {}
                    report_state["faywu"] = st.session_state.faywu or {}
                    report_state["gwas_results"] = st.session_state.gwas_results_lmm
                    report_state["vep_results"] = st.session_state.vep_results
                    report_state["go_enrichment"] = st.session_state.go_enrichment
                    report_state["kegg_enrichment"] = st.session_state.kegg_enrichment

                    html = build_imrad_html(report_state, {
                        "project_name": proj,
                        "geno": geno, "mind": mind, "maf": maf_t, "hwe": hwe_t,
                    })
                    tables = build_supplementary_tables(report_state)
                st.session_state.imrad_html = html
                st.session_state.supp_tables = tables
                st.success(f"✅ Rapport + {len(tables)} tables")
            if st.session_state.imrad_html:
                st.download_button("⬇ Rapport IMRaD (HTML)",
                                    data=st.session_state.imrad_html.encode("utf-8"),
                                    file_name=f"rapport_IMRaD_{datetime.now():%Y%m%d}.html",
                                    mime="text/html",
                                    use_container_width=True)
            if st.session_state.supp_tables:
                for name, df in st.session_state.supp_tables.items():
                    with st.expander(f"📊 {name}"):
                        st.dataframe(df, use_container_width=True)

    # ----- PIPELINE COMPLET -----
    if st.session_state.get("_run_pipe"):
        st.session_state["_run_pipe"] = False
        try:
            with st.spinner("🚀 Pipeline..."):
                g2, s2 = filter_auto(st.session_state.gt,
                                      st.session_state.snp_df)
                st.session_state.gt = g2
                st.session_state.snp_df = s2
                p = {"geno": geno, "mind": mind, "maf": maf_t,
                     "hwe": hwe_t, "het_sd": het_sd}
                r = qc_filters(g2, st.session_state.ind_df, s2, p)
                for k, v in zip(["gt_filt", "ind_filt", "snp_filt",
                                  "qc_stats", "gt_snp", "ind_snp",
                                  "snp_snp"], r):
                    st.session_state[k] = v
                k = ld_pruning(st.session_state.gt_filt, 50, 5, ld_r2)
                st.session_state.gt_pruned = st.session_state.gt_filt[:, k]
                st.session_state.snp_pruned = (
                    st.session_state.snp_filt[k].reset_index(drop=True))
                gu = st.session_state.gt_pruned
                sc, vp = pca_calc(gu, 10)
                st.session_state.pca_scores = sc
                st.session_state.pca_var = vp
                st.session_state.mds_coords = mds_calc(gu, 5)
            st.success("✅ Pipeline terminé !")
        except Exception as e:
            st.error(f"❌ {e}")

if __name__ == "__main__":
    main()
