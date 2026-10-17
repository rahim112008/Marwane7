"""
v71_report.py — Générateur rapport IMRaD + Tables S1-S10
FIXES v7.1.1 :
- Ajout import pandas as pd (manquant → NameError)
- Getter _safe_get robuste face aux valeurs None de st.session_state
- Gestion conditionnelle de toutes les métriques (GWAS, VEP, GO, KEGG)
- F-strings sans parenthèses imbriquées (évite confusion AST)
- build_supplementary_tables tolérant aux données manquantes
"""
import json
from datetime import datetime
from io import BytesIO
import pandas as pd


# ============================================================
# GETTER ROBUSTE
# ============================================================

def _safe_get(state, key, default=None):
    """
    Getter qui fonctionne pour dict ET st.session_state.
    Retourne `default` si la valeur est None ou la clé absente.

    Pourquoi ce helper ?
    -------------------
    dict.get(key, default) ne retourne `default` QUE si la clé est ABSENTE.
    Si la clé existe avec valeur None (cas de st.session_state dans v71_app.py),
    dict.get retourne None → provoque AttributeError sur .get().
    """
    # Tentative 1 : .get() si c'est un dict ou dict-like
    try:
        val = state.get(key, None)
        return default if val is None else val
    except (AttributeError, KeyError, TypeError):
        pass

    # Tentative 2 : accès par crochets (st.session_state supporte __getitem__)
    try:
        val = state[key]
        return default if val is None else val
    except (KeyError, TypeError, AttributeError):
        return default


# ============================================================
# RAPPORT IMRaD
# ============================================================

def build_imrad_html(state, config):
    """Génère un rapport IMRaD complet, tolérant aux données manquantes."""

    # --- Récupération SAFE de tous les états ---
    s        = _safe_get(state, "qc_stats", {})
    fst_data = _safe_get(state, "fst_result", {})
    boot     = _safe_get(state, "bootstrap_ci", {})
    taj      = _safe_get(state, "tajima", {})
    fuli     = _safe_get(state, "fuli", {})
    faywu    = _safe_get(state, "faywu", {})
    gwas     = _safe_get(state, "gwas_results", None)
    vep      = _safe_get(state, "vep_results", None)
    go       = _safe_get(state, "go_enrichment", None)
    kegg     = _safe_get(state, "kegg_enrichment", None)

    # --- Extraction SAFE des valeurs ---
    n_ind_init  = s.get("n_ind_init", "?")
    n_snp_init  = s.get("n_snp_init", "?")
    n_ind_final = s.get("n_ind_final", "?")
    n_snp_final = s.get("n_snp_final", "?")
    excl_ind    = s.get("excluded_ind", 0) or 0
    excl_snp    = s.get("excluded_snp", 0) or 0

    theta   = (fst_data.get("theta", 0) if fst_data else 0) or 0
    ci_low  = (boot.get("ci_low", 0)    if boot     else 0) or 0
    ci_high = (boot.get("ci_high", 0)   if boot     else 0) or 0
    n_boot  = (boot.get("n", 0)         if boot     else 0) or 0

    taj_D        = (taj.get("D", "NA") if taj else "NA") or "NA"
    taj_interp   = (taj.get("interpretation", "NA") if taj else "NA") or "NA"
    fuli_D       = (fuli.get("D_star", "NA") if fuli else "NA") or "NA"
    fuli_interp  = (fuli.get("interpretation", "NA") if fuli else "NA") or "NA"
    faywu_H      = (faywu.get("H", "NA") if faywu else "NA") or "NA"
    faywu_interp = (faywu.get("interpretation", "NA") if faywu else "NA") or "NA"

    # --- Résumé GWAS conditionnel ---
    gwas_info = "Non réalisé."
    if gwas is not None and hasattr(gwas, "empty") and not gwas.empty:
        try:
            n_tests = len(gwas)
            n_sig = int((gwas["P"] < 0.05 / n_tests).sum())
            gwas_info = (f"{n_tests} SNPs testés. "
                         f"{n_sig} significatifs Bonferroni.")
        except (KeyError, ZeroDivisionError, Exception):
            gwas_info = f"{len(gwas)} SNPs testés."

    # --- Résumé annotation (chaque bloc est indépendant) ---
    vep_info = ""
    if vep is not None and hasattr(vep, "empty") and not vep.empty:
        vep_info = f"<p>{len(vep)} variants annotés VEP.</p>"

    go_info = ""
    if go is not None and hasattr(go, "empty") and not go.empty:
        go_info = f"<p>{len(go)} termes GO enrichis.</p>"

    kegg_info = ""
    if kegg is not None and hasattr(kegg, "empty") and not kegg.empty:
        kegg_info = f"<p>{len(kegg)} voies KEGG enrichies.</p>"

    # --- Sections du rapport ---
    abstract = (
        "<h2>Abstract</h2>"
        f"<p>Nous avons analysé <b>{n_ind_init}</b> individus bovins "
        f"génotypés pour <b>{n_snp_init}</b> SNPs. Après QC "
        f"(<b>{n_ind_final}</b> individus, <b>{n_snp_final}</b> SNPs), "
        f"nous avons calculé FST (Weir &amp; Cockerham) = "
        f"<b>{theta:.4f}</b> (IC 95% : {ci_low:.4f}–{ci_high:.4f}), "
        f"Tajima's D = <b>{taj_D}</b>, Fu &amp; Li D* = <b>{fuli_D}</b>.</p>"
    )

    intro = (
        "<h2>1. Introduction</h2>"
        "<p>La caractérisation de la diversité génétique et des signatures "
        "de sélection chez les races bovines est essentielle pour la "
        "conservation et l'amélioration génétique. Cette étude utilise une "
        "approche intégrative combinant contrôle qualité rigoureux, "
        "analyses de structure des populations, détection de signatures "
        "de sélection et annotation fonctionnelle.</p>"
    )

    methods = (
        "<h2>2. Materials and Methods</h2>"
        "<h3>2.1 Échantillons et génotypage</h3>"
        f"<p>{n_ind_init} individus génotypés pour {n_snp_init} SNPs.</p>"
        "<h3>2.2 Contrôle qualité</h3>"
        f"<p>Missingness SNP &le; {config.get('geno', 0.05)}, "
        f"missingness individu &le; {config.get('mind', 0.05)}, "
        f"MAF &ge; {config.get('maf', 0.05)}, "
        f"HWE p &ge; {config.get('hwe', '1e-6')}, "
        f"hétérozygotie ±{config.get('het_sd', 3)}σ.</p>"
        "<h3>2.3 FST (Weir &amp; Cockerham 1984)</h3>"
        f"<p>θ estimé par la méthode des moments. "
        f"IC 95% par bootstrap (n={n_boot}).</p>"
        "<h3>2.4 Tests de neutralité</h3>"
        "<p>Tajima's D (1989), Fu &amp; Li D* (1993), Fay &amp; Wu H (2000).</p>"
        "<h3>2.5 GWAS</h3>"
        "<p>Modèle mixte (LMM, EMMA-style) avec matrice de parenté "
        "génomique et 10 PCs comme covariables. Seuils Bonferroni et FDR (BH).</p>"
        "<h3>2.6 Annotation fonctionnelle</h3>"
        "<p>VEP (Ensembl), enrichissement GO (g:Profiler) et KEGG.</p>"
    )

    results = (
        "<h2>3. Results</h2>"
        "<h3>3.1 Contrôle qualité</h3>"
        f"<p>{n_ind_final} individus ({excl_ind} exclus) et "
        f"{n_snp_final} SNPs ({excl_snp} exclus) retenus.</p>"
        "<h3>3.2 Diversité génétique</h3>"
        f"<p><b>FST (Weir &amp; Cockerham)</b> = {theta:.4f} "
        f"[IC 95% : {ci_low:.4f}–{ci_high:.4f}, bootstrap n={n_boot}]</p>"
        f"<p><b>Tajima's D</b> = {taj_D} ({taj_interp})</p>"
        f"<p><b>Fu &amp; Li D*</b> = {fuli_D} ({fuli_interp})</p>"
        f"<p><b>Fay &amp; Wu H</b> = {faywu_H} ({faywu_interp})</p>"
        "<h3>3.3 GWAS</h3>"
        f"<p>{gwas_info}</p>"
        "<h3>3.4 Annotation</h3>"
        f"{vep_info}{go_info}{kegg_info}"
    )

    discussion = (
        "<h2>4. Discussion</h2>"
        "<p>Les valeurs de FST modérées suggèrent une différentiation "
        "génétique entre les races, cohérente avec l'histoire de sélection. "
        "Les tests de neutralité indiquent des signatures de sélection "
        "ou de démographie. Les analyses GWAS et l'annotation fonctionnelle "
        "identifient des gènes candidats à valider expérimentalement.</p>"
    )

    conclusion = (
        "<h2>5. Conclusion</h2>"
        "<p>Cette étude fournit une caractérisation complète de la diversité "
        "génétique bovine avec rigueur statistique (IC bootstrap, corrections "
        "multi-tests, tests de neutralité). Les résultats sont prêts pour "
        "validation expérimentale et application en sélection.</p>"
    )

    refs = (
        "<h2>6. References</h2><ol>"
        "<li>Weir BS, Cockerham CC (1984). Estimating F-statistics for the "
        "analysis of population structure. <i>Evolution</i> 38:1358-1370.</li>"
        "<li>Tajima F (1989). Statistical method for testing the neutral "
        "mutation hypothesis by DNA polymorphism. <i>Genetics</i> 123:585-595.</li>"
        "<li>Fu YX, Li WH (1993). Statistical tests of neutrality of mutations. "
        "<i>Genetics</i> 133:693-709.</li>"
        "<li>Fay JC, Wu CI (2000). Hitchhiking under positive Darwinian "
        "selection. <i>Genetics</i> 155:1405-1413.</li>"
        "<li>Benjamini Y, Hochberg Y (1995). Controlling the false discovery "
        "rate. <i>J R Stat Soc B</i> 57:289-300.</li>"
        "</ol>"
    )

    # --- Assemblage final ---
    project = config.get("project_name", "Bovine")
    date_now = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = (
        "<!DOCTYPE html><html lang='fr'><head><meta charset='UTF-8'>"
        f"<title>Rapport IMRaD — {project}</title>"
        "<style>"
        "body{font-family:'Times New Roman',serif;max-width:900px;"
        "margin:40px auto;padding:20px;color:#222;line-height:1.6}"
        "h1{color:#1a5276;border-bottom:3px solid #1a5276;padding-bottom:10px}"
        "h2{color:#2471a3;margin-top:30px;border-bottom:1px solid #ccc}"
        "h3{color:#2e86c1}"
        "table{border-collapse:collapse;width:100%;margin:15px 0;font-size:0.9em}"
        "th,td{border:1px solid #ccc;padding:6px 10px;text-align:left}"
        "th{background:#eaf2f8}"
        ".meta{background:#f4f6f7;padding:15px;border-radius:5px;margin:20px 0}"
        "</style></head><body>"
        "<h1>🐄 Rapport Scientifique — Bovine SNP Platform v7.1</h1>"
        "<div class='meta'>"
        f"<p><b>Projet :</b> {project}</p>"
        f"<p><b>Date :</b> {date_now}</p>"
        "<p><b>Logiciel :</b> Bovine SNP Platform v7.1</p>"
        "</div>"
        f"{abstract}{intro}{methods}{results}{discussion}{conclusion}{refs}"
        "<hr><p style='font-size:0.85em;color:#666;margin-top:40px'>"
        "Rapport généré automatiquement par Bovine SNP Platform v7.1."
        "</p></body></html>"
    )
    return html


# ============================================================
# TABLES SUPPLÉMENTAIRES S1-S10
# ============================================================

def build_supplementary_tables(state):
    """Génère les tables S1-S10, tolérant aux données manquantes."""
    tables = {}

    # S1 : Individus
    ind_df = _safe_get(state, "ind_df", None)
    gt     = _safe_get(state, "gt", None)
    if (ind_df is not None
            and hasattr(ind_df, "copy")
            and gt is not None
            and hasattr(gt, "shape")):
        try:
            df = ind_df.copy()
            df["n_snp"] = gt.shape[1]
            tables["S1_Individus"] = df.head(200)
        except Exception:
            pass

    # S2 : QC cascade
    s = _safe_get(state, "qc_stats", {})
    if s.get("trace"):
        try:
            tables["S2_QC_cascade"] = pd.DataFrame(
                list(s["trace"].items()),
                columns=["Filtre", "Exclus"])
        except Exception:
            pass

    # S3 : FST pairwise
    fst_pair = _safe_get(state, "fst_pair_matrix", None)
    if (fst_pair is not None
            and isinstance(fst_pair, tuple)
            and len(fst_pair) == 2):
        try:
            M, pops = fst_pair
            tables["S3_FST_pairwise"] = (
                pd.DataFrame(M, index=pops, columns=pops).round(4))
        except Exception:
            pass

    # S4 : Bootstrap
    boot = _safe_get(state, "bootstrap_ci", {})
    if boot:
        try:
            tables["S4_Bootstrap_FST"] = pd.DataFrame([boot])
        except Exception:
            pass

    # S5 : Tests de neutralité
    rows = []
    for name, key in [("Tajima D", "tajima"),
                      ("Fu & Li D*", "fuli"),
                      ("Fay & Wu H", "faywu")]:
        d = _safe_get(state, key, {})
        if d:
            for k, v in d.items():
                rows.append({"Test": name, "Paramètre": k, "Valeur": v})
    if rows:
        try:
            tables["S5_Neutralite"] = pd.DataFrame(rows)
        except Exception:
            pass

    # S6 : GWAS top
    gwas = _safe_get(state, "gwas_results", None)
    if gwas is not None and hasattr(gwas, "empty") and not gwas.empty:
        try:
            tables["S6_GWAS_top"] = gwas.head(100)
        except Exception:
            pass

    # S7 : VEP
    vep = _safe_get(state, "vep_results", None)
    if vep is not None and hasattr(vep, "empty") and not vep.empty:
        try:
            tables["S7_VEP"] = vep.head(100)
        except Exception:
            pass

    # S8 : GO enrichment
    go = _safe_get(state, "go_enrichment", None)
    if go is not None and hasattr(go, "empty") and not go.empty:
        try:
            tables["S8_GO"] = go.head(50)
        except Exception:
            pass

    # S9 : KEGG enrichment
    kegg = _safe_get(state, "kegg_enrichment", None)
    if kegg is not None and hasattr(kegg, "empty") and not kegg.empty:
        try:
            tables["S9_KEGG"] = kegg.head(50)
        except Exception:
            pass

    # S10 : Session info
    try:
        tables["S10_Session"] = pd.DataFrame([
            {"Paramètre": "Date", "Valeur": datetime.now().isoformat()},
            {"Paramètre": "Version", "Valeur": "v7.1"},
            {"Paramètre": "Python",
             "Valeur": _safe_get(state, "python_version", "?")},
        ])
    except Exception:
        pass

    return tables
