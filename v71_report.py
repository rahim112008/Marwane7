"""
v71_report.py — Générateur rapport IMRaD + Tables S1-S10
"""
import json
from datetime import datetime
from io import BytesIO


def build_imrad_html(state, config):
    """Génère un rapport IMRaD complet."""
    s = state.get("qc_stats", {})
    fst_data = state.get("fst_result", {})
    boot = state.get("bootstrap_ci", {})
    taj = state.get("tajima", {})
    fuli = state.get("fuli", {})
    faywu = state.get("faywu", {})
    gwas = state.get("gwas_results")
    vep = state.get("vep_results")
    go = state.get("go_enrichment")
    kegg = state.get("kegg_enrichment")

    # ====== ABSTRACT ======
    abstract = f"""
    <h2>Abstract</h2>
    <p>Nous avons analysé <b>{s.get('n_ind_init', '?')}</b> individus
    bovins génotypés pour <b>{s.get('n_snp_init', '?')}</b> SNPs.
    Après QC (<b>{s.get('n_ind_final', '?')}</b> individus,
    <b>{s.get('n_snp_final', '?')}</b> SNPs), nous avons calculé
    FST (Weir & Cockerham) = <b>{fst_data.get('theta', 0):.4f}</b>
    (IC 95% : {boot.get('ci_low', 0):.4f}–{boot.get('ci_high', 0):.4f}),
    Tajima's D = <b>{taj.get('D', 'NA')}</b>,
    Fu & Li D* = <b>{fuli.get('D_star', 'NA')}</b>.
    {"GWAS LMM : " + str(len(gwas)) + " SNPs testés." if gwas is not None else ""}
    </p>
    """

    # ====== INTRODUCTION ======
    intro = """
    <h2>1. Introduction</h2>
    <p>La caractérisation de la diversité génétique et des signatures
    de sélection chez les races bovines est essentielle pour la
    conservation et l'amélioration génétique. Cette étude utilise
    une approche intégrative combinant contrôle qualité rigoureux,
    analyses de structure des populations, détection de signatures
    de sélection et annotation fonctionnelle.</p>
    """

    # ====== METHODS ======
    methods = f"""
    <h2>2. Materials and Methods</h2>
    <h3>2.1 Échantillons et génotypage</h3>
    <p>{s.get('n_ind_init', '?')} individus génotypés pour
    {s.get('n_snp_init', '?')} SNPs (puce bovine).</p>

    <h3>2.2 Contrôle qualité</h3>
    <p>Missingness SNP &le; {config.get('geno', 0.05)},
    missingness individu &le; {config.get('mind', 0.05)},
    MAF &ge; {config.get('maf', 0.05)},
    HWE p &ge; {config.get('hwe', '1e-6')},
    hétérozygotie &plusmn;{config.get('het_sd', 3)}&sigma;.</p>

    <h3>2.3 FST (Weir &amp; Cockerham 1984)</h3>
    <p>θ estimé par la méthode des moments. IC 95% par bootstrap
    (n=1000 réplications).</p>

    <h3>2.4 Tests de neutralité</h3>
    <p>Tajima's D (1989), Fu &amp; Li D* (1993), Fay &amp; Wu H (2000).</p>

    <h3>2.5 GWAS</h3>
    <p>Modèle mixte (LMM, EMMA-style) avec matrice de parenté génomique
    et 10 PCs comme covariables. Seuils Bonferroni et FDR (BH).</p>

    <h3>2.6 Annotation fonctionnelle</h3>
    <p>VEP (Ensembl), enrichissement GO (g:Profiler) et KEGG.</p>
    """

    # ====== RESULTS ======
    results = f"""
    <h2>3. Results</h2>
    <h3>3.1 Contrôle qualité</h3>
    <p>{s.get('n_ind_final', '?')} individus ({s.get('excluded_ind', 0)} exclus)
    et {s.get('n_snp_final', '?')} SNPs ({s.get('excluded_snp', 0)} exclus)
    retenus.</p>

    <h3>3.2 Diversité génétique</h3>
    <p><b>FST (Weir &amp; Cockerham)</b> = {fst_data.get('theta', 0):.4f}
    [IC 95% : {boot.get('ci_low', 0):.4f}–{boot.get('ci_high', 0):.4f},
    bootstrap n={boot.get('n', 0)}]</p>
    <p><b>Tajima's D</b> = {taj.get('D', 'NA')} ({taj.get('interpretation', 'NA')})</p>
    <p><b>Fu &amp; Li D*</b> = {fuli.get('D_star', 'NA')} ({fuli.get('interpretation', 'NA')})</p>
    <p><b>Fay &amp; Wu H</b> = {faywu.get('H', 'NA')} ({faywu.get('interpretation', 'NA')})</p>

    <h3>3.3 GWAS</h3>
    {"<p>" + str(len(gwas)) + " SNPs testés. " +
     (str(int((gwas['P'] < 0.05/len(gwas)).sum())) + " significatifs Bonferroni."
      if gwas is not None and len(gwas) > 0 else "Aucun résultat.") + "</p>"
     if gwas is not None else "<p>Non réalisé.</p>"}

    <h3>3.4 Annotation</h3>
    {"<p>" + str(len(vep)) + " variants annotés VEP.</p>" if vep is not None else ""}
    {"<p>" + str(len(go)) + " termes GO enrichis.</p>" if go is not None and len(go) > 0 else ""}
    {"<p>" + str(len(kegg)) + " voies KEGG enrichies.</p>" if kegg is not None and len(kegg) > 0 else ""}
    """

    # ====== DISCUSSION ======
    discussion = """
    <h2>4. Discussion</h2>
    <p>Les valeurs de FST modérées suggèrent une différentiation
    génétique entre les races, cohérente avec l'histoire de sélection.
    Les tests de neutralité indiquent des signatures de sélection
    ou de démographie. Les analyses GWAS et l'annotation fonctionnelle
    identifient des gènes candidats à valider.</p>
    """

    # ====== CONCLUSION ======
    conclusion = """
    <h2>5. Conclusion</h2>
    <p>Cette étude fournit une caractérisation complète de la diversité
    génétique bovine avec rigueur statistique (IC bootstrap, corrections
    multi-tests, tests de neutralité). Les résultats sont prêts pour
    validation expérimentale et application en sélection.</p>
    """

    # ====== REFERENCES ======
    refs = """
    <h2>6. References</h2>
    <ol>
    <li>Weir BS, Cockerham CC (1984). Estimating F-statistics for the
    analysis of population structure. <i>Evolution</i> 38:1358-1370.</li>
    <li>Tajima F (1989). Statistical method for testing the neutral
    mutation hypothesis by DNA polymorphism. <i>Genetics</i> 123:585-595.</li>
    <li>Fu YX, Li WH (1993). Statistical tests of neutrality of mutations.
    <i>Genetics</i> 133:693-709.</li>
    <li>Fay JC, Wu CI (2000). Hitchhiking under positive Darwinian
    selection. <i>Genetics</i> 155:1405-1413.</li>
    <li>Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate.
    <i>J R Stat Soc B</i> 57:289-300.</li>
    </ol>
    """

    html = f"""
    <!DOCTYPE html>
    <html lang="fr"><head><meta charset="UTF-8">
    <title>Rapport IMRaD — {config.get('project_name', 'Bovine')}</title>
    <style>
      body {{ font-family: 'Times New Roman', serif; max-width: 900px;
             margin: 40px auto; padding: 20px; color: #222;
             line-height: 1.6; }}
      h1 {{ color: #1a5276; border-bottom: 3px solid #1a5276;
           padding-bottom: 10px; }}
      h2 {{ color: #2471a3; margin-top: 30px; border-bottom: 1px solid #ccc; }}
      h3 {{ color: #2e86c1; }}
      table {{ border-collapse: collapse; width: 100%; margin: 15px 0;
              font-size: 0.9em; }}
      th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
      th {{ background: #eaf2f8; }}
      .meta {{ background: #f4f6f7; padding: 15px; border-radius: 5px;
              margin: 20px 0; }}
    </style>
    </head><body>

    <h1>🐄 Rapport Scientifique — Bovine SNP Platform v7.1</h1>

    <div class="meta">
    <p><b>Projet :</b> {config.get('project_name', 'Bovine')}</p>
    <p><b>Date :</b> {datetime.now():%Y-%m-%d %H:%M}</p>
    <p><b>Logiciel :</b> Bovine SNP Platform v7.1</p>
    <p><b>Version Python :</b> {config.get('python_version', '3.x')}</p>
    </div>

    {abstract}
    {intro}
    {methods}
    {results}
    {discussion}
    {conclusion}
    {refs}

    <hr>
    <p style="font-size: 0.85em; color: #666; margin-top: 40px;">
    Rapport généré automatiquement par Bovine SNP Platform v7.1.
    </p>
    </body></html>
    """
    return html


def build_supplementary_tables(state):
    """Génère les tables S1-S10 au format dict de DataFrames."""
    tables = {}

    # S1 : Individus
    if state.get("ind_df") is not None:
        df = state["ind_df"].copy()
        df["n_snp"] = state["gt"].shape[1]
        tables["S1_Individus"] = df.head(200)

    # S2 : QC cascade
    s = state.get("qc_stats", {})
    if s.get("trace"):
        tables["S2_QC_cascade"] = pd.DataFrame(
            list(s["trace"].items()), columns=["Filtre", "Exclus"])

    # S3 : FST pairwise
    if state.get("fst_pair_matrix") is not None:
        M, pops = state["fst_pair_matrix"]
        tables["S3_FST_pairwise"] = pd.DataFrame(
            M, index=pops, columns=pops).round(4)

    # S4 : Bootstrap
    boot = state.get("bootstrap_ci", {})
    if boot:
        tables["S4_Bootstrap_FST"] = pd.DataFrame([boot])

    # S5 : Tests neutralité
    taj = state.get("tajima", {})
    fuli = state.get("fuli", {})
    faywu = state.get("faywu", {})
    if taj or fuli or faywu:
        rows = []
        for name, d in [("Tajima D", taj), ("Fu & Li D*", fuli), ("Fay & Wu H", faywu)]:
            if d:
                for k, v in d.items():
                    rows.append({"Test": name, "Paramètre": k, "Valeur": v})
        tables["S5_Neutralite"] = pd.DataFrame(rows)

    # S6 : GWAS top
    gwas = state.get("gwas_results")
    if gwas is not None and not gwas.empty:
        tables["S6_GWAS_top"] = gwas.head(100)

    # S7 : VEP
    vep = state.get("vep_results")
    if vep is not None and not vep.empty:
        tables["S7_VEP"] = vep.head(100)

    # S8 : GO
    go = state.get("go_enrichment")
    if go is not None and not go.empty:
        tables["S8_GO"] = go.head(50)

    # S9 : KEGG
    kegg = state.get("kegg_enrichment")
    if kegg is not None and not kegg.empty:
        tables["S9_KEGG"] = kegg.head(50)

    # S10 : Session info
    tables["S10_Session"] = pd.DataFrame([
        {"Paramètre": "Date", "Valeur": datetime.now().isoformat()},
        {"Paramètre": "Version", "Valeur": "v7.1"},
        {"Paramètre": "Python", "Valeur": state.get("python_version", "?")},
    ])

    return tables


def tables_to_html(tables):
    """Convertit les tables en HTML."""
    html = ""
    for name, df in tables.items():
        html += f"<h3>Table {name}</h3>"
        html += df.to_html(index=True, max_rows=100, escape=False)
        html += "<br>"
    return html
