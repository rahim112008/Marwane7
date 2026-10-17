"""
v71_annotation.py — VEP REST API + GO enrichment + KEGG
"""
import numpy as np
import pandas as pd
import requests
import time
from scipy import stats


# ============ VEP REST API ============

def vep_annotate(snps_df, species="bos_taurus", max_retries=3):
    """
    Annote via Ensembl VEP REST API.
    snps_df doit avoir colonnes CHR, BP, REF, ALT (ou VCF).
    """
    results = []
    for _, snp in snps_df.iterrows():
        chrom = str(snp["CHR"]).replace("chr", "")
        pos = int(snp["BP"])
        ref = snp.get("A1", "A")
        alt = snp.get("A2", "G")
        if ref == alt or ref in ("", "nan"):
            ref, alt = "A", "G"

        url = f"https://rest.ensembl.org/vep/{species}/region"
        payload = {
            "variants": [f"{chrom} {pos} . {ref} {alt} . . ."],
        }
        try:
            r = requests.post(url, json=payload,
                              headers={"Content-Type": "application/json",
                                       "Accept": "application/json"},
                              timeout=30)
            if r.status_code == 200:
                data = r.json()
                if data:
                    t = data[0].get("transcript_consequences", [{}])
                    if t:
                        info = t[0]
                        results.append({
                            "CHR": chrom, "BP": pos, "REF": ref, "ALT": alt,
                            "gene": info.get("gene_symbol", "?"),
                            "transcript": info.get("transcript_id", "?"),
                            "consequence": ", ".join(info.get("consequence_terms", [])),
                            "impact": info.get("impact", "?"),
                            "sift": info.get("sift_prediction", "?"),
                            "polyphen": info.get("polyphen_prediction", "?"),
                            "cadd": info.get("cadd_raw", "?"),
                        })
        except Exception:
            continue
        time.sleep(0.1)  # Rate limit
    return pd.DataFrame(results)


# ============ GO ENRICHMENT ============

def go_enrichment(genes, background=None, organism="bos_taurus",
                   source="go"):
    """
    Test d'enrichissement GO par hypergeométrie.
    Utilise g:Profiler REST API.
    """
    if not genes:
        return pd.DataFrame()
    url = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
    payload = {
        "organism": organism,
        "query": list(genes),
        "sources": [source] if source else ["GO:BP", "GO:MF", "GO:CC"],
        "user_threshold": 0.05,
        "significance_method": "fdr",
    }
    try:
        r = requests.post(url, json=payload, timeout=60)
        if r.status_code == 200:
            data = r.json()
            results = data.get("result", [])
            df = pd.DataFrame([{
                "term_id": t.get("native", "?"),
                "term_name": t.get("name", "?"),
                "source": t.get("source", "?"),
                "p_value": t.get("p_value", 1),
                "padj": t.get("p_values", {}).get("false_discovery_rate", 1),
                "intersection_size": t.get("intersection_size", 0),
                "term_size": t.get("term_size", 0),
                "query_size": t.get("query_size", 0),
                "genes": ",".join(t.get("intersections", [])),
            } for t in results])
            if not df.empty:
                df = df.sort_values("p_value").reset_index(drop=True)
            return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})
    return pd.DataFrame()


# ============ KEGG ENRICHMENT ============

def kegg_enrichment(genes, organism="bta"):
    """KEGG pathway enrichment via KEGG REST API + hypergeométrie."""
    if not genes:
        return pd.DataFrame()
    # Query KEGG for pathways linked to each gene
    try:
        gene_links = {}
        for g in genes:
            url = f"https://rest.kegg.jp/link/{organism}/{organism}:{g}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200 and r.text:
                for line in r.text.strip().split("\n"):
                    parts = line.split("\t")
                    if len(parts) == 2:
                        pathway = parts[1]
                        gene_links.setdefault(pathway, []).append(g)
            time.sleep(0.05)

        if not gene_links:
            return pd.DataFrame()

        # Background : tous les gènes KEGG de l'organisme
        bg_url = f"https://rest.kegg.jp/link/{organism}/pathway"
        r_bg = requests.get(bg_url, timeout=30)
        all_pathways = set()
        if r_bg.status_code == 200:
            for line in r_bg.text.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) == 2:
                    all_pathways.add(parts[1])

        n_genes = len(genes)
        n_bg = 20000  # Approximation gènes bovins
        results = []
        for pathway, gene_list in gene_links.items():
            # Nom du pathway
            name_r = requests.get(f"https://rest.kegg.jp/list/{pathway}", timeout=10)
            name = (name_r.text.strip().split("\t")[1]
                    if name_r.status_code == 200 and "\t" in name_r.text
                    else pathway)
            k = len(set(gene_list))
            # Enrichissement
            p = stats.hypergeom.sf(k-1, n_bg, k, n_genes) if n_genes > 0 else 1.0
            results.append({
                "pathway_id": pathway,
                "pathway_name": name,
                "n_genes": k,
                "p_value": float(p),
                "genes": ",".join(gene_list),
            })
        df = pd.DataFrame(results)
        if not df.empty:
            df["padj"] = np.minimum(df["p_value"] * len(df), 1.0)
            df = df.sort_values("p_value").reset_index(drop=True)
        return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


# ============ STRING NETWORK ============

def string_network(genes, species=9913, required_score=400):
    """Réseau STRING (Bos taurus = 9913)."""
    if not genes:
        return pd.DataFrame()
    url = "https://string-db.org/api/json/network"
    params = {
        "identifiers": "\r".join(genes),
        "species": species,
        "required_score": required_score,
    }
    try:
        r = requests.post(url, data=params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            return pd.DataFrame([{
                "preferredName_A": d["preferredName_A"],
                "preferredName_B": d["preferredName_B"],
                "score": d["score"],
                "nscore": d.get("nscore", 0),
            } for d in data])
    except Exception:
        pass
    return pd.DataFrame()
