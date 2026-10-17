# Snakefile — Workflow reproductible v7.1
from snakemake.utils import min_version
import os

min_version("7.0")

configfile: "config.yaml"

PROJECT = config["project_name"]
INPUT_PED = config["input_ped"]
INPUT_MAP = config["input_map"]
OUTDIR = config.get("outdir", "results")

rule all:
    input:
        f"{OUTDIR}/01_qc/qc_stats.json",
        f"{OUTDIR}/02_fst/fst_bootstrap.json",
        f"{OUTDIR}/03_neutrality/neutrality.json",
        f"{OUTDIR}/04_gwas/gwas_summary.json",
        f"{OUTDIR}/05_report/rapport_IMRaD.html",

rule qc:
    input: ped=INPUT_PED, map=INPUT_MAP
    output: json=f"{OUTDIR}/01_qc/qc_stats.json"
    log: f"{OUTDIR}/logs/qc.log"
    shell:
        """
        python -m v71_pipeline.qc --ped {input.ped} --map {input.map} \
            --outdir {OUTDIR}/01_qc > {log} 2>&1
        """

rule fst_weir_cockerham:
    input: json=f"{OUTDIR}/01_qc/qc_stats.json"
    output: json=f"{OUTDIR}/02_fst/fst_bootstrap.json"
    log: f"{OUTDIR}/logs/fst.log"
    shell:
        """
        python -m v71_pipeline.fst --input {input.json} \
            --n-bootstrap 1000 --out {output.json} > {log} 2>&1
        """

rule neutrality:
    input: json=f"{OUTDIR}/01_qc/qc_stats.json"
    output: json=f"{OUTDIR}/03_neutrality/neutrality.json"
    log: f"{OUTDIR}/logs/neutrality.log"
    shell:
        """
        python -m v71_pipeline.neutrality --input {input.json} \
            --tests tajima,fuli,faywu --out {output.json} > {log} 2>&1
        """

rule gwas_lmm:
    input:
        qc=f"{OUTDIR}/01_qc/qc_stats.json",
        pheno=config["pheno_file"]
    output: json=f"{OUTDIR}/04_gwas/gwas_summary.json"
    log: f"{OUTDIR}/logs/gwas.log"
    shell:
        """
        python -m v71_pipeline.gwas_lmm --input {input.qc} \
            --pheno {input.pheno} --out {output.json} > {log} 2>&1
        """

rule imrad_report:
    input:
        qc=f"{OUTDIR}/01_qc/qc_stats.json",
        fst=f"{OUTDIR}/02_fst/fst_bootstrap.json",
        neutral=f"{OUTDIR}/03_neutrality/neutrality.json",
        gwas=f"{OUTDIR}/04_gwas/gwas_summary.json"
    output: html=f"{OUTDIR}/05_report/rapport_IMRaD.html"
    log: f"{OUTDIR}/logs/report.log"
    shell:
        """
        python -m v71_pipeline.report --qc {input.qc} --fst {input.fst} \
            --neutral {input.neutral} --gwas {input.gwas} \
            --out {output.html} > {log} 2>&1
        """
