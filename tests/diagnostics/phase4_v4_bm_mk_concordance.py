"""Phase 4 V4: paired-AQ vs paired-LIMPA concordance on real bm_mk.

Per docs/paired_diffdist_design.md:
> Validation 4 — Concordance vs paired LIMPA on real bm_mk (descriptive):
>   - Direction agreement on shared FDR < 0.1 hits
>   - Spearman ρ on lfc estimates across detected proteins
>   - Tabulate; investigate disagreements case by case

Runs paired AQ for one or more bone-axis cells (GC_F, GC_M, HU_F, HU_M) using
the within-subject femur-vs-humerus design, then compares the protein-level
results.tsv to paired LIMPA's existing output for the same cell.

Cell --> subject_id mapping is derived from sample names (e.g. FGC1_femur and
FGC1_humerus are paired). Subject_id column is added to the samplemap before
calling alphaquant.run_pipeline with ``ion_test_method='paired_diffdist'``.

Usage:
    python phase4_v4_bm_mk_concordance.py --cell GC_F           # one cell
    python phase4_v4_bm_mk_concordance.py --cell GC_F GC_M HU_F HU_M  # all four

Outputs (per cell, under ``<output_dir>/phase4_v4_<cell>/``):
    paired_aq.results.tsv             — paired AQ output (canonical AQ schema)
    paired_aq_run.log                 — alphaquant.run_pipeline stdout
    concordance_<cell>.tsv            — per-protein side-by-side AQ vs LIMPA
    concordance_<cell>_summary.json   — Spearman ρ, hit overlaps, n_proteins
    concordance_<cell>.png            — p-value histogram + scatter (4 panels)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Ensure working-tree alphaquant takes precedence over site-packages
_REPO = "/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/alphaquant"
sys.path.insert(0, _REPO)

LOG = logging.getLogger("phase4_v4")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# bm_mk inputs
BM_MK_PARQUET = Path(
    "/fs/pool/pool-mann-edwin/DIA_output/bm_mk_2026Apr/publication_figures_v7/filtered.parquet"
)
BM_MK_SAMPLEMAP = Path(
    "/fs/pool/pool-mann-edwin/DIA_output/bm_mk_2026Apr/publication_figures_v7/.aq_array_state/samplemap.tsv"
)
LIMPA_DIR = Path(
    "/fs/pool/pool-mann-edwin/DIA_output/bm_mk_2026Apr/publication_figures_v7/aq_runner_output_paired_limpa"
)

# Cell name → (sex, group). e.g. GC_F = F mice in GC group.
CELLS = {
    "GC_F": ("F", "GC"),
    "GC_M": ("M", "GC"),
    "HU_F": ("F", "HU"),
    "HU_M": ("M", "HU"),
}


def derive_subject_id(sm: pd.DataFrame) -> pd.DataFrame:
    """Add subject_id from sample name (e.g., FGC1, FHU3, MGC2).

    bm_mk run-name pattern: `..._E990_<SubjectID>_<bone>_rep<N>_<plate>_<...>`.
    Extracts <SubjectID> as the token starting with [F|M][GC|HU][digit].
    """
    pat = re.compile(r"_([FM](?:GC|HU)\d+)_")
    sm = sm.copy()
    sids = []
    for s in sm["sample"]:
        m = pat.search(s)
        sids.append(m.group(1) if m else None)
    sm["subject_id"] = sids
    n_missing = sm["subject_id"].isna().sum()
    if n_missing:
        LOG.warning("derive_subject_id: %d/%d samples without subject_id "
                    "(first 3): %s",
                    n_missing, len(sm),
                    sm[sm["subject_id"].isna()]["sample"].head(3).tolist())
    return sm


def subset_to_cell(sm: pd.DataFrame, sex: str, group: str) -> pd.DataFrame:
    """Subset samplemap to femur+humerus rows of a single (sex, group) cell."""
    out = sm[(sm["sex"] == sex) & (sm["group"] == group)
             & (sm["bone"].isin(["femur", "humerus"]))].copy()
    # Condition: <sex>_<group>_<bone> (matches bm_mk samplemap condition col)
    out["condition"] = out["sex"] + "_" + out["group"] + "_" + out["bone"]
    return out.reset_index(drop=True)


def read_paired_limpa_for_cell(cell: str) -> pd.DataFrame | None:
    """Read paired LIMPA results.tsv for a cell. Returns None if missing."""
    candidates = [
        LIMPA_DIR / cell / f"{cell}.results.tsv",
        LIMPA_DIR / cell / f"{cell}.proteins.tsv",
    ]
    for p in candidates:
        if p.is_file():
            df = pd.read_csv(p, sep="\t")
            df = df.rename(columns={
                "protein": "protein",
                "p_value": "p_aq_limpa",
                "log2fc": "lfc_aq_limpa",
                "fdr": "fdr_aq_limpa",
            })
            return df[["protein", "p_aq_limpa", "lfc_aq_limpa",
                       "fdr_aq_limpa"]].dropna(subset=["p_aq_limpa"])
    LOG.warning("read_paired_limpa_for_cell(%s): no results.tsv found at %s",
                cell, candidates)
    return None


def run_paired_aq(parquet: Path, sm_cell: pd.DataFrame, cell: str,
                  cell_sex: str, cell_group: str, results_dir: Path,
                  paired_permutation_max_perms: int = 64) -> Path:
    """Invoke alphaquant.run_pipeline with paired_diffdist on a single cell.

    Returns the path to the AQ results.tsv (or raises if it doesn't appear).
    """
    import alphaquant.run_pipeline as aq_pipeline
    LOG.info("loading alphaquant from: %s", aq_pipeline.__file__)

    cond_a = f"{cell_sex}_{cell_group}_femur"
    cond_b = f"{cell_sex}_{cell_group}_humerus"

    LOG.info("[%s] paired AQ:  %s vs %s, n_a=%d, n_b=%d, n_subjects=%d",
             cell, cond_a, cond_b,
             (sm_cell["condition"] == cond_a).sum(),
             (sm_cell["condition"] == cond_b).sum(),
             sm_cell["subject_id"].nunique())

    results_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    aq_pipeline.run_pipeline(
        input_file=str(parquet),
        samplemap_df=sm_cell[["sample", "condition", "subject_id"]].copy(),
        results_dir=str(results_dir),
        condpairs_list=[(cond_a, cond_b)],
        ion_test_method="paired_diffdist",
        subject_id_col="subject_id",
        paired_permutation_max_perms=paired_permutation_max_perms,
        use_ml=False,
        use_multiprocessing=False,
        runtime_plots=False,
        # Min-valid: with 4 mice × 2-3 microdiss/bone, ≥3 in each arm is reasonable
        valid_values_filter_mode="per_condition",
        min_valid_values_c1=3,
        min_valid_values_c2=3,
    )
    LOG.info("[%s] paired AQ done in %.0fs", cell, time.time() - t0)

    # Locate the canonical results.tsv
    candidates = list(results_dir.rglob("*.results.tsv"))
    if not candidates:
        raise FileNotFoundError(
            f"paired AQ produced no .results.tsv under {results_dir}; "
            f"check {results_dir}/paired_aq_run.log"
        )
    # Prefer the condpair-named one
    preferred = [p for p in candidates
                 if cond_a in p.name and cond_b in p.name]
    return preferred[0] if preferred else candidates[0]


def compare_to_limpa(aq_path: Path, limpa: pd.DataFrame, cell: str,
                     out_dir: Path) -> dict:
    """Compute per-protein concordance metrics + scatter figure."""
    aq = pd.read_csv(aq_path, sep="\t")
    # Adapt AQ schema to common columns
    rename_map = {}
    for src, dst in [
        ("protein", "protein"), ("Protein.Group", "protein"), ("ProteinGroup", "protein"),
    ]:
        if src in aq.columns:
            rename_map[src] = dst
            break
    if "protein" not in aq.columns and rename_map:
        aq = aq.rename(columns=rename_map)
    # canonical AQ schema for differential output
    p_col = "p_val" if "p_val" in aq.columns else (
        "pvalue" if "pvalue" in aq.columns else "p_value")
    lfc_col = "log2fc" if "log2fc" in aq.columns else "fc"
    fdr_col = "fdr" if "fdr" in aq.columns else "padj"
    aq = aq.rename(columns={p_col: "p_aq_paired",
                              lfc_col: "lfc_aq_paired",
                              fdr_col: "fdr_aq_paired"})
    aq = aq[["protein", "p_aq_paired", "lfc_aq_paired", "fdr_aq_paired"]].dropna(
        subset=["p_aq_paired"])

    # Inner join on protein
    merged = aq.merge(limpa, on="protein", how="inner")
    n_shared = len(merged)
    n_aq_only = len(aq) - n_shared
    n_limpa_only = len(limpa) - n_shared

    # Spearman ρ on -log10(p) and on lfc
    rho_p, p_p = spearmanr(-np.log10(merged["p_aq_paired"].clip(1e-300, 1)),
                            -np.log10(merged["p_aq_limpa"].clip(1e-300, 1)))
    rho_lfc, p_lfc = spearmanr(merged["lfc_aq_paired"], merged["lfc_aq_limpa"])

    # Hits at FDR < 0.1
    aq_hits = set(merged.loc[merged["fdr_aq_paired"] < 0.1, "protein"])
    limpa_hits = set(merged.loc[merged["fdr_aq_limpa"] < 0.1, "protein"])
    intersect = aq_hits & limpa_hits
    union = aq_hits | limpa_hits
    jaccard = len(intersect) / len(union) if union else float("nan")

    # Direction agreement on shared hits at FDR<0.1
    shared_hits = merged[merged["protein"].isin(intersect)]
    dir_agree = ((np.sign(shared_hits["lfc_aq_paired"])
                   == np.sign(shared_hits["lfc_aq_limpa"])).sum()
                 / max(len(shared_hits), 1))

    summary = {
        "cell": cell,
        "n_proteins_aq_paired": int(len(aq)),
        "n_proteins_limpa": int(len(limpa)),
        "n_shared": n_shared,
        "n_aq_only": n_aq_only,
        "n_limpa_only": n_limpa_only,
        "spearman_rho_neglog10p": float(rho_p),
        "spearman_rho_lfc": float(rho_lfc),
        "n_aq_hits_fdr01": len(aq_hits),
        "n_limpa_hits_fdr01": len(limpa_hits),
        "n_shared_hits_fdr01": len(intersect),
        "jaccard_hits_fdr01": float(jaccard),
        "direction_agreement_shared_hits": float(dir_agree),
    }
    LOG.info("[%s] CONCORDANCE: rho_p=%.3f  rho_lfc=%.3f  Jaccard=%.3f  "
             "dir_agree=%.3f  n_shared=%d",
             cell, rho_p, rho_lfc, jaccard, dir_agree, n_shared)

    # Save TSV + summary
    merged.to_csv(out_dir / f"concordance_{cell}.tsv", sep="\t", index=False)
    (out_dir / f"concordance_{cell}_summary.json").write_text(
        json.dumps(summary, indent=2))

    # 4-panel figure: p-hist (AQ), p-hist (LIMPA), -log10p scatter, lfc scatter
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=130)
    ax = axes[0, 0]
    ax.hist(aq["p_aq_paired"], bins=40, color="#1f77b4", alpha=0.8,
             edgecolor="white")
    ax.set_title(f"paired AQ p-value distribution\n(n={len(aq)} proteins)",
                  fontsize=10)
    ax.set_xlabel("p")
    ax.set_ylabel("count")
    ax.axvline(0.05, color="red", lw=1, ls="--", alpha=0.5)

    ax = axes[0, 1]
    ax.hist(limpa["p_aq_limpa"], bins=40, color="#2ca02c", alpha=0.8,
             edgecolor="white")
    ax.set_title(f"paired LIMPA p-value distribution\n(n={len(limpa)} proteins)",
                  fontsize=10)
    ax.set_xlabel("p")
    ax.set_ylabel("count")
    ax.axvline(0.05, color="red", lw=1, ls="--", alpha=0.5)

    ax = axes[1, 0]
    ax.scatter(-np.log10(merged["p_aq_limpa"].clip(1e-300, 1)),
                -np.log10(merged["p_aq_paired"].clip(1e-300, 1)),
                s=4, alpha=0.4, color="#444")
    lo, hi = 0, max(-np.log10(merged["p_aq_paired"].clip(1e-300, 1)).max(),
                       -np.log10(merged["p_aq_limpa"].clip(1e-300, 1)).max())
    ax.plot([lo, hi], [lo, hi], color="red", lw=1, ls="--", alpha=0.5)
    ax.set_xlabel("-log10 p (paired LIMPA)")
    ax.set_ylabel("-log10 p (paired AQ)")
    ax.set_title(f"-log10p concordance\nSpearman ρ={rho_p:.3f}", fontsize=10)

    ax = axes[1, 1]
    ax.scatter(merged["lfc_aq_limpa"], merged["lfc_aq_paired"],
                s=4, alpha=0.4, color="#444")
    lim = float(np.nanmax(np.abs([merged["lfc_aq_limpa"].min(),
                                     merged["lfc_aq_limpa"].max(),
                                     merged["lfc_aq_paired"].min(),
                                     merged["lfc_aq_paired"].max()])))
    ax.plot([-lim, lim], [-lim, lim], color="red", lw=1, ls="--", alpha=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("log2FC (paired LIMPA)")
    ax.set_ylabel("log2FC (paired AQ)")
    ax.set_title(f"log2FC concordance\nSpearman ρ={rho_lfc:.3f}", fontsize=10)

    fig.suptitle(
        f"Phase 4 V4 — paired-AQ vs paired-LIMPA concordance ({cell})\n"
        f"FDR<0.1 hits: AQ={len(aq_hits)}, LIMPA={len(limpa_hits)}, shared={len(intersect)}, "
        f"Jaccard={jaccard:.3f}, direction_agree={dir_agree:.3f}",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / f"concordance_{cell}.png")
    plt.close(fig)

    return summary


def main(cells: Iterable[str], output_dir: Path,
         paired_permutation_max_perms: int = 64) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("inputs: parquet=%s  samplemap=%s  limpa_dir=%s",
             BM_MK_PARQUET, BM_MK_SAMPLEMAP, LIMPA_DIR)
    for p in (BM_MK_PARQUET, BM_MK_SAMPLEMAP, LIMPA_DIR):
        if not p.exists():
            raise FileNotFoundError(f"missing input: {p}")

    sm_full = pd.read_csv(BM_MK_SAMPLEMAP, sep="\t")
    sm_full = derive_subject_id(sm_full)

    summaries = []
    for cell in cells:
        if cell not in CELLS:
            raise ValueError(f"unknown cell {cell!r}; expected one of "
                              f"{sorted(CELLS)}")
        cell_sex, cell_group = CELLS[cell]
        cell_dir = output_dir / f"phase4_v4_{cell}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        sm_cell = subset_to_cell(sm_full, cell_sex, cell_group)
        sm_cell.to_csv(cell_dir / "samplemap_cell.tsv", sep="\t", index=False)

        # Run paired AQ
        try:
            aq_path = run_paired_aq(
                BM_MK_PARQUET, sm_cell, cell, cell_sex, cell_group, cell_dir,
                paired_permutation_max_perms=paired_permutation_max_perms,
            )
            LOG.info("[%s] paired AQ results: %s", cell, aq_path)
        except Exception as e:
            LOG.exception("[%s] paired AQ failed: %s", cell, e)
            continue

        # Compare
        limpa = read_paired_limpa_for_cell(cell)
        if limpa is None:
            LOG.warning("[%s] no paired LIMPA reference; skipping concordance",
                         cell)
            continue
        summary = compare_to_limpa(aq_path, limpa, cell, cell_dir)
        summary["aq_results_path"] = str(aq_path)
        summaries.append(summary)

    # Aggregate summary
    if summaries:
        agg = pd.DataFrame(summaries)
        agg.to_csv(output_dir / "phase4_v4_summary.tsv", sep="\t", index=False)
        LOG.info("\n=== AGGREGATE PHASE 4 V4 SUMMARY ===\n%s",
                  agg.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", nargs="+", default=["GC_F"],
                     choices=sorted(CELLS),
                     help="One or more cells to test (default GC_F).")
    ap.add_argument("--output-dir", type=Path,
                     default=Path("/fs/pool/pool-mann-edwin/DIA_output/bm_mk_2026Apr/phase4_v4_validation"),
                     help="Output directory.")
    ap.add_argument("--max-perms", type=int, default=64,
                     help="paired_permutation_max_perms passed to AQ (default 64).")
    args = ap.parse_args()
    main(args.cell, args.output_dir, args.max_perms)
