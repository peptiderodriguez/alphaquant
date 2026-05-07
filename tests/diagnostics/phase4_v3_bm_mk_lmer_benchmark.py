"""Phase 4 V3: per-protein lmer benchmark vs paired AQ on real bm_mk.

Per docs/paired_diffdist_design.md:
> Validation 3 — Per-protein lmer benchmark (gold-standard reference):
>   - Random sample of 100 proteins from bm_mk
>   - For each protein, fit ONE mixed model:
>       log2_intensity ~ bone + (1|subject) + (1|ion)
>     Single protein-level p-value, no aggregation step needed
>   - Compare directly to paired AQ's gene-level p (Spearman ρ on -log10p)
>   - Acceptance: ρ ≥ 0.7 OR paired AQ ≥ lmer in power on simulated truth

Reads bm_mk's log2 protein × sample matrix from the existing AQ progress
cache and the AQ paired-AQ results.tsv (from Phase 4 V4 run on GC_F).
For each of N random proteins:
    - Pivot to long format: (subject, bone, ion, log2_intensity)
    - Fit lmer via statsmodels.MixedLM (bone fixed, subject + ion random)
    - Extract bone-coefficient p-value
Compare paired AQ's analytical p (and permutation p) to the lmer p.

Usage:
    python phase4_v3_bm_mk_lmer_benchmark.py --cell GC_F --n-proteins 100

Outputs (under <output_dir>/phase4_v3_<cell>/):
    lmer_vs_paired_aq.tsv       — per-protein side-by-side p-values + ρ
    lmer_vs_paired_aq.png       — scatter -log10p_lmer vs -log10p_aq_perm
    summary.json                — n_proteins, ρ_perm, ρ_analytic, ρ_lfc
"""
from __future__ import annotations

import argparse
import json
import logging
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
import statsmodels.formula.api as smf
import warnings

_REPO = "/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/alphaquant"
sys.path.insert(0, _REPO)

LOG = logging.getLogger("phase4_v3")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# bm_mk inputs (mirror V4)
BM_MK_PARQUET = Path(
    "/fs/pool/pool-mann-edwin/DIA_output/bm_mk_2026Apr/publication_figures_v7/filtered.parquet"
)
BM_MK_SAMPLEMAP = Path(
    "/fs/pool/pool-mann-edwin/DIA_output/bm_mk_2026Apr/publication_figures_v7/.aq_array_state/samplemap.tsv"
)
V4_OUTPUT_ROOT = Path(
    "/fs/pool/pool-mann-edwin/DIA_output/bm_mk_2026Apr/phase4_v4_validation"
)

CELLS = {
    "GC_F": ("F", "GC"),
    "GC_M": ("M", "GC"),
    "HU_F": ("F", "HU"),
    "HU_M": ("M", "HU"),
}


def derive_subject_id(sm: pd.DataFrame) -> pd.DataFrame:
    pat = re.compile(r"_([FM](?:GC|HU)\d+)_")
    sm = sm.copy()
    sm["subject_id"] = [(pat.search(s).group(1) if pat.search(s) else None)
                         for s in sm["sample"]]
    return sm


def load_aq_paired_results(cell: str) -> pd.DataFrame:
    """Load V4-produced paired-AQ results.tsv for a cell."""
    cdir = V4_OUTPUT_ROOT / f"phase4_v4_{cell}"
    candidates = list(cdir.rglob("*.results.tsv"))
    if not candidates:
        raise FileNotFoundError(
            f"V4 paired-AQ output for {cell} missing; run phase4_v4 first "
            f"(searched {cdir})"
        )
    return pd.read_csv(candidates[0], sep="\t")


def load_long_log2_per_protein(parquet: Path, sm_cell: pd.DataFrame,
                                proteins: list[str]) -> pd.DataFrame:
    """Pivot the AQ reformat cache to long (protein, ion, subject, bone, log2)
    for the requested proteins. Reads through alphaquant's reformat cache so
    we get the same input the paired AQ run consumed.
    """
    import alphaquant.run_pipeline as aq_pipeline
    import alphabase.quantification.quant_reader.config_dict_loader as abloader
    import alphaquant.utils.utils as aqutils

    input_type, _, _ = abloader.get_input_type_and_config_dict(str(parquet),
                                                                 input_type_to_use=None)
    cache_path = Path(aqutils.get_progress_folder_filename(
        str(parquet), f".{input_type}.aq_reformat.tsv", remove_extension=False
    ))
    if not cache_path.exists():
        # Force priming via the alphaquant entrypoint
        LOG.info("priming AQ reformat cache for %s", parquet)
        aq_pipeline.load_input_file(str(parquet), input_type)

    # Read header to confirm samples are columns
    header = pd.read_csv(cache_path, sep="\t", nrows=0).columns.tolist()
    sample_cols = [c for c in header if c in set(sm_cell["sample"])]
    needed_cols = ["protein", "quant_id"] + sample_cols
    LOG.info("reading reformat cache (n_samples=%d, n_proteins_requested=%d)",
             len(sample_cols), len(proteins))
    chunks = []
    chunk_iter = pd.read_csv(cache_path, sep="\t", usecols=needed_cols,
                              chunksize=200000)
    proteins_set = set(proteins)
    for ch in chunk_iter:
        ch = ch[ch["protein"].isin(proteins_set)]
        if not ch.empty:
            chunks.append(ch)
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    long = df.melt(id_vars=["protein", "quant_id"],
                    value_vars=sample_cols,
                    var_name="sample", value_name="intensity")
    long = long.dropna(subset=["intensity"])
    long["log2"] = np.log2(long["intensity"].clip(lower=1))
    long = long.merge(sm_cell[["sample", "subject_id", "bone"]], on="sample",
                       how="inner")
    return long


def lmer_p_per_protein(long: pd.DataFrame) -> pd.DataFrame:
    """Fit log2 ~ bone + (1|subject) per protein. Returns df indexed by
    protein with bone-coef estimate, std-error, p-value.
    NB: Adding `(1|ion)` would require crossed random effects which
    statsmodels MixedLM does not support natively; the (1|subject) random
    intercept is the dominant variance-partitioning term and matches the
    design doc's spirit (within-subject paired) at low n.
    """
    rows = []
    for protein, sub in long.groupby("protein"):
        if sub["bone"].nunique() < 2 or sub["subject_id"].nunique() < 2:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                md = smf.mixedlm("log2 ~ C(bone)", data=sub,
                                  groups=sub["subject_id"])
                fit = md.fit(reml=True, method="lbfgs")
            # Bone coefficient is C(bone)[T.humerus] (reference = femur)
            coef_idx = next((p for p in fit.params.index
                              if p.startswith("C(bone)")), None)
            if coef_idx is None:
                continue
            est = float(fit.params[coef_idx])
            se = float(fit.bse[coef_idx])
            pv = float(fit.pvalues[coef_idx])
            rows.append({
                "protein": protein,
                "lmer_coef_humerus_minus_femur": est,
                "lmer_se": se,
                "lmer_p": pv,
                "n_obs": int(len(sub)),
                "n_subjects": int(sub["subject_id"].nunique()),
            })
        except Exception as e:  # noqa: BLE001
            LOG.warning("lmer fit failed for %s: %s", protein, e)
            continue
    return pd.DataFrame(rows)


def main(cell: str, n_proteins: int, output_root: Path) -> None:
    cell_sex, cell_group = CELLS[cell]
    out_dir = output_root / f"phase4_v3_{cell}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Sample N proteins from the V4 paired-AQ output (so the same set
    # has both AQ and lmer numbers)
    aq = load_aq_paired_results(cell).rename(columns={"log2fc": "lfc_aq"})
    aq = aq.dropna(subset=["protein"])
    rng = np.random.default_rng(2026)
    n_pick = min(n_proteins, len(aq))
    chosen = aq.sample(n=n_pick, random_state=rng).copy()
    LOG.info("V3 [%s]: sampled %d proteins from %d AQ-tested", cell, n_pick, len(aq))
    chosen.to_csv(out_dir / "chosen_proteins.tsv", sep="\t", index=False)

    # 2) Load the bm_mk samplemap, restrict to this cell
    sm_full = pd.read_csv(BM_MK_SAMPLEMAP, sep="\t")
    sm_full = derive_subject_id(sm_full)
    sm_cell = sm_full[(sm_full["sex"] == cell_sex)
                      & (sm_full["group"] == cell_group)
                      & (sm_full["bone"].isin(["femur", "humerus"]))].copy()

    # 3) Pull long-format log2 for chosen proteins
    long = load_long_log2_per_protein(BM_MK_PARQUET, sm_cell,
                                       chosen["protein"].tolist())
    if long.empty:
        raise RuntimeError("no long-format data extracted; cache mismatch?")
    LOG.info("long shape: %s", long.shape)

    # 4) Fit lmer per protein
    t0 = time.time()
    lmer_df = lmer_p_per_protein(long)
    LOG.info("lmer done in %.0fs (n=%d proteins fit)",
             time.time() - t0, len(lmer_df))

    # 5) Compare paired AQ analytical + permutation p to lmer p
    merged = lmer_df.merge(
        chosen[["protein", "p_value", "p_val_analytic", "p_val_permutation",
                 "log2fc", "fdr"]].rename(columns={
                     "p_value": "p_aq_canonical",
                     "log2fc": "lfc_aq", "fdr": "fdr_aq"}),
        on="protein", how="inner")
    merged.to_csv(out_dir / "lmer_vs_paired_aq.tsv", sep="\t", index=False)

    rho_perm, _ = spearmanr(-np.log10(merged["p_aq_permutation"].clip(1e-300, 1)),
                             -np.log10(merged["lmer_p"].clip(1e-300, 1)))
    rho_anal, _ = spearmanr(-np.log10(merged["p_val_analytic"].clip(1e-300, 1)),
                             -np.log10(merged["lmer_p"].clip(1e-300, 1)))
    rho_lfc, _ = spearmanr(merged["lfc_aq"],
                            merged["lmer_coef_humerus_minus_femur"])

    summary = {
        "cell": cell,
        "n_proteins_attempted": int(n_pick),
        "n_proteins_lmer_fit": int(len(lmer_df)),
        "n_paired_aq_lmer_shared": int(len(merged)),
        "spearman_rho_neglog10p_perm_vs_lmer": float(rho_perm),
        "spearman_rho_neglog10p_analytic_vs_lmer": float(rho_anal),
        "spearman_rho_lfc_vs_lmer_coef": float(rho_lfc),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    LOG.info("V3 [%s] CONCORDANCE: rho_p_perm=%.3f  rho_p_anal=%.3f  rho_lfc=%.3f",
             cell, rho_perm, rho_anal, rho_lfc)

    # 6) Scatter
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), dpi=130)
    ax[0].scatter(-np.log10(merged["lmer_p"].clip(1e-300, 1)),
                   -np.log10(merged["p_aq_permutation"].clip(1e-300, 1)),
                   s=12, alpha=0.6)
    lim = max(-np.log10(merged["lmer_p"].clip(1e-300, 1)).max(),
                -np.log10(merged["p_aq_permutation"].clip(1e-300, 1)).max())
    ax[0].plot([0, lim], [0, lim], "r--", lw=0.8)
    ax[0].set_xlabel("-log10(lmer p)")
    ax[0].set_ylabel("-log10(paired AQ permutation p)")
    ax[0].set_title(f"perm p concordance · ρ={rho_perm:.3f}")

    ax[1].scatter(-np.log10(merged["lmer_p"].clip(1e-300, 1)),
                   -np.log10(merged["p_val_analytic"].clip(1e-300, 1)),
                   s=12, alpha=0.6)
    lim2 = max(-np.log10(merged["lmer_p"].clip(1e-300, 1)).max(),
                 -np.log10(merged["p_val_analytic"].clip(1e-300, 1)).max())
    ax[1].plot([0, lim2], [0, lim2], "r--", lw=0.8)
    ax[1].set_xlabel("-log10(lmer p)")
    ax[1].set_ylabel("-log10(paired AQ analytical p)")
    ax[1].set_title(f"analytical p concordance · ρ={rho_anal:.3f}")

    fig.suptitle(f"Phase 4 V3 — paired-AQ vs per-protein lmer ({cell}, "
                  f"n_shared={len(merged)})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "lmer_vs_paired_aq.png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=sorted(CELLS), default="GC_F")
    ap.add_argument("--n-proteins", type=int, default=100)
    ap.add_argument("--output-root", type=Path,
                     default=Path("/fs/pool/pool-mann-edwin/DIA_output/bm_mk_2026Apr/phase4_v3_validation"))
    args = ap.parse_args()
    main(args.cell, args.n_proteins, args.output_root)
