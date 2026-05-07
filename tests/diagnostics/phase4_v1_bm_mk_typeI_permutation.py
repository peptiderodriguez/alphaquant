"""Phase 4 V1: Type-I calibration on real bm_mk via within-subject permutation.

Per docs/paired_diffdist_design.md:
> Validation 1 — Type-I calibration (paired-correct null):
>   - Within-subject permute arm labels, run paired AQ, record gene-level p
>   - Acceptance: p-distribution uniform on [1/2ⁿ, 1]
>   - QQ plot diagonal in [0.1, 1] range (above the discrete floor)
> Compute: 16 perms × 4 cells × ~5 min ≈ 5 hr sequential, ~1.5 hr on Slurm array

Slurm-array driver. For task_id ∈ [0, 64):
    cell_idx = task_id // 16     (0 = GC_F, 1 = GC_M, 2 = HU_F, 3 = HU_M)
    perm_id  = task_id % 16      (16 = 2^4 sign-flip patterns at n=4)

Sign-flip permutation: for each subject in {S1, S2, S3, S4} apply sign=-1
to swap that subject's (femur, humerus) condition labels (so the paired test
now sees humerus values in the femur arm and vice-versa). Under H0 (no
true bone effect), the resulting gene-level p-distribution should be
uniform on the achievable discrete grid [1/2^n, 1].

Usage:
    # one task (debug)
    python phase4_v1_bm_mk_typeI_permutation.py --task-id 0

    # 64-task Slurm array
    sbatch --array=0-63 ... submit_phase4_v1.sh

Outputs (per task):
    <output_root>/<cell>/perm_<NN>/results.tsv      — gene-level paired-AQ output
    <output_root>/<cell>/perm_<NN>/permvec.json     — sign-flip vector + meta

Aggregator (run after array completes):
    python phase4_v1_bm_mk_typeI_permutation.py --aggregate
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kstest

_REPO = "/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/alphaquant"
sys.path.insert(0, _REPO)

LOG = logging.getLogger("phase4_v1")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

BM_MK_PARQUET = Path(
    "/fs/pool/pool-mann-edwin/DIA_output/bm_mk_2026Apr/publication_figures_v7/filtered.parquet"
)
BM_MK_SAMPLEMAP = Path(
    "/fs/pool/pool-mann-edwin/DIA_output/bm_mk_2026Apr/publication_figures_v7/.aq_array_state/samplemap.tsv"
)
DEFAULT_OUT = Path(
    "/fs/pool/pool-mann-edwin/DIA_output/bm_mk_2026Apr/phase4_v1_validation"
)

CELLS = ["GC_F", "GC_M", "HU_F", "HU_M"]
CELL_FACTORS = {"GC_F": ("F", "GC"), "GC_M": ("M", "GC"),
                "HU_F": ("F", "HU"), "HU_M": ("M", "HU")}


def derive_subject_id(sm: pd.DataFrame) -> pd.DataFrame:
    pat = re.compile(r"_([FM](?:GC|HU)\d+)_")
    sm = sm.copy()
    sm["subject_id"] = [(pat.search(s).group(1) if pat.search(s) else None)
                         for s in sm["sample"]]
    return sm


def subset_to_cell(sm: pd.DataFrame, sex: str, group: str) -> pd.DataFrame:
    out = sm[(sm["sex"] == sex) & (sm["group"] == group)
             & (sm["bone"].isin(["femur", "humerus"]))].copy()
    out["condition"] = out["sex"] + "_" + out["group"] + "_" + out["bone"]
    return out.reset_index(drop=True)


def enumerate_signflip_vectors(n_subjects: int) -> list[tuple[int, ...]]:
    """All 2^n sign-flip vectors (each entry ∈ {+1, -1})."""
    return [tuple(v) for v in product((+1, -1), repeat=n_subjects)]


def apply_signflip_to_samplemap(sm_cell: pd.DataFrame, signflip_vec: dict[str, int]) -> pd.DataFrame:
    """Swap the `condition` (and `bone`) for samples of subjects with sign=-1.

    This is the H0 permutation: under no-true-bone-effect, swapping a
    subject's femur/humerus labels produces a draw from the same null.
    """
    out = sm_cell.copy()
    out["bone_orig"] = out["bone"]
    new_bone = []
    new_condition = []
    for _, row in out.iterrows():
        sign = signflip_vec.get(row["subject_id"], +1)
        if sign == -1:
            swapped = "humerus" if row["bone"] == "femur" else "femur"
        else:
            swapped = row["bone"]
        new_bone.append(swapped)
        new_condition.append(f"{row['sex']}_{row['group']}_{swapped}")
    out["bone"] = new_bone
    out["condition"] = new_condition
    return out


def run_permuted_paired_aq(parquet: Path, sm_perm: pd.DataFrame, cell: str,
                            cell_sex: str, cell_group: str, results_dir: Path
                            ) -> Path:
    import alphaquant.run_pipeline as aq_pipeline
    cond_a = f"{cell_sex}_{cell_group}_femur"
    cond_b = f"{cell_sex}_{cell_group}_humerus"
    LOG.info("[%s] paired AQ (permuted): %s vs %s, n_a=%d, n_b=%d, n_subj=%d",
             cell, cond_a, cond_b,
             (sm_perm["condition"] == cond_a).sum(),
             (sm_perm["condition"] == cond_b).sum(),
             sm_perm["subject_id"].nunique())

    results_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    aq_pipeline.run_pipeline(
        input_file=str(parquet),
        samplemap_df=sm_perm[["sample", "condition", "subject_id"]].copy(),
        results_dir=str(results_dir),
        condpairs_list=[(cond_a, cond_b)],
        ion_test_method="paired_diffdist",
        subject_id_col="subject_id",
        paired_permutation_max_perms=64,
        use_ml=False,
        use_multiprocessing=False,
        runtime_plots=False,
        valid_values_filter_mode="per_condition",
        min_valid_values_c1=3, min_valid_values_c2=3,
    )
    LOG.info("[%s] paired AQ done in %.0fs", cell, time.time() - t0)
    candidates = list(results_dir.rglob("*.results.tsv"))
    if not candidates:
        raise FileNotFoundError(f"no .results.tsv under {results_dir}")
    preferred = [p for p in candidates
                 if cond_a in p.name and cond_b in p.name]
    return preferred[0] if preferred else candidates[0]


def run_one_task(task_id: int, output_root: Path) -> None:
    cell = CELLS[task_id // 16]
    perm_id = task_id % 16
    cell_sex, cell_group = CELL_FACTORS[cell]
    cell_dir = output_root / cell / f"perm_{perm_id:02d}"

    sentinel = cell_dir / ".done"
    if sentinel.exists():
        LOG.info("task_id=%d (%s perm=%d) already done — skipping", task_id, cell, perm_id)
        return

    sm_full = pd.read_csv(BM_MK_SAMPLEMAP, sep="\t")
    sm_full = derive_subject_id(sm_full)
    sm_cell = subset_to_cell(sm_full, cell_sex, cell_group)

    subjects = sorted(sm_cell["subject_id"].dropna().unique())
    flips = enumerate_signflip_vectors(len(subjects))
    if perm_id >= len(flips):
        raise IndexError(f"perm_id {perm_id} out of range for n_subjects={len(subjects)} "
                          f"({len(flips)} permutations)")
    flip = flips[perm_id]
    flip_dict = dict(zip(subjects, flip))

    sm_perm = apply_signflip_to_samplemap(sm_cell, flip_dict)
    cell_dir.mkdir(parents=True, exist_ok=True)
    sm_perm.to_csv(cell_dir / "samplemap_permuted.tsv", sep="\t", index=False)

    meta = {
        "cell": cell, "perm_id": perm_id,
        "subjects": subjects, "signflip_vec": list(flip),
        "n_signs_neg": int(sum(1 for s in flip if s == -1)),
        "is_identity": all(s == +1 for s in flip),
    }
    (cell_dir / "permvec.json").write_text(json.dumps(meta, indent=2))

    aq_path = run_permuted_paired_aq(BM_MK_PARQUET, sm_perm, cell,
                                       cell_sex, cell_group, cell_dir)
    canonical = cell_dir / "results.tsv"
    if aq_path.resolve() != canonical.resolve():
        canonical.write_bytes(aq_path.read_bytes())
    sentinel.touch()
    LOG.info("task_id=%d (%s perm=%d) DONE → %s", task_id, cell, perm_id, canonical)


def aggregate(output_root: Path) -> None:
    rows = []
    for cell in CELLS:
        cdir = output_root / cell
        if not cdir.is_dir():
            continue
        for pdir in sorted(cdir.glob("perm_*")):
            r = pdir / "results.tsv"
            meta_p = pdir / "permvec.json"
            if not r.is_file() or not meta_p.is_file():
                continue
            meta = json.loads(meta_p.read_text())
            df = pd.read_csv(r, sep="\t")
            for c in df.columns:
                if "p" in c.lower():
                    pass  # let downstream choose canonical column
            # The canonical p column for paired-diffdist is `p_value`
            # (= permutation-calibrated). The analytical p is `p_val_analytic`.
            ps = df["p_value"].dropna().values if "p_value" in df.columns else np.array([])
            ps_analytic = (df["p_val_analytic"].dropna().values
                            if "p_val_analytic" in df.columns else np.array([]))
            rows.append({
                "cell": cell, "perm_id": meta["perm_id"],
                "is_identity": meta["is_identity"],
                "n_signs_neg": meta["n_signs_neg"],
                "n_proteins": int(len(df)),
                "p_perm": ps.tolist(),
                "p_analytic": ps_analytic.tolist(),
            })
    if not rows:
        LOG.warning("no perm outputs found under %s", output_root)
        return

    # Pool p-values from non-identity permutations only (identity perm is the
    # original test, not an H0 draw).
    pooled_perm = np.concatenate([np.asarray(r["p_perm"], dtype=float)
                                    for r in rows if not r["is_identity"]])
    pooled_anal = np.concatenate([np.asarray(r["p_analytic"], dtype=float)
                                    for r in rows if not r["is_identity"]])
    pooled_perm = pooled_perm[~np.isnan(pooled_perm)]
    pooled_anal = pooled_anal[~np.isnan(pooled_anal)]
    LOG.info("pooled non-identity perm p: n=%d", len(pooled_perm))
    LOG.info("pooled non-identity analytic p: n=%d", len(pooled_anal))

    # Tail-rate stats
    def tail_stats(p, name):
        return {
            f"n_{name}": int(len(p)),
            f"frac_p<0.05_{name}": float(np.mean(p < 0.05)),
            f"frac_p<0.01_{name}": float(np.mean(p < 0.01)),
            f"frac_p<0.001_{name}": float(np.mean(p < 0.001)),
            f"frac_p<0.0625_{name}": float(np.mean(p < 0.0625)),  # 1/16 floor
            f"mean_p_{name}": float(np.mean(p)),
            f"median_p_{name}": float(np.median(p)),
        }

    summary = {
        "n_tasks": len(rows),
        "n_cells": len(set(r["cell"] for r in rows)),
        "perm": tail_stats(pooled_perm, "perm"),
        "analytic": tail_stats(pooled_anal, "analytic"),
    }
    # KS test against uniform on [1/16, 1] for permutation p (truncated uniform)
    # and on [0, 1] for analytical p.
    if len(pooled_perm):
        # Map [1/16, 1] permutation p back to [0, 1] for KS
        pp = (pooled_perm - 1/16) / (1 - 1/16)
        pp = pp[(pp >= 0) & (pp <= 1)]
        if len(pp):
            ks = kstest(pp, "uniform")
            summary["ks_perm_truncated"] = {"stat": float(ks.statistic),
                                              "p": float(ks.pvalue)}
    if len(pooled_anal):
        ks2 = kstest(pooled_anal, "uniform")
        summary["ks_analytic"] = {"stat": float(ks2.statistic),
                                    "p": float(ks2.pvalue)}

    (output_root / "v1_summary.json").write_text(json.dumps(summary, indent=2))
    LOG.info("\n%s", json.dumps(summary, indent=2))

    # Diagnostic figure: histograms + QQ for both p columns
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=130)
    ax = axes[0, 0]
    ax.hist(pooled_perm, bins=32, color="#1f77b4", alpha=0.7,
             edgecolor="white")
    ax.axvline(0.05, color="red", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("paired AQ permutation p")
    ax.set_ylabel("count")
    ax.set_title(f"H0 perm-p hist (n={len(pooled_perm):,})\n"
                  f"frac p<0.05 = {summary['perm']['frac_p<0.05_perm']:.3f}")
    ax = axes[0, 1]
    ax.hist(pooled_anal, bins=32, color="#2ca02c", alpha=0.7,
             edgecolor="white")
    ax.axvline(0.05, color="red", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("paired AQ analytical p")
    ax.set_ylabel("count")
    ax.set_title(f"H0 analytical-p hist (n={len(pooled_anal):,})\n"
                  f"frac p<0.05 = {summary['analytic']['frac_p<0.05_analytic']:.3f}")
    ax = axes[1, 0]
    pp_sorted = np.sort(pooled_perm)
    if len(pp_sorted):
        u = np.linspace(1/16, 1, len(pp_sorted))
        ax.plot(u, pp_sorted, lw=0.6)
        ax.plot([0, 1], [0, 1], "r--", lw=0.8)
        ax.set_xlabel("expected (truncated uniform [1/16, 1])")
        ax.set_ylabel("observed paired AQ permutation p")
        ax.set_title("QQ — permutation p")
    ax = axes[1, 1]
    pa_sorted = np.sort(pooled_anal)
    if len(pa_sorted):
        u = np.linspace(0, 1, len(pa_sorted))
        ax.plot(u, pa_sorted, lw=0.6)
        ax.plot([0, 1], [0, 1], "r--", lw=0.8)
        ax.set_xlabel("expected (uniform [0, 1])")
        ax.set_ylabel("observed paired AQ analytical p")
        ax.set_title("QQ — analytical p")
    fig.suptitle("Phase 4 V1 — Type-I calibration on real bm_mk via "
                  "within-subject sign-flip permutation\n"
                  "(non-identity permutations pooled across 4 cells)",
                  fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_root / "v1_typeI_diagnostic.png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=None,
                     help="Slurm-array task id ∈ [0, 64).")
    ap.add_argument("--aggregate", action="store_true",
                     help="Run aggregator after array completes.")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args.output_root)
    elif args.task_id is not None:
        run_one_task(args.task_id, args.output_root)
    else:
        ap.error("either --task-id N or --aggregate is required")
