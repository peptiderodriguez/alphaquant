# Paired-diffdist: extending AlphaQuant for within-subject contrasts

**Status**: design, not yet implemented
**Branch**: `paired-diffdist`
**Plan version**: 4 (consolidated 2026-05-06)

This design extends AlphaQuant's ion-level test to handle within-subject
paired contrasts (e.g., bm_mk's femur-vs-humerus per mouse; pre/post
within subject) while preserving AQ's tree aggregation, intensity-dependent
empirical-null calibration, and FDR philosophy.

The design has been through four critique rounds (two by an external
proteomics-statistics collaborator, two AI self-critiques). Their substantive
contributions are reflected throughout — most importantly: covariance-adjusted
Stouffer (not naïve Stouffer), shrinkage covariance estimation, mandatory
permutation calibration at low n, equicorrelation-not-identity fallback,
single-mixed-model benchmark (not per-ion lmer + aggregate). Where the
collaborator's recommendations conflict with AQ's existing framework, the
trade-offs are stated explicitly.

## Statistical estimand

For a within-subject axis where each subject _i_ contributes measurements
under both arms _A_ and _B_ (bm_mk: each mouse contributes femur and
humerus; pre/post: each subject contributes baseline and follow-up),
test per protein:

```
y_{i, t, k} = μ + β · Arm_{it} + b_i + u_k + ε_{itk}
H0: β = 0
```

where _i_ = subject, _t_ ∈ {A, B}, _k_ = ion, b_i ~ N(0, σ²_subject) is
the subject random intercept, u_k is the ion-level effect (nuisance,
absorbed via tree aggregation), and ε is residual noise.

This estimand maps both pre/post and within-animal paired bones — both
are within-subject repeated measures with between-subject variation as
the nuisance factor.

## Architecture changes

| Layer | Current (unpaired) | Paired |
|---|---|---|
| Ion-level test statistic | All N×M cross-replicate log-FCs vs intensity-binned empirical null | Within-subject paired log-FCs vs within-subject same-arm empirical null |
| Empirical null source | Same-condition cross-replicate log-FCs | Within-subject same-arm log-FCs (techreps within (subject, arm) cell) |
| Tree aggregation (Stouffer Z) | Assumes ion-level p independence | Covariance-adjusted Stouffer Z + shrinkage R̂; mandatory permutation calibration at low n |
| Missingness | Counting-based pool for missing-in-one-arm | Per-ion: only subjects with both arms detected; flag `n_paired_subjects` |
| FDR | BH at protein level | Same, with `fdr_family ∈ {protein, proteoform, ptm}` for hierarchical claims |
| Output schema | Standard | + `test_method`, `n_paired_subjects`, `permutation_p_floor` columns |

Tree-walking, outlier filtering, ML reordering, output table generation,
plotting are paired-agnostic and unchanged.

## Why covariance-adjusted Stouffer + permutation (not full mixed-model)

The collaborator's gold-standard recommendation is per-protein mixed-model:

```
log2_intensity ~ bone + (1 | subject) + (1 | ion)
```

This produces a single protein-level p-value directly. We **benchmark**
against it in Phase 4, but make covariance-adjusted Stouffer + permutation
the **primary** approach because:

1. **Calibration consistency** — AQ's framework is built around
   intensity-dependent empirical nulls at the ion level. Mixed-model breaks
   that calibration philosophy and would require parallel infrastructure.
2. **Computational cost** — lmer per protein at 5000 proteins × 4 contrasts
   ≈ 20K model fits per dataset. Covariance-adjusted Stouffer is closed-form;
   ~10-50× faster.
3. **Refactor surface** — Stouffer-with-Σ plugs into the existing
   `aggregate_node_properties` Stouffer call at one site. Mixed-model
   touches the entire stats pipeline.

If Phase 4 benchmarking shows mixed-model dominates by >20% TPR at matched
FDR, we revisit and consider shipping mixed-model as a second test method
(`ion_test_method='lmer_paired'`).

## Statistical mechanism (the crucial detail)

Two complementary pieces:

**(a) Covariance-adjusted Stouffer Z** for the test statistic:

```
Z_combined = Σ Z_i  /  √(N + Σ_{i≠j} R̂_ij)
```

where R̂_ij is the within-protein ion correlation matrix, estimated from
same-condition replicate data via Ledoit-Wolf shrinkage.

Note: this is the Z-aggregation analog of Brown's method. Brown's specifically
combines dependent p-values via Fisher's chi-squared with df adjustment;
what we use is the Stouffer-with-covariance equivalent. Both are valid
formulations; Stouffer-with-Σ is natural here because AQ's ion-level
output is z-scores from an empirical-null framework.

**(b) Patient-block sign-flip permutation** for FDR-honest p-value
calibration (mandatory at low n):

```
For each within-subject sign-flip permutation:
  apply permutation → recompute paired ion log-FCs → recompute ion-level Z
  → recompute Z_combined via mechanism (a)

p_observed = (# perms with |Z_perm| ≥ |Z_observed|) / N_perms
```

At n=4 mice: 16 permutations → discrete floor at 1/16 = 0.0625.
At n=6: 64 permutations → floor 0.016. At n=8: 256 → floor 0.004.
Reported in the output column `permutation_p_floor` so consumers know
the resolution limit.

**Why both are needed**: at low n (bm_mk's n=4), R̂ is fundamentally
under-identified — even with shrinkage. Ledoit-Wolf regularizes the
estimate but cannot create the information that's not in 4 mice. The
permutation-derived p-value is FDR-honest by construction regardless
of how good or bad R̂ is. At higher n (≥ 8), R̂ becomes identifiable
and the analytic p from (a) is the primary inference; permutation
becomes optional / spot-check.

**Critical**: do NOT use identity-Σ as a fallback when shrinkage is
unreliable. Identity-Σ assumes independence; under positive ion
correlation (the typical case in proteomics) it **underestimates the
combined Z's variance** → inflates Z → anti-conservative. Safer fallback
when shrinkage is suspect: equicorrelation matrix `Σ = ρ·11ᵀ + (1-ρ)·I`
with conservative ρ = 0.3-0.5. Always raise an explicit warning when
falling back.

## Phases

### Phase 1 — Plumbing (~2 days)

Thread subject IDs from samplemap through contrast routing. No statistical
change.

**Files**:
- `alphaquant/diffquant/condpair_analysis.py::analyze_condpair`:
  - Detect paired-eligible contrasts (samplemap has `subject_id` repeats
    in both arms)
  - Dispatch to paired test when `runconfig.ion_test_method == 'paired_diffdist'`
  - Mixed paired/unpaired contrasts in same run supported
- `alphaquant/config/variables.py`: add `'paired_diffdist'` to
  `ion_test_method` enum

**Tests** (`tests/unit_tests/test_paired_routing.py`, property-based):
- Contrast with `subject_id` repeats → paired routing
- Contrast without `subject_id` → unpaired routing (back-compat)
- Mixed paired/unpaired contrasts in single run → both work
- Existing unpaired tests pass (regression)

**Exit**: dispatch correct; no behavior change for unpaired contrasts.

### Phase 2 — Paired ion test + null builder (~3-4 days)

**New files** (composition over inheritance — no `BackgroundDistribution`
refactor):
- `alphaquant/diffquant/diff_analysis_paired.py::DifferentialIonPaired`
- `alphaquant/diffquant/background_distributions_paired.py::PairedBackgroundDistribution`

`PairedBackgroundDistribution` builds the empirical null from within-subject
same-arm log-FCs across all ions in an intensity bin (techreps within
(subject, arm) cells for bm_mk).

**Missingness handling**: per ion, only subjects with both arms detected
contribute paired log-FCs. Output column `n_paired_subjects` is
propagated up the tree. Ions with `n_paired_subjects < 2` produce NaN
p-value (not garbage).

**Tests** (`tests/unit_tests/test_paired_ion.py`, property-based):
- Strong synthetic signal → p < 0.01
- Null synthetic ion → p uniform on [0, 1]
- 1-subject outlier ion → bounded influence (sensitivity threshold)
- n=2 edge case: runs without crashing, reports discrete-null limitation
- Existing `DifferentialIon` tests pass (regression)

### Phase 2.5 — Null-shape diagnostic (~1 day)

`tests/diagnostics/test_paired_null_shape.py`:
- Run paired AQ on bm_mk
- Emit per-intensity-bin null histograms + QQ plots
- Manual gate: unimodal, centered ≈ 0, intensity-dependent variance,
  no heavy tails

If null is bimodal or skewed, debug before continuing.

### Phase 3 — Covariance-adjusted Stouffer + permutation (~4-5 days)

**New files**:
- `alphaquant/cluster/correlation_aware_stouffer.py`:
  - `compute_correlation_matrix(protein_ions, samplemap, method)`:
    - `method='ledoit_wolf'` (default): shrunk sample correlation
    - `method='equicorrelation'`: fixed ρ fallback (warns)
    - `method='sample'`: raw sample correlation (warns at low n)
    - Never returns identity silently; explicit `'identity'` raises a
      warning that this is anti-conservative under positive ion correlation
  - `covariance_adjusted_stouffer_z(z_values, R)`: closed-form
    Σz / √(1ᵀ R 1)
- `alphaquant/cluster/paired_permutation_null.py`:
  - `paired_signflip_permutations(samplemap, max_perms=64)`:
    - Within-subject arm-label sign-flips
    - n=4: 16 distinct (full enum); n=6: 64 distinct (full enum);
      n=8: 256 distinct (full enum); higher n: random sample of `max_perms`
  - `permutation_calibrated_p(observed_z, null_z_distribution)`:
    - p = (# perms with |z_perm| ≥ |z_obs|) / N_perms
    - Documents discrete floor in result metadata
- `alphaquant/cluster/cluster_utils.py::sum_and_re_scale_zvalues_paired`:
  - Used by `aggregate_node_properties` when contrast is paired
  - Calls (a) covariance-adjusted Stouffer + (b) permutation calibration
    when N_subjects ≤ 8 (configurable threshold)

**Tests**:
- Independent ions (R̂ = 0): covariance-adjusted Stouffer = classical Stouffer
- Highly correlated ions (R̂ ≈ 1): combined Z ≈ Z_single (no inflation)
- Sample-cov vs Ledoit-Wolf at n=4: shrunk reduces MSE vs known truth
- Permutation calibration: p-distribution uniform on achievable resolution
- bm_mk fixture: uncorrected vs corrected; uncorrected shows inflated p,
  corrected + permuted is calibrated
- n=4 edge case: discrete p-values, floor reported

**Exit**: paired AQ calibrated against permutation null at low n;
covariance-adjusted Stouffer matches classical when R̂ is zero; identity-Σ
never a silent fallback.

### Phase 3.5 — Low-n cautionary note (added 2026-05-07 from V4 GC_F)

**The analytical p-value column (`p_val_analytic`) is anti-conservative at
n_subjects ≤ 8 and MUST NOT be the FDR-control input.**

Empirical evidence from Phase 4 V4 on real bm_mk GC_F (n=4 paired mice,
femur vs humerus, ~5,360 protein-level tests):

| metric                  | paired AQ permutation | paired AQ analytical | paired LIMPA |
|---|---|---|---|
| min p-value             | 0.0625 (= 1/16)       | 2.22e-16 (numerical) | 0.000779     |
| hits at FDR < 0.10      | 0                     | **469**              | 0            |
| Spearman ρ vs LIMPA p   | 0.21                  | 0.21                 | —            |
| Spearman ρ on log2FC    | 0.70                  | 0.70                 | —            |

The analytical p-values land at the float64 floor (2.22e-16) for hundreds of
proteins whose log2FC is essentially zero (Glce 0.10, Terf2ip 0.016, Mical3
0.24). With 4 paired subjects, the cov-adjusted Stouffer Z + Ledoit-Wolf
shrinkage operates on a covariance matrix that is fundamentally
under-determined (the LW shrinkage target dominates), and the resulting
analytical p-values are inflated whenever any small but spuriously-correlated
within-protein ion pattern aligns. Synthetic Type-I checks (Phase 2.5) did
not catch this because synthetic ions have iid noise — the real-data
correlation structure that the LW shrinkage was designed to absorb is
exactly what generates the spurious significance at low n.

The permutation-calibrated p (`p_value` / `p_val_permutation` columns,
identical) is the safety valve and **the canonical output column**:
- Resolution = 1 / 2^n_subjects (1/16 at n=4, 1/64 at n=6, 1/256 at n=8)
- Floor reported in `permutation_p_floor` per row
- Cannot make implausibly-significant claims at low n
- Convergent with paired LIMPA where biology is weak (both gave 0 hits at
  FDR < 0.1 in GC_F — agreement on negative result)

**Operational rule (locked in)**:

| n_paired_subjects | safe FDR-control column         | analytical p column     |
|---|---|---|
| ≤ 8               | `p_value` (= permutation)       | diagnostic only         |
| 9 — 15            | `p_value` (= permutation, sampled at `paired_permutation_max_perms`) | use with caution; cross-check |
| ≥ 16              | analytical OR permutation       | analytical preferred (more power, finer resolution; permutation grid coarsens) |

Surfaces of this rule:
- The `p_value` column in `*.results.tsv` is the **permutation** p, not the
  analytical p — `tables/diffquant_table.py` writes it that way deliberately.
- Downstream HTML / GO ORA / volcano plots all read `p_value` and `fdr` —
  they automatically inherit the safe column.
- The `p_val_analytic` column is preserved as a research diagnostic (see
  `n_perms_used`, `permutation_p_floor`, `correlation_estimator` columns
  for full provenance per protein).

**Phase 4 V4 acceptance criterion is now informed by this**: when both
methods report 0 hits at FDR < 0.1 (as in GC_F), agreement on log2FC
estimates (Spearman ρ ≥ 0.5 on shared proteins) is the appropriate
convergence signal. The hit-overlap Jaccard is undefined when both sets
are empty and should not be reported as 0/0 → bug.

**Action items propagated to Phase 5**:
- Document this rule in the AlphaQuant user guide.
- Emit an `LOGGER.warning` in `condpair_analysis.py` whenever
  `n_paired_subjects ≤ 8` and the user accesses `p_val_analytic`
  programmatically (e.g., via `aq_runner` configuration with a
  custom p-column — see Phase 5).

### Phase 4 — Validation cascade (~5-7 days)

**Validation 1 — Type-I calibration** (paired-correct null):
- Within-subject permute arm labels, run paired AQ, record gene-level p
- Acceptance: p-distribution uniform on `[1/2ⁿ, 1]`
- QQ plot diagonal in `[0.1, 1]` range (above the discrete floor)

**Validation 2 — Power simulation**:
- Synthetic dataset matching bm_mk: n=4 mice × 2 arms × 3 techreps × 5000
  proteins, ICC ≈ 0.4
- Spike in 200 proteins with within-subject correlated effects of size
  0.3-1.0 log2
- Compare unpaired AQ, paired LIMPA, paired AQ
- Acceptance: paired AQ TPR at FDR < 0.1 ≥ paired LIMPA TPR at matched FDR
  (FDR < 0.1 because the 1/16 floor caps FDR < 0.05 resolution)

**Validation 3 — Per-protein lmer benchmark** (gold-standard reference):
- Random sample of 100 proteins from bm_mk
- For each protein, fit ONE mixed model:
  ```
  log2_intensity ~ bone + (1 | subject) + (1 | ion)
  ```
  Single protein-level p-value, no aggregation step needed
- Compare directly to paired AQ's gene-level p (Spearman ρ on -log10p)
- Acceptance: ρ ≥ 0.7 OR paired AQ ≥ lmer in power on simulated truth
- **Decision rule**: if lmer dominates by >20% TPR at matched FDR,
  escalate (ship lmer as primary path)

**Validation 4 — Concordance vs paired LIMPA on real bm_mk** (descriptive):
- Direction agreement on shared FDR < 0.1 hits
- Spearman ρ on lfc estimates across detected proteins
- Tabulate; investigate disagreements case by case

**Compute budget**:
- Type-I: 16 perms × 4 cells × ~5 min ≈ 5 hr sequential, ~1.5 hr on Slurm array
- Power simulation: 100 sim datasets × 3 methods × ~5 min ≈ 25 hr sequential,
  ~5 hr on Slurm array
- Lmer benchmark: 100 proteins × ~10 sec ≈ 17 min sequential
- Concordance: 4 cells × 1 paired AQ run ≈ 20 min

Total ~10-12 Slurm-hours; fits in a half-day on a busy cluster.

### Phase 5 — Integration + docs (~2-3 days)

**Integration**:
- `alphaquant/diffquant/diff_analysis.py` — add `test_method`,
  `n_paired_subjects`, `permutation_p_floor` columns to results.tsv
- `alphaquant/tables/diffquant_table.py` — add `fdr_family` column;
  separate BH within each family (protein / proteoform / ptm)
- `aq_runner/runner.py` — auto-detect paired-eligible contrasts from
  samplemap when `ion_test_method='paired_diffdist'`
- `ehr_proteomics_analysis/registry.py` — `paired_test_method:
  'limpa' | 'aq_paired'` (default `'limpa'` for back-compat)
- `_shared/run_publication_contam.sh` (bm_mk) — add Stage 8: paired AQ
  alongside paired LIMPA
- `_shared/make_narrative_html_v2.py::build_bm_mk_v7` — add paired-AQ
  comparison panel: AQ-unpaired vs paired LIMPA vs paired AQ

**Documentation**:
- `alphaquant/CLAUDE.md` — add `paired_diffdist` to ion_test_method options
- `alphaquant/DEVELOPERS.md` — paired test extension architecture,
  covariance-adjusted Stouffer rationale, permutation-calibration design,
  troubleshooting guide
- `alphaquant/docs/paired_diffdist_design.md` — this file
- `alphaquant/docs/paired_diffdist_methods.md` (NEW) — citable methods
  note; cites MSstats as gold-standard mixed-model alternative; explains
  the Stouffer-with-Σ vs Brown's method distinction; acknowledges
  external collaborator
- `alphaquant/docs/paired_diffdist_validation.md` (NEW) — Type-I + power +
  lmer-benchmark + concordance evidence

## Forking + sync strategy

- **origin**: `git@github.com:peptiderodriguez/alphaquant.git` (fork)
- **upstream**: `https://github.com/MannLabs/alphaquant.git` (read-only)
- **Feature branch**: `paired-diffdist`
- **Sync cadence**: monthly merge from `upstream/main`
- **Upstream PR**: open draft after Phase 4 validation passes; don't
  block on its merge
- **Local install**: `pip install -e <fork>/alphaquant` ensures local
  env uses the fork; downstream consumers (aq_runner,
  ehr_proteomics_analysis) unaffected

## Total timeline

| Phase | Time | Cumulative |
|---|---|---|
| 1. Plumbing | 2 days | 2 |
| 2. Paired ion test + null + missingness | 3-4 days | 6 |
| 2.5. Null-shape diagnostic | 1 day | 7 |
| 3. Covariance-adjusted Stouffer + permutation | 4-5 days | 12 |
| 4. Validation (Type-I + power + lmer + concordance) | 5-7 days | 19 |
| 5. Integration + claim-level FDR + docs | 2-3 days | 22 |
| **Total** | **17-22 days** | **~3-4 weeks** |

Add 30% buffer for unknowns → **4-5 weeks calendar time**.

## Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| R̂ fundamentally under-identified at n=4 even with shrinkage | High | High | Permutation calibration is **mandatory** at low n; serves as the FDR-honest backbone regardless of R̂ quality |
| Identity-Σ silently used as fallback (anti-conservative) | Was a v3 bug | High | Removed in v4; equicorrelation is the explicit fallback; identity raises warning |
| Per-protein lmer dominates by >20% TPR | Medium | Medium | Phase 4 V3 catches this; honest acceptance — switch to lmer-as-primary OR ship paired AQ as alt method |
| Paired empirical null bimodal/non-Gaussian | Low | High | Phase 2.5 diagnostic catches this |
| Missing-paired-data exclusion shrinks usable proteins | Medium | Low | Document `n_paired_subjects` in output |
| n=4 1/16 discrete floor surprises users | High | Low | Documented prominently; reported in `permutation_p_floor` column |
| Upstream PR rejected | Medium | Low | Maintain fork; monthly upstream sync |

## Decision gates

- **End of Phase 2.5**: empirical null shape sane? If bimodal/skewed, debug.
- **End of Phase 3**: covariance-adjusted Stouffer behaves correctly on
  synthetic data? Permutation calibration produces uniform p? If not, debug.
- **End of Phase 4 V1 (Type-I)**: p-distribution uniform above the
  discrete floor? If not, the test is broken — fix or stop.
- **End of Phase 4 V2 (power)**: paired AQ ≥ paired LIMPA on simulated
  within-subject biology? If yes, ship as default. If matches, ship as
  alternative test method (user choice). If worse, ship as research-config.
- **End of Phase 4 V3 (lmer benchmark)**: lmer doesn't dominate by >20%?
  If lmer dominates by larger margin, escalate to consider mixed-model
  primary path.

## Open questions

1. **Subject ID column convention**: standardize on `subject_id`
   (lowercase, snake-case) across alphaquant + aq_runner +
   ehr_proteomics_analysis.

2. **Within-protein ion correlation reference data**: same-condition
   replicate data is the natural source. For bm_mk: 3-4 microdiss
   techreps per (mouse, bone) cell × 4 cells = ~60 same-condition pairs.
   Probably enough for shrinkage; verify in Phase 3.

3. **ANCOVA for genuine pre/post designs**: the bone-axis paired bone
   contrast doesn't have a "baseline" — it's symmetric. But for genuine
   pre/post designs (longitudinal psilo data?), the natural model is
   `y_post ~ y_pre + arm + (1|subject)`. Out of scope here; flag for
   future paired contrasts.

4. **Block bootstrap as alternative to permutation**: for higher n
   where 2ⁿ permutations is prohibitive, patient-block bootstrap is an
   alternative resampling scheme. Could be third test method
   (`'paired_bootstrap'`). Phase 6 / future work.

5. **External collaborator co-authorship**: their critique substantively
   shaped this design (covariance-aware aggregation, mandatory permutation
   at low n, equicorrelation fallback, single-mixed-model benchmark). If
   we publish methods, they should be co-authors on the methods note.

## Pre-implementation checklist

- [x] Variance decomposition diagnostic on bm_mk (median ICC 0.29-0.42 → worth building)
- [x] v4 design plan (this doc)
- [x] External collaborator review incorporated (2 rounds)
- [x] Fork created: `peptiderodriguez/alphaquant`
- [x] Branch created: `paired-diffdist`
- [x] **Phase 1 plumbing** (2026-05-06):
      - `is_paired_eligible` + `get_paired_sample_triples` helpers in diffutils.py
      - `DifferentialIonPaired` stub in diff_analysis.py
      - Per-contrast paired-eligibility dispatch in condpair_analysis.py
      - `subject_id_col` kwarg + `paired_diffdist` option in run_pipeline.py
      - 18 property-based tests in test_paired_routing.py (all passing
        after Phase 1 reviewer issues addressed: signature lock, ion_test_method
        validation, cond1==cond2 guard, dispatch-level mock tests)
      - Existing diff_analysis tests still pass (no regression)
- [x] **Phase 2 paired ion test** (2026-05-06):
      - `PairedBackGroundDistribution` (subclasses BackGroundDistribution,
        overrides anchor-FC generation to pool within-(subject, arm) cell
        log-FCs)
      - `PairedConditionBackgrounds` (orchestrator, intensity-binned per-ion
        null lookup)
      - `DifferentialIonPaired` body: classical Stouffer over within-subject
        log-FCs, capped-z outlier correction (Phase 3 will add covariance
        correction)
      - `_build_paired_subject_pairs_for_ion` helper in condpair_analysis.py
        (per-ion microdiss collapse + drop subjects with all-NaN in one arm)
      - Dispatch wired up: `paired_normed = PairedConditionBackgrounds(...)`
        once per condpair, per-ion lookup in the loop
      - 17 new Phase 2 tests in test_paired_phase2.py (all passing)
      - 36 total alphaquant tests pass (18 routing + 17 phase2 + 2 diff +
        1 condpair); no regressions
      - Numerical concerns documented for Phase 4 validation (independence
        assumption, asymptotic-normality at low n, capped-z K choice, 0.5
        factor reuse from parent, empirical-null SD≈1 assumption)
- [x] Phase 2.5 null-shape diagnostic (2026-05-06):
      - Initial diagnostic on bm_mk F·GC bone-axis flagged calibration
        bug (issue C1 from reviewer): null SD ≈ σ_within (per-microdiss
        scale) while test-statistic SD = sqrt(2/m)·σ_within (cell-mean
        scale). Reported |z_i| systematically too small under H0 →
        over-conservative p's.
      - Fix: rebuild null from per-subject paired log-FCs (mean(arm1) −
        mean(arm2)) pooled across ions in the bin — i.e., the test
        statistic itself. Override `generate_fc2counts_from_anchor_fcs` to
        bin raw values (no 0.5·pairwise-diff rescaling). Variance
        algebraically aligned to test stat under H0 (= 2σ²_within / m).
      - Synthetic H0 calibration check (5000 ions, n=4 mice, m=3):
        - Bin SD ≈ 0.146, matches theoretical sqrt(2·0.18²/3) = 0.147 ✓
        - Post-lookup z mean=0.07, SD=0.98 (target 0, 1) ✓
        - |z|>1.96 = 4.97% (target 5%), |z|>2.58 = 0.94% (target 1%) ✓
        - Gene-level p mean=0.51, KS vs uniform[0,1] p=0.05 (cannot
          reject uniform) ✓
      - Reviewer follow-ups also addressed:
        - M1: outlier_scaling_factor median-zero → mean-fallback +
          floor (no longer collapses)
        - C2: paired_diff_dist.SD docstring clarified (FC-scale, NOT
          z-scale)
        - m1: degenerate-bin docstring (returns 0 for all fc, not ±max_z)
        - m4: per-bin warnings aggregated to single end-of-init summary
        - G2: synthetic-H0 uniform-p test added (test_paired_phase2.py)
        - G3: outlier_correction with median_abs_z=0 test added
        - G4: numeric regression test for unpaired fallback added
        - G5: KS test against uniform on n=6 sign-flips added
      - Real bm_mk QQ plots remain S-shaped — that's now EXPECTED
        (empirical null on real data inherits heavy tails of true
        differential ions). Rank-based z-lookup handles this correctly;
        Type-I calibration is verified on synthetic H0 instead.
- [ ] Phase 3 covariance-adjusted Stouffer + permutation
- [ ] Phase 4 validation cascade
- [ ] Phase 5 integration + docs
