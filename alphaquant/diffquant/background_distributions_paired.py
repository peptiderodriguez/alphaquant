"""Paired-design empirical background distributions (Phase 2 + C1 fix).

These mirror :class:`alphaquant.diffquant.background_distributions.ConditionBackgrounds`
and :class:`BackGroundDistribution`, but the empirical null is built from
**within-subject paired log-FCs** instead of cross-replicate same-condition
log-FCs.

Variance algebra (post C1 fix — 2026-05-06)
------------------------------------------

Setup (bm_mk-shaped data; symbols generalise):
    - subjects i = 1..n
    - arms t ∈ {arm1, arm2}, microdiss reps k = 1..m_{i,t} per (i, t) cell
    - Under H0:    y_{i,t,k} = μ + b_i + ε_{i,t,k},
                    b_i ~ N(0, σ²_subj),  ε_{i,t,k} ~ N(0, σ²_within)

Test statistic (per-ion, per-subject paired log-FC):
    fc_i = mean_k(y_{i,1,k}) − mean_k(y_{i,2,k})
    Under H0, b_i cancels (within-subject pairing), so
        fc_i ~ N(0, 2 σ²_within / m_i)
    where m_i = harmonic mean of m_{i,1} and m_{i,2}; for bm_mk m=3 → Var≈(2/3)σ²_within.

Old (Phase 2) null builder (BUG — issue C1):
    - Anchor-FC step (paired override): for each (i, t) cell, anchor=v_a; emit
      (v_b − v_a). Single-pair within-cell difference, Var = 2 σ²_within.
    - Bin step (parent's `generate_fc2counts_from_anchor_fcs`): bin
      0.5·(anchor_fcs[j-1] − anchor_fcs[j]). Across-cell pair: Var = σ²_within.
      Same-cell pair (shared anchor): Var = 0.5 σ²_within. After shuffle,
      mostly across-cell → binned Var ≈ σ²_within.
    - But the test statistic Var = 2 σ²_within / m. So the null is too WIDE
      (by factor m/2 = 1.5 for m=3). z_i = lookup(fc_i) is shrunk by
      sqrt(2/m). Under H0, E[z_i²] = 2/m < 1 — the test is OVER-CONSERVATIVE
      (claim that z_i ~ N(0,1) under H0 was broken).

Fix (Option (d) — direct null at the test-statistic scale):
    Build the null FROM THE TEST STATISTIC ITSELF, pooled across ions in
    the intensity bin. For each ion k in the bin and each subject i with
    BOTH arms detected for that ion, compute:
        paired_fc_{i,k} = mean(arm1 microdiss) − mean(arm2 microdiss).
    Pool {paired_fc_{i,k}} across (i, k) → empirical null for the bin.

    Under H0 (most ions in the bin null), paired_fc_{i,k} ~ N(0, 2σ²_within/m).
    The empirical null IS the test statistic distribution under H0.
    Therefore lookup(fc_i) returns z_i ~ N(0, 1) under H0, by construction
    (rank-based z via empirical CDF). No fudge factor; calibration is
    correct.

    Trade-off: a few truly differential ions in the bin INFLATE the null's
    tails. The test loses some power (lookup says "this big fc_i is just a
    tail draw"). This is the SAME tradeoff AQ's unpaired empirical-null
    framework already makes — and it errs on the conservative side, never
    anti-conservative.

    For shape-related deviations from Gaussian (e.g., leptokurtic high-int
    bins observed in the bm_mk Phase 2.5 diagnostic), the rank-based z
    handles them correctly: empirical CDF → standard-normal quantile is
    well-defined regardless of the empirical distribution's shape.

Phase 2 scope:
    - Per-ion analytical p-value via classical Stouffer (sum z / sqrt(N)).
    - Cross-ion covariance correction (Phase 3) and permutation calibration
      (Phase 3) are NOT implemented here.

This module is intentionally a thin extension — composition over rewriting.
``PairedBackGroundDistribution`` subclasses ``BackGroundDistribution`` and
overrides only the anchor-FC generation step + the binning step (the latter
because the new anchor_fcs are ALREADY at test-statistic scale, so the
parent's 0.5·pairwise-diff rescaling no longer applies).
"""
from __future__ import annotations

import random
import typing

import numpy as np

import alphaquant.diffquant.diffutils as aqdiffutils
import alphaquant.diffquant.background_distributions as aqbg
import alphaquant.config.config as aqconfig

import logging

aqconfig.setup_logging()
LOGGER = logging.getLogger(__name__)


class PairedBackGroundDistribution(aqbg.BackGroundDistribution):
    """Empirical null for one intensity bin, built from **within-subject paired
    log-FCs** pooled across ions in the bin.

    Mirrors the public z-score-lookup interface of
    :class:`BackGroundDistribution` (``calc_zscore_from_fc``, ``var``, ``SD``,
    ``min_fc``, ``max_fc``, ``zscores``, ``cumulative``) so it plugs in
    transparently to ``DifferentialIonPaired``.

    The behavioural differences vs. parent:

      - Parent (``BackGroundDistribution``): for each ion in the bin, take the
        full vector of same-condition cross-replicate values, pick one as the
        "anchor", emit ``other_value - anchor`` log-FCs (pairwise within-rep
        FCs). Then the binning step takes 0.5·(consecutive-anchor-fc diffs)
        to rescale back to a single-replicate (within-condition) variance.
      - This class: for each ion in the bin, for **each subject with BOTH
        arms detected**, emit ``mean(arm1 microdiss) - mean(arm2 microdiss)``
        — i.e., the per-subject paired log-FC, the test statistic itself.
        Pool across subjects AND ions in the bin. The binning step then
        bins these raw values directly (no 0.5·diff rescaling).

    SD note (closes reviewer issue C2): ``self.SD`` reports the sample SD
    of the binned log-FC distribution on the **fold-change scale** (units of
    log2 intensity), NOT a unit-normal-scale SD. Under H0 (most ions in the
    bin null), this is approximately ``sqrt(2 σ²_within / m)`` — the SD of
    the per-subject paired log-FC. ``calc_zscore_from_fc(fc)`` returns the
    rank-based z (standard-normal quantile of fc's empirical CDF position),
    which IS unit-normal-scaled by construction.

    Args:
        start_idx, end_idx, ion2noNanvals, idx2ion, p2z: as in the parent
            class. ``ion2noNanvals`` is the COMBINED-arm non-NaN values per ion
            (used only for the missing-value fraction diagnostic).
        ion2pairedfcs: ``{ion_name: list[float]}`` where each float is one
            per-subject paired log-FC for this ion (mean(arm1 microdiss) −
            mean(arm2 microdiss)). Subjects with all-NaN in either arm for
            this ion contribute nothing. Ions absent from this dict (or
            with empty lists) contribute no anchor-FCs to the bin's null.
    """

    def __init__(
        self,
        start_idx: int,
        end_idx: int,
        ion2noNanvals: typing.Dict[str, np.ndarray],
        idx2ion: dict,
        p2z: dict,
        ion2pairedfcs: typing.Dict[str, typing.List[float]],
    ):
        # Stash the paired-fcs dict on self BEFORE super().__init__ runs,
        # because the parent's __init__ calls our overridden
        # generate_anchorfcs_from_intensity_range, which needs it.
        self._ion2pairedfcs = ion2pairedfcs
        # Pre-check: if NO ion in the bin has any paired-fc, anchor_fcs will
        # be empty. The parent's `generate_fc2counts_from_anchor_fcs` would
        # then crash on `min(self.fc2counts.keys())`. Bail out with a
        # degenerate but non-crashing distribution: zero-variance, no FC range.
        # Callers should treat such bins as non-informative (any non-zero fc
        # lookup returns 0 under our degenerate setup, i.e., the test cannot
        # discriminate signal from noise for ions in this bin).
        any_paired_fc = False
        for i in range(int(start_idx), int(end_idx)):
            ion = idx2ion.get(i)
            fcs = ion2pairedfcs.get(ion)
            if fcs:
                any_paired_fc = True
                break
        if not any_paired_fc:
            self._init_degenerate(start_idx, end_idx, ion2noNanvals, idx2ion)
            LOGGER.warning(
                "PairedBackGroundDistribution: bin [%d, %d) has no paired log-FC "
                "for any ion (no subject has BOTH arms detected). Empirical null "
                "is degenerate (SD=0, calc_zscore_from_fc returns 0 for any fc). "
                "Paired test on ions in this bin will be unstable and report "
                "n_paired_subjects=0.",
                int(start_idx), int(end_idx),
            )
            return
        super().__init__(start_idx, end_idx, ion2noNanvals, idx2ion, p2z)

    def _init_degenerate(self, start_idx, end_idx, ion2noNanvals, idx2ion):
        """Set the minimum fields required for downstream lookups when the bin
        has no usable paired log-FC data.

        ``calc_zscore_from_fc`` (the only method DifferentialIonPaired calls
        on us) goes through ``aqdiffutils.z_from_fc_lookup``. With min_fc=0,
        max_fc=0, max_z=0.0 and a 1-element cumulative:

          - |fc| < fc_conversion_factor → returns 0.0 (small-fc branch)
          - rank < 0 (fc < 0) → returns -max_z = 0.0
          - rank >= len(cumulative)=1 (fc > 0) → returns max_z = 0.0

        I.e., the lookup returns 0 for ALL fc — the test cannot discriminate
        signal from noise for ions in this bin (closes m1 docstring fix).
        """
        self.fc2counts = {0: 1}
        self.cumulative = np.array([1], dtype=np.int64)
        self.min_fc = 0
        self.max_fc = 0
        self.min_z = 0
        self.max_z = 0.0
        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx)
        self.var = 0.0
        self.SD = 0.0
        self.ions = {idx2ion.get(idx) for idx in range(int(start_idx), int(end_idx))}
        self.fraction_missingval = self.calc_missingval_fraction(ion2noNanvals, idx2ion)
        self.zscores = np.zeros(1)

    def generate_anchorfcs_from_intensity_range(
        self, ion2noNanvals: dict, idx2ion: dict
    ) -> list:
        """Anchor-FC generation from per-subject paired log-FCs.

        For each ion in the intensity bin, pull the precomputed list of
        per-subject paired log-FCs (``mean(arm1 microdiss) − mean(arm2
        microdiss)``, one per subject with both arms detected for that ion).
        Pool across subjects AND ions in the bin.

        These anchor_fcs are ALREADY at the test-statistic scale (the
        per-subject paired log-FC is exactly the per-ion test statistic) —
        the binning step (overridden below) bins these directly without the
        parent's 0.5·pairwise-diff rescaling.

        Returns:
            list: pooled per-subject paired log-FCs, in arbitrary order.
            Returns an empty list if no ion in the bin has any subject with
            both arms detected (degenerate bin — handled by ``__init__``
            before this gets called).
        """
        anchor_fcs: typing.List[float] = []
        for idx in range(self.start_idx, self.end_idx):
            ion = idx2ion.get(idx)
            fcs = self._ion2pairedfcs.get(ion)
            if not fcs:
                continue
            anchor_fcs.extend(float(f) for f in fcs if np.isfinite(f))
        return anchor_fcs

    def generate_fc2counts_from_anchor_fcs(self, anchor_fcs: list):
        """Bin per-subject paired log-FCs directly at the test-statistic scale.

        The parent class's implementation takes consecutive-pair differences
        with a 0.5 factor: ``0.5·(anchor_fcs[i-1] − anchor_fcs[i])``. That
        rescaling is appropriate when ``anchor_fcs`` are themselves single-
        rep within-condition differences (as in the unpaired
        ``BackGroundDistribution``), because the 0.5·pairwise-diff
        operation rescales 2-sample-difference variance back to
        1-sample-variance — which is what unpaired AQ's
        ``SubtractedBackgrounds`` then re-adds across two conditions.

        For the paired case, ``anchor_fcs`` are ALREADY per-subject paired
        log-FCs (the test statistic), with variance ≈ 2σ²_within/m under H0.
        Applying the 0.5·pairwise-diff rescaling would shrink the null's
        variance by a factor (m/2) — exactly the bug the C1 review flagged.
        So we bin the raw anchor_fcs directly: each value contributes one
        count at its own bin index.

        Args:
            anchor_fcs (list): per-subject paired log-FCs at the test-stat
                scale.

        Updates:
            ``self.fc2counts`` (binned histogram of paired log-FCs),
            ``self.min_fc``, ``self.max_fc``.
        """
        for fc in anchor_fcs:
            fc_binned = np.rint(self.fc_resolution_factor * fc).astype(np.int64)
            self.fc2counts[fc_binned] = self.fc2counts.get(fc_binned, 0) + 1

        self.min_fc = min(self.fc2counts.keys())
        self.max_fc = max(self.fc2counts.keys())


class PairedConditionBackgrounds:
    """Orchestrator for paired empirical-null intensity binning.

    Mirrors :class:`ConditionBackgrounds` but pools data from BOTH arms
    (the paired null is built across paired-subject log-FCs, which span
    both arms by construction).

    For bm_mk-shaped data:
        - ``df_arm1``, ``df_arm2``: wide-format (ions × samples) log2 dataframes,
          ONE arm per dataframe — the same ``df_c1_normed`` / ``df_c2_normed``
          objects that ``ConditionBackgrounds`` consumes in
          ``analyze_condpair``.
        - ``paired_triples``: from
          :func:`alphaquant.diffquant.diffutils.get_paired_sample_triples`,
          giving ``(subject_id, [c1_samples], [c2_samples])`` per paired
          subject. Used to identify (subject, arm) cells.
        - ``p2z``: shared p→z cache, as in ``ConditionBackgrounds``.

    The class:
        1. Builds a COMBINED (ions × all_samples) dataframe from the two arms.
        2. Computes a per-ion ``ion2pairedfcs``: list of per-subject paired
           log-FCs (= mean(arm1 microdiss) − mean(arm2 microdiss)). Subjects
           missing all values in either arm contribute nothing.
        3. Sorts ions by combined median intensity and bins them into
           intensity contexts using the same logic as ``ConditionBackgrounds``.
        4. Builds one ``PairedBackGroundDistribution`` per bin.
        5. Exposes ``get_paired_diff_dist(ion)`` for the dispatch site.

    Attributes:
        ion2background: ``{ion: PairedBackGroundDistribution}`` lookup.
        backgrounds: list of bin-level distributions.
        context_ranges: list of ``[start_idx, end_idx]`` per bin.
        ion2nonNanvals, ion2allvals, idx2ion, normed_condition_df:
            as in ``ConditionBackgrounds`` (over the combined dataframe).
        ion2pairedfcs: ``{ion: list[float]}`` per-ion per-subject paired
            log-FCs.
        ion2cellvals: kept as a SECONDARY diagnostic (within-(subject, arm)
            cell values) for inspection / future use; NOT consumed by the
            null builder anymore.
    """

    def __init__(
        self,
        df_arm1: "pd.DataFrame",
        df_arm2: "pd.DataFrame",
        paired_triples: typing.List[
            typing.Tuple[str, typing.List[str], typing.List[str]]
        ],
        p2z: dict,
    ):
        import pandas as pd

        self.backgrounds: typing.List[PairedBackGroundDistribution] = []
        self.ion2background: typing.Dict[str, PairedBackGroundDistribution] = {}
        self.context_ranges: typing.List[typing.List[int]] = []

        # Combine the two arms column-wise. Both are ions × samples log2 frames.
        # Inner join on index keeps only ions detected in both arms (the
        # only ions for which any paired test is possible).
        common_idx = df_arm1.index.intersection(df_arm2.index)
        df_arm1 = df_arm1.loc[common_idx]
        df_arm2 = df_arm2.loc[common_idx]
        # Strip any pre-existing diagnostic non-sample columns (e.g. a
        # "median" col left by upstream normalization) BEFORE concat so they
        # neither (a) collide into duplicate column names that break
        # sort_values, nor (b) get treated as sample intensities by the
        # downstream median/binning code.
        _NON_SAMPLE_COLS = {"median"}
        df_arm1 = df_arm1.drop(columns=[c for c in _NON_SAMPLE_COLS
                                         if c in df_arm1.columns])
        df_arm2 = df_arm2.drop(columns=[c for c in _NON_SAMPLE_COLS
                                         if c in df_arm2.columns])
        combined = pd.concat([df_arm1, df_arm2], axis=1)

        self._init_ion2nonNanvals(combined)

        # Pre-compute per-ion per-subject paired log-FCs (test-statistic-scale
        # null source). Replaces the old ion2cellvals as the primary input
        # to the empirical null. ion2cellvals is retained as a diagnostic
        # but is no longer consumed by the null builder.
        self.ion2pairedfcs = self._build_ion2pairedfcs(
            df_arm1, df_arm2, paired_triples
        )
        self.ion2cellvals = self._build_ion2cellvals(combined, paired_triples)

        # Aggregated per-bin warning state (closes reviewer m4).
        self._degenerate_bin_count = 0

        self._select_intensity_ranges(p2z)

        self.all_intensities = np.concatenate(
            list(self.ion2nonNanvals.values())
        ) if self.ion2nonNanvals else np.array([])

        if self._degenerate_bin_count > 0:
            LOGGER.warning(
                "PairedConditionBackgrounds: %d of %d intensity bin(s) were "
                "degenerate (no paired log-FC for any ion in the bin). "
                "Ions in those bins will produce non-informative paired tests.",
                self._degenerate_bin_count, len(self.backgrounds),
            )

    # ────────────────────────────────────────────────────────────────────
    # init helpers (mirror ConditionBackgrounds)
    # ────────────────────────────────────────────────────────────────────
    def _init_ion2nonNanvals(self, combined_df):
        # Sort by median intensity across the combined samples — same
        # ordering ConditionBackgrounds uses.
        # Use a sentinel column name that cannot collide with sample names
        # (the legacy "median" string was vulnerable when an upstream stage
        # left a column also called "median" in the per-arm normalized
        # frames — pandas raised "column label 'median' is not unique").
        df = combined_df.copy()
        _SORT_COL = "__sort_median__"
        df[_SORT_COL] = df.median(numeric_only=True, axis=1)
        df = df.sort_values(by=_SORT_COL).drop(_SORT_COL, axis=1)
        self.normed_condition_df = df
        self.ion2nonNanvals = aqdiffutils.get_non_nas_from_pd_df(df)
        self.ion2allvals = aqdiffutils.get_ionints_from_pd_df(df)
        self.idx2ion = dict(zip(range(len(df.index)), df.index))

    def _build_ion2pairedfcs(self, df_arm1, df_arm2, paired_triples):
        """For each ion, collect per-subject paired log-FCs.

        For each subject with BOTH arms having ≥ 1 detected microdiss values
        for the ion, compute ``mean(arm1 microdiss) − mean(arm2 microdiss)``.
        Subjects missing all values in either arm contribute nothing.

        Returns:
            dict ``{ion: list[float]}``. Empty for ions with no
            fully-observed subject.
        """
        cols_a1 = set(df_arm1.columns)
        cols_a2 = set(df_arm2.columns)

        # Per-subject sample lists, filtered to columns actually present.
        subj_arm_samples: typing.List[
            typing.Tuple[str, typing.List[str], typing.List[str]]
        ] = []
        for subj, c1_samples, c2_samples in paired_triples:
            c1 = [s for s in c1_samples if s in cols_a1]
            c2 = [s for s in c2_samples if s in cols_a2]
            if not c1 or not c2:
                continue
            subj_arm_samples.append((subj, c1, c2))

        ion2pairedfcs: typing.Dict[str, typing.List[float]] = {}
        if not subj_arm_samples:
            LOGGER.warning(
                "PairedConditionBackgrounds: no subject has detected samples "
                "in both arms (column intersection is empty). Paired empirical "
                "null will be empty — paired test cannot calibrate. This "
                "indicates a data-shape mismatch between samplemap pairing "
                "and detected runs."
            )
            return ion2pairedfcs

        # Pull each subject's per-arm sub-frame once; vectorise the per-ion
        # mean computation. Equivalent to per-ion .loc[:, samples].dropna().mean()
        # but ~50× faster on large frames.
        for subj, c1, c2 in subj_arm_samples:
            arm1_sub = df_arm1.loc[:, c1].to_numpy(dtype=float)
            arm2_sub = df_arm2.loc[:, c2].to_numpy(dtype=float)

            # Per-row nan-mean (rows with all-NaN yield NaN; we drop those).
            # `np.nanmean` over an all-NaN slice raises RuntimeWarning; we
            # silence it because we explicitly handle the NaN result below
            # via `np.isfinite(fc)`.
            import warnings as _warnings
            with np.errstate(invalid="ignore"), _warnings.catch_warnings():
                _warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_a1 = np.nanmean(arm1_sub, axis=1)
                mean_a2 = np.nanmean(arm2_sub, axis=1)
            paired_fc = mean_a1 - mean_a2

            for row_idx, ion in enumerate(df_arm1.index):
                fc = paired_fc[row_idx]
                if not np.isfinite(fc):
                    continue
                ion2pairedfcs.setdefault(ion, []).append(float(fc))
        return ion2pairedfcs

    def _build_ion2cellvals(self, combined_df, paired_triples):
        """Diagnostic-only: per-ion within-(subject, arm) cell value arrays.

        Retained for backwards compatibility with the Phase 2 diagnostic.
        Cells with < 2 detected values are dropped. NOT consumed by the null
        builder after the C1 fix.
        """
        cols = set(combined_df.columns)
        cells: typing.List[typing.List[str]] = []
        for subj, c1_samples, c2_samples in paired_triples:
            c1 = [s for s in c1_samples if s in cols]
            c2 = [s for s in c2_samples if s in cols]
            if len(c1) >= 2:
                cells.append(c1)
            if len(c2) >= 2:
                cells.append(c2)

        ion2cellvals: typing.Dict[str, typing.List[np.ndarray]] = {}
        if not cells:
            return ion2cellvals

        for cell_samples in cells:
            cell_df = combined_df.loc[:, cell_samples]
            cell_arr = cell_df.to_numpy(dtype=float)
            for row_idx, ion in enumerate(combined_df.index):
                row = cell_arr[row_idx]
                row = row[~np.isnan(row)]
                if row.size < 2:
                    continue
                ion2cellvals.setdefault(ion, []).append(row)
        return ion2cellvals

    # ────────────────────────────────────────────────────────────────────
    # bin selection (mirrors ConditionBackgrounds.select_intensity_ranges)
    # ────────────────────────────────────────────────────────────────────
    def _select_intensity_ranges(self, p2z):
        total_available_comparisons = 0
        cumulative_counts = np.zeros(self.normed_condition_df.shape[0])
        num_contexts = 10

        for idx, count in enumerate(self.normed_condition_df.count(axis=1)):
            total_available_comparisons += count - 1
            cumulative_counts[idx] = int(total_available_comparisons / 2)

        if total_available_comparisons <= 0:
            LOGGER.warning(
                "PairedConditionBackgrounds: no comparisons available (combined "
                "frame has no multi-value rows). No background bins built."
            )
            return

        context_size = np.max(
            [1000, int(total_available_comparisons / (1 + num_contexts / 2))]
        )
        if context_size > total_available_comparisons:
            context_size = int(total_available_comparisons / 2)
        halfcontext_size = int(context_size / 2)
        context_boundaries = np.zeros(3).astype(int)

        middle_idx = int(np.searchsorted(cumulative_counts, halfcontext_size))
        end_idx = int(np.searchsorted(cumulative_counts, context_size))

        context_boundaries[0] = 0
        context_boundaries[1] = middle_idx
        context_boundaries[2] = end_idx
        while context_boundaries[1] < len(cumulative_counts):
            bgdist = PairedBackGroundDistribution(
                context_boundaries[0],
                context_boundaries[2],
                self.ion2nonNanvals,
                self.idx2ion,
                p2z,
                ion2pairedfcs=self.ion2pairedfcs,
            )
            if bgdist.SD == 0.0:
                self._degenerate_bin_count += 1
            self.context_ranges.append(
                [int(context_boundaries[0]), int(context_boundaries[2])]
            )
            self._assign_ions2bgdists(
                int(context_boundaries[0]),
                int(context_boundaries[2]),
                bgdist,
            )
            self.backgrounds.append(bgdist)
            context_boundaries[0] = context_boundaries[1]
            context_boundaries[1] = context_boundaries[2]
            end_idx = np.searchsorted(
                cumulative_counts,
                context_size + cumulative_counts[context_boundaries[0]],
            )
            if end_idx > len(cumulative_counts) - (
                context_boundaries[1] - context_boundaries[0]
            ) / 1.5:
                end_idx = len(cumulative_counts)
            context_boundaries[2] = end_idx

    def _assign_ions2bgdists(self, boundaries1, boundaries2, bgdist):
        for idx in range(boundaries1, boundaries2):
            self.ion2background[self.idx2ion.get(idx)] = bgdist

    # ────────────────────────────────────────────────────────────────────
    # public lookup
    # ────────────────────────────────────────────────────────────────────
    def get_paired_diff_dist(
        self, ion
    ) -> typing.Optional[PairedBackGroundDistribution]:
        """Return the bin-level paired empirical null for this ion, or None
        if the ion was not assigned a bin (e.g., dropped during sorting,
        not in the combined frame, or ``_select_intensity_ranges`` produced
        no bins because the data is too sparse).
        """
        return self.ion2background.get(ion)
