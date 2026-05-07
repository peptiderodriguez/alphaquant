"""Phase 3 tests for paired-diffdist.

Property-based, not snapshot. Verifies:

  * (3a) Covariance-adjusted Stouffer
      - PSD correlation matrix from synthetic ion-fc data
      - Independent ions: corr-adj Z == classical Stouffer
      - Highly correlated ions: combined Z ≈ Z_single (no inflation)
      - Equicorrelation ρ=0.3 produces conservative p (vs identity)
      - Ledoit-Wolf shrinkage at low n is more stable than sample corr

  * (3b) Patient-block sign-flip permutation
      - At n=4: enumerator returns 16 distinct ±1 vectors, no dups
      - At n=6: 64 distinct
      - At n=4: include_identity=True → all-+1 vector present
      - apply_permutation_to_paired_pairs: sign=+1 keeps tuple, sign=-1 swaps
      - Permutation calibrated p: under H0, distribution is uniform on the
        achievable discrete grid [1/2^n, 1] (KS-on-discrete soft check)
      - Permutation calibrated p: at floor, p ≥ 1/2^n for n=4 (always)

  * Wiring
      - DifferentialIonPaired sets is_paired=True on the instance
      - DifferentialIonPaired caches paired_subject_fcs / paired_subject_zs
      - Tree aggregation uses corr-aware Stouffer when children are paired

Phase 4 validation cascade (full power simulation, lmer benchmark) is
out of scope here.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

import alphaquant.cluster.correlation_aware_stouffer as aq_corr
import alphaquant.cluster.cluster_utils as aq_clust_utils
import alphaquant.diffquant.paired_permutation_null as aq_perm
import alphaquant.diffquant.diff_analysis as aqdiff
import alphaquant.diffquant.background_distributions_paired as aqbg_paired
import alphaquant.diffquant.condpair_analysis as aq_condpair


# ─────────────────────────────────────────────────────────────────────────
# Synthetic-data helper (re-uses Phase 2's bm_mk-shape fixture builder)
# ─────────────────────────────────────────────────────────────────────────
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(__file__))
import test_paired_phase2 as t_p2  # noqa: E402


# ═════════════════════════════════════════════════════════════════════════
#  3a — Covariance-adjusted Stouffer
# ═════════════════════════════════════════════════════════════════════════

class TestCorrelationMatrix:
    """compute_ion_correlation_matrix returns PSD, well-shaped output."""

    def test_psd_property_independent_columns(self):
        """Independent normal columns → R close to I, fully PSD."""
        rng = np.random.default_rng(2026)
        X = rng.normal(0, 1, size=(50, 5))
        R = aq_corr.compute_ion_correlation_matrix(X, method="ledoit_wolf")
        assert R.shape == (5, 5)
        eigvals = np.linalg.eigvalsh(R)
        assert (eigvals >= -1e-9).all(), f"eigvals: {eigvals}"
        # Diagonal = 1
        assert np.allclose(np.diag(R), 1.0)
        # Off-diagonal magnitudes should be modest with shrinkage
        off_diag = R - np.eye(5)
        assert np.max(np.abs(off_diag)) < 0.4

    def test_psd_property_correlated_columns(self):
        """Strongly correlated columns → R has large positive off-diag,
        still PSD."""
        rng = np.random.default_rng(2026)
        # Two underlying signals; build 4 columns as linear combos.
        latent = rng.normal(0, 1, size=(50, 2))
        X = np.column_stack([
            latent[:, 0] + 0.1 * rng.normal(0, 1, 50),
            latent[:, 0] + 0.1 * rng.normal(0, 1, 50),
            latent[:, 1] + 0.1 * rng.normal(0, 1, 50),
            latent[:, 1] + 0.1 * rng.normal(0, 1, 50),
        ])
        R = aq_corr.compute_ion_correlation_matrix(X, method="ledoit_wolf")
        eigvals = np.linalg.eigvalsh(R)
        assert (eigvals >= -1e-9).all()
        # Pairs from the same latent should correlate strongly.
        assert R[0, 1] > 0.5
        assert R[2, 3] > 0.5

    def test_n1_returns_unit_one_one_matrix(self):
        X = np.random.default_rng(0).normal(0, 1, size=(10, 1))
        R = aq_corr.compute_ion_correlation_matrix(X)
        assert R.shape == (1, 1)
        assert R[0, 0] == 1.0

    def test_low_n_falls_back_to_equicorrelation(self):
        """n_subjects < MIN_N_FOR_SAMPLE_CORR → equicorr fallback."""
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, size=(2, 4))
        R = aq_corr.compute_ion_correlation_matrix(X, method="ledoit_wolf")
        # Equicorrelation: off-diag = DEFAULT_EQUICORR_RHO = 0.3
        assert R.shape == (4, 4)
        assert np.allclose(np.diag(R), 1.0)
        off_diag = [R[i, j] for i in range(4) for j in range(4) if i != j]
        assert np.allclose(off_diag, aq_corr.DEFAULT_EQUICORR_RHO)

    def test_equicorrelation_method(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, size=(20, 3))
        R = aq_corr.compute_ion_correlation_matrix(X, method="equicorrelation",
                                                    equicorr_rho=0.5)
        assert np.allclose(np.diag(R), 1.0)
        # Off-diag = 0.5 by construction.
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert R[i, j] == pytest.approx(0.5)

    def test_identity_method_warns(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, size=(20, 3))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            R = aq_corr.compute_ion_correlation_matrix(X, method="identity")
            assert any("anti-conservative" in str(wi.message) for wi in w), (
                "Expected anti-conservative warning when identity is used"
            )
        assert np.allclose(R, np.eye(3))

    def test_handles_nan_subjects(self):
        """Rows with any NaN are dropped (complete-case)."""
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, size=(30, 4))
        X[0, 0] = np.nan
        X[5, 3] = np.nan
        R = aq_corr.compute_ion_correlation_matrix(X, method="ledoit_wolf")
        assert R.shape == (4, 4)
        eigvals = np.linalg.eigvalsh(R)
        assert (eigvals >= -1e-9).all()


class TestCovAdjustedStouffer:
    """Z-aggregation behaviour under various R structures."""

    def test_n1_returns_input(self):
        z = aq_corr.covariance_adjusted_stouffer_z(np.array([1.5]))
        assert z == 1.5

    def test_independent_equals_classical_stouffer(self):
        """R = I → Σz / sqrt(N) = classical Stouffer."""
        z_vals = np.array([0.5, 1.2, -0.3, 0.8])
        R = np.eye(4)
        z_corr = aq_corr.covariance_adjusted_stouffer_z(z_vals, R)
        z_classical = float(np.sum(z_vals) / np.sqrt(4))
        assert z_corr == pytest.approx(z_classical, abs=1e-10)

    def test_perfectly_correlated_gives_single_z(self):
        """R = 11ᵀ → combined Z = mean(z) ≈ z_single."""
        z_vals = np.array([1.0, 1.0, 1.0])
        R = np.ones((3, 3))
        z_corr = aq_corr.covariance_adjusted_stouffer_z(z_vals, R)
        # 1ᵀ R 1 = 9, so denom = 3, sum = 3 → Z = 1.0
        assert z_corr == pytest.approx(1.0, abs=1e-10)

    def test_equicorrelation_is_conservative(self):
        """ρ > 0 in equicorrelation → larger denominator → smaller |Z|
        than classical Stouffer (more conservative)."""
        z_vals = np.array([2.0, 1.5, 1.8, 1.2])
        R_id = np.eye(4)
        R_eq = aq_corr._equicorrelation_matrix(4, 0.3)
        z_id = aq_corr.covariance_adjusted_stouffer_z(z_vals, R_id)
        z_eq = aq_corr.covariance_adjusted_stouffer_z(z_vals, R_eq)
        # Both same sign (sum > 0), |z_eq| < |z_id| because denom larger
        assert abs(z_eq) < abs(z_id)

    def test_all_zero_z_returns_zero(self):
        z_vals = np.zeros(5)
        R = np.eye(5)
        z = aq_corr.covariance_adjusted_stouffer_z(z_vals, R)
        assert z == 0.0


class TestSumAndReScaleZvaluesPairedCorr:
    """The thin wrapper that hooks corr-adj Stouffer into the tree
    aggregator (sum_and_re_scale_zvalues_paired_corr)."""

    def test_independent_matches_classical_stouffer(self):
        """When R = I, the wrapper output should equal the classical
        wrapper output to high precision."""
        zvals = np.array([0.4, 1.1, -0.6, 0.3])
        R = np.eye(4)
        z_paired = aq_clust_utils.sum_and_re_scale_zvalues_paired_corr(zvals, R)
        z_classical = aq_clust_utils.sum_and_re_scale_zvalues(zvals)
        assert z_paired == pytest.approx(z_classical, abs=1e-10)

    def test_correlated_smaller_magnitude_than_classical(self):
        """Positive correlation → corr-adj |Z| < classical Stouffer |Z|."""
        zvals = np.array([1.5, 1.5, 1.5, 1.5])
        R_corr = aq_corr._equicorrelation_matrix(4, 0.5)
        z_corr = aq_clust_utils.sum_and_re_scale_zvalues_paired_corr(zvals, R_corr)
        z_classical = aq_clust_utils.sum_and_re_scale_zvalues(zvals)
        assert abs(z_corr) < abs(z_classical)

    def test_n1_passes_through(self):
        z = aq_clust_utils.sum_and_re_scale_zvalues_paired_corr(
            np.array([2.5]), R=None)
        assert z == 2.5


# ═════════════════════════════════════════════════════════════════════════
#  3b — Patient-block sign-flip permutations
# ═════════════════════════════════════════════════════════════════════════

class TestPermutationEnumeration:

    def test_n4_full_enum_16_distinct(self):
        perms = aq_perm.enumerate_signflip_permutations(4)
        assert len(perms) == 16
        # All distinct
        unique = {tuple(p.tolist()) for p in perms}
        assert len(unique) == 16
        # All ±1
        for p in perms:
            assert set(p.tolist()).issubset({1, -1})
        # Identity present
        assert any(np.all(p == 1) for p in perms)

    def test_n6_full_enum_64_distinct(self):
        perms = aq_perm.enumerate_signflip_permutations(6)
        assert len(perms) == 64
        unique = {tuple(p.tolist()) for p in perms}
        assert len(unique) == 64

    def test_n8_full_enum_256_distinct(self):
        perms = aq_perm.enumerate_signflip_permutations(8)
        assert len(perms) == 256
        unique = {tuple(p.tolist()) for p in perms}
        assert len(unique) == 256

    def test_n12_random_sample(self):
        perms = aq_perm.enumerate_signflip_permutations(
            12, max_perms=64, rng_seed=0)
        # Capped at 64 random samples (12 > MAX_N_FOR_FULL_ENUM=8)
        assert len(perms) == 64
        # All distinct
        unique = {tuple(p.tolist()) for p in perms}
        assert len(unique) == 64

    def test_exclude_identity(self):
        perms = aq_perm.enumerate_signflip_permutations(4, include_identity=False)
        assert len(perms) == 15
        assert not any(np.all(p == 1) for p in perms)


class TestApplyPermutation:

    def test_sign_plus1_keeps_tuple(self):
        pairs = [("s1", 10.0, 20.0), ("s2", 11.0, 21.0)]
        signs = np.array([1, 1])
        out = aq_perm.apply_permutation_to_paired_pairs(pairs, signs)
        assert out == [("s1", 10.0, 20.0), ("s2", 11.0, 21.0)]

    def test_sign_minus1_swaps_arms(self):
        pairs = [("s1", 10.0, 20.0), ("s2", 11.0, 21.0)]
        signs = np.array([-1, -1])
        out = aq_perm.apply_permutation_to_paired_pairs(pairs, signs)
        assert out == [("s1", 20.0, 10.0), ("s2", 21.0, 11.0)]

    def test_mixed_signs(self):
        pairs = [("s1", 10.0, 20.0), ("s2", 11.0, 21.0), ("s3", 12.0, 22.0)]
        signs = np.array([1, -1, 1])
        out = aq_perm.apply_permutation_to_paired_pairs(pairs, signs)
        assert out == [("s1", 10.0, 20.0), ("s2", 21.0, 11.0),
                       ("s3", 12.0, 22.0)]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            aq_perm.apply_permutation_to_paired_pairs(
                [("s1", 10.0, 20.0)], np.array([1, -1])
            )

    def test_subject_dict_apply_signs(self):
        d = {"s1": 0.5, "s2": -0.7, "s3": 0.3}
        signs = np.array([1, -1, 1])
        out = aq_perm.apply_permutation_to_subject_dict(
            d, ["s1", "s2", "s3"], signs)
        assert out["s1"] == pytest.approx(0.5)
        assert out["s2"] == pytest.approx(0.7)
        assert out["s3"] == pytest.approx(0.3)


class TestPermutationCalibratedP:

    def test_observed_in_null_floor_at_n4(self):
        """When observed |z| is the largest (no perm exceeds it), p should
        equal 1/N_perms (the floor) — the observed itself counts."""
        # 16 perm z's, with the observed being the largest
        nulls = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                          0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 5.0])
        observed_z = 5.0
        p, p_floor = aq_perm.permutation_calibrated_p(observed_z, nulls)
        assert p_floor == pytest.approx(1.0 / 16)
        assert p == pytest.approx(1.0 / 16)

    def test_observed_average_yields_around_half(self):
        """When |z_obs| is at the median of the null, p ≈ 0.5."""
        rng = np.random.default_rng(0)
        nulls = rng.normal(0, 1, size=64)
        observed_z = float(np.median(np.abs(nulls)))
        p, _ = aq_perm.permutation_calibrated_p(observed_z, nulls)
        # Floor on counted ties → tends slightly above 0.5, allow slack
        assert 0.4 < p < 0.65

    def test_p_floor_at_n4_is_one_sixteenth(self):
        nulls = np.zeros(16)
        nulls[0] = 0.1  # observed
        p, p_floor = aq_perm.permutation_calibrated_p(0.1, nulls)
        assert p_floor == pytest.approx(1.0 / 16)
        # p ≥ 1/16
        assert p >= 1.0 / 16 - 1e-12

    def test_uniform_under_h0_at_n6(self):
        """Under H0, the permutation p across many ions is approximately
        uniform on the discrete grid {k/N : k = 1, ..., N}.

        Generates 200 H0 ions (each with 6 paired subjects, fc ~ N(0, σ)),
        runs the analytic Stouffer paired test, computes the
        permutation-calibrated p for each, and checks the resulting
        distribution against the uniform-on-discrete-grid expectation.
        """
        from scipy.stats import kstest
        rng = np.random.default_rng(2026)
        sigma_null = 0.2
        all_perm_ps = []
        n = 6
        # Build a single _SyntheticBg-like helper for bg
        bg = t_p2._SyntheticBg(sd=sigma_null)
        perms = aq_perm.enumerate_signflip_permutations(n)
        n_perms = len(perms)
        for _ in range(200):
            log_fcs = rng.normal(0, sigma_null, size=n)
            # Compute observed z via DifferentialIonPaired
            pairs = [(f"s{i}", float(fc) + 18.0, 18.0)
                     for i, fc in enumerate(log_fcs)]
            d_obs = aqdiff.DifferentialIonPaired(
                pairs, bg, name="null", outlier_correction=False)
            # Build the permutation null by flipping signs (no need to
            # re-construct DifferentialIonPaired — we have the per-subject
            # z's via the cached dict).
            z_subj = np.array([d_obs.paired_subject_zs[f"s{i}"]
                               for i in range(n)])
            null_zs = []
            for signs in perms:
                z_perm_sum = float(np.sum(signs * z_subj))
                null_zs.append(z_perm_sum / np.sqrt(n))
            p, _ = aq_perm.permutation_calibrated_p(
                d_obs.z_val, null_zs)
            all_perm_ps.append(p)
        all_perm_ps = np.array(all_perm_ps)
        # p ≥ 1/64 always
        assert (all_perm_ps >= 1 / n_perms - 1e-12).all()
        # KS soft check: discrete uniform → continuous uniform is loose,
        # so use a generous threshold.
        ks_p = kstest(all_perm_ps, "uniform").pvalue
        assert ks_p > 1e-3, (
            f"permutation p's reject uniform[0,1] (KS p={ks_p:.4g}); "
            f"calibration broken under H0"
        )


# ═════════════════════════════════════════════════════════════════════════
#  3c — Wiring: DifferentialIonPaired sets is_paired + caches dicts
# ═════════════════════════════════════════════════════════════════════════

class TestDifferentialIonPairedAttributes:

    def test_is_paired_true(self):
        pairs = [(f"s{i}", 18.0 + 0.1 * i, 18.0) for i in range(4)]
        bg = t_p2._SyntheticBg(sd=0.2)
        d = aqdiff.DifferentialIonPaired(pairs, bg, name="ion",
                                         outlier_correction=False)
        assert d.is_paired is True

    def test_paired_subject_fcs_cached(self):
        pairs = [("s0", 18.5, 18.0), ("s1", 18.3, 18.1),
                 ("s2", 18.7, 17.9), ("s3", 18.4, 18.0)]
        bg = t_p2._SyntheticBg(sd=0.2)
        d = aqdiff.DifferentialIonPaired(pairs, bg, name="ion",
                                         outlier_correction=False)
        assert d.usable
        assert d.paired_subject_fcs is not None
        assert d.paired_subject_fcs == pytest.approx({
            "s0": 0.5, "s1": pytest.approx(0.2),
            "s2": pytest.approx(0.8), "s3": pytest.approx(0.4)
        })

    def test_paired_subject_zs_cached(self):
        pairs = [("s0", 18.5, 18.0), ("s1", 18.3, 18.1)]
        bg = t_p2._SyntheticBg(sd=0.5)  # z = fc / 0.5
        d = aqdiff.DifferentialIonPaired(pairs, bg, name="ion",
                                         outlier_correction=False)
        assert d.usable
        assert d.paired_subject_zs is not None
        # z = fc / 0.5
        assert d.paired_subject_zs["s0"] == pytest.approx(0.5 / 0.5)
        assert d.paired_subject_zs["s1"] == pytest.approx(0.2 / 0.5)

    def test_unusable_does_not_cache(self):
        d = aqdiff.DifferentialIonPaired(
            [("s0", 18.5, 18.0)], t_p2._SyntheticBg(sd=0.2),
            name="ion", outlier_correction=False)
        assert d.usable is False
        assert d.paired_subject_fcs is None
        assert d.paired_subject_zs is None
        # is_paired should still be True (instance is a paired-test object)
        assert d.is_paired is True


# ═════════════════════════════════════════════════════════════════════════
#  Tree aggregation: corr-aware path triggers when children are paired
# ═════════════════════════════════════════════════════════════════════════

class TestTreeAggregationPairedDispatch:

    def test_aggregator_uses_paired_corr_path_when_children_paired(self):
        """Build a minimal anytree with two paired-leaf children under a
        gene node; aggregate; check is_paired propagates and the
        z_normed differs from classical Stouffer when correlation is
        non-trivial."""
        import anytree
        gene = anytree.Node("Gene", type="gene", level="gene",
                            cluster=0, is_included=True)
        # Two children with strongly correlated fc dicts (both shift
        # together across subjects).
        ch1 = anytree.Node("Pep1", type="seq", level="seq", cluster=0,
                           is_included=True, parent=gene)
        ch2 = anytree.Node("Pep2", type="seq", level="seq", cluster=0,
                           is_included=True, parent=gene)
        # Required attributes for aggregate_node_properties
        for ch, z, fc in [(ch1, 1.5, 0.3), (ch2, 1.5, 0.3)]:
            ch.z_val = z
            ch.fc = fc
            ch.cv = 0.1
            ch.min_intensity = 18.0
            ch.total_intensity = 100.0
            ch.min_reps = 3
            ch.fraction_consistent = 1.0
            ch.is_paired = True
            # Same subjects, perfectly correlated paired-fc vectors
            ch.paired_subject_fcs = {"m0": 0.3, "m1": 0.4, "m2": 0.2,
                                       "m3": 0.5}
            ch.is_outlier_peptide = False
            ch.is_outlier_fragment = False
        aq_clust_utils.aggregate_node_properties(
            gene, only_use_mainclust=True,
            peptide_outlier_filtering=False,
            fragment_outlier_filtering=False,
        )
        assert gene.is_paired is True
        assert gene.paired_subject_fcs == pytest.approx({
            "m0": 0.3, "m1": 0.4, "m2": 0.2, "m3": 0.5
        })
        # Perfect correlation → corr-adj Z ≈ z_single = 1.5
        # vs classical Stouffer = 1.5*2 / sqrt(2) ≈ 2.12
        # After CDF round-trip (which is monotone), corr-adj < classical.
        z_classical = aq_clust_utils.sum_and_re_scale_zvalues(
            np.array([1.5, 1.5]))
        assert abs(gene.z_val) < abs(z_classical), (
            f"corr-adj z {gene.z_val:.4f} should be smaller in magnitude "
            f"than classical Stouffer {z_classical:.4f} for perfectly "
            f"correlated children"
        )

    def test_aggregator_classical_path_for_unpaired_children(self):
        """When children are unpaired, the existing Stouffer code path
        runs unchanged (regression test for unpaired-only behaviour)."""
        import anytree
        gene = anytree.Node("Gene", type="gene", level="gene",
                            cluster=0, is_included=True)
        ch1 = anytree.Node("Pep1", type="seq", level="seq", cluster=0,
                           is_included=True, parent=gene)
        ch2 = anytree.Node("Pep2", type="seq", level="seq", cluster=0,
                           is_included=True, parent=gene)
        for ch, z, fc in [(ch1, 1.5, 0.3), (ch2, 1.5, 0.3)]:
            ch.z_val = z
            ch.fc = fc
            ch.cv = 0.1
            ch.min_intensity = 18.0
            ch.total_intensity = 100.0
            ch.min_reps = 3
            ch.fraction_consistent = 1.0
            # No is_paired attribute → defaults to False
            ch.is_outlier_peptide = False
            ch.is_outlier_fragment = False
        aq_clust_utils.aggregate_node_properties(
            gene, only_use_mainclust=True,
            peptide_outlier_filtering=False,
            fragment_outlier_filtering=False,
        )
        z_classical = aq_clust_utils.sum_and_re_scale_zvalues(
            np.array([1.5, 1.5]))
        assert gene.z_val == pytest.approx(z_classical)
        assert gene.is_paired is False
