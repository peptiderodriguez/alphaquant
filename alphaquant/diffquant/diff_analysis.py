
from statistics import NormalDist
import typing
import numpy as np
import math
import statistics
from scipy.stats import ttest_ind
from scipy.stats import t as student_t
import alphaquant.diffquant.diffutils as aqdiffutils

class DifferentialIon():
    """Computes differential statistics for an ion using empirical background distributions.

    This is the default statistical test in AlphaQuant. It uses intensity-dependent empirical
    background distributions to compute p-values and z-scores for each ion. The method accounts
    for technical variation by comparing observed fold changes against distributions derived from
    similarly abundant ions in the dataset.

    Args:
        noNanvals_from: Array of log2 intensities from condition 1 (no NaN values)
        noNanvals_to: Array of log2 intensities from condition 2 (no NaN values)
        diffDist: BackGroundDistribution object containing empirical error distribution
        name: Ion identifier (e.g., peptide sequence + charge state)
        outlier_correction: If True, inflates variance estimates when replicates show
                          unusually high variability (default: True)

    Attributes:
        name: Ion identifier
        p_val: Two-sided p-value testing null hypothesis of no change
        fc: Log2 fold change (condition1 - condition2)
        z_val: Z-score (signed, magnitude indicates significance)
        usable: Boolean indicating whether statistics could be computed
    """

    def __init__(self,noNanvals_from, noNanvals_to, diffDist, name, outlier_correction = True):


        self.name = name
        self.p_val = None
        self.fc = None
        self.z_val = None
        self.usable = False

        self._calc_diffreg_peptide(noNanvals_from, noNanvals_to, diffDist, outlier_correction)


    def _calc_diffreg_peptide(self, noNanvals_from, noNanvals_to, diffDist, outlier_correction):
        """Calculates differential expression statistics using empirical background distributions.

        This method computes all pairwise fold changes between replicates of the two conditions,
        converts them to z-scores using the empirical background distribution, and aggregates
        them into a single p-value. The variance is scaled based on the number of replicates
        and optionally adjusted for outliers.

        Args:
            noNanvals_from: Log2 intensities from condition 1
            noNanvals_to: Log2 intensities from condition 2
            diffDist: Empirical background distribution for variance estimation
            outlier_correction: Whether to apply robust variance inflation
        """

        nrep_from = len(noNanvals_from)
        nrep_to = len(noNanvals_to)

        if ((nrep_from==0) or (nrep_to ==0)):
            return
        var_from = diffDist.var_from
        var_to = diffDist.var_to

        perEvidenceVariance = diffDist.var + (nrep_to-1) * var_from + (nrep_from-1) * var_to
        totalVariance = perEvidenceVariance*nrep_to * nrep_from
        outlier_scaling_factor = 1.0
        if outlier_correction:
            outlier_scaling_factor = calc_outlier_scaling_factor(noNanvals_from, noNanvals_to, diffDist)

        fc_sum =0
        z_sum=0
        unscaled_zs = []
        for from_intens in noNanvals_from:
            for to_intens in noNanvals_to:
                fc = from_intens - to_intens
                fc_sum+=fc
                z_unscaled = diffDist.calc_zscore_from_fc(fc)
                unscaled_zs.append(z_unscaled)
                z_sum += z_unscaled/outlier_scaling_factor

        scaled_SD =  math.sqrt(totalVariance/diffDist.var)*outlier_scaling_factor

        self.fc = fc_sum/(nrep_from * nrep_to)
        self.p_val = 2.0 * (1.0 -  NormalDist(mu=0, sigma= scaled_SD).cdf(abs(z_sum)))
        self.z_val = z_sum/scaled_SD
        self.usable = True




  #self.var_from = from_dist.var
   #     self.var_to

class DifferentialIonPaired():
    """Paired-design ion-level test for within-subject contrasts (Phase 2).

    Compares within-subject log-FCs (e.g. log2(femur/humerus) per mouse for
    bm_mk's bone axis) against an empirical null built from within-(subject,
    arm) cell log-FCs by ``PairedConditionBackgrounds``. Removes between-
    subject variance from the test denominator. See
    ``docs/paired_diffdist_design.md`` for the full statistical design.

    Statistical mechanism (Phase 2 — classical Stouffer; Phase 3 will add
    cross-ion covariance correction and permutation calibration):
        1. Compute paired log-FC per subject: ``fc_i = val_arm1_i - val_arm2_i``.
        2. Look up per-subject z-score from the paired empirical null:
           ``z_i = paired_diff_dist.calc_zscore_from_fc(fc_i)``.
        3. Combine: ``Z_combined = sum(z_i) / sqrt(N) / outlier_scaling``,
           where N is the number of paired subjects and outlier_scaling
           inflates the SD if any single |z_i| is implausibly large vs the
           bin's empirical SD (capped-z heuristic, see below).
        4. Two-sided p-value: ``p = 2 * (1 - Phi(|Z_combined|))``, here
           expressed equivalently as
           ``p = 2 * (1 - N(0, sigma=sqrt(N)*outlier_scaling).cdf(|Σz|))``.

    Outlier correction (when ``outlier_correction=True``): the cap is
    applied multiplicatively on z-scale. If the largest ``|z_i|`` exceeds
    ``_OUTLIER_CAP_K × median |z_i|``, ``outlier_scaling_factor`` is set so
    that the offending ``|z_i|`` would be capped at ``_OUTLIER_CAP_K × median``.
    This is the paired-test analog of ``calc_outlier_scaling_factor``: it
    inflates the denominator when one subject's paired log-FC is wildly
    extreme, making the test conservative when one subject contributes an
    implausibly large per-subject z.

    Note on ``paired_diff_dist.SD`` (closes reviewer C2): the bin's ``SD``
    attribute is the sample SD of the empirical null on the
    **fold-change scale** (units of log2 intensity), NOT the unit-normal
    z-scale. Under H0, ``calc_zscore_from_fc(fc)`` returns a rank-based z
    that IS unit-normal-scaled by construction (empirical CDF →
    standard-normal quantile). So the cap on ``|z_i|`` is on the z-scale
    and ``paired_diff_dist.SD`` is unrelated to the cap arithmetic.

    Args:
        paired_subject_pairs: list of ``(subject_id, val_arm1, val_arm2)``
            tuples. Each tuple is one paired within-subject observation:
            ``val_arm1`` and ``val_arm2`` are scalars (mean of microdiss
            log2-intensities within (subject, arm) cell, computed by the
            caller). Subjects missing one arm for this ion are excluded
            upstream — they don't appear in this list.
        paired_diff_dist: ``PairedBackGroundDistribution`` — empirical null
            built from within-(subject, arm) cell log-FCs. May be None
            (e.g., empty bin or sparse data) — in that case the test is
            unusable.
        name: ion identifier (e.g. peptide sequence + charge state)
        outlier_correction: capped-z outlier correction (default True).

    Attributes:
        name: ion identifier
        p_val: two-sided paired p-value (None if not usable)
        fc: mean within-subject log-FC (None if not usable)
        z_val: signed paired z-score (None if not usable)
        n_paired_subjects: number of paired observations
            (= ``len(paired_subject_pairs)``)
        usable: True iff a paired test could be computed (requires
            ``len(paired_subject_pairs) >= 2`` and a non-None
            ``paired_diff_dist``).
    """

    # Outlier-cap multiplier on |z_i| relative to the median |z_i|. Picked
    # to match the spirit of ``calc_outlier_scaling_factor`` (which caps at
    # the highest between-rep SD vs the bg SD). 3.0 is a soft cap;
    # adjustable per future analysis without touching the call site.
    _OUTLIER_CAP_K = 3.0

    def __init__(self, paired_subject_pairs, paired_diff_dist, name,
                 outlier_correction=True):
        # Always set attribute defaults first — defensive against any
        # caller that wraps construction in try/except and inspects attrs.
        self.name = name
        self.p_val = None
        self.fc = None
        self.z_val = None
        self.n_paired_subjects = (
            len(paired_subject_pairs) if paired_subject_pairs is not None else 0)
        # Phase 3: tag all paired ions so the tree aggregator (and
        # downstream output writer) can detect "this protein was tested
        # with the paired pipeline" and switch to covariance-adjusted
        # Stouffer + permutation calibration. The attribute is also
        # propagated to base-ion nodes via
        # ``cluster_utils.assign_properties_to_base_ions`` so it survives
        # the diffion → tree-leaf transition.
        self.is_paired = True
        # Phase 3: store the per-subject paired log-FC dict so the tree
        # aggregator can compute the within-protein ion-z correlation
        # matrix (Phase 3a) and apply sign-flip permutations (Phase 3b)
        # without re-doing the per-ion microdiss collapse. None when
        # unusable. Keys are subject_id strings.
        self.paired_subject_fcs: typing.Optional[typing.Dict[str, float]] = None
        # Per-subject z-scores (after empirical-null lookup), same keys
        # as paired_subject_fcs. Used by the permutation calibrator to
        # avoid re-doing the rank lookup for each permutation (the
        # lookup is invariant under sign-flip up to sign).
        self.paired_subject_zs: typing.Optional[typing.Dict[str, float]] = None
        self.usable = False

        if paired_subject_pairs is None or len(paired_subject_pairs) < 2:
            # n=1 is a point estimate — paired test is not meaningful at
            # n=1 (z and SD coincide; ratio is degenerate). Mark unusable
            # and exit cleanly. Caller will route into the
            # missingval-style accumulator.
            return
        if paired_diff_dist is None:
            return

        self._calc_paired_diffreg(
            paired_subject_pairs, paired_diff_dist, outlier_correction
        )

    def _calc_paired_diffreg(
        self, paired_subject_pairs, paired_diff_dist, outlier_correction
    ):
        """Compute paired Stouffer Z + p-value with capped-z outlier control."""
        subject_ids = [s for (s, _, _) in paired_subject_pairs]
        log_fcs = np.array(
            [val_arm1 - val_arm2 for (_, val_arm1, val_arm2) in paired_subject_pairs],
            dtype=float,
        )
        if not np.all(np.isfinite(log_fcs)):
            # Defensive: caller should have dropped non-finite entries, but
            # if any survive we can't run the test reliably.
            return

        unscaled_zs = np.array(
            [paired_diff_dist.calc_zscore_from_fc(float(fc)) for fc in log_fcs],
            dtype=float,
        )

        N = len(unscaled_zs)
        outlier_scaling_factor = 1.0
        if outlier_correction:
            outlier_scaling_factor = self._calc_paired_outlier_scaling(unscaled_zs)

        z_sum = float(np.sum(unscaled_zs))
        # Under H0, the N independent z-scores have variance N (each ~ N(0,1)
        # by empirical-null calibration). Stouffer denominator = sqrt(N).
        # outlier_scaling_factor inflates the denominator (more conservative).
        scaled_SD = math.sqrt(N) * outlier_scaling_factor

        self.fc = float(np.mean(log_fcs))
        # Two-sided p: |z_sum| / scaled_SD compared to N(0, 1), equivalently
        # |z_sum| compared to N(0, sigma=scaled_SD).
        self.p_val = 2.0 * (
            1.0 - NormalDist(mu=0, sigma=scaled_SD).cdf(abs(z_sum))
        )
        self.z_val = z_sum / scaled_SD
        # Phase 3: cache per-subject {subject: paired_fc} and
        # {subject: z} dicts for the tree aggregator (correlation matrix
        # construction) and the permutation calibrator (sign-flip
        # recomputation without re-doing the empirical-null lookup).
        self.paired_subject_fcs = {
            subj: float(fc) for subj, fc in zip(subject_ids, log_fcs)
        }
        self.paired_subject_zs = {
            subj: float(z) for subj, z in zip(subject_ids, unscaled_zs)
        }
        self.usable = True

    # Floor for the "typical |z|" reference when the median is zero
    # (e.g., majority of subjects fall in the small-fc zone where the
    # lookup returns 0). Closes reviewer M1: a zero median used to short-
    # circuit outlier correction even when one or more z's were huge.
    _MEDIAN_ZERO_FLOOR = 0.5

    @classmethod
    def _calc_paired_outlier_scaling(cls, unscaled_zs):
        """Capped-z outlier scaling.

        If the maximum absolute z-score is much larger than the typical
        absolute z-score, inflate the denominator so the offending z would
        be capped at ``_OUTLIER_CAP_K × typical |z|``. Returns 1.0 (no
        inflation) when N < 3 or when no individual z is extreme.

        Reference for "typical": median |z|, falling back to mean |z| when
        the median is zero (closes reviewer M1 — without the fallback, a
        zero median collapsed the cap to 0 and the function returned 1.0
        even when an outlier was extreme), and finally to a small floor
        ``_MEDIAN_ZERO_FLOOR`` when both are zero (e.g., a single huge
        outlier dominates and the rest are exactly at the lookup's
        small-fc-zero zone).

        This is the paired-test analog of ``calc_outlier_scaling_factor``
        — it makes the test conservative when one subject contributes an
        unrealistically large per-subject z (e.g., a subject whose paired
        log-FC is implausibly large, suggesting unmodelled biology or a
        noisy measurement).
        """
        abs_zs = np.abs(unscaled_zs)
        N = len(abs_zs)
        if N < 3:
            return 1.0
        median_abs_z = float(np.median(abs_zs))
        max_abs_z = float(np.max(abs_zs))
        if max_abs_z <= 0:
            # Truly all-zero z's — no outlier to correct.
            return 1.0
        # Build a stable "typical |z|" reference even when median is 0.
        if median_abs_z > 0:
            typical = median_abs_z
        else:
            mean_abs_z = float(np.mean(abs_zs))
            typical = mean_abs_z if mean_abs_z > 0 else cls._MEDIAN_ZERO_FLOOR
        cap = cls._OUTLIER_CAP_K * typical
        if max_abs_z <= cap:
            return 1.0
        # Inflate the denominator by the same factor that the largest z
        # exceeds the cap. This is conservative: at most we shrink z_val
        # so the offending z would be at the cap.
        return float(max_abs_z / cap)


class DifferentialIonTTest():
    """Example implementation of differential testing using Welch's t-test.

    This is an alternative statistical test provided as example code to demonstrate how
    developers can implement their own methods. It uses Welch's two-sample t-test with
    robust variance estimation (similar to MS-EmpiRe). This implementation has not been
    extensively benchmarked and is included primarily for educational purposes.

    To use this method instead of DifferentialIon, set ion_test_method='ttest' when
    calling run_pipeline().

    Args:
        noNanvals_from: Array of log2 intensities from condition 1 (no NaN values)
        noNanvals_to: Array of log2 intensities from condition 2 (no NaN values)
        name: Ion identifier (e.g., peptide sequence + charge state)
        p2z: Dictionary cache for p-value to z-value conversions
        outlier_correction: If True, inflates standard error using robust estimators
                          (MAD/IQR) when sample variance is unusually low (default: True)

    Attributes:
        name: Ion identifier
        p_val: Two-sided p-value from Welch's t-test
        fc: Log2 fold change (condition1 - condition2)
        z_val: Z-score derived from p-value
        usable: Boolean indicating whether statistics could be computed
    """

    def __init__(self, noNanvals_from, noNanvals_to, name, p2z = None, outlier_correction: bool = True):


        self.name = name
        self.p_val = None
        self.fc = None
        self.z_val = None
        self.usable = False

        self._calc_ttest_peptide(noNanvals_from, noNanvals_to, p2z, outlier_correction)


    def _calc_ttest_peptide(self, noNanvals_from, noNanvals_to, p2z, outlier_correction):
        """Calculates differential expression statistics using Welch's t-test.

        Computes the t-statistic and p-value using scipy's Welch's t-test, with optional
        robust variance inflation. The p-value is then converted to a z-score for
        compatibility with the tree aggregation framework.

        Args:
            noNanvals_from: Log2 intensities from condition 1
            noNanvals_to: Log2 intensities from condition 2
            p2z: Cache dictionary for p-value to z-value conversions
            outlier_correction: Whether to apply robust standard error inflation
        """

        nrep_from = len(noNanvals_from)
        nrep_to = len(noNanvals_to)

        if ((nrep_from==0) or (nrep_to ==0)):
            return

        mean_from = float(np.mean(noNanvals_from))
        mean_to = float(np.mean(noNanvals_to))
        self.fc = mean_from - mean_to

        # Compute Welch's t-statistic and degrees of freedom
        n1 = nrep_from
        n2 = nrep_to
        try:
            res = ttest_ind(noNanvals_from, noNanvals_to, equal_var=False, nan_policy='omit')
            t_stat = float(res.statistic) if res.statistic is not None else 0.0
            p_val = float(res.pvalue) if res.pvalue is not None else 1.0
        except Exception:
            t_stat = 0.0
            p_val = 1.0

        s1 = float(np.std(noNanvals_from, ddof=1)) if n1 > 1 else 0.0
        s2 = float(np.std(noNanvals_to, ddof=1)) if n2 > 1 else 0.0
        se_standard_sq = (s1*s1)/max(1, n1) + (s2*s2)/max(1, n2)
        se_standard = math.sqrt(se_standard_sq) if se_standard_sq > 0 else 0.0

        # Welch-Satterthwaite degrees of freedom
        num_df = se_standard_sq * se_standard_sq
        den_df = 0.0
        if n1 > 1:
            den_df += ((s1*s1)/n1) * ((s1*s1)/n1) / (n1 - 1)
        if n2 > 1:
            den_df += ((s2*s2)/n2) * ((s2*s2)/n2) / (n2 - 1)
        df = (num_df/den_df) if den_df > 0 else max(n1 + n2 - 2, 1)

        # Robust SE inflation analogous to diffdist outlier scaling
        if outlier_correction and se_standard > 0 and n1 > 1 and n2 > 1:
            se_robust = _calc_robust_se_ttest(noNanvals_from, noNanvals_to)
            if se_robust > 0:
                scaling_factor = max(1.0, min(5.0, se_robust / se_standard))
                t_adj = t_stat / scaling_factor
                p_val = 2.0 * float(student_t.sf(abs(t_adj), df))

        abs_z = aqdiffutils.two_sided_p_to_abs_z(p_val, p2z if p2z is not None else {})
        sign = 1.0 if self.fc >= 0 else -1.0
        self.z_val = sign * abs_z
        self.p_val = p_val
        self.usable = True


def calc_outlier_scaling_factor(noNanvals_from, noNanvals_to, diffDist):
    """Computes a variance inflation factor to account for outlier replicates.

    Compares the between-replicate variance to the expected technical variance from the
    empirical background distribution. If replicates are more variable than expected
    (e.g., due to biological variability or technical outliers), the variance estimate
    is inflated accordingly. This makes the test more conservative when data quality is poor.

    Args:
        noNanvals_from: Log2 intensities from condition 1
        noNanvals_to: Log2 intensities from condition 2
        diffDist: Background distribution providing expected technical variance

    Returns:
        float: Scaling factor (>=1.0) to multiply variance by. Returns 1.0 if no
               correction needed, higher values when outliers are detected.
    """
    sd_from = math.sqrt(diffDist.var_from)
    sd_to = math.sqrt(diffDist.var_to)
    median_from = statistics.median(noNanvals_from)
    median_to = statistics.median(noNanvals_to)

    between_rep_SD_from = math.sqrt(sum(np.square(noNanvals_from-median_from))/len(noNanvals_from)) if len(noNanvals_from)>1 else sd_from
    between_rep_SD_to = math.sqrt(sum(np.square(noNanvals_to-median_to))/len(noNanvals_to)) if len(noNanvals_to)>1 else sd_to

    highest_SD_from = max(between_rep_SD_from, sd_from)
    highest_SD_to = max(between_rep_SD_to, sd_to)
    highest_SD_combined = math.sqrt(highest_SD_from**2 + highest_SD_to**2)

    scaling_factor = max(1.0, highest_SD_combined/diffDist.SD)
    return scaling_factor

def _robust_sd(x):
    """Computes a robust estimate of standard deviation using MAD or IQR.

    This function provides a robust alternative to standard deviation that is less
    sensitive to outliers. It tries three methods in order of robustness:
    1. MAD (Median Absolute Deviation) scaled to match SD for normal distributions
    2. IQR (Interquartile Range) scaled to match SD for normal distributions
    3. Regular sample standard deviation (fallback)

    Args:
        x: Array of values

    Returns:
        float: Robust estimate of standard deviation, or 0.0 if n < 2
    """
    x = np.asarray(x)
    n = x.size
    if n == 0:
        return 0.0
    if n == 1:
        return 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad > 0:
        return 1.4826 * mad
    # Fallback to IQR if MAD is zero
    q75, q25 = np.percentile(x, [75, 25])
    iqr = float(q75 - q25)
    if iqr > 0:
        return iqr / 1.349
    # Final fallback to sample SD
    return float(np.std(x, ddof=1))

def _calc_robust_se_ttest(noNanvals_from, noNanvals_to):
    """Computes robust standard error for t-test using inflated variance estimates.

    This function calculates the standard error for Welch's t-test, but uses the maximum
    of the regular standard deviation and a robust estimate (MAD/IQR-based). This provides
    protection against underestimation of variance when sample SD is unusually low.

    Args:
        noNanvals_from: Log2 intensities from condition 1
        noNanvals_to: Log2 intensities from condition 2

    Returns:
        float: Robust standard error estimate, or 0.0 if insufficient replicates
    """
    n1 = len(noNanvals_from)
    n2 = len(noNanvals_to)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1 = float(np.std(noNanvals_from, ddof=1))
    s2 = float(np.std(noNanvals_to, ddof=1))
    s1_rob = _robust_sd(noNanvals_from)
    s2_rob = _robust_sd(noNanvals_to)
    s1_infl = max(s1, s1_rob)
    s2_infl = max(s2, s2_rob)
    se_sq = (s1_infl*s1_infl)/n1 + (s2_infl*s2_infl)/n2
    return math.sqrt(se_sq) if se_sq > 0 else 0.0
