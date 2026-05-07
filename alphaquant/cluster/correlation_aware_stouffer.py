"""Covariance-adjusted Stouffer aggregation for paired-diffdist (Phase 3a).

Background
----------
Classical Stouffer aggregation, used by AQ's tree
(``cluster_utils.sum_and_re_scale_zvalues``):

    Z_combined = (Σ z_i) / sqrt(N)

assumes the per-ion z-scores are independent. Under the **paired** test
(``DifferentialIonPaired``), the per-ion z-scores within a single protein
are NOT independent: they are computed from the same subjects' paired
log-FCs, so subjects with extreme random effects (or extreme
within-subject variability for a particular protein) push every ion of
that protein in the same direction. Independence-assuming Stouffer thus
under-states the protein-level z's variance, **inflating** combined Z
and **deflating** the protein-level p — i.e., anti-conservative.

The covariance-adjusted Stouffer fix:

    Z_combined = (Σ z_i) / sqrt(1ᵀ R 1)
              = (Σ z_i) / sqrt(N + Σ_{i≠j} R_ij)

where R is the within-protein ion-z correlation matrix. When R = I
(independent z's), the denominator reduces to sqrt(N), recovering
classical Stouffer. When R = 11ᵀ (perfect correlation), the denominator
becomes N, so Z_combined ≈ z_single (no fake amplification from
correlated evidence). This is the Z-aggregation analog of Brown's method.

Estimation of R
----------------
At low subject counts (n ≤ 8) the per-protein sample correlation matrix
is fundamentally under-identified — even shrinkage cannot create
information that is not in the sample. The
``compute_ion_correlation_matrix`` function offers three estimators:

    - ``'ledoit_wolf'`` (default): Ledoit-Wolf-shrunken sample
      correlation. Falls back to ``equicorrelation`` with ρ=0.3 when n is
      too small for shrinkage to be meaningful (n < 3 paired subjects).
    - ``'equicorrelation'``: fixed ρ·11ᵀ + (1-ρ)·I with ρ=0.3.
      Conservative (ρ > 0 inflates the denominator), explicit, and
      well-defined at any n.
    - ``'sample'``: raw sample correlation. Warns at low n.

**Identity-Σ is never a silent fallback.** Identity assumes
independence; under positive ion correlation (the typical case in
proteomics) it under-estimates Var(Σz) → inflates Z → anti-conservative.
If the caller explicitly asks for ``'identity'`` we raise a warning
explaining the risk.

Source data for R
------------------
Per-subject paired log-FC vectors per ion: ``fc_{subj, ion} = mean(arm1
microdiss) − mean(arm2 microdiss)``. The same vectors built by
``PairedConditionBackgrounds._build_ion2pairedfcs``. We compute the
sample Pearson correlation across the SUBJECT axis between every pair of
ions in the protein (after dropping subjects that are missing any of the
ions). The correlation matrix is N_ions × N_ions, symmetric, PSD by
construction.

Variance algebra (verifies Phase 2 calibration is preserved)
-----------------------------------------------------------
Phase 2 calibrated each per-ion z to be unit-normal under H0 via
rank-based lookup against the paired empirical null. Therefore:

    Var(z_i)         = 1 (by construction)
    Cov(z_i, z_j)    = R_ij (estimated from same data)
    Var(Σ z_i)       = Σ_i Var(z_i) + Σ_{i≠j} Cov(z_i, z_j)
                     = N + Σ_{i≠j} R_ij
                     = 1ᵀ R 1                                        (1)

Therefore Z_combined = (Σ z_i) / sqrt(1ᵀ R 1) ~ N(0, 1) under H0.
Phase 2's per-ion calibration is preserved; the only change is the
variance of the sum is correctly estimated rather than under-estimated.

When R = I (independence): Var(Σ z_i) = N, recovering Stouffer's denominator.
When R_ij = 1 ∀ i,j (perfect correlation): Var(Σ z_i) = N², so denominator = N,
and Z_combined = (Σ z_i)/N ≈ mean(z_i) ≈ z_single. No spurious amplification.
When R_ij = 0.3 ∀ i≠j (typical proteomic equicorrelation):
    Var(Σ z_i) = N + 0.3·N·(N-1) → denominator > sqrt(N) → conservative.
"""
from __future__ import annotations

import typing
import warnings

import numpy as np

import alphaquant.config.config as aqconfig
import logging

aqconfig.setup_logging()
LOGGER = logging.getLogger(__name__)


# Default equicorrelation ρ for the conservative fallback. Chosen to be
# in the typical range observed in proteomics same-condition ion pairs
# (literature 0.2–0.5; we err towards the low end so the fallback is not
# overly conservative). Adjustable via the parameter on the relevant
# functions; the default lives here so consumers can override at one place.
DEFAULT_EQUICORR_RHO = 0.3

# Minimum number of subjects (paired observations) for sample correlation
# to be considered "estimable". Below this, even Ledoit-Wolf shrinkage
# (which formally needs n ≥ 2 to compute the shrunk estimator) is
# fundamentally under-identified, and we should fall back to
# equicorrelation. Three is the lowest n where a non-trivial shrunk
# correlation matrix has any sample information at all; permutation
# calibration (Phase 3b) is the FDR-honest backbone at low n regardless.
MIN_N_FOR_SAMPLE_CORR = 3


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def compute_ion_correlation_matrix(
    paired_fc_matrix: np.ndarray,
    method: str = "ledoit_wolf",
    equicorr_rho: float = DEFAULT_EQUICORR_RHO,
) -> np.ndarray:
    """Compute the within-protein ion-level z-score correlation matrix.

    Args:
        paired_fc_matrix: shape (n_subjects, n_ions). Each column is one
            ion's per-subject paired log-FC vector
            (mean(arm1 microdiss) − mean(arm2 microdiss)). NaN entries
            (subjects missing for that ion) are tolerated; rows with any
            NaN are dropped pairwise per (i, j) for sample correlation,
            globally for Ledoit-Wolf (which requires complete cases).
        method: one of ``'ledoit_wolf'`` (default), ``'equicorrelation'``,
            ``'sample'``, ``'identity'``.
        equicorr_rho: ρ used by equicorrelation fallback. Default 0.3.

    Returns:
        np.ndarray: (n_ions, n_ions) symmetric PSD correlation matrix
        (diagonal = 1.0). For n_ions=1, returns ``[[1.0]]`` (no
        aggregation needed downstream).

    Notes:
        - The correlation of the per-subject paired log-FCs is taken as
          a proxy for the correlation of the per-ion z-scores. Under H0
          and with the empirical-null calibration, the rank-based
          z_i = lookup(fc_i) is a monotone transform of fc_i; Pearson
          correlation is not preserved exactly under monotone transforms
          (Spearman is), but for approximately-normal fc's (the
          empirical null is well-calibrated to N) the difference is
          small. We use Pearson for simplicity and consistency with
          Ledoit-Wolf which is defined on Pearson cov.
        - "Identity" is a safety-rail option: it raises a warning that
          using identity is anti-conservative under positive ion
          correlation, then returns I. Callers should NOT pick this in
          production unless they understand the risk.
    """
    paired_fc_matrix = np.asarray(paired_fc_matrix, dtype=float)
    if paired_fc_matrix.ndim != 2:
        raise ValueError(
            f"paired_fc_matrix must be 2D (n_subjects, n_ions); "
            f"got shape {paired_fc_matrix.shape}"
        )
    n_subj, n_ions = paired_fc_matrix.shape
    if n_ions == 0:
        return np.zeros((0, 0))
    if n_ions == 1:
        return np.array([[1.0]])

    if method == "identity":
        warnings.warn(
            "compute_ion_correlation_matrix(method='identity') returns the "
            "identity matrix, which assumes independence of ion-level "
            "z-scores. Under positive within-protein ion correlation (the "
            "typical case in proteomics) this is anti-conservative — combined "
            "Z is inflated and protein-level p is deflated. Prefer "
            "'ledoit_wolf' or 'equicorrelation'. See "
            "alphaquant/cluster/correlation_aware_stouffer.py docstring.",
            stacklevel=2,
        )
        return np.eye(n_ions)

    if method == "equicorrelation":
        return _equicorrelation_matrix(n_ions, equicorr_rho)

    if method == "sample":
        if n_subj < MIN_N_FOR_SAMPLE_CORR:
            LOGGER.warning(
                "compute_ion_correlation_matrix(method='sample'): only "
                "%d paired subjects (< %d) — sample correlation matrix "
                "is fundamentally under-identified at this n. Returning "
                "equicorrelation fallback (rho=%.2f) instead.",
                n_subj, MIN_N_FOR_SAMPLE_CORR, equicorr_rho,
            )
            return _equicorrelation_matrix(n_ions, equicorr_rho)
        R = _sample_correlation(paired_fc_matrix)
        return _project_to_psd(R)

    if method == "ledoit_wolf":
        if n_subj < MIN_N_FOR_SAMPLE_CORR:
            # Even shrinkage cannot help below n=3. Fall back to
            # equicorrelation; permutation calibration (Phase 3b) carries
            # the FDR-honest claim at this n.
            return _equicorrelation_matrix(n_ions, equicorr_rho)
        R = _ledoit_wolf_correlation(paired_fc_matrix)
        return _project_to_psd(R)

    raise ValueError(
        f"Unknown method={method!r}; must be one of "
        "{'ledoit_wolf', 'equicorrelation', 'sample', 'identity'}."
    )


def covariance_adjusted_stouffer_z(
    z_values: np.ndarray,
    R: typing.Optional[np.ndarray] = None,
) -> float:
    """Closed-form covariance-adjusted Stouffer combined Z.

    Args:
        z_values: 1D array of per-ion z-scores. Length N.
        R: (N, N) correlation matrix. If None, defaults to identity
            (classical Stouffer). For N=1, R is ignored.

    Returns:
        float: combined Z, scaled to be ~ N(0, 1) under H0 when R is
        well-estimated and the per-ion z's are unit-normal.

    Examples:
        >>> covariance_adjusted_stouffer_z(np.array([1.0]))
        1.0
        >>> # Independent ions (R=I) → classical Stouffer
        >>> abs(covariance_adjusted_stouffer_z(
        ...     np.array([1.0, 1.0, 1.0]), np.eye(3)) -
        ...     3.0/np.sqrt(3)) < 1e-12
        True
        >>> # Perfectly correlated ions: combined Z = mean = single Z
        >>> abs(covariance_adjusted_stouffer_z(
        ...     np.array([1.0, 1.0, 1.0]), np.ones((3, 3))) - 1.0) < 1e-12
        True
    """
    z = np.asarray(z_values, dtype=float)
    N = z.size
    if N == 0:
        return 0.0
    if N == 1:
        return float(z[0])
    if R is None:
        R = np.eye(N)
    R = np.asarray(R, dtype=float)
    if R.shape != (N, N):
        raise ValueError(
            f"R has shape {R.shape}; expected {(N, N)} for {N} z-values"
        )
    z_sum = float(np.sum(z))
    var_sum = float(np.ones(N) @ R @ np.ones(N))
    if var_sum <= 0:
        # Degenerate R (would only happen with a pathological non-PSD
        # matrix or all-zero rows). Fall back to classical Stouffer.
        LOGGER.warning(
            "covariance_adjusted_stouffer_z: 1ᵀ R 1 = %.6g ≤ 0; falling "
            "back to classical Stouffer (sqrt(N)) as denominator.",
            var_sum,
        )
        var_sum = float(N)
    return z_sum / np.sqrt(var_sum)


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────

def _equicorrelation_matrix(n: int, rho: float) -> np.ndarray:
    """Build R = ρ·11ᵀ + (1-ρ)·I (n × n)."""
    R = np.full((n, n), float(rho))
    np.fill_diagonal(R, 1.0)
    return R


def _sample_correlation(X: np.ndarray) -> np.ndarray:
    """Sample Pearson correlation across rows.

    X: (n_subjects, n_ions). Drops subjects (rows) with any NaN — i.e.,
    complete-case correlation. For pairwise complete-case use
    ``_pairwise_correlation`` (slower; not used in the default path).
    """
    mask = ~np.any(np.isnan(X), axis=1)
    Xc = X[mask]
    if Xc.shape[0] < 2:
        # Not enough complete cases — fall back to equicorrelation
        # (caller will see this when constructing R; we return I-ish
        # equicorr-like behavior so cov-adjusted Stouffer doesn't crash).
        return _equicorrelation_matrix(X.shape[1], DEFAULT_EQUICORR_RHO)
    # np.corrcoef is row-variance-correlation when rowvar=True; we want
    # column-variable correlation (each column is one ion).
    R = np.corrcoef(Xc, rowvar=False)
    # Defensively replace NaN (zero-variance ion) with 0 off-diagonal,
    # 1 on-diagonal.
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)
    return R


def _ledoit_wolf_correlation(X: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf-shrunken sample correlation.

    Uses ``sklearn.covariance.LedoitWolf`` if available; otherwise falls
    back to a direct implementation of the LW shrinkage intensity for the
    correlation matrix toward the identity target. Both produce the same
    PSD-projected, shrunk correlation matrix.

    Args:
        X: (n_subjects, n_ions). Complete cases only (rows with any NaN
            dropped).

    Returns:
        (n_ions, n_ions) shrunk correlation matrix.
    """
    mask = ~np.any(np.isnan(X), axis=1)
    Xc = X[mask]
    n_subj, n_ions = Xc.shape
    if n_subj < 2:
        return _equicorrelation_matrix(n_ions, DEFAULT_EQUICORR_RHO)

    # Strategy: standardise each column to zero-mean unit-variance, then
    # apply LW to the correlation matrix (LW is for covariance — applied
    # to standardised data it shrinks correlation).
    col_mean = np.mean(Xc, axis=0)
    col_sd = np.std(Xc, axis=0, ddof=0)
    # Avoid /0 for zero-variance ions (degenerate but possible).
    col_sd[col_sd == 0] = 1.0
    Xs = (Xc - col_mean) / col_sd

    try:
        from sklearn.covariance import LedoitWolf  # type: ignore
        lw = LedoitWolf().fit(Xs)
        R = lw.covariance_
    except ImportError:  # pragma: no cover (sklearn is a hard dep)
        R = _ledoit_wolf_shrinkage_direct(Xs)

    # Force diagonal to 1.0 (numerical hygiene; LW on standardised data
    # should already give ~1.0 but rounding can deviate).
    np.fill_diagonal(R, 1.0)
    R = 0.5 * (R + R.T)  # ensure symmetry
    return R


def _ledoit_wolf_shrinkage_direct(Xs: np.ndarray) -> np.ndarray:
    """Direct Ledoit-Wolf shrinkage to identity target (fallback if
    sklearn is unavailable).

    Implements the closed-form shrinkage intensity from
    Ledoit & Wolf (2004): minimum-MSE shrinkage of the sample covariance
    toward the identity target. For standardised input Xs (each col
    zero-mean unit-var), this gives the shrunk correlation matrix.

    Args:
        Xs: (n, p) standardised data.

    Returns:
        (p, p) shrunk correlation matrix.
    """
    n, p = Xs.shape
    S = (Xs.T @ Xs) / n  # MLE sample correlation (since Xs is standardised)

    # Target: identity (mu * I, where mu = trace(S)/p ≈ 1 for standardised data).
    mu = np.trace(S) / p
    F = mu * np.eye(p)

    # Variance of the off-diagonal elements (Ledoit-Wolf eq. 13).
    # π̂ = (1/n) · sum over (i, j) of Var(S_ij)
    #    where Var(S_ij) ≈ (1/n) · Σ_k (X_{ki} X_{kj} - S_ij)^2
    Y = Xs ** 2
    pi_mat = (Y.T @ Y) / n - S ** 2
    pi_hat = np.sum(pi_mat)

    # Misfit between sample and target (γ̂).
    gamma_hat = np.sum((S - F) ** 2)

    # Shrinkage intensity (clipped to [0, 1]).
    if gamma_hat <= 0:
        kappa = 0.0
    else:
        kappa = pi_hat / gamma_hat
    delta = max(0.0, min(1.0, kappa / n))

    R_shrunk = delta * F + (1 - delta) * S
    return R_shrunk


def _project_to_psd(R: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Project a (near-)symmetric matrix to the closest PSD matrix by
    clipping negative eigenvalues to ``eps``.

    Numerical hygiene: sample correlation matrices computed at low n can
    have tiny negative eigenvalues from rounding. Without projection,
    1ᵀ R 1 could be marginally negative — covariance-adjusted Stouffer
    would then divide by sqrt of a negative number.

    Args:
        R: (n, n) symmetric matrix.
        eps: floor for eigenvalues.

    Returns:
        (n, n) symmetric PSD matrix.
    """
    R_sym = 0.5 * (R + R.T)
    eigvals, eigvecs = np.linalg.eigh(R_sym)
    eigvals_clipped = np.clip(eigvals, eps, None)
    R_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    # Re-normalise diagonal to 1 (because clipping may shift it slightly).
    d = np.sqrt(np.clip(np.diag(R_psd), eps, None))
    R_psd = R_psd / np.outer(d, d)
    np.fill_diagonal(R_psd, 1.0)
    R_psd = 0.5 * (R_psd + R_psd.T)
    return R_psd


# ──────────────────────────────────────────────────────────────────────
# Tree-aware helpers (used by aggregate_node_properties)
# ──────────────────────────────────────────────────────────────────────

def build_paired_fc_matrix_from_subject_dicts(
    child_subject_fc_dicts: typing.Sequence[typing.Dict[str, float]],
    subject_order: typing.Optional[typing.Sequence[str]] = None,
) -> typing.Tuple[np.ndarray, typing.List[str]]:
    """Stack per-child {subject_id: paired_fc} dicts into a (n_subj, n_children)
    matrix with NaNs filled for subjects missing from a given child.

    Args:
        child_subject_fc_dicts: list of dicts, one per child, mapping
            subject_id → per-subject paired log-FC. Subjects can be a
            non-overlapping set (some ions may have measurements only in
            a subset of subjects).
        subject_order: optional fixed subject ordering. If None, uses
            the union of subjects across all children, sorted.

    Returns:
        (matrix, subject_order) where matrix shape is
        (len(subject_order), n_children). Missing entries are NaN.
    """
    if not child_subject_fc_dicts:
        return np.zeros((0, 0)), []
    if subject_order is None:
        subjects: set = set()
        for d in child_subject_fc_dicts:
            subjects.update(d.keys())
        subject_order = sorted(subjects)
    n_subj = len(subject_order)
    n_children = len(child_subject_fc_dicts)
    M = np.full((n_subj, n_children), np.nan, dtype=float)
    for j, d in enumerate(child_subject_fc_dicts):
        for i, subj in enumerate(subject_order):
            if subj in d:
                M[i, j] = float(d[subj])
    return M, list(subject_order)
