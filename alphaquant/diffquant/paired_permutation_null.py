"""Patient-block sign-flip permutation calibration for paired-diffdist (Phase 3b).

Background
----------
Even with a well-calibrated empirical null and covariance-adjusted
Stouffer aggregation (Phase 3a), the analytical p-value relies on the
asymptotic claim that Z_combined ~ N(0, 1) under H0. At the low subject
counts where paired-diffdist is most useful (n ≤ 8, e.g., bm_mk's n=4
mice per cell), the asymptotic claim is shaky. The permutation null is
the FDR-honest backbone:

    For each within-subject sign-flip permutation:
        flip arm-labels for the ±1-marked subjects → recompute paired
        ion log-FCs → recompute ion-level z → recompute Z_combined via
        covariance-adjusted Stouffer.

    p_observed = (# perms with |Z_perm| ≥ |Z_observed|) / N_perms

The exchangeability that makes the sign-flip null valid:

    Under H0, swapping arm labels for any subject leaves the joint
    distribution of (arm1, arm2) within that subject unchanged
    (each subject contributes the same paired difference with equal
    probability under +1 and -1 sign). The sign-flip permutation set is
    therefore an EXACT null for the test statistic.

Discrete p-floor at low n
-------------------------
At n=4: 2⁴ = 16 distinct sign-flips → minimum reportable p = 1/16 =
0.0625. The user's `permutation_p_floor` column documents this. For
FDR < 0.05 thresholding to be meaningful at n=4, the user should
loosen to FDR < 0.10 (consistent with the 0.0625 floor).

At n=6: 2⁶ = 64 → floor 0.016
At n=8: 2⁸ = 256 → floor 0.004
At n>8: random sample of `max_perms` permutations (default 64 — keeps
the floor at 1/64 = 0.016, sufficient for typical biology).

Identity-permutation handling
-----------------------------
The all-+1 sign vector is the OBSERVED data (no flip). It must be
INCLUDED in the permutation count when computing p (so p ≥ 1/N_perms),
but NOT separately added — it is one of the 2ⁿ enumerated permutations.
We document this rather than excluding it; including it makes the test
strictly conservative when |Z_observed| is the largest (then p = 1/N).
"""
from __future__ import annotations

import itertools
import typing

import numpy as np

import alphaquant.config.config as aqconfig
import logging

aqconfig.setup_logging()
LOGGER = logging.getLogger(__name__)


# Above this subject count, full enumeration of 2^n is prohibitive.
# Switch to random sampling of `max_perms` distinct sign vectors.
MAX_N_FOR_FULL_ENUM = 8
DEFAULT_MAX_PERMS = 64


# ──────────────────────────────────────────────────────────────────────
# Permutation enumeration
# ──────────────────────────────────────────────────────────────────────

def enumerate_signflip_permutations(
    n_subjects: int,
    max_perms: int = DEFAULT_MAX_PERMS,
    rng_seed: typing.Optional[int] = 2026,
    include_identity: bool = True,
) -> typing.List[np.ndarray]:
    """Generate within-subject arm-label sign-flip permutations.

    Args:
        n_subjects: number of paired subjects. Determines the
            permutation space size 2^n_subjects.
        max_perms: cap on the number of permutations returned. For
            n_subjects ≤ MAX_N_FOR_FULL_ENUM, the full set of 2^n is
            returned regardless. For n_subjects > MAX_N_FOR_FULL_ENUM,
            a random sample of max_perms distinct sign vectors is
            returned.
        rng_seed: seed for the random sampler (only used when n is
            large). Default 2026 for reproducibility.
        include_identity: if True (default), include the all-+1 vector
            in the returned set — needed for the (# perms ≥ |z_obs|)
            count to include the observed value.

    Returns:
        list of np.ndarray of length n_subjects, each entry ±1.

    Notes:
        - At n=4: 16 permutations (full enum).
        - At n=6: 64 permutations (full enum).
        - At n=8: 256 permutations (full enum).
        - At n=12: 64 random samples of the 4096 possible (default).

        The permutation set is symmetric under global sign-flip
        (s and -s give identical |Z|), which means effectively half the
        distinct |Z| values appear in pairs. We enumerate the full
        space anyway because the cost is dominated by the
        recompute-Z-per-permutation step, not the enumeration.
    """
    if n_subjects <= 0:
        return []
    if n_subjects <= MAX_N_FOR_FULL_ENUM:
        perms = [
            np.array(s, dtype=np.int8)
            for s in itertools.product([1, -1], repeat=n_subjects)
        ]
        if not include_identity:
            perms = [s for s in perms if not np.all(s == 1)]
        return perms

    # Random sample for large n.
    rng = np.random.default_rng(rng_seed)
    n_to_sample = min(max_perms, 2 ** n_subjects)
    seen = set()
    perms = []
    if include_identity:
        identity = tuple([1] * n_subjects)
        seen.add(identity)
        perms.append(np.array(identity, dtype=np.int8))
    # Reject-sample distinct random sign vectors.
    attempts = 0
    max_attempts = max(1000, 100 * n_to_sample)
    while len(perms) < n_to_sample and attempts < max_attempts:
        candidate = tuple(int(s) for s in rng.choice([1, -1], size=n_subjects))
        if candidate not in seen:
            seen.add(candidate)
            perms.append(np.array(candidate, dtype=np.int8))
        attempts += 1
    if len(perms) < n_to_sample:
        LOGGER.warning(
            "enumerate_signflip_permutations: requested %d perms but only "
            "found %d distinct after %d attempts; returning what we have.",
            n_to_sample, len(perms), attempts,
        )
    return perms


def apply_permutation_to_paired_pairs(
    paired_subject_pairs: typing.Sequence[typing.Tuple[str, float, float]],
    signs: np.ndarray,
) -> typing.List[typing.Tuple[str, float, float]]:
    """Apply a sign-flip vector to a list of paired (subject, arm1, arm2).

    For sign=+1: keep tuple as-is (val_arm1, val_arm2).
    For sign=-1: swap to (val_arm2, val_arm1).

    Args:
        paired_subject_pairs: list of (subject_id, val_arm1, val_arm2)
            tuples. The same shape that ``DifferentialIonPaired``
            accepts.
        signs: 1D int array of ±1, length == len(paired_subject_pairs).

    Returns:
        list of permuted tuples (same order as input).

    Raises:
        ValueError if ``len(signs) != len(paired_subject_pairs)``.
    """
    if len(signs) != len(paired_subject_pairs):
        raise ValueError(
            f"signs length {len(signs)} != paired_subject_pairs length "
            f"{len(paired_subject_pairs)}"
        )
    out = []
    for (subj, v1, v2), s in zip(paired_subject_pairs, signs):
        if int(s) == 1:
            out.append((subj, v1, v2))
        elif int(s) == -1:
            out.append((subj, v2, v1))
        else:
            raise ValueError(f"sign must be ±1; got {s}")
    return out


def apply_permutation_to_subject_dict(
    subject_fc_dict: typing.Mapping[str, float],
    subject_order: typing.Sequence[str],
    signs: np.ndarray,
) -> typing.Dict[str, float]:
    """Apply a sign-flip vector to a per-subject paired-fc dict.

    Equivalent to :func:`apply_permutation_to_paired_pairs` but operates
    on the dict-of-{subject: fc} representation used by
    ``correlation_aware_stouffer.build_paired_fc_matrix_from_subject_dicts``.
    Each subject's fc is multiplied by its sign (±1).

    Args:
        subject_fc_dict: {subject_id: per-subject paired log-FC}.
            Subjects missing from `subject_order` are passed through
            unchanged (they don't contribute to the test statistic
            anyway, but we keep them for downstream invariants).
        subject_order: canonical subject ordering — the i-th sign in
            ``signs`` applies to ``subject_order[i]``.
        signs: 1D int array of ±1, length == len(subject_order).

    Returns:
        new dict with permuted fc values for each subject in
        ``subject_order``.
    """
    if len(signs) != len(subject_order):
        raise ValueError(
            f"signs length {len(signs)} != subject_order length "
            f"{len(subject_order)}"
        )
    out = dict(subject_fc_dict)  # carry through any extras
    for subj, s in zip(subject_order, signs):
        if subj in out:
            out[subj] = float(s) * out[subj]
    return out


# ──────────────────────────────────────────────────────────────────────
# P-value calibration
# ──────────────────────────────────────────────────────────────────────

def permutation_calibrated_p(
    observed_z: float,
    null_z_distribution: typing.Sequence[float],
) -> typing.Tuple[float, float]:
    """Two-sided permutation-calibrated p-value.

    p = (# perms with |z_perm| ≥ |z_obs|) / N_perms

    The observed value MUST be included in ``null_z_distribution`` for
    the p-value to be conservative — when included, the count is at
    least 1 (the observed itself), so p ≥ 1/N_perms. This is the
    standard Phipson-Smyth (2010) correction; we don't add +1 because
    the observed is one of the enumerated sign-flip permutations.

    Args:
        observed_z: gene-level z (analytic, covariance-adjusted Stouffer)
            from the un-permuted data.
        null_z_distribution: list / array of permutation gene-level z's
            (one per permutation). Must include the observed value
            (i.e., the all-+1 sign vector should have produced it).

    Returns:
        (p_value, p_floor) where:
            - p_value = (# |z_perm| >= |z_obs|) / N_perms
            - p_floor = 1 / N_perms (smallest achievable p at this N)

    Notes:
        Returns (1.0, 1/N_perms) when N_perms == 0 (degenerate; should
        not happen in practice — caller should check).
    """
    arr = np.asarray(list(null_z_distribution), dtype=float)
    n_perms = len(arr)
    if n_perms == 0:
        return 1.0, 1.0
    p_floor = 1.0 / n_perms
    # |z_perm| >= |z_obs| comparison; ties favour conservative (count them in).
    n_ge = int(np.sum(np.abs(arr) >= abs(observed_z) - 1e-15))
    n_ge = max(n_ge, 1)  # safety floor — observed itself should give ≥ 1
    p = n_ge / n_perms
    return float(p), float(p_floor)


def compute_n_perms_for_n_subjects(
    n_subjects: int,
    max_perms: int = DEFAULT_MAX_PERMS,
) -> int:
    """Return the actual N_perms that will be run for ``n_subjects``.

    Useful for pre-allocating `n_perms_used` per-protein output column
    without having to call ``enumerate_signflip_permutations`` first.
    """
    if n_subjects <= 0:
        return 0
    if n_subjects <= MAX_N_FOR_FULL_ENUM:
        return 2 ** n_subjects
    return min(max_perms, 2 ** n_subjects)
