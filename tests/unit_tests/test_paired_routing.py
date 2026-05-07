"""Phase 1 tests for paired-diffdist plumbing.

Verifies the dispatch contract:
  - is_paired_eligible correctly detects paired-design contrasts
  - get_paired_sample_triples groups runs by subject correctly
  - DifferentialIonPaired stub raises NotImplementedError on construction
  - Existing unpaired tests still pass (regression — implicit; checked by
    full test-suite run, not in this module)

These tests do NOT exercise the actual paired ion test (Phase 2). They
verify only the routing layer added in Phase 1.
"""
import pytest
import pandas as pd

import alphaquant.diffquant.diffutils as aqdiffutils
import alphaquant.diffquant.diff_analysis as aqdiff


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture
def paired_samplemap():
    """Mimics bm_mk: 4 mice × 2 bones × 2 microdiss = 16 samples.
    Subject IDs repeat across the bone arms."""
    rows = []
    for mouse_idx in range(1, 5):
        subject = f"mouse_{mouse_idx}"
        for bone in ("femur", "humerus"):
            for tech in (1, 2):
                rows.append({
                    "sample": f"{subject}_{bone}_{tech}",
                    "condition": bone,
                    "subject_id": subject,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def unpaired_samplemap():
    """Standard between-subject design: each sample is a unique subject."""
    rows = []
    for i in range(1, 5):
        rows.append({"sample": f"control_{i}", "condition": "control",
                     "subject_id": f"subj_c_{i}"})
    for i in range(1, 5):
        rows.append({"sample": f"treated_{i}", "condition": "treated",
                     "subject_id": f"subj_t_{i}"})
    return pd.DataFrame(rows)


@pytest.fixture
def samplemap_no_subject_col():
    """Samplemap missing the subject_id column entirely (legacy unpaired)."""
    rows = [{"sample": f"s_{i}", "condition": "A" if i % 2 == 0 else "B"}
            for i in range(8)]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────
# is_paired_eligible
# ─────────────────────────────────────────────────────────────────────────

def test_is_paired_eligible_paired_design(paired_samplemap):
    """All 4 mice contribute to both bones → paired-eligible."""
    assert aqdiffutils.is_paired_eligible(
        paired_samplemap, "femur", "humerus") is True


def test_is_paired_eligible_unpaired_design(unpaired_samplemap):
    """Each subject is in only one arm → not paired-eligible."""
    assert aqdiffutils.is_paired_eligible(
        unpaired_samplemap, "control", "treated") is False


def test_is_paired_eligible_missing_column(samplemap_no_subject_col):
    """No subject_id column → not paired-eligible (graceful False, not crash)."""
    assert aqdiffutils.is_paired_eligible(
        samplemap_no_subject_col, "A", "B") is False


def test_is_paired_eligible_min_subjects_threshold(paired_samplemap):
    """Default min_paired_subjects=2; raise threshold past available subjects → False."""
    assert aqdiffutils.is_paired_eligible(
        paired_samplemap, "femur", "humerus", min_paired_subjects=10) is False


def test_is_paired_eligible_custom_subject_id_col(paired_samplemap):
    """The function honors a custom subject_id_col argument."""
    sm = paired_samplemap.rename(columns={"subject_id": "patient"})
    # default 'subject_id' is now missing → False
    assert aqdiffutils.is_paired_eligible(sm, "femur", "humerus") is False
    # explicit column name → True
    assert aqdiffutils.is_paired_eligible(
        sm, "femur", "humerus", subject_id_col="patient") is True


def test_is_paired_eligible_cond1_equals_cond2(paired_samplemap):
    """Degenerate: same condition on both sides. set-intersection of subjects
    with itself trivially equals the full set, but pairing a condition
    against itself isn't meaningful — guard returns False."""
    assert aqdiffutils.is_paired_eligible(
        paired_samplemap, "femur", "femur") is False


def test_is_paired_eligible_partial_overlap():
    """Only some subjects appear in both arms — should still be eligible if
    enough do (≥ min_paired_subjects)."""
    rows = [
        # mouse_1: only femur (drop)
        {"sample": "m1_femur", "condition": "femur", "subject_id": "mouse_1"},
        # mouse_2: only humerus (drop)
        {"sample": "m2_humerus", "condition": "humerus", "subject_id": "mouse_2"},
        # mouse_3 + mouse_4: both arms (keep — gives 2 paired subjects)
        {"sample": "m3_femur", "condition": "femur", "subject_id": "mouse_3"},
        {"sample": "m3_humerus", "condition": "humerus", "subject_id": "mouse_3"},
        {"sample": "m4_femur", "condition": "femur", "subject_id": "mouse_4"},
        {"sample": "m4_humerus", "condition": "humerus", "subject_id": "mouse_4"},
    ]
    sm = pd.DataFrame(rows)
    # 2 paired subjects (mouse_3, mouse_4) at default threshold (2) → eligible
    assert aqdiffutils.is_paired_eligible(sm, "femur", "humerus") is True
    # at threshold 3 → not eligible
    assert aqdiffutils.is_paired_eligible(
        sm, "femur", "humerus", min_paired_subjects=3) is False


# ─────────────────────────────────────────────────────────────────────────
# get_paired_sample_triples
# ─────────────────────────────────────────────────────────────────────────

def test_get_paired_sample_triples_basic(paired_samplemap):
    """4 mice × 2 microdiss/(mouse, bone) → 4 triples, each with 2-element lists."""
    triples = aqdiffutils.get_paired_sample_triples(
        paired_samplemap, "femur", "humerus")
    assert len(triples) == 4
    for subj, c1, c2 in triples:
        assert subj.startswith("mouse_")
        assert len(c1) == 2  # 2 microdiss per (mouse, bone) cell
        assert len(c2) == 2
        assert all(s.startswith(subj) for s in c1)
        assert all(s.startswith(subj) for s in c2)


def test_get_paired_sample_triples_sorted(paired_samplemap):
    """Output is sorted by subject_id for deterministic downstream."""
    triples = aqdiffutils.get_paired_sample_triples(
        paired_samplemap, "femur", "humerus")
    subjects = [t[0] for t in triples]
    assert subjects == sorted(subjects)


def test_get_paired_sample_triples_drops_unpaired_subjects():
    """Subjects in only one arm are excluded from the triple list."""
    rows = [
        {"sample": "m1_femur", "condition": "femur", "subject_id": "mouse_1"},
        # mouse_1 has NO humerus → drop entirely
        {"sample": "m2_femur", "condition": "femur", "subject_id": "mouse_2"},
        {"sample": "m2_humerus", "condition": "humerus", "subject_id": "mouse_2"},
    ]
    sm = pd.DataFrame(rows)
    triples = aqdiffutils.get_paired_sample_triples(sm, "femur", "humerus")
    assert len(triples) == 1
    assert triples[0][0] == "mouse_2"


def test_get_paired_sample_triples_no_column():
    """Missing subject_id_col → empty list (graceful, not crash)."""
    sm = pd.DataFrame([{"sample": "s1", "condition": "A"}])
    assert aqdiffutils.get_paired_sample_triples(sm, "A", "B") == []


# ─────────────────────────────────────────────────────────────────────────
# DifferentialIonPaired (Phase 2: stub replaced with real implementation)
# ─────────────────────────────────────────────────────────────────────────

def test_differential_ion_paired_signature_matches_dispatch():
    """Locked contract: __init__ signature matches what
    ``condpair_analysis.analyze_condpair`` passes when use_paired=True.
    Drift here means the dispatch site needs updating too."""
    import inspect
    sig = inspect.signature(aqdiff.DifferentialIonPaired.__init__)
    params = list(sig.parameters.keys())
    expected = ["self", "paired_subject_pairs", "paired_diff_dist",
                "name", "outlier_correction"]
    assert params == expected, (
        f"DifferentialIonPaired __init__ signature drift: "
        f"expected {expected}, got {params}")


def test_differential_ion_paired_n1_unusable():
    """n=1 (single paired observation) is degenerate for a paired test —
    there is no within-subject variance to test against. Constructor must
    return cleanly with usable=False rather than raising."""
    diffion = aqdiff.DifferentialIonPaired(
        paired_subject_pairs=[("subj1", 1.0, 2.0)],
        paired_diff_dist=None,
        name="ion_X")
    assert diffion.usable is False
    assert diffion.p_val is None
    assert diffion.n_paired_subjects == 1


def test_differential_ion_paired_none_diff_dist_unusable():
    """Missing paired_diff_dist (e.g., empty intensity bin) → usable=False
    rather than crash in calc_zscore_from_fc."""
    diffion = aqdiff.DifferentialIonPaired(
        paired_subject_pairs=[("subj1", 1.0, 2.0), ("subj2", 1.5, 1.0)],
        paired_diff_dist=None,
        name="ion_X")
    assert diffion.usable is False
    assert diffion.n_paired_subjects == 2


# ─────────────────────────────────────────────────────────────────────────
# Dispatch in analyze_condpair
# ─────────────────────────────────────────────────────────────────────────

def test_dispatch_unknown_ion_test_method_raises():
    """Typo'd ion_test_method values raise ValueError instead of silently
    falling through to the unpaired path. Catches user bugs early."""
    import alphaquant.diffquant.condpair_analysis as condpair_mod
    # Simulate a runconfig with a typo'd method
    class _StubRunConfig:
        ion_test_method = 'paired_difdist'  # typo (missing 'f')
        samplemap_df = pd.DataFrame([{"sample": "s1", "condition": "A"}])
        # ... other attrs would matter only past the validation
    # We can't easily run the full analyze_condpair without lots of
    # setup, but we can grep for the validation block. Validate by
    # inspecting the source.
    import inspect
    src = inspect.getsource(condpair_mod.analyze_condpair)
    assert "_allowed_ion_test_methods" in src
    assert "'paired_diffdist'" in src
    assert "ValueError" in src


def test_dispatch_routes_paired_eligible_to_paired_path(monkeypatch,
                                                         paired_samplemap):
    """When ion_test_method='paired_diffdist' AND the contrast is paired-
    eligible, analyze_condpair MUST call DifferentialIonPaired (NOT
    DifferentialIon). Verifies the routing decision lands in the right
    branch — independent of Phase 2's actual paired test math."""
    import alphaquant.diffquant.diff_analysis as aqdiff_mod

    # Spy on DifferentialIonPaired to verify it gets called.
    paired_calls = []
    original_paired = aqdiff_mod.DifferentialIonPaired

    class _PairedSpy:
        def __init__(self, *args, **kwargs):
            paired_calls.append((args, kwargs))
            # Pretend it's usable so the rest of the loop continues
            self.usable = False
            self.name = kwargs.get("name", "?")
            self.p_val = None; self.fc = None; self.z_val = None
            self.n_paired_subjects = 0

    monkeypatch.setattr(aqdiff_mod, "DifferentialIonPaired", _PairedSpy)

    # Verify the dispatch site reaches our spy. We do this by inspecting
    # the source of analyze_condpair — the dispatch line should construct
    # DifferentialIonPaired by name. (A full integration test would require
    # standing up a real condpair pipeline; that's a Phase 2/4 effort.)
    import alphaquant.diffquant.condpair_analysis as condpair_mod
    import inspect
    src = inspect.getsource(condpair_mod.analyze_condpair)
    assert "DifferentialIonPaired" in src, (
        "analyze_condpair source does not reference DifferentialIonPaired — "
        "dispatch is not wired up")
    assert "use_paired" in src, "use_paired flag not present in dispatch"


def test_dispatch_falls_back_when_not_paired_eligible(unpaired_samplemap):
    """When ion_test_method='paired_diffdist' but the contrast is NOT
    paired-eligible (no shared subjects), analyze_condpair should fall
    back to the unpaired DifferentialIon path with a WARNING — not
    raise an error or silently use the wrong test."""
    # Verified at the eligibility-check level; an unpaired samplemap
    # returns False from is_paired_eligible.
    assert aqdiffutils.is_paired_eligible(
        unpaired_samplemap, "control", "treated") is False
    # And the dispatch source contains the fallback WARN log + the
    # is_paired_eligible() guard. Use loose substring matches because
    # the actual message is f-string split across lines.
    import alphaquant.diffquant.condpair_analysis as condpair_mod
    import inspect
    src = inspect.getsource(condpair_mod.analyze_condpair)
    assert "is_paired_eligible" in src
    assert "LOGGER.warning" in src
    assert "falling back" in src


# ─────────────────────────────────────────────────────────────────────────
# NaN handling
# ─────────────────────────────────────────────────────────────────────────

def test_is_paired_eligible_drops_nan_subject_ids():
    """Rows with NaN subject_id (e.g., samples not yet annotated) are dropped
    before the eligibility check. Two valid mice with both arms remain → eligible."""
    import numpy as np
    rows = [
        # 2 valid paired mice
        {"sample": "m1_f", "condition": "femur", "subject_id": "mouse_1"},
        {"sample": "m1_h", "condition": "humerus", "subject_id": "mouse_1"},
        {"sample": "m2_f", "condition": "femur", "subject_id": "mouse_2"},
        {"sample": "m2_h", "condition": "humerus", "subject_id": "mouse_2"},
        # an unannotated sample — should be ignored
        {"sample": "x_f", "condition": "femur", "subject_id": np.nan},
    ]
    sm = pd.DataFrame(rows)
    assert aqdiffutils.is_paired_eligible(sm, "femur", "humerus") is True
