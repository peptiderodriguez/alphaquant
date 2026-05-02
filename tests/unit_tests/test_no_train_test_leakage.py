"""Regression test: train-test leakage CANNOT recur in alphaquant's
ML quality scoring.

After the multi-agent audit (D1 + D2 → consolidation into
``_train_with_nested_cv``), we have one canonical training loop. These
tests pin down the leakage-free invariants so a future "small refactor"
cannot silently reintroduce the bug class without breaking a test.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# D2 invariant — RandomizedSearchCV / GridSearchCV is fit on the OUTER
# TRAINING FOLD ONLY. Pre-fix: train_fast_gradient_boosting fit it on
# the FULL X, leaking test-fold info into hyperparameter selection.
# ---------------------------------------------------------------------------
def test_train_fast_gradient_boosting_search_fit_only_on_training_fold():
    """The inner RandomizedSearchCV.fit must never receive the full X."""
    from alphaquant.classify import training_functions as tf
    from sklearn.model_selection import RandomizedSearchCV

    seen_shapes: list[int] = []
    real_fit = RandomizedSearchCV.fit

    def spy_fit(self, X, y=None, **kw):
        seen_shapes.append(X.shape[0])
        return real_fit(self, X, y, **kw)

    rng = np.random.default_rng(0)
    n = 500
    X = rng.normal(size=(n, 8))
    y = rng.normal(size=n)

    # 5-fold CV → outer training set has 4/5 = 400 rows. Search must NEVER
    # see all 500. Allow ±1 for partition rounding.
    monkey = pytest.MonkeyPatch()
    monkey.setattr(RandomizedSearchCV, "fit", spy_fit)
    try:
        tf.train_fast_gradient_boosting(X, y, shorten_features_for_speed=True,
                                         num_splits=5, n_iter=2)
    finally:
        monkey.undo()

    assert seen_shapes, "RandomizedSearchCV.fit must have been called"
    assert all(s <= n * 4 / 5 + 2 for s in seen_shapes), (
        f"D2 leakage: search fit saw {max(seen_shapes)} rows of {n}; "
        f"must be ≤ outer training fold size (~{int(n*4/5)})."
    )


# ---------------------------------------------------------------------------
# D1 invariant — train_random_forest_ensemble returns OOF predictions.
# Pre-fix: it averaged 5 K-fold-trained models on the FULL X, biasing each
# ion's prediction toward "memorized" labels.
# ---------------------------------------------------------------------------
def test_train_random_forest_ensemble_returns_oof_predictions():
    """Each row of y_pred_cv must come from a model that did NOT see that
    row in its training fold. We verify this end-to-end by:
      1. running the trainer on synthetic data,
      2. independently re-running KFold on the same X with same seed,
      3. for each test fold, predicting from the corresponding model and
         asserting equality with y_pred_cv on the test indices.
    """
    from alphaquant.classify.classify_fragments import train_random_forest_ensemble
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(42)
    n, d = 200, 5
    X = rng.normal(size=(n, d)).astype(np.float64)
    y = rng.normal(size=n).astype(np.float64)

    models, _, y_pred_cv = train_random_forest_ensemble(
        X, y, shorten_features_for_speed=True, num_splits=5,
    )

    assert len(models) == 5
    assert y_pred_cv.shape == (n,)

    # Recompute OOF predictions independently — must match exactly.
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_check = np.zeros_like(y_pred_cv)
    for fold, (_, test_idx) in enumerate(kf.split(X)):
        oof_check[test_idx] = models[fold].predict(X[test_idx])

    # rtol/atol at ~ machine epsilon: parallel RF prediction (n_jobs=-1) can
    # reorder float sums and yield differences of ~1e-16. The OOF property
    # is structural (each row predicted from a model that didn't train on
    # it); float-noise tolerance doesn't weaken the invariant.
    np.testing.assert_allclose(oof_check, y_pred_cv, rtol=1e-10, atol=1e-12,
                                 err_msg="D1: y_pred_cv is not the OOF prediction")


def test_train_random_forest_ensemble_returns_three_tuple():
    """Returns (models, test_set_predictions, y_pred_cv) — tuple shape and
    length, mirroring the canonical helper. Pre-fix: returned only models."""
    from alphaquant.classify.classify_fragments import train_random_forest_ensemble

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4)).astype(np.float64)
    y = rng.normal(size=120).astype(np.float64)

    out = train_random_forest_ensemble(X, y, shorten_features_for_speed=True,
                                         num_splits=5)
    assert len(out) == 3
    models, test_predictions, y_pred_cv = out
    assert isinstance(models, list) and len(models) == 5
    assert isinstance(test_predictions, list) and len(test_predictions) == 5
    assert hasattr(y_pred_cv, "shape") and y_pred_cv.shape == (120,)


# ---------------------------------------------------------------------------
# D1 + D2 — set_params is used (not setattr) for hyperparameter assignment.
# This is a structural guard against the silent-no-op-on-typo bug class.
# ---------------------------------------------------------------------------
def test_set_params_used_for_best_params_assignment():
    """If a future refactor hands `setattr` (or any other un-validated
    mechanism) the hyperparameters, sklearn won't catch typos in the
    param dict. Verify set_params is actually invoked."""
    from alphaquant.classify import training_functions as tf
    from sklearn.ensemble import HistGradientBoostingRegressor

    set_params_calls: list[dict] = []
    real = HistGradientBoostingRegressor.set_params

    def spy(self, **params):
        set_params_calls.append(params)
        return real(self, **params)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = rng.normal(size=300)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(HistGradientBoostingRegressor, "set_params", spy)
    try:
        tf.train_fast_gradient_boosting(X, y, shorten_features_for_speed=True,
                                         num_splits=5, n_iter=2)
    finally:
        monkey.undo()

    # We expect at least one set_params call carrying real best_params keys.
    assert set_params_calls, "set_params was never called — D2 fix may have regressed to setattr"
    assert any("learning_rate" in p or "max_depth" in p or "max_iter" in p
                for p in set_params_calls), (
        f"set_params was called but never with model hyperparameters; "
        f"saw: {set_params_calls}"
    )
