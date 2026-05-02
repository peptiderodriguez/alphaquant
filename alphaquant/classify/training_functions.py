import alphaquant.config.config as aqconfig

import sklearn.ensemble
import sklearn.linear_model
import sklearn.impute
import sklearn.metrics
import sklearn.model_selection

from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

import logging
aqconfig.setup_logging()
LOGGER = logging.getLogger(__name__)

import numpy as np


# ---------------------------------------------------------------------------
# Canonical nested-CV training helper.
#
# Every training function in this module is a thin wrapper around this one.
# The contract:
#   - Outer KFold over X (num_splits).
#   - For each outer fold, hyper-parameter search (Grid or Randomized) is fit
#     on the *training fold only* — never on the full X. This prevents the
#     search itself from peeking at the held-out test fold (the bug that
#     motivated the refactor: see audit fix D2).
#   - Test predictions go into y_pred_cv at the test indices, producing a
#     truly out-of-fold (OOF) prediction vector that is leakage-free for the
#     downstream consumer (e.g. classify_precursors → ml_score).
#   - Per-fold random_state = base_random_state + fold_num, mirroring the
#     original per-fold determinism convention.
# ---------------------------------------------------------------------------

def _train_with_nested_cv(
    X, y,
    *,
    model_factory,                 # callable: random_state -> estimator
    param_search=None,             # None | {"distributions": dict, "n_iter": int} | {"grid": dict}
    num_splits=5,
    inner_cv=3,
    random_state=42,
    compute_perm_importance=False,
    y_abs=False,
    feature_selector=None,         # callable(X, y) -> X_transformed
    n_jobs=-1,
    scoring=None,
    logger=LOGGER,
):
    """Outer K-fold with optional inner hyper-parameter search per fold.

    Hyper-parameter search runs INSIDE the outer fold's training set only.
    Returns ``(models, test_set_predictions, y_pred_cv)`` where ``y_pred_cv``
    is OOF and leakage-free for every ion in the input.

    Parameters
    ----------
    X, y
        Feature matrix and target vector.
    model_factory
        ``callable(random_state) -> estimator``. Used both to build the
        per-fold base estimator that is wrapped in {Grid,Randomized}SearchCV
        AND to build the final per-fold estimator from ``best_params_``.
    param_search
        ``None`` for no hyper-parameter search (just fit ``model_factory``).
        ``{"distributions": dict, "n_iter": int}`` runs RandomizedSearchCV.
        ``{"grid": dict}`` runs GridSearchCV.
    num_splits
        Outer K-fold splits.
    inner_cv
        Inner CV used by the hyper-parameter search.
    random_state
        Outer KFold seed; per-fold model seed is ``random_state + fold_num``.
    compute_perm_importance
        If True, replaces ``model.feature_importances_`` with permutation
        importance computed on the held-out test fold (used by the histGB
        path which lacks native importances).
    y_abs
        If True, replaces ``y`` with ``np.abs(y)`` before training. Several
        legacy callers want to predict magnitudes only.
    feature_selector
        Optional ``callable(X, y) -> X_transformed`` applied BEFORE the outer
        K-fold split. Caller is responsible for guaranteeing the selector is
        leakage-free for its own use case (the legacy ``shorten_features_for_speed``
        f-regression selector is technically supervised but is preserved
        verbatim for behavioural parity with the pre-refactor code).
    n_jobs, scoring
        Passed through to {Grid,Randomized}SearchCV.
    logger
        Where per-fold progress messages go.
    """
    if y_abs:
        y = np.abs(y)

    if feature_selector is not None:
        X = feature_selector(X, y)

    models = []
    test_set_predictions = []
    y_pred_cv = np.zeros_like(y, dtype=float)

    kf = sklearn.model_selection.KFold(
        n_splits=num_splits, shuffle=True, random_state=random_state,
    )

    for fold_num, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        per_fold_seed = random_state + fold_num

        if param_search is None:
            model = model_factory(per_fold_seed)
            model.fit(X_train, y_train)
            best_params = None
        else:
            base_estimator = model_factory(per_fold_seed)
            inner_kf = sklearn.model_selection.KFold(
                n_splits=inner_cv, shuffle=True, random_state=random_state,
            )
            if "distributions" in param_search:
                searcher = RandomizedSearchCV(
                    estimator=base_estimator,
                    param_distributions=param_search["distributions"],
                    n_iter=param_search.get("n_iter", 10),
                    cv=inner_kf,
                    scoring=scoring or "neg_mean_squared_error",
                    n_jobs=n_jobs,
                    verbose=0,
                    random_state=random_state,
                    return_train_score=True,
                    error_score=param_search.get("error_score", "raise"),
                )
            elif "grid" in param_search:
                searcher = GridSearchCV(
                    estimator=base_estimator,
                    param_grid=param_search["grid"],
                    cv=inner_kf,
                    scoring=scoring or "r2",
                    n_jobs=n_jobs,
                    verbose=0,
                    return_train_score=True,
                )
            else:
                raise ValueError(
                    f"param_search must contain 'distributions' or 'grid'; "
                    f"got keys {list(param_search.keys())}"
                )

            # CRITICAL: fit the search on TRAINING fold only — never on full X.
            # This is the leakage-fix that motivated the refactor.
            searcher.fit(X_train, y_train)
            best_params = searcher.best_params_
            logger.info(f"Fold {fold_num} Best parameters found: {best_params}")
            # Re-instantiate from best_params with the per-fold seed so the
            # final fold model uses the same seeding convention as the
            # no-search code path. Use set_params (not setattr) so sklearn
            # validates the names; setattr would silently no-op on typos.
            model = model_factory(per_fold_seed)
            model.set_params(**best_params)
            model.fit(X_train, y_train)

        if compute_perm_importance:
            perm_importance = permutation_importance(
                model, X_test, y_test, n_repeats=5,
                random_state=random_state,
                scoring=scoring or "neg_mean_squared_error",
            )
            feature_importances = perm_importance.importances_mean
            feature_importances = np.maximum(feature_importances, 0)
            if np.sum(feature_importances) > 0:
                feature_importances = feature_importances / np.sum(feature_importances)
            else:
                feature_importances = np.zeros_like(feature_importances)
            model.feature_importances_ = feature_importances

        y_pred_test = model.predict(X_test)
        y_pred_cv[test_index] = y_pred_test

        models.append(model)
        test_set_predictions.append((y_test, y_pred_test))

        fold_mse = np.mean((y_test - y_pred_test) ** 2)
        logger.info(f"Fold {fold_num} MSE: {fold_mse}")
        if len(y_test) >= 2 and np.std(y_test) > 0 and np.std(y_pred_test) > 0:
            correlation = np.corrcoef(y_test, y_pred_test)[0, 1]
            logger.info(f"Overall correlation in fold {fold_num}: {correlation}")

    return models, test_set_predictions, y_pred_cv


# ---------------------------------------------------------------------------
# Public API — thin wrappers preserving original signatures + return tuple.
# ---------------------------------------------------------------------------

def train_random_forest_with_grid_search(X, y, shorten_features_for_speed, num_splits=5):
    if shorten_features_for_speed:
        max_features_choices = ['sqrt']
    else:
        max_features_choices = [None, 'sqrt']
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_leaf': [1, 5],
        'max_features': max_features_choices,
        'max_samples': [None, 0.8],
    }

    def factory(seed):
        return sklearn.ensemble.RandomForestRegressor(random_state=seed)

    return _train_with_nested_cv(
        X, y,
        model_factory=factory,
        param_search={"grid": param_grid},
        num_splits=num_splits,
        inner_cv=3,
        random_state=42,
        y_abs=True,
        scoring='r2',
    )


def train_gradient_boosting_with_grid_search(X, y, shorten_features_for_speed, num_splits=5):
    if shorten_features_for_speed:
        max_features_choices = ['sqrt']
    else:
        max_features_choices = [None, 'sqrt']
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_leaf': [1, 5],
        'max_features': max_features_choices,
        'subsample': [1.0, 0.8],
    }

    def factory(seed):
        return sklearn.ensemble.GradientBoostingRegressor(random_state=seed)

    return _train_with_nested_cv(
        X, y,
        model_factory=factory,
        param_search={"grid": param_grid},
        num_splits=num_splits,
        inner_cv=3,
        random_state=42,
        y_abs=True,
        scoring='r2',
    )


def train_gradient_boosting_with_random_search(X, y, shorten_features_for_speed, num_splits=5, n_iter=10):
    if shorten_features_for_speed:
        max_features_choices = ['sqrt']
    else:
        max_features_choices = [None, 'sqrt']
    param_distributions = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_leaf': [1, 5],
        'max_features': max_features_choices,
        'subsample': [1.0, 0.8],
    }

    def factory(seed):
        return sklearn.ensemble.GradientBoostingRegressor(random_state=seed)

    return _train_with_nested_cv(
        X, y,
        model_factory=factory,
        param_search={"distributions": param_distributions, "n_iter": n_iter},
        num_splits=num_splits,
        inner_cv=3,
        random_state=42,
        y_abs=True,
        scoring='neg_mean_squared_error',
    )


def train_fast_gradient_boosting(X, y, shorten_features_for_speed, num_splits=3, n_iter=10):
    """HistGradientBoostingRegressor with per-fold RandomizedSearchCV.

    Audit fix D2: the previous implementation called
    ``RandomizedSearchCV.fit(X, y)`` on the FULL dataset to pick best_params,
    then re-trained per fold. That's hyperparameter selection on the test
    folds — a leakage path. We now run the random search inside each outer
    fold so neither the chosen hyperparameters nor the OOF predictions are
    informed by the held-out data.
    """
    LOGGER.info("Starting train_fast_gradient_boosting (per-fold RandomizedSearchCV; no full-X search)")

    # f-regression feature selection is supervised; preserved verbatim from
    # the original implementation for behavioural parity. It is applied
    # before the outer split (matching legacy behaviour) — see helper docstring.
    if shorten_features_for_speed:
        from sklearn.feature_selection import SelectKBest, f_regression

        def selector_fn(X_in, y_in):
            sel = SelectKBest(score_func=f_regression, k=int(X_in.shape[1] * 0.5))
            return sel.fit_transform(X_in, y_in)
        feature_selector = selector_fn
    else:
        feature_selector = None

    param_distributions = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5],
        'min_samples_leaf': [1, 5],
        'max_iter': [100, 200],
        'max_bins': [255],
        'max_leaf_nodes': [31],
        'l2_regularization': [0, 0.1],
        'early_stopping': [True],
    }

    def factory(seed):
        return HistGradientBoostingRegressor(random_state=seed)

    return _train_with_nested_cv(
        X, y,
        model_factory=factory,
        param_search={
            "distributions": param_distributions,
            "n_iter": n_iter,
            "error_score": "raise",
        },
        num_splits=num_splits,
        inner_cv=3,
        random_state=42,
        compute_perm_importance=True,   # histGB has no native feature_importances_
        y_abs=True,
        feature_selector=feature_selector,
        n_jobs=1,                       # legacy behaviour: search uses n_jobs=1
        scoring='neg_mean_squared_error',
    )


def train_random_forest_simple(X, y, shorten_features_for_speed, num_splits=5):
    # NOTE: ``shorten_features_for_speed`` is unused in the original; kept in
    # the signature for backwards compatibility.
    _ = shorten_features_for_speed

    def factory(seed):
        return sklearn.ensemble.RandomForestRegressor(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1,
            max_features='sqrt',
            max_depth=5,
            min_samples_leaf=5,
            bootstrap=True,
            max_samples=0.8,
        )

    return _train_with_nested_cv(
        X, y,
        model_factory=factory,
        param_search=None,
        num_splits=num_splits,
        random_state=42,
        y_abs=True,
    )
