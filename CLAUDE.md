# CLAUDE.md

## Project overview

AlphaQuant is a Python proteomics package implementing **tree-based quantification**. MS data is organized as fragments → MS1 isotopes → charge states → modifications → peptides → genes/proteins. Ion-level statistics are aggregated upward (Stouffer's Z for z-values, median for fold changes); peptides within a protein are clustered to infer proteoforms.

Part of the AlphaPept ecosystem (Mann Labs). Python 3.10/3.11/3.12.

## Common commands

Install (developer mode):
```bash
conda create --name alphaquant python=3.11 -y && conda activate alphaquant
pip install -e ".[development]"                  # loose deps + dev/test
# pinned alternative: pip install -e ".[stable,development]"
```

Tests (run from `tests/`, alphaquant env active):
```bash
coverage run --source=../alphaquant -m pytest -k 'not slow'   # unit
pytest unit_tests/test_diff_analysis.py::TestName::test_case  # single test
pytest --nbmake e2e_tests_small/{mixed_species,phospho,different_input_tables}.ipynb
pytest --nbmake . --ignore=tree_example.ipynb                 # all example_nbs (cd ../example_nbs first)
pytest pytest/test_requirements.py                            # strict-pin ↔ loose-pin alignment
```
Tests that need data fetch it from `datashare.biochem.mpg.de` (links in `alphaquant/config/download_links_*.yaml`); the e2e notebooks download their own via `DataShareDownloader`. `tests/download_testfiles.py` is currently BROKEN — it calls `benchm.testfile_handling.TestFileDownloader`, whose module was deleted in b954be1 — and CI does not run it.

GUI: `alphaquant gui` (Panel dashboard; port from `$PORT`, else random — 41215 is set only in the Docker image).

Programmatic:
```python
import alphaquant.run_pipeline as aq_pipeline
aq_pipeline.run_pipeline(input_file=..., samplemap_file=..., results_dir=...)
```

Version bumps: `bump-my-version bump {patch,minor,major,prerelease}` or the **Bump version** GitHub Actions workflow. `.bumpversion.toml` updates `alphaquant/__init__.py` *and* specific installer/release files under `release/` (exact list in `.bumpversion.toml`). Never hand-edit version strings.

## High-level architecture

`alphaquant/run_pipeline.py::run_pipeline` orchestrates top-down and dispatches each condition pair to `alphaquant/diffquant/condpair_analysis.py::analyze_condpair` — the single place where all subpackages wire together. **Read it first when tracing data flow.**

Per-condition-pair stages (paths below are relative to `alphaquant/`):
1. **Read & filter** — input → AlphaQuant long format via `alphabase.quantification.quant_reader`, driven by `alphaquant/config/quant_reader_config.yaml` (a parallel `_lightweight.yaml` is maintained, but `run_pipeline.py` hard-wires the full config). Engines: DIA-NN, AlphaDIA, Spectronaut, MaxQuant, FragPipe, AlphaPept.
2. **Normalize** — `alphaquant/norm/normalization.py` (within- and between-condition).
3. **Background distributions** — `diffquant/background_distributions.py` builds intensity-dependent empirical nulls per ion.
4. **Ion-level test** — `diffquant/diff_analysis.py::DifferentialIon` (empirical, the default) or `DifferentialIonTTest` (Welch). Output per ion: `(p_val, fc, z_val)`. Gated by `runconfig.ion_test_method ∈ {'diffdist','ttest','paired_diffdist'}` (allowed set in `diffquant/condpair_analysis.py`). `'paired_diffdist'` → `DifferentialIonPaired`, a within-subject test for repeated-measures cohorts; it also swaps the stage-3 null for a paired one and the stage-5 Stouffer for a correlation-aware one, and adds a sign-flip permutation step before the tables. **At n ≤ 8 subjects its analytic p is anti-conservative and must not drive FDR — the permutation p is the canonical `p_value` column.** Design + validation: `docs/paired_diffdist_design.md`. `calc_outlier_scaling_factor` (ion-level variance inflation) also runs here.
5. **Tree aggregation & clustering** — `cluster/cluster_ions.py` walks bottom→top; `cluster_utils.aggregate_node_properties` combines child stats (Stouffer rescale in `sum_and_re_scale_zvalues`, median fc); `remove_outlier_fragion_childs` trims fragments before peptide aggregation. Proteoform clustering: `cluster/{cluster_sorting,proteoform_statistics}.py`.
6. **ML quality scoring** — `classify/` trains/applies a quality model; ions reordered by `cluster/ml_reorder.py`.
7. **Outlier filtering** — `cluster/outlier_filtering.py` (peptide-level; the ion-level factor runs during stage 4).
8. **Output tables + FDR** — `tables/diffquant_table.py` (proteins; BH on intensity-based set, adjusted BH on missing-value counting set) + `tables/proteoformtable.py` (proteoform regulation vs. reference). Tables carry raw p-values for external re-correction.
9. **Plots** — `plotting/` (volcano, fc-vs-intensity, tree views, proteoform views).

Other subpackages:
- `multicond/` — median-condition construction + multi-condition analysis layered on pairwise results.
- `ptm/` — phospho/PTM site mapping, PTM-site proteome normalization, phospho inference.
- `benchm/` — benchmarking helpers.
- `ui/` — `cli.py` (click; exposed as `alphaquant`) + Panel GUI (`gui.py`, `dashboard_parts_*.py`).
- `config/variables.py` — runtime globals; `config.py` configures logging.

`DEVELOPERS.md` documents the extension points (ion test class, Z aggregation, FDR step, outlier handling) with file/line references. Statistical-method changes require ground-truth benchmarking before becoming default.

## Conventions

- **Parallel requirements per extra**: `requirements/requirements_X_loose.txt` (unpinned) + `requirements_X.txt` (`==` pins). `tests/pytest/test_requirements.py` enforces same package set + strict ⊨ loose. Annotate intentional exceptions with `# test: tolerate_version`.
- **GUI is optional** — `gui`/`gui-stable` extras; deps in `requirements_gui[_loose].txt`.
- **Spectronaut needs a specific export schema** — `.rs` files in `alphaquant/config/` (`spectronaut_tableconfig_{precursor,fragion,ptm_fragion}.rs`) must be loaded into Spectronaut first. Mismatched columns silently produce wrong inputs.
- **CI**: `install_and_unit_tests.yml` (unit, Ubuntu × Py 3.10/3.11/3.12, every push to `main`); `e2e_tests_quick_multiple_platforms.yml` (e2e notebooks, all 3 OS). PR label `code-review` triggers `code_review.yml` (automated AI review).
- **Releases**: PyInstaller from `release/{linux,macos,windows,pyinstaller}/`. Editing version strings or bundle resources outside `bump-my-version` desyncs the installers.
- **Classify training goes through one helper.** Every training function in `classify/training_functions.py` (plus `train_random_forest_ensemble` in `classify_fragments.py`) wraps `_train_with_nested_cv`, which guarantees: (1) hyper-parameter search fits only on the outer fold's training set (audit fix D2 — no leakage from held-out folds); (2) returns leakage-free OOF predictions (`y_pred_cv`) for the training set, with `predict_on_models` reserved for the genuinely-unseen remainder (audit fix D1) — PROVIDED no supervised `feature_selector` is passed, since one runs on the full X,y BEFORE the outer split (`training_functions.py:83-98`). `shorten_features_for_speed=True` is such a selector; the production caller passes `False`. New training methods MUST wrap `_train_with_nested_cv` — never reimplement the K-fold loop. See `DEVELOPERS.md`.
