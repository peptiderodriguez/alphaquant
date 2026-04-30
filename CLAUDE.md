# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

AlphaQuant is a Python proteomics package implementing **tree-based quantification**: a hierarchical model where MS data is organized as fragments → MS1 isotopes → charge states → modifications → peptides → genes/proteins. Statistics computed at ion level are aggregated up the tree (Stouffer's Z-method for z-values, median for fold changes), and peptides within a protein are clustered to infer proteoforms.

Part of the AlphaPept ecosystem (Mann Labs). Supported Python: 3.10/3.11/3.12.

## Common commands

Developer install (creates conda env, editable install):
```bash
conda create --name alphaquant python=3.11 -y
conda activate alphaquant
pip install -e ".[development]"          # loose deps + dev/test extras
# or pinned: pip install -e ".[stable,development]"
```

Tests (run from `tests/`, requires the `alphaquant` conda env to be active):
```bash
# unit tests (skips anything marked 'slow')
coverage run --source=../alphaquant -m pytest -k 'not slow'

# single test
pytest unit_tests/test_diff_analysis.py::TestName::test_case

# quick e2e tests (notebook execution via nbmake)
pytest --nbmake e2e_tests_small/mixed_species.ipynb
pytest --nbmake e2e_tests_small/phospho.ipynb
pytest --nbmake e2e_tests_small/different_input_tables.ipynb

# all example notebooks (cd ../example_nbs first)
pytest --nbmake . --ignore=tree_example.ipynb

# requirements alignment check (strict pin ↔ loose unpinned)
pytest pytest/test_requirements.py
```

Some unit tests download data from S3 via `tests/download_testfiles.py`; CI does this before running tests.

GUI / CLI:
```bash
alphaquant gui     # launches the Panel-based dashboard at :41215
```

Programmatic entry point:
```python
import alphaquant.run_pipeline as aq_pipeline
aq_pipeline.run_pipeline(input_file=..., samplemap_file=..., results_dir=...)
```

Version bumps go through `bump-my-version` driven by `.bumpversion.toml` — it updates `alphaquant/__init__.py` *and* installer/release files (`release/linux/control`, `release/macos/{distribution.xml,Info.plist,build_package_macos.sh}`, `release/windows/alphaquant_innoinstaller.iss`). Don't edit version strings by hand; use the **Bump version** GitHub Actions workflow or `bump-my-version bump {patch,minor,major,prerelease}`.

## High-level architecture

The pipeline is orchestrated top-down by `alphaquant/run_pipeline.py::run_pipeline`, which dispatches each condition pair to `alphaquant/diffquant/condpair_analysis.py::analyze_condpair`. That function is the single place where all subpackages are wired together — read it first when tracing data flow.

Per-condition-pair stages (in order):
1. **Read & filter** — input tables are normalized to AlphaQuant's internal long format via `alphabase.quantification.quant_reader` driven by `alphaquant/config/quant_reader_config.yaml` (a separate `_lightweight.yaml` exists for low-memory paths). Search-engine-specific column mapping happens here; supported engines: DIA-NN, AlphaDIA, Spectronaut, MaxQuant, FragPipe, AlphaPept.
2. **Normalize** — `alphaquant/norm/normalization.py` (within- and between-condition).
3. **Background distributions** — `alphaquant/diffquant/background_distributions.py` builds intensity-dependent empirical null distributions per ion.
4. **Ion-level test** — `alphaquant/diffquant/diff_analysis.py::DifferentialIon` (default; uses the empirical background) or `DifferentialIonTTest` (Welch's t-test variant). Each ion produces `(p_val, fc, z_val)`. The choice is gated by `runconfig.ion_test_method` (`'diffdist'` vs `'ttest'`).
5. **Tree aggregation & clustering** — `alphaquant/cluster/cluster_ions.py` walks levels bottom→top; `cluster_utils.py::aggregate_node_properties` combines child stats (Stouffer rescale in `sum_and_re_scale_zvalues`, median fc). `cluster_utils.py::remove_outlier_fragion_childs` trims extreme fragments before peptide aggregation. Proteoform clustering lives in `alphaquant/cluster/{cluster_sorting,proteoform_statistics}.py`.
6. **ML quality scoring** — `alphaquant/classify/` trains/applies a model that re-orders ions by quality (`cluster/ml_reorder.py`).
7. **Outlier filtering** — `alphaquant/cluster/outlier_filtering.py` (peptide-level) plus `diff_analysis.calc_outlier_scaling_factor` (ion-level variance inflation).
8. **Output tables + FDR** — `alphaquant/tables/diffquant_table.py` (proteins; BH for the intensity-based set, adjusted BH for the missing-value-only "counting" set) and `alphaquant/tables/proteoformtable.py` (proteoform regulation vs. reference). All output tables carry raw p-values so external recorrection is always possible.
9. **Plots** — `alphaquant/plotting/` (volcano, fc vs intensity, tree visualizations, proteoform views).

Other top-level subpackages:
- `alphaquant/multicond/` — median-condition construction and multi-condition analysis on top of pairwise results.
- `alphaquant/ptm/` — phospho/PTM site mapping, proteome normalization for PTM sites, phospho inference.
- `alphaquant/benchm/` — benchmarking helpers and the `TestFileDownloader` used by tests.
- `alphaquant/ui/` — `cli.py` (click entry point, exposed as the `alphaquant` script) and the Panel GUI in `gui.py` + `dashboard_parts_*.py`.
- `alphaquant/config/variables.py` — global runtime variables; `config.py` configures logging.

When extending statistical behavior, see `DEVELOPERS.md` — it documents the exact extension points (ion test class, Z aggregation, FDR step, outlier handling) with file/line references. Any change to these methods needs benchmarking on ground-truth datasets before being defaulted on.

## Conventions worth knowing

- **Two parallel requirements files per extra**: `requirements/requirements_X_loose.txt` (unpinned, package names only) and `requirements/requirements_X.txt` (strict `==` pins). `tests/pytest/test_requirements.py` enforces that they contain the same package set and that strict versions satisfy loose constraints. Annotate intentional exceptions with the comment `test: tolerate_version`.
- **GUI is optional** — installed via the `gui`/`gui-stable` extras; the GUI deps live in `requirements_gui[_loose].txt`.
- **Spectronaut input expects a specific export schema** — the `.rs` files in `alphaquant/config/` (`spectronaut_tableconfig_{precursor,fragion,ptm_fragion}.rs`) must be loaded into Spectronaut before exporting; mismatched columns will silently produce wrong inputs.
- **CI**: `install_and_unit_tests.yml` runs unit tests on Ubuntu × Py 3.10/3.11/3.12 on every push to `main`; `e2e_tests_quick_multiple_platforms.yml` runs the e2e notebooks across all three OS. Adding a `code-review` label to a PR triggers an automated AI review (`code_review.yml`).
- **Releases are built with PyInstaller** from `release/{linux,macos,windows,pyinstaller}/`; one-click installers are produced per OS. Touching version strings or bundle resources here without going through `bump-my-version` will desync the installers.
