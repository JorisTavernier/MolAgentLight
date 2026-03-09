# Changelog

## [Unreleased] — 2026-03-09

### Bug Fixes

- **[C1] Remove debug prints** — Removed leftover `print('bloep')` and `print('bloep1')` from `evaluate_regression_model.py:240-243` that polluted subagent output.

- **[C2] Fix convert_log10 hardcoding in predict.py** — `convert_log10` was hardcoded to `True` for ALL regression models, silently corrupting predictions for non-log-transformed targets (e.g., TPSA, pKa). Now reads `use_log10` from `train_info.json` and only back-transforms when the model was actually trained on log10 data.

- **[C3] Fix NaN crash in generate_dashboard.py** — `json.dumps(allow_nan=False)` crashed when any metric was `NaN` or `Inf` (e.g., constant predictions, <3 test points). Added `sanitize_for_json()` to replace `NaN`/`Inf` with `None` before serialization.

- **[C4] Fix n_jobs free tier mismatch** — `step-1-configure.md` had `"free": inner=2` which contradicted SKILL.md, reference.md, and the actual training scripts (all use `inner=1`). Aligned to `inner=1`.

### Inconsistency Fixes

- **[S1] Restore step-6-review as real user checkpoint** — Step 6 said "Do NOT ask the user" despite being labeled a "User Checkpoint". Restored `AskUserQuestion` with 4 options: refit with test set (default), refit without test set, skip refit, or abort. Users can now stop the pipeline after seeing bad metrics.

- **[S2] Add evaluation_results to reference.md schema** — The canonical state schema was missing the `evaluation_results` field that step-5 writes and the visualize skill reads.

- **[S3] Align model_registry.py ID format** — Model IDs used dash separator (`20260207-1430`) while run folders used underscore (`20260207_1430`). Aligned to underscore for consistency.

- **[S4] Align detect_dataset.py load thresholds** — Script used `<200 → free` but reference.md documented `<100 → free, <500 → cheap`. Aligned script to match reference.md thresholds.

- **[S5] Fix browser open command in visualize SKILL.md** — Hardcoded `start` (Windows-only) replaced with platform-aware instructions for macOS (`open`), Linux (`xdg-open`), and Windows (`start`).

- **[S6] Fix predict.py docstring** — Docstring showed space-separated `--smiles-list "S1" "S2"` which Click's `multiple=True` doesn't support. Fixed to use repeated `--smiles-list` flags. Also fixed `uv run python` to `uv run` (PEP 723 compatibility).

### UX Improvements

- **[U1] Wire target transforms into step-2-prepare** — Detection recommended log10/Yeo-Johnson transforms but never instructed step-2 to apply `--use-log10`. Added explicit instructions to check `config.target_transformations` and add the flag when a `log10` transform is recommended.

- **[U2] Add include_test_in_refit user choice** — `include_test_in_refit` was hardcoded to `true` with no user input. Now part of the step-6 review question: users can choose to keep the test set clean for future reporting.

- **[U3] Timestamp predict output files** — Prediction output always overwrote the same `predictions.csv` and `predictions_info.json`. Now uses timestamped filenames (e.g., `tpsa_predictions_20260309_150000.csv`) to preserve history.

- **[U4] Add SMILES validation in predict.py** — Added NaN/empty SMILES filtering before model inference. Drops invalid entries with a warning instead of crashing deep in the RDKit stack.

### Quality Review Fixes (Phase 6)

- **[QR1] Eliminate double train_info file read in predict.py** — `train_info.json` was loaded twice: once for blender properties and again for `convert_log10`. Merged into a single load, reusing the result for both.

- **[QR2] Extend sanitize_for_json to numpy scalars** — `np.floating` and `np.integer` types (returned by sklearn/numpy metric computations) were passed through as-is, causing `json.dumps(default=str)` to stringify them. Now explicitly converts `np.floating` → `float` (or `None` if NaN/Inf) and `np.integer` → `int`.

- **[QR3] Fix SMILES NaN filter to use pd.notna()** — Previous filter `str(s) != 'nan'` missed `pd.NA` and `pd.NaT` values which convert to non-`'nan'` strings. Replaced with `pd.notna(s)` which handles all pandas NA types.

- **[QR4] Update predict SKILL.md Step 6 output filenames** — Step 6 referenced hardcoded `predictions.csv` / `{property}_predictions.csv`. These filenames are now timestamped (e.g., `predictions_20260309_150000.csv`). Updated to glob for the latest file or read the path from the companion `predictions_info_*.json`.

- **[QR5] Add evaluation_results to step-1-configure.md outputs schema** — The `outputs` object initialized in step 1.7 was missing the `"evaluation_results": {}` field that step-5 writes and the visualize skill reads. Aligned with `reference.md`.

### Files Modified

| File | Changes |
|------|---------|
| `skills/train-pipeline/scripts/evaluate_regression_model.py` | Remove debug prints |
| `skills/predict/scripts/predict.py` | Fix convert_log10, add timestamps, add SMILES validation, fix docstring, eliminate double train_info read, use pd.notna() |
| `skills/visualize/scripts/generate_dashboard.py` | Fix NaN crash with sanitize_for_json, add numpy scalar handling |
| `skills/train-pipeline/steps/step-1-configure.md` | Fix n_jobs free tier, add evaluation_results to outputs schema |
| `skills/train-pipeline/steps/step-2-prepare.md` | Add target transform instructions |
| `skills/train-pipeline/steps/step-6-review.md` | Restore as real user checkpoint |
| `skills/train-pipeline/reference.md` | Add evaluation_results to schema |
| `skills/train-pipeline/scripts/model_registry.py` | Align ID format (dash → underscore) |
| `skills/train-pipeline/scripts/detect_dataset.py` | Align load thresholds |
| `skills/visualize/SKILL.md` | Platform-aware browser open |
| `skills/predict/SKILL.md` | Update Step 6 output filename references |
