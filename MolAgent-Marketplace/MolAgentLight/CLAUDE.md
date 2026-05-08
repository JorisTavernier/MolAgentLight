# AutoMol Plugin — Development Guide

## Architecture

This is a Claude Code plugin for the AutoMol molecular ML library. It has 3 skills (`train-pipeline`, `predict`, `visualize`) and a SessionStart hook.

### Key Directories

- `skills/train-pipeline/scripts/` — Python scripts invoked with `uv run script.py`. With the `.venv` activated, `uv run` uses it for all scripts.
- `skills/predict/scripts/` — Prediction script, invoked the same way.
- `skills/visualize/scripts/` — Dashboard generation script (PEP 723, self-contained dependencies).
- `MolagentFiles/` — All pipeline outputs. Only `model_registry.json` is version-controlled.
- `hooks/setup-automol-env.sh` — Exports `AUTOMOL_ROOT` (repo root) and `MOLAGENT_PLUGIN_ROOT` (plugin root) at session start.

### Execution Pattern

All scripts: `uv run <script> --option value`

With the `.venv` activated (SessionStart hook), `uv run` uses the venv for all scripts. Do not use `uv run python` — for PEP 723 scripts it bypasses inline metadata and fails.

Virtual env activation is needed when the system Python doesn't have AutoMol installed:
- Linux/macOS: `source ${AUTOMOL_VENV:-.venv}/bin/activate`
- Windows: `${AUTOMOL_VENV:-.venv}/Scripts/activate`

### Windows Troubleshooting

If `uv` is not found after installing it with `pip`, add the Python Scripts directory to your PATH:

```bash
export PATH="$HOME/AppData/Roaming/Python/PythonXX/Scripts:$PATH"
```

Replace `PythonXX` with your Python version, e.g. `Python312`.

If you are behind a corporate firewall or proxy, set native TLS so `uv` uses the system certificate store:

```bash
export UV_NATIVE_TLS=true
```

### Pipeline Version

Current: **2.0** — Run folder isolation, Task-based dispatch, unified `"outputs"` key.

### Run Folder Convention

Each pipeline run creates a self-contained folder: `MolagentFiles/{dataset}-{props}-{YYYYMMDD}_{HHMM}/`. All pipeline outputs (prepared CSV, models, state, model card) live inside the run folder. The global `MolagentFiles/model_registry.json` indexes all runs.

### Task-Based Dispatch

Steps 2, 3, 4, 5, and 7 are dispatched as `Task(subagent_type="general-purpose", model="sonnet")`. The orchestrator (SKILL.md) reads the step file content, combines it with the pipeline state, and sends it to a general-purpose subagent. Inline steps (0, 1, 6, 8) run in the main context for user interaction.


### Pipeline state key

The pipeline state uses `"outputs"` (not `"files"`) for file paths. Always read with fallback: `state.get("outputs", state.get("files", {}))`.

### Nested model files

Model files are stored per-property: `outputs.model_files.gamma1`, not as a flat string. Scripts must handle both `dict` and `str` formats.

### Refitted model filenames

Refitted models have `_refitted` in the filename (`gamma1_refitted_stackingregmodel.pt`). When extracting property names, strip this suffix or predictions will silently fail to produce output columns.

### SMILES column

The standardized column after preparation is `Stand_SMILES`. Raw input files may use `smiles`, `SMILES`, or others. The predict script auto-detects — don't force the training column name on new data.

### Classification: quantile-binning collapse

When preparing binary classification data, quantile-binning may collapse `Class_{prop}` to a single class if the original column is already binary (0/1). `train_classification.py` auto-detects this and falls back to the original column. When this happens, `labelnames` must be regenerated for the fallback property.

### Classification: JSON labelnames keys

`labelnames` loaded from JSON have string keys (`"0"`) but upstream scikit-learn indexes with `int`. Always convert: `{int(k): v for k, v in mapping.items()}`.

### blender_properties None vs empty list

Use `is not None` (not truthiness) when checking `blender_properties`. An empty list `[]` means "no blenders specified" but `None` means "auto-detect from prep info".

## Pipeline State Schema

```json
{
  "pipeline_version": "2.0",
  "run_id": "raw_data-gamma1_gamma2_gamma3-20260207_1430",
  "current_step": 8,
  "steps_completed": [0, 1, 2, 3, 4, 5, 6, 7, 8],
  "pipeline_complete": true,
  "config": {
    "data_file": "path/to/data.csv",
    "smiles_column": "smiles",
    "target_properties": ["prop1", "prop2"],
    "task_type": "regression|classification",
    "feature_keys": ["Bottleneck"],
    "computational_load": "free|cheap|moderate|expensive",
    "split_strategy": "mixed",
    "use_advanced": false,
    "output_folder": "MolagentFiles/raw_data-prop1_prop2-20260207_1430/"
  },
  "outputs": {
    "prepared_csv": "MolagentFiles/.../automol_prepared_X.csv",
    "split_csv": "MolagentFiles/.../automol_split_X.csv",
    "model_files": { "prop1": "MolagentFiles/.../prop1_stackingregmodel.pt" },
    "merged_model_file": "MolagentFiles/.../merged_stackingregmodel.pt",
    "refitted_model_files": { "prop1": "MolagentFiles/.../prop1_refitted_stackingregmodel.pt" },
    "merged_refitted_model_file": "MolagentFiles/.../merged_refitted_stackingregmodel.pt",
    "train_info": { "prop1": "MolagentFiles/.../prop1_train_info.json" },
    "evaluation_results": { "prop1": "MolagentFiles/.../prop1_evaluation_predictions.csv" }
  },
  "metrics": {
    "prop1": { "r2": 0.85, "mse": 0.12 }
  }
}
```

## File Naming Conventions

| Step | Pattern |
|------|---------|
| Prepare | `automol_prepared_{stem}.csv` + `_info.json` |
| Split | `automol_split_{stem}.csv` + `_info.json` |
| Train (reg) | `{property}_stackingregmodel.pt` + `{property}_train_info.json` |
| Train (clf) | `{property}_stackingclfmodel.pt` + `{property}_train_info.json` |
| Evaluate | `{property}_evaluation_predictions.csv` |
| Refit (reg) | `{property}_refitted_stackingregmodel.pt` |
| Refit (clf) | `{property}_refitted_stackingclfmodel.pt` |
| Merge | `merged_stackingregmodel.pt` / `merged_refitted_stackingregmodel.pt` |
| Predict | `{property}_predictions.csv` (individual) / `predictions.csv` (merged) |
| Dashboard | `dashboard.html` (in run folder) |

## Merged Models

When training multiple properties, per-property `.pt` files each contain the full model with identical pretrained encoder (~10MB). The `merge_models.py` script uses upstream `merge_model()`/`delete_properties()` to combine them into a single file, eliminating encoder duplication.

- `merged_model_file` and `merged_refitted_model_file` are optional state fields — absent/null means individual files only
- The predict script auto-discovers properties from `model.models.keys()` — merged and individual files work transparently
- The registry uses `model_format: "merged"` (single string) or `"individual"` (list of paths)
- `merge_models.py` verifies encoder identity across all input files before merging

## Environment Variables

The plugin honors a layered env-var contract — Nexus values take precedence over user values, which take precedence over defaults.

| Variable | Source | Purpose |
|----------|--------|---------|
| `CLAUDE_PLUGIN_ROOT` | Claude Code | Set when loaded as a plugin. The hook reads this. |
| `AUTOMOL_ROOT` | SessionStart hook | Project / repo root (`CLAUDE_PROJECT_DIR` or `$PWD`). |
| `MOLAGENT_PLUGIN_ROOT` | SessionStart hook → `.claude/settings.local.json` | Plugin root directory (where `skills/` lives). Available in all Bash calls and subagents. Takes effect on next session start. |
| `MOLAGENT_OUTPUT_ROOT` | SessionStart hook → `.claude/settings.local.json` | Where pipeline output and the model registry live. Default: `$AUTOMOL_ROOT/MolagentFiles`. Override-aware: if `PHARMAOS_MOLAGENT_ROOT` is set (Nexus), that wins. |
| `PHARMAOS_MOLAGENT_ROOT` | Nexus host (per-project) | Nexus injects this so each project gets its own output namespace. Honored automatically by the hook. |
| `MOLAGENT_REGISTRY_PATH` | User | Full path override for `model_registry.json`. Defaults to `${MOLAGENT_OUTPUT_ROOT}/model_registry.json`. |
| `MOLAGENT_DETERMINISTIC` | User | Opt-in. When `true` (also `1/yes/on`): seeds `random` / `numpy` / `torch`, sets `PYTHONHASHSEED`, forces all `n_jobs` to 1. Reproducible but slower (no parallel CV). Default off. |
| `MOLAGENT_LOG_DIR` | User | Where the Stop-hook validator log goes (default: `$TMPDIR/molagent`). The legacy plugin-cache path is read-only when installed via `/plugin install`. |
| `AUTOMOL_VENV` | User | Virtual environment path (default: `$AUTOMOL_ROOT/.venv`). Set this to a stable path to avoid per-project re-installs in Nexus. |

**Output-root resolution priority** (used by every Python script):

```
1. PHARMAOS_MOLAGENT_ROOT (Nexus)
2. MOLAGENT_OUTPUT_ROOT (user / hook-defaulted)
3. ./MolagentFiles
```

All this is encapsulated in `skills/train-pipeline/scripts/_paths.py::get_output_root()`. The `_determinism.py` sibling provides `maybe_seed_everything()` and `force_serial_jobs()`.

## Nexus Integration

The `nexus` block in `.claude-plugin/plugin.json` declares one artifact type — `molagent_dashboard` — pointing at the playground template at `playgrounds/dashboard_playground.html`. That is the SAME file that the `visualize` skill substitutes into `dashboard.html` via `generate_dashboard.py`. One template, two delivery paths:

- **Standalone**: `generate_dashboard.py` substitutes `{{INITIAL_DATA}}` with the pre-computed JSON blob, writes `{run_folder}/dashboard.html`.
- **Nexus iframe**: the Nexus host loads the template into an iframe `srcdoc` and substitutes `{{INITIAL_DATA}}`, `{{ARTIFACT_TITLE}}`, `{{ARTIFACT_ID}}`, `{{RUN_ID}}` from the artifact body.

The playground emits `artifact:height` (iframe sizing) and `artifact:prompt_draft` (when the user clicks "Ask Claude about these outliers"); both are no-ops when running standalone. Theming follows the Atelier (warm paper, amber `#d4a056`) / Jarvis (dark cyan `#38bdf8`) Nexus token system; `prefers-color-scheme` auto-switches and a manual toggle persists to `localStorage`.

