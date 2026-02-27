# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the repository for **MolAgent** — automated molecular property prediction. All code lives under `MolAgent-Marketplace/`:

- **`MolAgent-Marketplace/plugins/molagent-taskmanager/`** — Claude Code plugin that exposes AutoMol through natural language skills
- **`MolAgent-Marketplace/plugins/molagent-taskmanager/AutoMol/automol/`** — Bundled copy of the AutoMol core ML library (stacking ensemble models, molecular feature generators)

## Installation

Install the plugin via the Claude Code marketplace:

```
/plugin marketplace add ./MolAgent-Marketplace
/plugin install molagent-taskmanager@molagent-marketplace
```

> **Note:** After installing the plugin, restart Claude Code so the SessionStart hook runs and sets up the virtual environment with AutoMol.

For manual installation of the core library:

```bash
pip install -e MolAgent-Marketplace/plugins/molagent-taskmanager/AutoMol/automol/
```

The SessionStart hook in `MolAgent-Marketplace/plugins/molagent-taskmanager/hooks/setup-automol-env.sh` auto-creates a `.venv` if missing.

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

## Running Scripts

All scripts are invoked the same way:

```bash
uv run path/to/script.py --option value
```

When the `.venv` is activated (done automatically by the SessionStart hook), `uv run` uses it for all scripts. Scripts with PEP 723 inline metadata (`detect_dataset.py`, `merge_models.py`, `model_registry.py`) also work without the venv — uv resolves their deps automatically. Do **not** use `uv run python script.py` — for PEP 723 scripts that bypasses the inline metadata and fails.

Virtual env activation is needed when system Python lacks AutoMol:
```bash
# Linux/macOS
source ${AUTOMOL_VENV:-.venv}/bin/activate
# Windows
${AUTOMOL_VENV:-.venv}/Scripts/activate
```

## Architecture

### AutoMol Library

The core library (`MolAgent-Marketplace/plugins/molagent-taskmanager/AutoMol/automol/automol/`) provides:

- **stacking.py** — Main `StackingRegressor` and `StackingClassifier` classes with ensemble methods
- **feature_generators.py** — Molecular feature generators:
  - `BottleneckTransformer` / `OnnxBottleneckTransformer` — Pretrained ONNX encoder
  - `ECFPGenerator` — Extended connectivity fingerprints
  - `RDKITGenerator` — RDKit descriptors
- **model_search.py** — Nested cross-validation model search with HyperOpt
- **stacking_util.py** — Model serialization (save/load `.pt` files), `merge_model()` for multi-property models

Key pattern: Models are saved as `.pt` files containing the full ensemble. Multi-property models can be merged to eliminate encoder duplication.

### Claude Code Plugin

Located in `MolAgent-Marketplace/plugins/molagent-taskmanager/`. Two skills:

- **train-pipeline** — End-to-end training: detect → prepare → split → train → evaluate → refit → merge
- **predict** — Inference with auto-discovered models

Pipeline outputs go to `MolagentFiles/{run_id}/` with a global `model_registry.json`.

See `MolAgent-Marketplace/plugins/molagent-taskmanager/CLAUDE.md` for detailed plugin architecture.

## Key Conventions

- SMILES column after preparation is `Stand_SMILES` (input may be `smiles`, `SMILES`, etc.)
- Model files: `{property}_stackingregmodel.pt` (regression) or `{property}_stackingclfmodel.pt` (classification)
- Refitted models have `_refitted` suffix in filename
- Merged models: `merged_stackingregmodel.pt`
- Pipeline state stored in `pipeline_state.json` with `"outputs"` key (not `"files"`)

## Marketplace

`MolAgent-Marketplace/` is the sole distribution directory. It contains the marketplace catalog and the plugin with its bundled AutoMol library. Development happens directly in `MolAgent-Marketplace/plugins/molagent-taskmanager/`.

- **`MolAgent-Marketplace/.claude-plugin/marketplace.json`** — Marketplace catalog
- **`MolAgent-Marketplace/plugins/molagent-taskmanager/`** — The plugin (skills, hooks, scripts)
- **`MolAgent-Marketplace/plugins/molagent-taskmanager/AutoMol/automol/`** — Bundled AutoMol library (auto-installed by the SessionStart hook)

## Computational Load Presets

Training scripts accept `--computational-load` with 4 levels:

| Level | Description |
|-------|-------------|
| `free` | Minimal — single method, no hyperparameter search |
| `cheap` | Light — inner methods, basic search |
| `moderate` | Medium — stacking, randomized search |
| `expensive` | Full — stacking of stacking, hyperopt |
