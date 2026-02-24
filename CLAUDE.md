# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a monorepo for **MolAgent** — automated molecular property prediction with two main components:

1. **AutoMol/** — Core ML library for QSAR/drug discovery (stacking ensemble models, molecular feature generators)
2. **MolAgent-TaskManager/** — Claude Code plugin that exposes AutoMol through natural language skills

## Installation

```bash
# Core library
pip install -e AutoMol/automol/

```

The SessionStart hook in `MolAgent-TaskManager/hooks/setup-automol-env.sh` auto-creates a `.venv` if missing.

## Running Scripts

All Python scripts in `MolAgent-TaskManager/skills/` use Click CLI with PEP 723 inline metadata:

```bash
uv run python path/to/script.py --option value
```

Virtual env activation is needed when system Python lacks AutoMol:
```bash
source ${AUTOMOL_VENV:-.venv}/bin/activate
```

## Architecture

### AutoMol Library

The core library (`AutoMol/automol/automol/`) provides:

- **stacking.py** — Main `StackingRegressor` and `StackingClassifier` classes with ensemble methods
- **feature_generators.py** — Molecular feature generators:
  - `BottleneckTransformer` / `OnnxBottleneckTransformer` — Pretrained ONNX encoder
  - `ECFPGenerator` — Extended connectivity fingerprints
  - `RDKITGenerator` — RDKit descriptors
- **model_search.py** — Nested cross-validation model search with HyperOpt
- **stacking_util.py** — Model serialization (save/load `.pt` files), `merge_model()` for multi-property models

Key pattern: Models are saved as `.pt` files containing the full ensemble. Multi-property models can be merged to eliminate encoder duplication.

### Claude Code Plugin

Located in `MolAgent-TaskManager/`. Two skills:

- **train-pipeline** — End-to-end training: detect → prepare → split → train → evaluate → refit → merge
- **predict** — Inference with auto-discovered models

Pipeline outputs go to `MolagentFiles/{run_id}/` with a global `model_registry.json`.

See `MolAgent-TaskManager/CLAUDE.md` for detailed plugin architecture.

## Key Conventions

- SMILES column after preparation is `Stand_SMILES` (input may be `smiles`, `SMILES`, etc.)
- Model files: `{property}_stackingregmodel.pt` (regression) or `{property}_stackingclfmodel.pt` (classification)
- Refitted models have `_refitted` suffix in filename
- Merged models: `merged_stackingregmodel.pt`
- Pipeline state stored in `pipeline_state.json` with `"outputs"` key (not `"files"`)

## Computational Load Presets

Training scripts accept `--computational-load` with 4 levels:

| Level | Description |
|-------|-------------|
| `free` | Minimal — single method, no hyperparameter search |
| `cheap` | Light — inner methods, basic search |
| `moderate` | Medium — stacking, randomized search |
| `expensive` | Full — stacking of stacking, hyperopt |
