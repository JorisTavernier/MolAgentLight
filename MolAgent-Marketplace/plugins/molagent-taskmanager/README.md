# AutoMol Plugin for Claude Code

A Claude Code plugin that wraps the [AutoMol](https://github.com/openanalytics/AutoMol) molecular property prediction library. Train ensemble stacking models on SMILES data and run inference — all through natural language.

## Quick Start

### As a Claude Code Plugin

Install via the marketplace:

```
/plugin marketplace add ./MolAgent-Marketplace
/plugin install molagent-taskmanager@molagent-marketplace
```

> **Note:** After installing the plugin, restart Claude Code so the SessionStart hook runs and sets up the virtual environment with AutoMol.

Then:

```
# Train a model
> Use the train-pipeline skill on my_molecules.csv

# Make predictions
> Use the predict skill on new_data.csv
```

### Standalone (development)

```bash
cd /path/to/MolAgent-Marketplace/plugins/molagent-taskmanager
claude
> /train-pipeline
> /predict
```

## Skills

### train-pipeline

Single-invocation training pipeline (plan + execute in one session):

1. **Plan** — Detects dataset properties (SMILES column, targets, task type), auto-configures defaults, asks one focused question, creates an isolated run folder under `MolagentFiles/`.
2. **Execute** — Runs 7 steps: prepare, split, train, evaluate, review, refit, generate model card. Registers the final model in `MolagentFiles/model_registry.json`.

Auto-executes when the user clearly requests training. Saves plan for later if user declines. Re-invoke to resume interrupted runs.

Supports regression, classification, and mixed tasks. Features include Bottleneck (2D) and optionally AffGraph/ProLIF (3D protein-ligand). Computational load is user-selectable: free, cheap, moderate, or expensive.

### predict

Single-phase inference skill:

1. Auto-discovers trained models from the registry
2. Selects model (auto if only one, asks if multiple)
3. Accepts CSV files or individual SMILES strings
4. Runs `predict.py` — merged models predict all properties in one call; individual models run once per property
5. Outputs `predictions.csv` (merged) or `{property}_predictions.csv` (individual)

## Model Merging

When training multiple target properties (e.g., gamma1, gamma2, gamma3), AutoMol saves one `.pt` file per property — but each contains the full model including the pretrained encoder (~10MB). For 3 properties, this means 3x encoder duplication: ~36MB instead of ~16MB.

The pipeline automatically merges per-property files into a single `.pt` file after training and refit steps using `merge_models.py`. This:

- Eliminates encoder duplication (~10MB saved per extra property)
- Enables single-call multi-property prediction
- Produces a standard AutoMol `.pt` file (no custom format)
- Is fully backward compatible — individual per-property files still work

The predict skill handles both merged and individual models transparently.

## Project Structure

```
molagent-taskmanager/
  .claude-plugin/
    plugin.json              # Plugin manifest
  AutoMol/                   # Bundled ML library (auto-installed by hook)
    automol/
  hooks/
    setup-automol-env.sh     # SessionStart hook (exports AUTOMOL_ROOT)
  skills/
    train-pipeline/
      SKILL.md               # Skill definition + Stop hook
      scripts/               # Python scripts (Click CLI), incl. merge_models.py
      steps/                 # Step-by-step execution guides
      validators/            # Pipeline state validator
    predict/
      SKILL.md               # Skill definition
      scripts/
        predict.py           # Inference script
  MolagentFiles/             # Pipeline outputs (gitignored except registry)
    model_registry.json      # Trained model registry
```

## Prerequisites

- Python 3.10+
- `uv` package manager
- AutoMol is bundled at `AutoMol/automol/` and auto-installed by the SessionStart hook

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

## Environment

The SessionStart hook exports two variables via `.claude/settings.local.json`:

- **`AUTOMOL_ROOT`** — Repository root. Falls back from `CLAUDE_PROJECT_DIR` to `PWD`.
- **`PLUGIN_ROOT`** — Plugin root where `skills/` lives. Falls back from `CLAUDE_PLUGIN_ROOT` to `$AUTOMOL_ROOT/MolAgent-Marketplace/plugins/molagent-taskmanager`.

On first install, the hook writes these to `.claude/settings.local.json`. **Restart Claude Code once** after installing — on next start, both vars are injected into every Bash tool call, including subagents.

Skill files use `$PLUGIN_ROOT/skills/...` for all script paths.

## Configuration

Each pipeline run stores its configuration in `MolagentFiles/{run_id}/pipeline_state.json`. Run folders are self-contained — all outputs (prepared data, models, model card) live inside the run folder. The model registry at `MolagentFiles/model_registry.json` is global and indexes all completed runs.

No manual configuration files are needed — the train-pipeline skill auto-detects dataset properties and presents sensible defaults.
