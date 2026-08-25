# MolAgent Marketplace

Claude Code plugin marketplace for [MolAgent](https://github.com/JorisTavernier/MolAgentLight) — automated molecular property prediction.

## Highlights

- **Three Bottleneck encoder keys** — `Bottleneck` (ChEMBL 37 E-logD, default), `Bottleneck_chembl37_base` (no logD supervision — use for logD/logP targets to avoid CV bias), `Bottleneck_chembl27` (legacy). See [Encoder Reference](#encoder-reference) below.
- **Remote MCP server with authentication** — deploy the server centrally; users connect with Bearer tokens via `claude mcp add --transport http`. Multi-user with per-user model/dataset isolation.
- **Dataset upload registry** — upload CSVs once (`upload_dataset`) and reference by `dataset_id` in training and prediction. Tracks `last_used`; admin cleanup removes stale data.
- **Browser web app** — SvelteKit frontend + FastAPI backend for the full pipeline without Claude Code. See `MolAgentLight/app/`.

## Available Plugins

| Plugin | Description | Version |
|--------|-------------|---------|
| **MolAgentLight** | Train and deploy molecular property prediction models from SMILES data using ensemble stacking methods, with interactive Plotly.js dashboards for evaluation results | 1.0.0 |

## Installation

### From a Local Clone

```bash
git clone https://github.com/JorisTavernier/MolAgentLight.git
cd MolAgentLight
```

Then in Claude Code:

```
/plugin marketplace add ./MolAgent-Marketplace
/plugin install MolAgentLight@molagent-marketplace
```

> **Note:** After installing the plugin, restart Claude Code so the SessionStart hook runs and sets up the virtual environment with AutoMol.

### From GitHub (direct)

In Claude Code:

```
/plugin marketplace add https://github.com/JorisTavernier/MolAgentLight/tree/main/MolAgent-Marketplace
/plugin install MolAgentLight@molagent-marketplace
```

> **Note:** After installing the plugin, restart Claude Code so the SessionStart hook runs and sets up the virtual environment with AutoMol.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- The plugin auto-creates a virtual environment and installs AutoMol on first use

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

## Usage

Once installed, the plugin provides three skills:

- **`/train-pipeline`** — End-to-end training: detect dataset, prepare, split, train, evaluate, refit, and merge models
- **`/predict`** — Inference with auto-discovered trained models
- **`/visualize`** — Generate an interactive HTML dashboard from evaluation results

### Train a Model

```
> Train a model on my_molecules.csv with target property potency
```

Or invoke directly:

```
> /train-pipeline
```

### Make Predictions

```
> Predict properties for new_molecules.csv
```

Or invoke directly:

```
> /predict
```

### Visualize Results

```
> Visualize the evaluation results from my last training run
```

Or invoke directly:

```
> /visualize
```

## Plugin Details

See [MolAgentLight README](MolAgentLight/README.md) for full documentation on:

- Supported skills and their options
- Computational load presets (free, cheap, moderate, expensive)
- Model merging for multi-property predictions
- Environment variables and configuration

## Encoder Reference

Three Bottleneck ONNX encoders are available as `feature_keys`:

| Key | Encoder | When to use |
|-----|---------|-------------|
| `Bottleneck` | ChEMBL 37 E-logD (v6_best) | **Default.** Best general accuracy for most endpoints (permeability, activity, toxicity, etc.). |
| `Bottleneck_chembl37_base` | ChEMBL 37 E-base (no logD) | Use when the target **is** logD, logP, or lipophilicity — avoids optimistic CV bias from logD supervision. |
| `Bottleneck_chembl27` | ChEMBL 27 legacy | Use only to reproduce results from models trained before the ChEMBL 37 upgrade. |

You can also combine with `rdkit` (RDKit 2D descriptors) or `fps_2048_2` (Morgan fingerprints, radius 2).

## Maintenance

Development happens directly in `MolAgentLight/`. The AutoMol library is bundled at `MolAgentLight/AutoMol/automol/` and is auto-installed by the SessionStart hook.
