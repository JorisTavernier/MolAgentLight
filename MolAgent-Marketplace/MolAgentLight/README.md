# AutoMol Plugin for Claude Code

A Claude Code plugin that wraps the [AutoMol](https://github.com/openanalytics/AutoMol) molecular property prediction library. Train ensemble stacking models on SMILES data and run inference — all through natural language.

## Quick Start

### As a Claude Code Plugin

Install via the marketplace:

```
/plugin marketplace add ./MolAgent-Marketplace
/plugin install MolAgentLight@molagent-marketplace
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
cd /path/to/MolAgent-Marketplace/MolAgentLight
claude
> /train-pipeline
> /predict
```

### Remote MCP Server (with authentication)

Start the server with auth enabled:
```bash
cd MolAgent-Marketplace/MolAgentLight
MOLAGENT_OUTPUT_ROOT=/absolute/path/to/output \
MOLAGENT_AUTH_REQUIRED=true \
uv run mcp/server.py --transport streamable-http --host 127.0.0.1 --port 8001
```

Create a user token (admin token is printed on first server start):
```bash
uv run mcp/admin_cli.py --url http://127.0.0.1:8001/mcp --token <ADMIN_TOKEN> create-user alice
```

Add the server to Claude Code with the user token:
```bash
claude mcp add --transport http automol-mcp http://127.0.0.1:8001/mcp \
  --header "Authorization: Bearer <USER_TOKEN>"
```

Or via `~/.claude/settings.json`:
```json
{
  "mcpServers": {
    "automol-mcp": {
      "type": "http",
      "url": "http://127.0.0.1:8001/mcp",
      "headers": { "Authorization": "Bearer <USER_TOKEN>" }
    }
  }
}
```

See `mcp/MCP_SERVER.md` for the full server API and admin reference.

### Talking to the MCP directly

The plugin's skills are thin guides over an MCP server (`mcp/server.py`, 13 tools — see `mcp/MCP_SERVER.md`). You don't need to invoke a skill by name; plain requests to Claude Code work too, since Claude maps your sentences to the underlying tool calls (`start_training_session`, `answer_training_question`, `train_and_visualize`, `predict`, `merge_models`, etc.). Some real examples:

```
> Train a model for data.csv to predict logD
```
Claude detects the SMILES column, task type, and defaults, then asks you to confirm or change anything before training.

```
> Use classification with a threshold of 3.5, and set the computational load to moderate
```
Claude maps this to `task="Classification"`, `class_values=[3.5]` (values above 3.5 → class 1, at/below → class 0), and `computational_load="moderate"`.

```
> Only use rdkit features, drop the Bottleneck encoder, and use an 80/20 split
```
Maps to `feature_keys=["rdkit"]` and `test_size=0.2`.

```
> Looks good, go ahead
```
Confirms the config and kicks off `train_and_visualize`.

```
> Predict logD for CCO and c1ccccc1 using the model we just trained
```
Claude resolves the model ID from the recent run and calls `predict`.

```
> Combine the solubility and logD models into one model
```
Calls `merge_models` with both model IDs.

## Encoder Reference

Three Bottleneck ONNX encoders are available as `feature_keys`. All produce 250-dim embeddings.

| Key | Encoder | Notes |
|-----|---------|-------|
| `Bottleneck` | ChEMBL 37 E-logD | **Default.** Trained with logD supervision. Best general accuracy for most endpoints. |
| `Bottleneck_chembl37_base` | ChEMBL 37 E-base | No logD supervision. **Use for logD, logP, or lipophilicity targets** — avoids optimistic CV bias when the target correlates with the encoder's training signal. |
| `Bottleneck_chembl27` | ChEMBL 27 (legacy) | Legacy encoder. Use only to reproduce results from models trained before the ChEMBL 37 upgrade. |

You can also combine with `rdkit` (RDKit 2D descriptors) or `fps_2048_2` (Morgan fingerprints, 2048 bits, radius 2).

## Skills

### train-pipeline

Single-invocation training pipeline (plan + execute in one session):

1. **Plan** — Detects dataset properties (SMILES column, targets, task type), auto-configures defaults, asks one focused question, creates an isolated run folder under `MolagentFiles/`.
2. **Execute** — Runs 7 steps: prepare, split, train, evaluate, review, refit, generate model card. Registers the final model in `MolagentFiles/model_registry.json`.

Auto-executes when the user clearly requests training. Saves plan for later if user declines. Re-invoke to resume interrupted runs.

Supports regression, classification, and mixed tasks. Features include the Bottleneck 2D encoders — `Bottleneck` (ChEMBL 37 E-logD, default), `Bottleneck_chembl37_base` (no logD supervision; use for logD/logP/lipophilicity), and `Bottleneck_chembl27` (legacy) — plus RDKit descriptors, Morgan fingerprints, and optionally AffGraph/ProLIF (3D protein-ligand). Computational load is user-selectable: free, cheap, moderate, or expensive.

### predict

Single-phase inference skill:

1. Auto-discovers trained models from the registry
2. Selects model (auto if only one, asks if multiple)
3. Accepts CSV files or individual SMILES strings
4. Runs `predict.py` — merged models predict all properties in one call; individual models run once per property
5. Outputs `predictions.csv` (merged) or `{property}_predictions.csv` (individual)

### visualize

Single-phase dashboard skill:

1. Auto-discovers completed runs with evaluation data
2. Selects run (auto if only one, asks if multiple)
3. Runs `generate_dashboard.py` — PEP 723 script with self-contained dependencies
4. Generates a self-contained HTML dashboard with Plotly.js charts and SmilesDrawer molecular structure tooltips
5. Opens the dashboard in the default browser

## Model Merging

When training multiple target properties (e.g., gamma1, gamma2, gamma3), AutoMol saves one `.pt` file per property — but each contains the full model including the pretrained encoder (~10MB). For 3 properties, this means 3x encoder duplication: ~36MB instead of ~16MB.

The pipeline automatically merges per-property files into a single `.pt` file after training and refit steps using `merge_models.py`. This:

- Eliminates encoder duplication (~10MB saved per extra property)
- Enables single-call multi-property prediction
- Produces a standard AutoMol `.pt` file (no custom format)
- Is fully backward compatible — individual per-property files still work

The predict skill handles both merged and individual models transparently.

## Web App

A browser UI is available at `app/` — no Claude Code required. It provides the same train / predict / visualize workflow through a SvelteKit frontend backed by a FastAPI server that proxies to the MCP server.

```bash
cd MolAgent-Marketplace/MolAgentLight/app
./start.sh        # starts backend (port 8000) and frontend (port 5173)
```

Open http://localhost:5173. The backend can connect to the MCP server locally (stdio) or remotely (streamable-http URL + token). See `app/README.md` for configuration.

## Project Structure

```
MolAgentLight/
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
    visualize/
      SKILL.md               # Skill definition
      scripts/
        generate_dashboard.py  # PEP 723 dashboard script
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
- **`MOLAGENT_PLUGIN_ROOT`** — Plugin root where `skills/` lives. Falls back from `CLAUDE_PLUGIN_ROOT` to `$AUTOMOL_ROOT/MolAgent-Marketplace/MolAgentLight`.

On first install, the hook writes these to `.claude/settings.local.json`. **Restart Claude Code once** after installing — on next start, both vars are injected into every Bash tool call, including subagents.

Skill files use `$MOLAGENT_PLUGIN_ROOT/skills/...` for all script paths.

### MCP Timeouts

Training runs can take minutes to hours. To prevent Claude Code from timing out MCP connections or tool calls, add these to your global `~/.claude/settings.json` under `"env"`:

```json
{
  "env": {
    "MCP_TIMEOUT": "1800000",
    "MCP_TOOL_TIMEOUT": "172800000"
  }
}
```

- **`MCP_TIMEOUT`** — Maximum time (ms) to wait for the MCP server to start and connect. `1800000` = 30 minutes (covers slow first-time dependency resolution by `uv`).
- **`MCP_TOOL_TIMEOUT`** — Maximum time (ms) a single MCP tool call can run before being aborted. `172800000` = 48 hours (covers `expensive` computational load training runs).

## Configuration

Each pipeline run stores its configuration in `MolagentFiles/{run_id}/pipeline_state.json`. Run folders are self-contained — all outputs (prepared data, models, model card) live inside the run folder. The model registry at `MolagentFiles/model_registry.json` is global and indexes all completed runs.

No manual configuration files are needed — the train-pipeline skill auto-detects dataset properties and presents sensible defaults.
