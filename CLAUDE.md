# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the repository for **MolAgent** — automated molecular property prediction. All code lives under `MolAgent-Marketplace/`:

- **`MolAgent-Marketplace/MolAgentLight/`** — Claude Code plugin that exposes AutoMol through natural language skills, including interactive Plotly.js dashboards for evaluation results
- **`MolAgent-Marketplace/MolAgentLight/AutoMol/automol/`** — Bundled copy of the AutoMol core ML library (stacking ensemble models, molecular feature generators)

## Installation

Install the plugins via the Claude Code marketplace:

```
/plugin marketplace add ./MolAgent-Marketplace
/plugin install MolAgentLight@molagent-marketplace
```

> **Note:** After installing the plugin, restart Claude Code so the SessionStart hook runs and sets up the virtual environment with AutoMol.

For manual installation of the core library:

```bash
pip install -e MolAgent-Marketplace/MolAgentLight/AutoMol/automol/
```

The SessionStart hook in `MolAgent-Marketplace/MolAgentLight/hooks/setup-automol-env.sh` auto-creates a `.venv` if missing.

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

On Windows, the MCP server may time out on first start while `uv` resolves dependencies. Increase `MCP_TIMEOUT` (milliseconds) in `~/.claude/settings.json` (global user settings):

```json
{ "env": { "MCP_TIMEOUT": "120000" } }
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

The core library (`MolAgent-Marketplace/MolAgentLight/AutoMol/automol/automol/`) provides:

- **stacking.py** — Main model classes (`FeatureGenerationStackingRegressors`, `FeatureGenerationStackingClassifiers`, etc.) with ensemble methods, plus `save_model()` / `load_model()` and per-class `merge_model()` methods
- **feature_generators.py** — Molecular feature generators:
  - `BottleneckTransformer` / `OnnxBottleneckTransformer` — Pretrained ONNX encoder
  - `ECFPGenerator` — Extended connectivity fingerprints
  - `RDKITGenerator` — RDKit descriptors
- **model_search.py** — Nested cross-validation model search with HyperOpt
- **stacking_util.py** — Hyperopt helpers, search option retrieval, `ModelAndParams` / `ResultDesigner` utility classes

Key pattern: Models are saved as `.pt` files containing the full ensemble. Multi-property models can be merged to eliminate encoder duplication.

### Claude Code Plugins

**MolAgentLight** — Located in `MolAgent-Marketplace/MolAgentLight/`. Three skills:

- **train-pipeline** — End-to-end training: detect → prepare → split → train → evaluate → refit → merge
- **predict** — Inference with auto-discovered models
- **visualize** — Generates an interactive HTML dashboard from evaluation results

Pipeline outputs go to `MolagentFiles/{run_id}/` with a global `model_registry.json`.

The dashboard script (`skills/visualize/scripts/generate_dashboard.py`) is a PEP 723 script with inline dependencies (click, pandas, numpy, scikit-learn, jinja2, scipy). It reads `pipeline_state.json` and evaluation CSVs, computes derived metrics in Python, and renders a self-contained HTML file with Plotly.js charts and SmilesDrawer molecular structure hover tooltips. Dashboard outputs go to `MolagentFiles/{run_id}/dashboard.html`.

See `MolAgent-Marketplace/MolAgentLight/CLAUDE.md` for detailed plugin architecture.

### MCP Server

The MCP server (`mcp/server.py`) exposes 13 tools: `list_options`, `start_training_session`, `answer_training_question`, `train_and_visualize`, `list_models`, `predict`, `merge_models`, `delete_model`, `download_model`, `upload_dataset`, `list_datasets`, `delete_dataset`, `admin_manage`. It runs via stdio when used as a Claude Code plugin (no auth), or via streamable-http for remote access (token auth required). See `mcp/MCP_SERVER.md` for the full API.

**Authentication:** When `MOLAGENT_AUTH_REQUIRED=true`, the MCP server validates Bearer tokens. Admin tokens can manage users and see all models/datasets; user tokens can only access their own. Tokens are stored as **SHA-256 hashes** in `${MOLAGENT_OUTPUT_ROOT}/auth_tokens.json` — the plaintext is shown only once, at creation, and cannot be recovered. On first run the admin token is auto-generated, printed to stderr, **and** written to a `0600` sidecar file `${MOLAGENT_OUTPUT_ROOT}/admin_token.txt` (so it's recoverable even when stderr is swallowed by a parent process). Copy it and delete the file. If lost, delete `auth_tokens.json` and restart to regenerate (this also wipes all user tokens). Over remote (streamable-http) the token is verified by the transport; over stdio, the web app passes a caller token via `MOLAGENT_CALLER_TOKEN` (with `MOLAGENT_AUTH_REQUIRED=true`) to run under a real per-user identity ("local + token" mode).

**Data Registry:** Uploaded datasets are tracked in `${MOLAGENT_OUTPUT_ROOT}/data_registry.json` with per-user isolation. Files are stored at `uploads/<owner_id>/<filename>`. The `upload_dataset` tool accepts base64-encoded CSV content (works in remote mode without shared filesystem). Both `start_training_session` and `predict` accept `dataset_id` as an alternative to direct file paths. A `last_used` timestamp is maintained on both model and dataset registry entries.

**Remote Upload (Claude Code CLI):** When the MCP is remote, do NOT pass `file_content_b64` through the tool call directly — base64 of files >30KB will be truncated by LLM I/O limits. The train-pipeline skill includes an in-process upload snippet (`uv run --with fastmcp python -c "..."`) that keeps base64 in Python memory and POSTs via `fastmcp.Client`. See `mcp/MCP_SERVER.md` or the train-pipeline SKILL.md for the full snippet.

**Admin Cleanup:** `admin_manage(action="purge_stale", max_age_days=N)` removes models/datasets not used within N days. `admin_manage(action="purge_orphans", max_age_days=N)` removes run folders not referenced by any registry entry (e.g. failed training runs). Both dry-run by default; pass `force=True` to execute deletion.

### Web App

The web app (`MolAgent-Marketplace/MolAgentLight/app/`) provides a browser UI for the same pipeline. It is a **thin HTTP bridge** to the MCP server — no pipeline logic in the app itself.

- **Frontend**: SvelteKit 5 + Tailwind CSS 4 at port 5173 → open http://localhost:5173
- **Backend**: FastAPI at port 8000, translates HTTP → MCP `call_tool`
- **MCP connection**: Local (stdio, spawns `mcp/server.py`) or Remote (streamable-http URL + auth token)
- **Progress**: MCP progress notifications update job state, frontend polls every 2s

Start both servers:
```bash
cd MolAgent-Marketplace/MolAgentLight/app
./start.sh
```

Or manually:
```bash
cd MolAgent-Marketplace/MolAgentLight/app
# Backend (terminal 1)
uv run --with fastapi --with uvicorn --with python-multipart --with pydantic-settings --with "fastmcp[tasks]" --with pandas \
  uvicorn backend.main:app --port 8000
# Frontend (terminal 2)
cd frontend && npm run dev
```

### MCP Server Standalone (Remote Mode)

```bash
cd MolAgent-Marketplace/MolAgentLight
MOLAGENT_OUTPUT_ROOT=/absolute/path/to/output \
MOLAGENT_AUTH_REQUIRED=true \
uv run mcp/server.py --transport streamable-http --host 127.0.0.1 --port 8001
```

Admin token is printed on first run. Manage users with the admin CLI:
```bash
uv run mcp/admin_cli.py --url http://127.0.0.1:8001/mcp --token <ADMIN_TOKEN> create-user alice
uv run mcp/admin_cli.py --url http://127.0.0.1:8001/mcp --token <ADMIN_TOKEN> list-users
uv run mcp/admin_cli.py --url http://127.0.0.1:8001/mcp --token <ADMIN_TOKEN> revoke <OWNER_ID>
uv run mcp/admin_cli.py --url http://127.0.0.1:8001/mcp --token <ADMIN_TOKEN> purge-stale --days 30
uv run mcp/admin_cli.py --url http://127.0.0.1:8001/mcp --token <ADMIN_TOKEN> purge-stale --days 30 --force
uv run mcp/admin_cli.py --url http://127.0.0.1:8001/mcp --token <ADMIN_TOKEN> purge-orphans --days 7
uv run mcp/admin_cli.py --url http://127.0.0.1:8001/mcp --token <ADMIN_TOKEN> purge-orphans --days 7 --force
```

### Installation Without Plugin Hook

If running the MCP server or web app without the Claude Code marketplace (i.e. the SessionStart hook hasn't run):

```bash
cd MolAgent-Marketplace/MolAgentLight
uv venv .venv
uv pip install -e AutoMol/automol/
uv pip install "fastmcp[tasks]" pandas pydantic
cd app/frontend && npm install
```

See `MolAgent-Marketplace/MolAgentLight/app/README.md` for full details.

## Key Conventions

- SMILES column after preparation is `Stand_SMILES` (input may be `smiles`, `SMILES`, etc.)
- Model files: `{property}_stackingregmodel.pt` (regression) or `{property}_stackingclfmodel.pt` (classification)
- Refitted models have `_refitted` suffix in filename
- Merged models: `merged_stackingregmodel.pt`
- Pipeline state stored in `pipeline_state.json` with `"outputs"` key (not `"files"`)

## Marketplace

`MolAgent-Marketplace/` is the sole distribution directory. It contains the marketplace catalog and the plugins with the bundled AutoMol library.

- **`MolAgent-Marketplace/.claude-plugin/marketplace.json`** — Marketplace catalog (also mirrored at root `.claude-plugin/marketplace.json`)
- **`MolAgent-Marketplace/MolAgentLight/`** — Training, prediction & visualization plugin (skills, hooks, scripts)
- **`MolAgent-Marketplace/MolAgentLight/AutoMol/automol/`** — Bundled AutoMol library (auto-installed by the SessionStart hook)

## Computational Load Presets

Training scripts accept `--computational-load` with 4 levels:

| Level | Description |
|-------|-------------|
| `free` | Single method with hyperparameter search (no ensemble/stacking) |
| `cheap` | Light — inner methods, basic search |
| `moderate` | Medium — stacking, randomized search |
| `expensive` | Full — stacking of stacking, hyperopt |

## Environment Variables (top-level summary)

| Variable | Purpose |
|----------|---------|
| `MOLAGENT_PLUGIN_ROOT` | Plugin root (set by SessionStart hook from `CLAUDE_PLUGIN_ROOT`). |
| `MOLAGENT_OUTPUT_ROOT` | Where pipeline output and the registry live (default: `./MolagentFiles`). |
| `PHARMAOS_MOLAGENT_ROOT` | Nexus-injected per-project output root. Wins over `MOLAGENT_OUTPUT_ROOT`. |
| `MOLAGENT_REGISTRY_PATH` | Override the registry JSON path. Default: `${MOLAGENT_OUTPUT_ROOT}/model_registry.json`. |
| `MOLAGENT_DETERMINISTIC` | Opt-in. `true` seeds RNGs and forces serial CV. Default off. |
| `MOLAGENT_LOG_DIR` | Stop-hook validator log directory. Default: `$TMPDIR/molagent`. |
| `AUTOMOL_VENV` | Venv path. Default: `$AUTOMOL_ROOT/.venv`. |
| `MOLAGENT_AUTH_REQUIRED` | Enable token auth on MCP server. Required for remote/HTTP mode. Default: off. |
| `MOLAGENT_TOKEN_STORE_PATH` | Override token store location. Default: `${MOLAGENT_OUTPUT_ROOT}/auth_tokens.json`. |
| `FASTMCP_DOCKET_REDELIVERY_TIMEOUT` | Docket redelivery timeout (seconds). Set to 86400 for long training. Default: 300. |

See `MolAgent-Marketplace/MolAgentLight/CLAUDE.md` for the full env-var contract and Nexus integration details.

## Nexus Integration

The plugin declares a `nexus` block in `.claude-plugin/plugin.json` exposing the `molagent_dashboard` artifact type. To install into a Nexus host (e.g., `drug_discovery_ui`):

1. Copy or symlink `MolAgent-Marketplace/MolAgentLight/` into the host's `plugins/` directory.
2. Add `'molagent_dashboard'` to the host's playground type registry (e.g., `MOLECULAR_TYPES` in `playgroundTemplates.ts` for `nexus-atrium`).
3. Restart the host so its bundled-plugin discovery picks up the new manifest.

The host injects `PHARMAOS_MOLAGENT_ROOT` per project; the SessionStart hook already honors it.
