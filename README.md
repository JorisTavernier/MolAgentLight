# MolAgent Light

<div align="center">

[![MolAgent](https://img.shields.io/badge/MolAgent-1.0.0-red.svg)](https://github.com/openanalytics/MolAgent)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![AutoMol](https://img.shields.io/badge/AutoMol-Pipeline-orange.svg)](https://github.com/openanalytics/AutoMol)
[![License](https://img.shields.io/badge/License-GPL--3.0-yellow.svg)](LICENSE)

**MolAgent Light is a Claude Code plugin for automated molecular property prediction**

[Installation](#installation) • [Examples](#examples) • [Skills](#skills) • [Visualization](#visualization) • [Citation](#citation)

</div>

---

## Overview

MolAgent Light enables expert-level molecular property prediction through natural language interaction with Claude Code. It provides:

- **Autonomous Model Construction** — Train ensemble stacking models for regression and classification
- **Intelligent Feature Selection** — Automatically selects optimal molecular representations: 2D descriptors, ECFP fingerprints, and a pretrained ONNX encoder
- **Portable ONNX Encoder** — The bottleneck encoder is exported to ONNX, making it lightweight, dependency-free, and portable across platforms
- **On-the-fly Predictions** — Make predictions on new molecules using trained models
- **Interactive Dashboards** — Visualize evaluation results with Plotly.js charts and molecular structure hover tooltips

This is a lightweight version of [MolAgent](https://github.com/openanalytics/MolAgent) without MCP server dependencies, designed specifically for Claude Code.

---

## Installation

### Prerequisites

- Python 3.10+ (earlier system Python is fine — `uv` will use a newer version in the virtual environment)
- [uv](https://docs.astral.sh/uv/) package manager

Clone the repository:

```bash
git clone https://github.com/JorisTavernier/MolAgentLight.git
cd MolAgentLight
```

### Install the Plugin

In Claude Code:

```
/plugin marketplace add ./MolAgent-Marketplace
/plugin install molagent-taskmanager@molagent-marketplace
```

> **Note:** After installing the plugin, restart Claude Code so the SessionStart hook runs and sets up the virtual environment with AutoMol and the environment variables (only the first time). Then restart Claude Code a **second** time and you are good to go.

---

## Examples

### Train a Model

Use natural language:

```
> Train a model on my_molecules.csv with target property potency
```

Or invoke the skill directly:

```
> /train-pipeline
```

The skill will:
1. Detect dataset properties (SMILES column, targets, task type)
2. Auto-configure sensible defaults
3. Ask for approval
4. Run the full pipeline: prepare → split → train → evaluate → refit

### Make Predictions

```
> Predict properties for new_molecules.csv using my trained model
```

Or:

```
> /predict
```

The predict skill will list trained models from the registry and run inference.

### Visualize Results

```
> Visualize the evaluation results from my last training run
```

Or:

```
> /visualize
```

The visualize skill auto-discovers completed runs with evaluation data and generates an interactive HTML dashboard.

---

## Skills

### `train-pipeline` — End-to-End Training

Runs the complete QSAR modeling pipeline from raw CSV to a production-ready ensemble model.

**Highlights:**
- Handles both **regression** and **classification** tasks
- Three complementary feature generators, automatically selected and combined:
  - **Bottleneck encoder** — Pretrained ONNX model; lightweight, portable, no GPU required
  - **ECFP fingerprints** — Extended connectivity fingerprints via RDKit
  - **RDKit descriptors** — 200+ physicochemical 2D descriptors
- **Nested cross-validation** with hyperparameter optimization (HyperOpt)
- **Stacking ensembles** — Combines base learners into a meta-model for improved accuracy
- **Computational load presets** — Scale from a quick free run to a full expensive search (see below)
- Resumes interrupted pipelines from the last completed step

**Outputs:**
- Trained models saved as `.pt` files in `MolagentFiles/{run_id}/`
- Global model registry at `MolagentFiles/model_registry.json`
- Evaluation metrics and model card per run

---

### `predict` — Inference

Single-phase inference on new molecules using any trained model.

**Highlights:**
- **Auto-discovers** trained models from the registry — no paths to specify
- Accepts a **CSV file** or individual **SMILES strings**
- Supports **merged multi-property models** (single encoder shared across targets)
- Outputs predictions to CSV with confidence-ready format

---

### `visualize` — Interactive Dashboard

Generates a self-contained HTML dashboard from evaluation results, powered by Plotly.js and SmilesDrawer.

**Regression plots:**
- Scatter with MAE bands
- Scatter with moving average error
- Scatter with cutoff classification (TP/FP/TN/FN)
- Residual plot
- Absolute error histogram
- Error bar plot (stacked bins)
- Threshold variation (accuracy/precision/recall sweep)
- Hit enrichment curve

**Classification plots:**
- Confusion matrix heatmap
- ROC curves (per class)
- Precision-recall curves
- Classification report heatmap
- F1 threshold tuning
- Calibration / reliability diagram
- Hit enrichment curve
- Probability bar plot

**Highlights:**
- **Molecular structure hover** — Hover over scatter points to see 2D molecular structures rendered client-side via SmilesDrawer
- **Interactive sliders** — Adjust the regression cutoff and error histogram bin count in real time
- **Property selector** — Switch between properties in multi-property runs
- **Overview / Detailed presets** — Grid of all plots or single fullscreen plot
- **Findings summary** — Copy-able text output with key metrics
- **Self-contained** — Single HTML file, works offline after first load

---

## Visualization

After training and evaluation, generate an interactive dashboard:

```
> /visualize
```

The dashboard opens in your default browser. All charts are interactive — zoom, pan, hover for details. Hover over scatter plot points to see the 2D molecular structure, true/predicted values, and error.

---

## Computational Load Presets

| Level | Description |
|-------|-------------|
| `free` | Minimal — single Light Gradient Boosting Machine, no ensemble|
| `cheap` | Light — blender, gridsearch |
| `moderate` | Medium — blender with tree methods, randomized search |
| `expensive` | Full — stacking method, hyperopt using bayesian optimization|

---

## Architecture

```
MolAgentLight/
└── MolAgent-Marketplace/           # Plugin marketplace
    ├── .claude-plugin/
    │   └── marketplace.json        # Marketplace catalog
    └── plugins/
        └── molagent-taskmanager/   # Training, prediction & visualization plugin
            ├── AutoMol/            # Bundled ML library
            │   └── automol/automol/
            │       ├── stacking.py
            │       ├── feature_generators.py
            │       └── model_search.py
            ├── skills/
            │   ├── train-pipeline/ # Training skill
            │   ├── predict/        # Prediction skill
            │   └── visualize/      # Dashboard skill
            │       └── scripts/
            │           └── generate_dashboard.py  # PEP 723 script
            └── hooks/              # Session setup
```

---

## Troubleshooting

### Custom Virtual Environment

By default the plugin creates and uses a `.venv` directory in the project root. To use a different virtual environment, set the `AUTOMOL_VENV` environment variable before starting Claude Code:

```bash
export AUTOMOL_VENV=/path/to/your/venv
```

### Windows: `uv` or `pip` Not Found

If `uv` is not found after installing it with `pip`, or `pip` is not found after installing Python, add the Python Scripts directory to your PATH:

```bash
export PATH="$HOME/AppData/Roaming/Python/PythonXX/Scripts:$PATH"
```

Replace `PythonXX` with your Python version, e.g. `Python312`.

### Windows: Corporate Firewall / Proxy

If you are behind a corporate firewall or proxy, set native TLS so `uv` uses the system certificate store:

```bash
export UV_NATIVE_TLS=true
```

---

## Citation

**MolAgent: Biomolecular Property Estimation in the Agentic Era**

Jose Carlos Gómez-Tamayo*, Joris Tavernier**, Roy Aerts***, Natalia Dyubankova*, Dries Van Rompaey*, Sairam Menon*, Marvin Steijaert**, Jörg Wegner*, Hugo Ceulemans*, Gary Tresadern*, Hans De Winter***, Mazen Ahmad*

> \*Johnson & Johnson \
> \** Open Analytics NV \
> \***Laboratory of Medicinal Chemistry, Department of Pharmaceutical Sciences, University of Antwerp

**Funding**: This work was partly funded by the Flanders innovation & entrepreneurship (VLAIO) project HBC.2021.112.

```bibtex
@article{molagent2025,
author = {Gómez-Tamayo, Jose Carlos and Tavernier, Joris and Aerts, Roy and Dyubankova, Natalia and Van Rompaey, Dries and Menon, Sairam and Steijaert, Marvin and Wegner, Jörg Kurt and Ceulemans, Hugo and Tresadern, Gary and De Winter, Hans and Ahmad, Mazen},
title = {MolAgent: Biomolecular Property Estimation in the Agentic Era},
journal = {Journal of Chemical Information and Modeling},
volume = {65},
number = {20},
pages = {10808-10818},
year = {2025},
doi = {10.1021/acs.jcim.5c01938},
note ={PMID: 41099298},
URL = {https://doi.org/10.1021/acs.jcim.5c01938}
}
```

---

## References

MolAgent relies on the following open-source projects:

1. **[AutoMol](https://github.com/openanalytics/AutoMol)** — Core ML pipeline for QSAR/drug discovery
2. **[scikit-learn](https://scikit-learn.org/)** — Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. JMLR, 12, 2825-2830.
3. **[molfeat](https://molfeat.datamol.io/)** — Noutahi, E., et al. (2023). datamol-io/molfeat. Zenodo. https://doi.org/10.5281/zenodo.8373019
4. **[RDKit](https://www.rdkit.org/)** — Open-source cheminformatics
5. **[Plotly.js](https://plotly.com/javascript/)** — Interactive charting library for the visualization dashboard
6. **[SmilesDrawer](https://github.com/reymond-group/smilesDrawer)** — Probst, D. & Reymond, J.-L. (2018). SmilesDrawer: Parsing and Drawing SMILES-Encoded Molecular Structures Using Client-Side JavaScript. JCIM, 58(1), 1-7.

---

## License

[GPL-3.0](LICENSE)

## Contact

- **Developers**: Joris Tavernier, Marvin Steijaert
- **Email**: joris.tavernier@openanalytics.eu
