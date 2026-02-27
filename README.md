# MolAgent Light

<div align="center">

[![MolAgent](https://img.shields.io/badge/MolAgent-1.0.0-red.svg)](https://github.com/openanalytics/MolAgent)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![AutoMol](https://img.shields.io/badge/AutoMol-Pipeline-orange.svg)](https://github.com/openanalytics/AutoMol)
[![License](https://img.shields.io/badge/License-GPL--3.0-yellow.svg)](LICENSE)

**MolAgent Light is a Claude Code plugin for automated molecular property prediction**

[Installation](#installation) • [Quick Start](#quick-start) • [Skills](#skills) • [Citation](#citation)

</div>

---

## Overview

MolAgent Light enables expert-level molecular property prediction through natural language interaction with Claude Code. It provides:

- **Autonomous Model Construction** — Train ensemble stacking models for regression and classification
- **Intelligent Feature Selection** — Automatically selects optimal molecular representations (2D descriptors, ECFP fingerprints, pretrained encoders)
- **On-the-fly Predictions** — Make predictions on new molecules using trained models

This is a lightweight version of [MolAgent](https://github.com/openanalytics/MolAgent) without MCP server dependencies, designed specifically for Claude Code.

---

## Installation

Clone the repository with submodules:

```bash
git clone https://github.com/JorisTavernier/MolAgentLight.git
cd MolAgentLight
```

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

Note that earlier version of python on your system is fine, the virtual uv  environment will use newer version.  
### Windows Troubleshooting

If `uv` is not found after installing it with `pip` or pip is not found after installing python, add the Python Scripts directory to your PATH:

```bash
export PATH="$HOME/AppData/Roaming/Python/PythonXX/Scripts:$PATH"
```

Replace `PythonXX` with your Python version, e.g. `Python312`.

If you are behind a corporate firewall or proxy, set native TLS so `uv` uses the system certificate store:

```bash
export UV_NATIVE_TLS=true
```

### Custom Virtual Environment

By default the plugin creates and uses a `.venv` directory in the project root. To use a different virtual environment, set the `AUTOMOL_VENV` environment variable before starting Claude Code:

```bash
export AUTOMOL_VENV=/path/to/your/venv
```

## Quick Start

### Install the Plugin

In Claude Code:

```
/plugin marketplace add ./MolAgent-Marketplace
/plugin install molagent-taskmanager@molagent-marketplace
```

> **Note:** After installing the plugin, restart Claude Code so the SessionStart hook runs and sets up the virtual environment with AutoMol and the environment variables (only the first time). Now restart Claude code a **second** time and you are good to go.   

### Train a Model

Prepare a csv file with the SMILES of the compounds an their property. Preferable use a column name for the SMILES like 'DRUG' or 'SMILES' then the code will pick it up, else you will need to provide the name. For example 
```
name,SMILES,potency,prop2,prop3,prop4,prop5
CHEMBL1733973,CC1COC(=O)O1,6.643689871786073,7.106531089985014,6.9811586501651055,4.517134550509942,0.0
CHEMBL11298,N[C@@H](CO)C(=O)O,0.8039345711747932,6.909154276280659,,4.2977726597961725,1.0
CHEMBL119604,OCCNCCO,4.6621819913829565,6.386019611731673,,4.699490823748647,0.0
CHEMBL15972,O=Cc1ccccc1,1.7722018651644058,6.597002439238543,6.84726059714884,5.1460695983691425,1.0
CHEMBL1235535,OCc1cccnc1,11.753512652733662,,6.7824928131407844,4.815961514096383,0.0
CHEMBL354077,Nc1ccncc1N,18.48577385876229,,6.599225758771932,5.063891096437568,0.0
```

Once in Claude Code, use natural language:

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
The predict skill will list the trained models from the registry. 

---

## Skills

### train-pipeline

End-to-end training pipeline for molecular property prediction.

**Features:**
- Regression and classification support
- Multiple feature generators: Bottleneck (pretrained encoder), ECFP fingerprints, RDKit descriptors
- Nested cross-validation with hyperparameter optimization
- Ensemble stacking methods
- Computational load presets: `free`, `cheap`, `moderate`, `expensive`

**Outputs:**
- Trained models saved as `.pt` files in `MolagentFiles/{run_id}/`
- Model registry at `MolagentFiles/model_registry.json`
- Evaluation metrics and model card

### predict

Single-phase inference skill.

**Features:**
- Auto-discovers trained models from registry
- Accepts CSV files or individual SMILES strings
- Supports merged multi-property models
- Outputs predictions to CSV

---

## Computational Load Presets

| Level | Description |
|-------|-------------|
| `free` | Minimal — single method, no hyperparameter search |
| `cheap` | Light — inner methods, basic search |
| `moderate` | Medium — stacking, randomized search |
| `expensive` | Full — stacking of stacking, hyperopt |

---

## Architecture

```
MolAgentLight/
└── MolAgent-Marketplace/           # Plugin marketplace
    ├── .claude-plugin/
    │   └── marketplace.json        # Marketplace catalog
    └── plugins/
        └── molagent-taskmanager/   # Claude Code plugin
            ├── AutoMol/            # Bundled ML library
            │   └── automol/automol/
            │       ├── stacking.py
            │       ├── feature_generators.py
            │       └── model_search.py
            ├── skills/
            │   ├── train-pipeline/ # Training skill
            │   └── predict/        # Prediction skill
            └── hooks/              # Session setup
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

---

## License

[GPL-3.0](LICENSE)

## Contact

- **Developers**: Joris Tavernier, Marvin Steijaert
- **Email**: joris.tavernier@openanalytics.eu
