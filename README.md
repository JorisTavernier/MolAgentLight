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

## Quick Start

### Running with Claude Code

Start Claude Code with the MolAgent plugin:

```bash
claude --plugin-dir ./MolAgent-TaskManager/
```

### Train a Model

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
MolAgent/
├── AutoMol/                    # Core ML library (submodule)
│   └── automol/automol/
│       ├── stacking.py         # Ensemble model classes
│       ├── feature_generators.py
│       └── model_search.py
└── MolAgent-TaskManager/      # Claude Code plugin
    ├── skills/
    │   ├── train-pipeline/     # Training skill
    │   └── predict/            # Prediction skill
    └── hooks/                  # Session setup
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
5. **[RDKit](https://www.rdkit.org/)** — Open-source cheminformatics

---

## License

[GPL-3.0](LICENSE)

## Contact

- **Developers**: Joris Tavernier, Marvin Steijaert, Jose Carlos Gómez-Tamayo, Mazen Ahmad
- **Email**: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu
