---
name: train-pipeline
description: Plan and execute an AutoMol training pipeline. Analyzes dataset, auto-configures a plan, presents config for user approval, then executes. Use when the user wants to train a complete model from raw data.
allowed-tools: Read, Glob, Bash, AskUserQuestion,
  mcp__plugin_MolAgentLight_automol-mcp__list_options,
  mcp__plugin_MolAgentLight_automol-mcp__start_training_session,
  mcp__plugin_MolAgentLight_automol-mcp__answer_training_question,
  mcp__plugin_MolAgentLight_automol-mcp__train_and_visualize,
  mcp__plugin_MolAgentLight_automol-mcp__list_models
---

# AutoMol — Training Pipeline

Orchestrate a full training pipeline using the `automol-mcp` MCP tools.

---

## Workflow

### Step 1: Start Session

Call `start_training_session(csv_file="<absolute_path>")` (or `dataset_id` for uploaded datasets).

This auto-detects: SMILES column, target properties, task type, recommended features, computational load, and split strategy. It returns a `session_id` and the detected config.

### Step 2: Present Config to User

Format the detected config clearly:

```
Dataset: {csv_file} ({characteristics.valid_smiles} molecules)
SMILES column: {detected.smiles_column}
Task: {detected.task} — {options.task[detected.task]}
Targets: {detected.properties}
Features: {detected.feature_keys}
Split: {detected.split_strategy} — {options.split_strategy[detected.split_strategy]}
Load: {detected.computational_load} — {options.computational_load[detected.computational_load]}
Log10: {detected.use_log10}

Warnings:
  {from response.question field}

Target Summary:
  {for each target: column, task_type, unique_values, null_count, key stats}
```

**Critical domain terms** (present these correctly to users):
- **RegressionClassification** is binary classification via regression estimators on 0/1 labels (predictions clipped to [0,1] as probabilities). It is NOT "train both regression and classification."
- **blender_properties** are auxiliary numeric columns used as extra input features alongside molecular representations — they are NOT targets.
- **feature_keys** are molecular representation methods (Bottleneck encoder, RDKit descriptors, fingerprints) — not CSV column names.

### Step 3: Apply Overrides or Confirm

If the user wants changes, parse their natural language into typed parameters and call `answer_training_question(session_id=..., field=value, ...)`.

Use `list_options(category=...)` to discover valid values for feature_keys, estimators, scorers, etc.

When ready: `answer_training_question(session_id=..., confirm=True)`. This returns the finalized config dict.

**Always pause for confirmation**: after presenting the detected config, ask the user whether to proceed or make changes — even if the original request clearly asked for training. The config summary surfaces important warnings (mixed target types, log10 flags, skewed targets, null rates) that the user should review before a potentially long run.

### Step 4: Execute Training

Call `train_and_visualize(config={...})` with the confirmed config.

This is long-running (minutes to hours depending on computational_load). It runs: prepare → split → train → evaluate → refit → dashboard generation → registry update.

### Step 5: Present Results

From the response, display:
- Metrics per property (R2/RMSE/MAE for regression; accuracy/AUC for classification)
- Model ID (for future predictions via the `predict` skill)
- Dashboard path (offer to open in browser)

```
Training Complete!
  Model ID: {model_id}
  Properties: {properties}

  Metrics:
    {prop1}: R2={r2}, RMSE={rmse}
    {prop2}: R2={r2}, RMSE={rmse}

  Dashboard: {dashboard_path}
  Next: use the predict skill to run inference on new molecules
```

---

## Error Handling

- If `start_training_session` fails with "CSV file not found": verify the path is absolute (Windows: `C:/Users/...`, not `/c/Users/...`).
- If `train_and_visualize` disconnects (long-running timeout): call `list_models()` to check if training completed in the background.
- If training fails mid-pipeline: start a new session. The MCP pipeline is atomic — there is no partial resume.

---

## Presentation Templates

**Computational load choices** (when user asks what to pick):
- `free` — ~0–2 min. Single LightGBM, good for fast signal checking.
- `cheap` — ~2–10 min. Light ensemble, quick prototyping.
- `moderate` — ~10–360 min. Stacking ensemble, good balance.
- `expensive` — 1–48 hrs. Full hyperopt search, best performance.

---

## Fallback (MCP unavailable)

If the MCP server is not connected (check `/mcp`), the pipeline can be run directly via scripts. See `skills/train-pipeline/steps/` for per-step instructions and `skills/train-pipeline/scripts/` for the underlying Python scripts. Each script is invoked as `uv run $MOLAGENT_PLUGIN_ROOT/skills/train-pipeline/scripts/<script>.py`.
