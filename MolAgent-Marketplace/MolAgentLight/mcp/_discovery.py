"""Dynamic discovery of available AutoMol options.

This module introspects the installed AutoMol library to report what
feature generators, estimators, dim-reduction methods, etc. are available.
An admin can modify this file to add/remove options without touching the
MCP server code.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def list_feature_generators() -> dict[str, str]:
    """Return {key: description} of available feature generators."""
    result: dict[str, str] = {}

    try:
        from automol.feature_generators import retrieve_default_offline_generators
        defaults = retrieve_default_offline_generators()
        for key in defaults:
            if key == "Bottleneck":
                result[key] = "250-dim ONNX pretrained transformer encoder (fast, recommended)"
            elif key == "rdkit":
                result[key] = "RDKit 2D molecular descriptors (~210 features)"
            elif key.startswith("fps_"):
                result[key] = f"Morgan circular fingerprints ({key})"
    except ImportError:
        result["Bottleneck"] = "250-dim ONNX pretrained transformer encoder (fast, recommended)"
        result["rdkit"] = "RDKit 2D molecular descriptors (~210 features)"
        result["fps_2048_2"] = "Morgan circular fingerprints (2048 bits, radius 2)"

    # Probe for molfeat
    try:
        import molfeat  # noqa: F401
        result["desc2D"] = "Molfeat 2D descriptors (molfeat)"
        result["ecfp-count"] = "Molfeat ECFP count fingerprints (molfeat)"
        result["fcfp-count"] = "Molfeat FCFP count fingerprints (molfeat)"
        result["maccs"] = "Molfeat MACCS keys (molfeat)"
        result["avalon"] = "Molfeat Avalon fingerprints (molfeat)"
        result["secfp"] = "Molfeat SECFP fingerprints (molfeat)"
        result["estate"] = "Molfeat E-state fingerprints (molfeat)"
        result["erg"] = "Molfeat ERG fingerprints (molfeat)"
    except ImportError:
        pass

    try:
        import molfeat.trans.pretrained  # noqa: F401
        result["ChemGPT-4.7M"] = "ChemGPT 4.7M HuggingFace transformer (molfeat)"
        result["ChemBERTa-77M-MTR"] = "ChemBERTa 77M multi-task (molfeat)"
        result["MolT5"] = "MolT5 HuggingFace transformer (molfeat)"
    except ImportError:
        pass

    result["_note"] = "ECFP pattern: fps_{nbits}_{radius} (e.g. fps_1024_3, fps_4096_2)"
    return result


@lru_cache(maxsize=4)
def list_base_estimators(task: str = "Regression") -> dict[str, str]:
    """Return {key: description} of available base estimators for the task."""
    descriptions_reg = {
        "lasso": "Lasso linear regression (L1 regularization)",
        "huber": "Huber regressor (robust to outliers)",
        "pls": "Partial Least Squares regression",
        "xgb": "XGBoost gradient boosting regressor",
        "lgbm": "LightGBM gradient boosting regressor",
        "mlp": "Multi-layer perceptron neural network",
        "svr": "Support Vector Regression (RBF kernel)",
        "rfr": "Random Forest regressor",
        "ada": "AdaBoost regressor",
        "bayesianridge": "Bayesian Ridge regression",
        "dtr": "Decision Tree regressor",
        "sgdr": "Stochastic Gradient Descent regressor",
        "kernelridge": "Kernel Ridge regression (RBF kernel)",
        "gp": "Gaussian Process regressor",
    }
    descriptions_clf = {
        "lr": "Logistic Regression",
        "SVC": "Support Vector Classifier",
        "knn": "K-Nearest Neighbors classifier",
        "sgdc": "Stochastic Gradient Descent classifier",
        "dtc": "Decision Tree classifier",
        "lgbm": "LightGBM gradient boosting classifier",
        "xgb": "XGBoost gradient boosting classifier",
        "rfc": "Random Forest classifier",
        "ada": "AdaBoost classifier",
        "mlp": "Multi-layer perceptron classifier",
        "nb": "Gaussian Naive Bayes",
    }

    if task == "Classification":
        desc_map = descriptions_clf
    else:
        desc_map = descriptions_reg

    # Try to get actual keys from the archive
    try:
        if task == "Classification":
            from automol.stacking_methodarchive import ClassifierArchive
            archive = ClassifierArchive()
        else:
            from automol.stacking_methodarchive import RegressorArchive
            archive = RegressorArchive()
        keys = list(archive.get_all_method_keys())
    except ImportError:
        keys = list(desc_map.keys())

    return {k: desc_map.get(k, k) for k in keys}


def list_blender_estimators(task: str = "Regression") -> dict[str, str]:
    """Return {key: description} of available blender/final estimators.

    Blenders come from the same archive as base estimators — any base
    estimator can be used as a blender.
    """
    return list_base_estimators(task)


@lru_cache(maxsize=1)
def list_dim_reduction_methods() -> dict[str, str]:
    """Return {key: description} of available dimensionality reduction methods."""
    result = {
        "passthrough": "No dimensionality reduction (use all features)",
        "pca": "Principal Component Analysis",
        "kpca": "Kernel PCA (RBF kernel)",
        "pca+kpca": "Combined PCA and Kernel PCA",
        "v_threshold": "Variance Threshold (remove low-variance features)",
        "SelectPercentile": "Percentile-based univariate feature selection",
        "Kbest": "Select K best features (univariate scoring)",
        "rfe": "Recursive Feature Elimination (SVM-based)",
        "frommodel": "Select features from model importance (LinearSVM or ExtraTrees)",
    }

    # Verify against installed archive
    try:
        from automol.stacking_methodarchive import ReducedimArchive
        archive = ReducedimArchive()
        keys = list(archive.get_all_method_keys())
        return {k: result.get(k, k) for k in keys}
    except ImportError:
        pass

    return result


def list_model_configs() -> dict[str, str]:
    """Return available model architecture configurations."""
    return {
        "single_method": "Single fitted method (fastest, no ensemble)",
        "inner_methods": "Average predictions from multiple inner-fold models",
        "inner_stacking": "Sklearn StackingRegressor per inner fold",
        "single_stack": "One StackingRegressor on all outer folds",
        "top_method": "Base estimators + final blender estimator (default for cheap/moderate)",
        "top_stacking": "Base estimators + StackingRegressor as final (default for expensive)",
        "stacking_stacking": "Stacking of stacking models (classification only, heavy)",
    }


def list_search_types() -> dict[str, str]:
    """Return available hyperparameter search strategies."""
    return {
        "grid": "Exhaustive grid search (slowest, deterministic)",
        "randomized": "Randomized search with configurable iterations",
        "hyperopt": "Bayesian optimization via HyperOpt (best for expensive budgets)",
    }


def list_scorers(task: str = "Regression") -> dict[str, str]:
    """Return available scoring metrics for the task."""
    if task == "Classification":
        return {
            "balanced_accuracy": "Balanced accuracy (default for classification)",
            "accuracy": "Standard accuracy",
            "f1_weighted": "Weighted F1 score",
            "roc_auc": "ROC AUC (binary only)",
            "precision_weighted": "Weighted precision",
            "recall_weighted": "Weighted recall",
        }
    return {
        "r2": "R-squared coefficient of determination (default)",
        "neg_mean_squared_error": "Negative MSE (lower is better)",
        "neg_mean_absolute_error": "Negative MAE (lower is better)",
        "neg_root_mean_squared_error": "Negative RMSE",
        "explained_variance": "Explained variance score",
    }


def get_all_options(task: str = "Regression") -> dict:
    """Return all option categories at once."""
    return {
        "feature_generators": list_feature_generators(),
        "base_estimators": list_base_estimators(task),
        "blender_estimators": list_blender_estimators(task),
        "dim_reduction": list_dim_reduction_methods(),
        "model_configs": list_model_configs(),
        "search_types": list_search_types(),
        "scorers": list_scorers(task),
    }
