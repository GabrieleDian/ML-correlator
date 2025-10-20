#!/usr/bin/env python3
"""
loop_predictor.py

Features:
- Target is binarised strictly: 0 if coefficient == 0, else 1 (supports fraction-like strings).
- --fraction_mode {none,float,binary} for feature columns:
    none   : leave as-is (coerce to numeric later with errors='coerce')
    float  : parse "a/b" or ints to float; others -> NaN
    binary : parse then map nonzero->1, zero->0; NaN stays NaN (XGBoost handles natively)
- Replaces np.inf/-np.inf and string forms ('inf','infinity','nan', any case) with np.nan.
- Drops all-NaN feature columns **with logging** (XGBoost handles missing values natively).
- Auto-detect loop CSVs; infer loop number from filename or 'loop' column.
- Experiments:
    1) Intra-loop (skips loop=5 in isolation)
    2) Pair mixed (l -> l+1)
    3) Directional chains (5,6->7; 5,6,7->8; ...)
- Cheap parameter search with RandomizedSearchCV on XGBoost
- Robust class-imbalance handling (adaptive CV folds; safe AUC evaluation)
- Loguru logging and CSV reports to configurable output directory (default: ./output)

Example:
  python loop_predictor.py --data_dir ./data --target_col COEFFICIENT --fraction_mode binary --output_dir ./results
"""

import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from fractions import Fraction
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split, cross_val_score
from xgboost import XGBClassifier


# -----------------------
# Config / Patterns
# -----------------------

NON_FEATURE_COLS_CANDIDATES = {
    "graph_id", "id", "ID", "name", "Name", "loop", "LOOP", "Loop"
}

# Accepts integers and "a/b" (with optional spaces and optional negative sign)
FRACTION_PATTERN = re.compile(r"^\s*-?\d+\s*(/\s*-?\d+\s*)?$")


# -----------------------
# Logging
# -----------------------

def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = log_dir / f"loop_predictor_{ts}.log"
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), colorize=True, level="INFO")
    logger.add(logfile, level="DEBUG", rotation="5 MB", retention=10)
    logger.info(f"Logging to: {logfile}")


# -----------------------
# Small helpers
# -----------------------

def find_loop_in_filename(fname: str) -> int:
    nums = re.findall(r"(\d+)", os.path.basename(fname))
    if not nums:
        return -1
    return int(nums[-1])


def sanitize_inf_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace np.inf/-np.inf and common string tokens with np.nan.
    Pandas-compatible (no 'flags' kwarg).
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.replace(
        to_replace=r"^\s*([iI][nN][fF]|-?[iI][nN][fF][iI][nN][iI][tT][yY]|[nN][aA][nN])\s*$",
        value=np.nan,
        regex=True,
    )
    return df


def parse_fraction_value(val):
    """
    Parse integers or 'a/b' into float. Returns np.nan on failure.
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating, bool)):
        return float(val)
    s = str(val).strip()
    if not s:
        return np.nan
    if not FRACTION_PATTERN.match(s):
        # try plain float string
        try:
            return float(s)
        except Exception:
            return np.nan
    try:
        if "/" in s:
            num, den = s.split("/")
            return float(Fraction(int(num.strip()), int(den.strip())))
        else:
            return float(int(s))
    except Exception:
        return np.nan


def fraction_to_float_or_nan(s):
    return parse_fraction_value(s)


def binarise_fraction_series_zero_vs_nonzero(series: pd.Series) -> pd.Series:
    """
    Map fraction-like strings (or numbers) to 0/1:
      0 if exactly zero, 1 if non-zero; NaN stays NaN.
    """
    parsed = series.map(fraction_to_float_or_nan)
    out = parsed.copy()
    mask = parsed.notna()
    out.loc[mask] = (parsed.loc[mask] != 0.0).astype(int)
    return out.astype("Int64")  # nullable ints to preserve NaN temporarily


def min_class_count(y) -> int:
    c = Counter(y)
    return min(c.values())


def has_two_classes(y) -> bool:
    return len(set(y)) >= 2


def create_stratification_labels(y, loop_labels=None):
    """
    Create combined stratification labels for multi-label stratification.
    If loop_labels is provided, combines target and loop labels.
    Otherwise, just uses target labels.
    """
    if loop_labels is None:
        return y
    
    # Create combined labels for stratification
    # Format: "target_loop" (e.g., "0_5", "1_6")
    combined_labels = [f"{target}_{loop}" for target, loop in zip(y, loop_labels)]
    return combined_labels


# -----------------------
# Fraction transforms for FEATURES
# -----------------------

def transform_fraction_columns(df: pd.DataFrame, target_col: str, mode: str) -> pd.DataFrame:
    """
    If mode == 'float': parse fraction-like strings to float.
    If mode == 'binary': parse and then map nonzero -> 1, zero -> 0; NaN stays NaN.
    mode == 'none': no special handling (beyond numeric coercion later).
    Operates on all non-target, non-known-id columns.
    """
    if mode not in {"none", "float", "binary"} or mode == "none":
        return df

    df = df.copy()
    drop_cols = {target_col} | NON_FEATURE_COLS_CANDIDATES
    candidate_cols = [c for c in df.columns if c not in drop_cols]

    converted_cols = []
    for c in candidate_cols:
        col = df[c]
        parsed = col.map(parse_fraction_value)

        # Conversion heuristic: if many values parse or we see fraction-like content
        orig_non_na = col.notna().sum()
        new_non_na = pd.Series(parsed).notna().sum()
        saw_fraction_like = col.astype(str).str.match(FRACTION_PATTERN).any()

        if (new_non_na >= orig_non_na * 0.5) or saw_fraction_like:
            if mode == "float":
                df[c] = parsed
                converted_cols.append(c)
            elif mode == "binary":
                parsed = pd.Series(parsed, index=col.index)
                bin_col = parsed.copy()
                mask = parsed.notna()
                bin_col.loc[mask] = (parsed.loc[mask] != 0.0).astype(int)
                # keep NaN, XGBoost will handle natively
                df[c] = bin_col.astype("float")
                converted_cols.append(c)

    if converted_cols:
        logger.info(f"Fraction transform ({mode}) applied to {len(converted_cols)} columns: {converted_cols[:8]}{'...' if len(converted_cols)>8 else ''}")
    else:
        logger.info("Fraction transform: no applicable columns detected.")
    return df


# -----------------------
# NA Feature dropper
# -----------------------

def drop_all_nan_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Aggressively drop ALL columns that are entirely NaN (excluding target/id columns).
    NEVER impute completely null features - just remove them.
    """
    drop_cols = set([target_col]) | NON_FEATURE_COLS_CANDIDATES
    nan_features = [c for c in df.columns if c not in drop_cols and df[c].isna().all()]
    if nan_features:
        logger.warning(
            f"AGGRESSIVELY DROPPING {len(nan_features)} all-NaN feature columns (NO IMPUTATION): "
            f"{nan_features[:8]}{'...' if len(nan_features) > 8 else ''}"
        )
        df = df.drop(columns=nan_features)
    else:
        logger.info("No all-NaN feature columns found.")
    return df


def ensure_no_all_nan_features(df: pd.DataFrame, target_col: str, context: str = "") -> pd.DataFrame:
    """
    Final safety check to ensure no all-NaN features remain before modeling.
    This is called right before feature selection to catch any remaining all-NaN columns.
    """
    drop_cols = set([target_col]) | NON_FEATURE_COLS_CANDIDATES
    nan_features = [c for c in df.columns if c not in drop_cols and df[c].isna().all()]
    if nan_features:
        logger.error(f"CRITICAL: Found {len(nan_features)} all-NaN features in {context}: {nan_features}")
        df = df.drop(columns=nan_features)
        logger.warning(f"Removed all-NaN features from {context}")
    return df


# -----------------------
# Data loading
# -----------------------

def load_loop_datasets(data_dir: Path, target_col: str, fraction_mode: str) -> Dict[int, pd.DataFrame]:
    """
    Load all CSVs, sanitize inf/nan, transform fraction-like feature columns,
    drop all-NaN feature columns, and binarise the target strictly: 0 if == 0, else 1.
    Map loop_number -> DataFrame.
    """
    loop_map: Dict[int, pd.DataFrame] = {}
    csv_paths = sorted([p for p in data_dir.glob("*.csv") if p.is_file()])
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    for p in csv_paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            logger.warning(f"Skipping {p} (read error): {e}")
            continue

        df = sanitize_inf_nan(df)

        # Infer loop
        loop = find_loop_in_filename(p.name)
        if loop < 0:
            if "loop" in df.columns:
                unique_loops = pd.unique(df["loop"].dropna())
                if len(unique_loops) == 1:
                    loop = int(unique_loops[0])
                else:
                    logger.warning(f"Skipping {p} (ambiguous 'loop' column).")
                    continue
            else:
                logger.warning(f"Skipping {p} (no loop number in filename or column).")
                continue

        if target_col not in df.columns:
            logger.warning(f"Skipping {p} (missing target column '{target_col}').")
            continue

        # Target binarisation: 0 if coefficient == 0, else 1 (supports fractions)
        t_bin = binarise_fraction_series_zero_vs_nonzero(df[target_col])

        # Drop rows where target couldn't be parsed
        num_bad = int(t_bin.isna().sum())
        if num_bad > 0:
            logger.warning(f"{p.name}: dropping {num_bad} rows with unparseable target values.")
        df = df.loc[t_bin.notna()].copy()
        df[target_col] = t_bin.loc[t_bin.notna()].astype(int)

        # Feature fraction transforms (optional)
        df = transform_fraction_columns(df, target_col, fraction_mode)

        # Drop all-NaN features (XGBoost handles missing values natively)
        df = drop_all_nan_features(df, target_col)

        loop_map[loop] = df
        logger.info(f"Loaded loop {loop}: {p.name} (rows={len(df)})")

    if not loop_map:
        raise RuntimeError("No valid loop datasets found.")
    return dict(sorted(loop_map.items(), key=lambda kv: kv[0]))


def select_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    drop_cols = set([target_col]) | NON_FEATURE_COLS_CANDIDATES

    numeric_df = df.copy()
    for col in numeric_df.columns:
        if col not in drop_cols:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")

    numeric_cols = numeric_df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in drop_cols]
    
    # Remove features that are entirely NaN after numeric conversion
    valid_features = []
    for feature in features:
        if not numeric_df[feature].isna().all():
            valid_features.append(feature)
        else:
            logger.debug(f"Skipping all-NaN feature after numeric conversion: {feature}")
    
    if not valid_features:
        raise ValueError("No valid numeric feature columns found after filtering.")
    return valid_features


# -----------------------
# Modeling
# -----------------------

def build_model(random_state: int) -> XGBClassifier:
    """
    Build XGBoost model directly - no imputation needed as XGBoost handles missing values natively.
    """
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1
    )
    return xgb


def cheap_param_distributions() -> dict:
    return {
        "n_estimators": [200, 400, 800],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.02, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "min_child_weight": [1, 2, 5, 10],
        "reg_alpha": [0.0, 0.001, 0.01, 0.1],
        "reg_lambda": [0.5, 1.0, 2.0],
    }


def randomized_search(model: XGBClassifier, X, y, n_iter: int, random_state: int, loop_order_labels=None):
    # Adaptive CV folds based on class counts
    if not has_two_classes(y):
        raise ValueError("Training labels are single-class; cannot perform CV.")
    min_count = min_class_count(y)
    n_splits = max(2, min(3, min_count))  # try up to 3-fold, but ensure ≥2 and feasible

    if n_splits < 2:
        raise ValueError("Too few samples per class for CV.")

    # Create stratification labels for CV
    stratify_labels = create_stratification_labels(y, loop_order_labels)
    
    # Check if stratification is feasible with combined labels
    unique_stratify_labels = set(stratify_labels)
    if len(unique_stratify_labels) > len(y) * 0.8:  # Too many unique combinations
        logger.warning("Too many unique stratification combinations for CV, using target-only stratification")
        stratify_labels = y

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    param_dist = cheap_param_distributions()
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        n_jobs=-1,
        cv=cv,
        random_state=random_state,
        verbose=0,
        refit=True,
    )
    search.fit(X, y)
    return search


def evaluate_model_safe(model: XGBClassifier, X_train, y_train, X_test, y_test, context: str):
    # Train AUC (train set enforced to have two classes before calling)
    train_pred = model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_pred)

    # Test AUC may be undefined if test is single-class
    if not has_two_classes(y_test):
        logger.warning(f"{context}: test set is single-class; test AUC undefined. Returning NaN.")
        return float(train_auc), float("nan")

    test_pred = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_pred)
    return float(train_auc), float(test_auc)


def compute_individual_loop_aucs(model: XGBClassifier, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                 train_loops: List[int], test_loop: int, target_col: str) -> Dict[str, float]:
    """
    Compute AUC for each individual loop in training and testing.
    Returns a dictionary with keys like 'train_loop_6_auc', 'test_loop_7_auc', etc.
    """
    individual_aucs = {}
    
    # Training loop AUCs
    for loop in train_loops:
        if 'loop' in train_df.columns:
            loop_mask = train_df['loop'] == loop
            if loop_mask.sum() > 0:
                loop_data = train_df[loop_mask]
                features = select_feature_columns(loop_data, target_col)
                X_loop = loop_data[features].values
                y_loop = loop_data[target_col].values
                
                if has_two_classes(y_loop):
                    loop_pred = model.predict_proba(X_loop)[:, 1]
                    loop_auc = roc_auc_score(y_loop, loop_pred)
                    individual_aucs[f'train_loop_{loop}_auc'] = float(loop_auc)
                else:
                    individual_aucs[f'train_loop_{loop}_auc'] = float('nan')
    
    # Test loop AUC
    if 'loop' in test_df.columns:
        loop_mask = test_df['loop'] == test_loop
        if loop_mask.sum() > 0:
            loop_data = test_df[loop_mask]
            features = select_feature_columns(loop_data, target_col)
            X_loop = loop_data[features].values
            y_loop = loop_data[target_col].values
            
            if has_two_classes(y_loop):
                loop_pred = model.predict_proba(X_loop)[:, 1]
                loop_auc = roc_auc_score(y_loop, loop_pred)
                individual_aucs[f'test_loop_{test_loop}_auc'] = float(loop_auc)
            else:
                individual_aucs[f'test_loop_{test_loop}_auc'] = float('nan')
    
    return individual_aucs


# -----------------------
# Experiment Runners
# -----------------------

def run_intra_loop(loop_df: pd.DataFrame, loop: int, target_col: str,
                   n_iter: int, random_state: int):
    # Final safety check for all-NaN features
    loop_df = ensure_no_all_nan_features(loop_df, target_col, f"intra-loop {loop}")
    
    features = select_feature_columns(loop_df, target_col)
    X = loop_df[features].values
    y = loop_df[target_col].values

    # Check if we have loop order information for stratification
    loop_order_col = None
    if 'loop_order' in loop_df.columns:
        loop_order_col = loop_df['loop_order'].values
    elif 'order' in loop_df.columns:
        loop_order_col = loop_df['order'].values
    
    # Create stratification labels (combine target and loop order if available)
    stratify_labels = create_stratification_labels(y, loop_order_col)
    
    # Need two classes and at least 2 per class to stratify split
    if not has_two_classes(y) or min_class_count(y) < 2:
        raise ValueError(f"Intra-loop {loop}: not enough examples per class for a stratified split.")

    # Check if stratification is feasible with combined labels
    unique_stratify_labels = set(stratify_labels)
    if len(unique_stratify_labels) > len(y) * 0.8:  # Too many unique combinations
        logger.warning(f"Intra-loop {loop}: too many unique stratification combinations, using target-only stratification")
        stratify_labels = y

    # Use 5-fold cross-validation with hyperparameter search
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    # Perform cross-validation with hyperparameter search
    train_scores = []
    test_scores = []
    best_params_per_fold = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Get loop order labels for this fold if available
        fold_loop_order_labels = None
        if loop_order_col is not None:
            fold_loop_order_labels = loop_order_col[train_idx]
        
        # Build model and do minimal hyperparameter search
        model = build_model(random_state + fold_idx)  # Different seed per fold
        search = randomized_search(model, X_train_fold, y_train_fold, n_iter, random_state + fold_idx, fold_loop_order_labels)
        
        # Store best parameters for this fold
        best_params_per_fold.append(search.best_params_)
        
        # Evaluate on training fold
        train_pred = search.best_estimator_.predict_proba(X_train_fold)[:, 1]
        train_auc = roc_auc_score(y_train_fold, train_pred)
        train_scores.append(train_auc)
        
        # Evaluate on validation fold
        val_pred = search.best_estimator_.predict_proba(X_val_fold)[:, 1]
        val_auc = roc_auc_score(y_val_fold, val_pred)
        test_scores.append(val_auc)
    
    mean_train_auc = np.mean(train_scores)
    mean_test_auc = np.mean(test_scores)
    std_train_auc = np.std(train_scores)
    std_test_auc = np.std(test_scores)
    
    # Find the fold with the best test AUC and use its parameters
    best_fold_idx = np.argmax(test_scores)
    best_params = best_params_per_fold[best_fold_idx]
    
    logger.info(f"Intra-loop {loop}: 5-fold CV train_auc = {mean_train_auc:.4f} ± {std_train_auc:.4f}, test_auc = {mean_test_auc:.4f} ± {std_test_auc:.4f}")
    logger.info(f"Intra-loop {loop}: Best fold {best_fold_idx} had test_auc = {test_scores[best_fold_idx]:.4f}")
    
    return {
        "experiment_type": "intra-loop",
        "train_loops": str(loop),
        "test_loops": str(loop),
        "n_train": int(len(y)),
        "n_test": int(len(y)),  # Same as train since it's CV
        "train_auc": mean_train_auc,
        "test_auc": mean_test_auc,
        "train_auc_std": std_train_auc,
        "test_auc_std": std_test_auc,
        "best_params": json.dumps(best_params),  # Best params from fold with highest test AUC
    }


def run_pair_mixed(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   l_train: int, l_test: int, target_col: str,
                   n_iter: int, random_state: int):
    # Final safety check for all-NaN features
    train_df = ensure_no_all_nan_features(train_df, target_col, f"pair train {l_train}")
    test_df = ensure_no_all_nan_features(test_df, target_col, f"pair test {l_test}")
    
    tr_feats = set(select_feature_columns(train_df, target_col))
    te_feats = set(select_feature_columns(test_df, target_col))
    common = sorted(list(tr_feats & te_feats))
    if not common:
        raise ValueError(f"No overlapping numeric features between loops {l_train} and {l_test}.")

    X_tr = train_df[common].values
    y_tr = train_df[target_col].values
    X_te = test_df[common].values
    y_te = test_df[target_col].values

    logger.info(f"Pair {l_train}->{l_test}: training samples={len(y_tr)}, test samples={len(y_te)}")
    
    if not has_two_classes(y_tr) or min_class_count(y_tr) < 2:
        class_counts = Counter(y_tr)
        logger.warning(f"Pair {l_train}->{l_test}: insufficient training samples per class for CV. Class distribution: {dict(class_counts)}")
        raise ValueError(f"Pair {l_train}->{l_test}: insufficient training samples per class for CV.")

    # Check if we have loop order information for stratification in training data
    train_loop_order_labels = None
    if 'loop_order' in train_df.columns:
        train_loop_order_labels = train_df['loop_order'].values
    elif 'order' in train_df.columns:
        train_loop_order_labels = train_df['order'].values

    model = build_model(random_state)
    search = randomized_search(model, X_tr, y_tr, n_iter, random_state, train_loop_order_labels)

    train_auc, test_auc = evaluate_model_safe(search.best_estimator_, X_tr, y_tr, X_te, y_te, context=f"pair {l_train}->{l_test}")
    
    # Compute individual loop AUCs
    individual_aucs = compute_individual_loop_aucs(search.best_estimator_, train_df, test_df, [l_train], l_test, target_col)
    
    result = {
        "experiment_type": "mixed",
        "train_loops": str(l_train),
        "test_loops": str(l_test),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "train_auc": train_auc,
        "test_auc": test_auc,
        "best_params": json.dumps(search.best_params_),
    }
    result.update(individual_aucs)
    return result


def run_mixed_loop_cv(train_loops: List[int], loop_map: Dict[int, pd.DataFrame], 
                      target_col: str, n_iter: int, random_state: int):
    """
    Train on a mix of multiple loops and test on the same loops using 5-fold cross-validation.
    """
    if len(train_loops) != 2:
        raise ValueError("Mixed loop CV currently supports exactly 2 loops")
    
    l1, l2 = train_loops
    if l1 not in loop_map or l2 not in loop_map:
        raise ValueError(f"Missing loop data for {train_loops}")
    
    # Combine training data from both loops
    train_frames = [loop_map[l] for l in train_loops]
    combined_df = pd.concat(train_frames, axis=0, ignore_index=True)
    
    # Extra safety: if concatenation introduces all-NaN columns, drop them
    combined_df = drop_all_nan_features(combined_df, target_col)
    
    # Final safety check for all-NaN features
    combined_df = ensure_no_all_nan_features(combined_df, target_col, f"mixed CV {l1},{l2}")
    
    # Get common features
    features = select_feature_columns(combined_df, target_col)
    X = combined_df[features].values
    y = combined_df[target_col].values
    
    if not has_two_classes(y) or min_class_count(y) < 2:
        raise ValueError(f"Mixed loops {train_loops}: insufficient samples per class for CV.")
    
    # Check if we have loop order information for stratification
    loop_order_col = None
    if 'loop_order' in combined_df.columns:
        loop_order_col = combined_df['loop_order'].values
    elif 'order' in combined_df.columns:
        loop_order_col = combined_df['order'].values
    
    # Create stratification labels (combine target and loop order if available)
    stratify_labels = create_stratification_labels(y, loop_order_col)
    
    # Check if stratification is feasible with combined labels
    unique_stratify_labels = set(stratify_labels)
    if len(unique_stratify_labels) > len(y) * 0.8:  # Too many unique combinations
        logger.warning(f"Mixed loops {train_loops}: too many unique stratification combinations, using target-only stratification")
        stratify_labels = y
    
    # Use 5-fold cross-validation with hyperparameter search
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    # Perform cross-validation with hyperparameter search
    train_scores = []
    test_scores = []
    best_params_per_fold = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Get loop order labels for this fold if available
        fold_loop_order_labels = None
        if loop_order_col is not None:
            fold_loop_order_labels = loop_order_col[train_idx]
        
        # Build model and do minimal hyperparameter search
        model = build_model(random_state + fold_idx)  # Different seed per fold
        search = randomized_search(model, X_train_fold, y_train_fold, n_iter, random_state + fold_idx, fold_loop_order_labels)
        
        # Store best parameters for this fold
        best_params_per_fold.append(search.best_params_)
        
        # Evaluate on training fold
        train_pred = search.best_estimator_.predict_proba(X_train_fold)[:, 1]
        train_auc = roc_auc_score(y_train_fold, train_pred)
        train_scores.append(train_auc)
        
        # Evaluate on validation fold
        val_pred = search.best_estimator_.predict_proba(X_val_fold)[:, 1]
        val_auc = roc_auc_score(y_val_fold, val_pred)
        test_scores.append(val_auc)
    
    mean_train_auc = np.mean(train_scores)
    mean_test_auc = np.mean(test_scores)
    std_train_auc = np.std(train_scores)
    std_test_auc = np.std(test_scores)
    
    # Find the fold with the best test AUC and use its parameters
    best_fold_idx = np.argmax(test_scores)
    best_params = best_params_per_fold[best_fold_idx]
    
    logger.info(f"Mixed CV {l1},{l2}: 5-fold CV train_auc = {mean_train_auc:.4f} ± {std_train_auc:.4f}, test_auc = {mean_test_auc:.4f} ± {std_test_auc:.4f}")
    logger.info(f"Mixed CV {l1},{l2}: Best fold {best_fold_idx} had test_auc = {test_scores[best_fold_idx]:.4f}")
    
    # Return separate rows for each loop in the mixed training set
    results = []
    for test_loop in train_loops:
        # Get data for this specific loop using the SAME features as training
        loop_df = loop_map[test_loop]
        
        # Check which features are available in this loop
        available_features = [f for f in features if f in loop_df.columns]
        missing_features = [f for f in features if f not in loop_df.columns]
        
        if missing_features:
            logger.warning(f"Mixed CV {train_loops} -> {test_loop}: Missing {len(missing_features)} features: {missing_features[:3]}{'...' if len(missing_features) > 3 else ''}")
        
        if not available_features:
            logger.error(f"Mixed CV {train_loops} -> {test_loop}: No common features available. Skipping.")
            continue
        
        # Use only available features for this loop
        X_loop = loop_df[available_features].values
        y_loop = loop_df[target_col].values
        
        # Use the best model from the best fold to evaluate on this specific loop
        # We'll retrain the best model on the full training set for consistency
        best_model = build_model(random_state + best_fold_idx)
        best_search = randomized_search(best_model, X, y, n_iter, random_state + best_fold_idx, loop_order_col)
        
        # Evaluate on this specific loop
        if has_two_classes(y_loop):
            # We need to handle the case where the model expects more features than available
            # Create a feature matrix with the same number of features as training
            X_loop_padded = np.full((X_loop.shape[0], len(features)), np.nan)
            feature_idx_map = {f: i for i, f in enumerate(features)}
            for i, feature in enumerate(available_features):
                if feature in feature_idx_map:
                    X_loop_padded[:, feature_idx_map[feature]] = X_loop[:, i]
            
            loop_pred = best_search.best_estimator_.predict_proba(X_loop_padded)[:, 1]
            loop_auc = roc_auc_score(y_loop, loop_pred)
        else:
            loop_auc = float('nan')
        
        result = {
            "experiment_type": "mixed_cv",
            "train_loops": ",".join(map(str, train_loops)),
            "test_loops": str(test_loop),
            "n_train": int(len(y)),
            "n_test": int(len(y_loop)),
            "train_auc": mean_train_auc,
            "test_auc": float(loop_auc),
            "train_auc_std": std_train_auc,
            "test_auc_std": float('nan'),  # Individual loop doesn't have std
            "best_params": json.dumps(best_params),
        }
        results.append(result)
    
    return results


def run_directional_chain(loop_map: Dict[int, pd.DataFrame], target_col: str,
                          n_iter: int, random_state: int):
    loops = sorted(loop_map.keys())
    if len(loops) < 3:
        logger.info("Not enough loops for directional chain (need ≥3). Skipping.")
        return []

    results = []
    # Start with loops 5,6 for training (as requested)
    train_loops = [5, 6]
    
    # Check if we have loops 5 and 6 available
    if 5 not in loop_map or 6 not in loop_map:
        logger.info("Missing loops 5 or 6 for directional chain training. Skipping.")
        return []
    
    # Start testing from loop 7 onwards
    for target_loop in loops:
        if target_loop <= 6:  # Skip loops 5 and 6 as test targets
            continue
        train_frames = [loop_map[l] for l in train_loops]
        train_df = pd.concat(train_frames, axis=0, ignore_index=True)

        # Extra safety: if concatenation introduces all-NaN columns, drop them
        train_df = drop_all_nan_features(train_df, target_col)

        test_df = loop_map[target_loop]
        
        # Final safety check for all-NaN features
        train_df = ensure_no_all_nan_features(train_df, target_col, f"chain train {train_loops}")
        test_df = ensure_no_all_nan_features(test_df, target_col, f"chain test {target_loop}")

        tr_feats = set(select_feature_columns(train_df, target_col))
        te_feats = set(select_feature_columns(test_df, target_col))
        common = sorted(list(tr_feats & te_feats))
        if not common:
            logger.warning(f"No overlapping features for chain train {train_loops} -> test {target_loop}. Skipping.")
            train_loops.append(target_loop)
            continue

        X_tr = train_df[common].values
        y_tr = train_df[target_col].values
        X_te = test_df[common].values
        y_te = test_df[target_col].values

        if not has_two_classes(y_tr) or min_class_count(y_tr) < 2:
            logger.warning(f"Chain {train_loops}->{target_loop}: insufficient training samples per class. Skipping.")
            train_loops.append(target_loop)
            continue

        # Check if we have loop order information for stratification in training data
        train_loop_order_labels = None
        if 'loop_order' in train_df.columns:
            train_loop_order_labels = train_df['loop_order'].values
        elif 'order' in train_df.columns:
            train_loop_order_labels = train_df['order'].values

        model = build_model(random_state)
        search = randomized_search(model, X_tr, y_tr, n_iter, random_state, train_loop_order_labels)

        train_auc, test_auc = evaluate_model_safe(search.best_estimator_, X_tr, y_tr, X_te, y_te, context=f"chain {','.join(map(str,train_loops))}->{target_loop}")
        
        # Compute individual loop AUCs
        individual_aucs = compute_individual_loop_aucs(search.best_estimator_, train_df, test_df, train_loops, target_loop, target_col)
        
        result = {
            "experiment_type": "chained",
            "train_loops": ",".join(map(str, train_loops)),
            "test_loops": str(target_loop),
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "train_auc": train_auc,
            "test_auc": test_auc,
            "best_params": json.dumps(search.best_params_),
        }
        result.update(individual_aucs)
        results.append(result)

        train_loops.append(target_loop)

    return results


# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="Loop predictor with cheap XGBoost search + reporting + logging.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing loop CSV files.")
    parser.add_argument("--target_col", type=str, default="target",
                        help="Column holding the coefficient (fraction/number). Will be binarised to 0/1.")
    parser.add_argument("--n_iter", type=int, default=16, help="RandomizedSearchCV iterations (default: 16).")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--fraction_mode", type=str, choices=["none", "float", "binary"], default="none",
                        help="Parse fraction-like strings in features. 'binary' maps nonzero->1 else 0; NaN preserved.")
    parser.add_argument("--output_dir", type=str, default="./output", 
                        help="Directory for reports and logs (default: ./output). Creates 'reports' and 'logs' subdirectories.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    reports_dir = output_dir / "reports"
    logs_dir = output_dir / "logs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(logs_dir)

    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Target column: {args.target_col}")
    logger.info(f"Search iterations: {args.n_iter}")
    logger.info(f"Fraction mode: {args.fraction_mode}")
    logger.info("Loading datasets...")

    loop_map = load_loop_datasets(data_dir, args.target_col, args.fraction_mode)
    loops = sorted(loop_map.keys())
    logger.info(f"Detected loops: {loops}")

    rows = []

    # 1) Intra-loop runs (exclude loop=5)
    logger.info("Running intra-loop experiments (excluding loop = 5)...")
    for l in loops:
        if l == 5:
            logger.info(f"Skipping intra-loop training for loop = {l}.")
            continue
        try:
            res = run_intra_loop(loop_map[l], l, args.target_col, args.n_iter, args.random_state)
            rows.append(res)
            logger.info(f"Intra loop {l}: test_auc={res['test_auc']:.4f}")
        except Exception as e:
            logger.exception(f"Intra loop {l} failed: {e}")

    # 2) Pair mixed loops (l -> l+1), skipping loop 5
    logger.info("Running pair mixed loop experiments (l -> l+1), skipping loop 5...")
    for i in range(len(loops) - 1):
        l_train, l_test = loops[i], loops[i + 1]
        
        # Skip experiments involving loop 5 - it has no zeros in COEFFICIENTS (binary classification issue)
        if l_train == 5 or l_test == 5:
            logger.info(f"Skipping pair {l_train}->{l_test} (loop 5 has no zeros in COEFFICIENTS)")
            continue
            
        try:
            res = run_pair_mixed(loop_map[l_train], loop_map[l_test], l_train, l_test,
                                 args.target_col, args.n_iter, args.random_state)
            rows.append(res)
            logger.info(f"Pair {l_train}->{l_test}: test_auc={res['test_auc']:.4f}")
        except Exception as e:
            logger.exception(f"Pair {l_train}->{l_test} failed: {e}")

    # 2b) Additional pair mixed loops (all non-sequential combinations), skipping loop 5
    logger.info("Running additional pair mixed loop experiments (skipping loop 5)...")
    additional_pairs = []
    for i in range(len(loops)):
        for j in range(i + 2, len(loops)):  # Skip sequential pairs (already covered)
            # Skip any pairs involving loop 5 (no zeros in COEFFICIENTS)
            if loops[i] != 5 and loops[j] != 5:
                additional_pairs.append((loops[i], loops[j]))
    
    for l_train, l_test in additional_pairs:
        if l_train in loop_map and l_test in loop_map:
            try:
                res = run_pair_mixed(loop_map[l_train], loop_map[l_test], l_train, l_test,
                                     args.target_col, args.n_iter, args.random_state)
                rows.append(res)
                logger.info(f"Pair {l_train}->{l_test}: test_auc={res['test_auc']:.4f}")
            except Exception as e:
                logger.exception(f"Pair {l_train}->{l_test} failed: {e}")
        else:
            logger.warning(f"Skipping pair {l_train}->{l_test} (missing loop data)")

    # 2c) Mixed loop CV experiments (train on mix, test on same mix via CV) - only neighboring loops, skipping loop 5
    logger.info("Running mixed loop CV experiments (neighboring loops only, skipping loop 5)...")
    mixed_pairs = []
    for i in range(len(loops) - 1):
        # Skip any pairs involving loop 5 (no zeros in COEFFICIENTS)
        if loops[i] != 5 and loops[i + 1] != 5:
            mixed_pairs.append((loops[i], loops[i + 1]))
    
    for l1, l2 in mixed_pairs:
        if l1 in loop_map and l2 in loop_map:
            try:
                res_list = run_mixed_loop_cv([l1, l2], loop_map, args.target_col, args.n_iter, args.random_state)
                rows.extend(res_list)  # Extend instead of append since we now get a list
                for res in res_list:
                    logger.info(f"Mixed CV {l1},{l2} -> {res['test_loops']}: test_auc={res['test_auc']:.4f}")
            except Exception as e:
                logger.exception(f"Mixed CV {l1},{l2} failed: {e}")
        else:
            logger.warning(f"Skipping mixed CV {l1},{l2} (missing loop data)")

    # 3) Directional chains
    logger.info("Running directional chain experiments...")
    try:
        chain_results = run_directional_chain(loop_map, args.target_col, args.n_iter, args.random_state)
        rows.extend(chain_results)
        for r in chain_results:
            logger.info(f"Chain {r['train_loops']} -> {r['test_loops']}: test_auc={r['test_auc']:.4f}")
    except Exception as e:
        logger.exception(f"Directional chain experiments failed: {e}")

    if not rows:
        logger.error("No results produced. Exiting without report.")
        return

    # Save report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"results_{ts}.csv"
    pd.DataFrame(rows).to_csv(report_path, index=False)
    logger.success(f"Saved report: {report_path.resolve()}")


if __name__ == "__main__":
    main()