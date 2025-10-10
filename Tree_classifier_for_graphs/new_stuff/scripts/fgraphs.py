#!/usr/bin/env python3
"""
loop_predictor.py

Features:
- Target is binarised strictly: 0 if coefficient == 0, else 1 (supports fraction-like strings).
- --fraction_mode {none,float,binary} for feature columns:
    none   : leave as-is (coerce to numeric later with errors='coerce')
    float  : parse "a/b" or ints to float; others -> NaN
    binary : parse then map nonzero->1, zero->0; NaN stays NaN (imputer handles)
- Replaces np.inf/-np.inf and string forms ('inf','infinity','nan', any case) with np.nan.
- Drops all-NaN feature columns **with logging** to avoid imputer warnings.
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
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
                # keep NaN, imputer will handle
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
    Drop feature columns that are entirely NaN (excluding target/id columns),
    and log the operation.
    """
    drop_cols = set([target_col]) | NON_FEATURE_COLS_CANDIDATES
    nan_features = [c for c in df.columns if c not in drop_cols and df[c].isna().all()]
    if nan_features:
        logger.warning(
            f"Dropping {len(nan_features)} all-NaN feature columns: "
            f"{nan_features[:8]}{'...' if len(nan_features) > 8 else ''}"
        )
        df = df.drop(columns=nan_features)
    else:
        logger.info("No all-NaN feature columns found.")
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

        # Drop all-NaN features to avoid imputer warnings
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
    if not features:
        raise ValueError("No numeric feature columns found after filtering.")
    return features


# -----------------------
# Modeling
# -----------------------

def build_model(random_state: int) -> Pipeline:
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1
    )
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", xgb),
    ])
    return pipe


def cheap_param_distributions() -> dict:
    return {
        "clf__n_estimators": [200, 400, 800],
        "clf__max_depth": [3, 4, 5, 6, 8],
        "clf__learning_rate": [0.01, 0.02, 0.05, 0.1, 0.2],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__colsample_bytree": [0.7, 0.85, 1.0],
        "clf__min_child_weight": [1, 2, 5, 10],
        "clf__reg_alpha": [0.0, 0.001, 0.01, 0.1],
        "clf__reg_lambda": [0.5, 1.0, 2.0],
    }


def randomized_search(pipe: Pipeline, X, y, n_iter: int, random_state: int, loop_order_labels=None):
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
        pipe,
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


def evaluate_model_safe(model: Pipeline, X_train, y_train, X_test, y_test, context: str):
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


# -----------------------
# Experiment Runners
# -----------------------

def run_intra_loop(loop_df: pd.DataFrame, loop: int, target_col: str,
                   n_iter: int, random_state: int):
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
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Get loop order labels for this fold if available
        fold_loop_order_labels = None
        if loop_order_col is not None:
            fold_loop_order_labels = loop_order_col[train_idx]
        
        # Build model and do minimal hyperparameter search
        pipe = build_model(random_state + fold_idx)  # Different seed per fold
        search = randomized_search(pipe, X_train_fold, y_train_fold, n_iter, random_state + fold_idx, fold_loop_order_labels)
        
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
    
    logger.info(f"Intra-loop {loop}: 5-fold CV train_auc = {mean_train_auc:.4f} ± {std_train_auc:.4f}, test_auc = {mean_test_auc:.4f} ± {std_test_auc:.4f}")
    
    return {
        "train_loops": str(loop),
        "test_loops": str(loop),
        "n_train": int(len(y)),
        "n_test": int(len(y)),  # Same as train since it's CV
        "train_auc": mean_train_auc,
        "test_auc": mean_test_auc,
        "train_auc_std": std_train_auc,
        "test_auc_std": std_test_auc,
        "best_params": json.dumps({}),  # No single best params for CV
    }


def run_pair_mixed(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   l_train: int, l_test: int, target_col: str,
                   n_iter: int, random_state: int):
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

    pipe = build_model(random_state)
    search = randomized_search(pipe, X_tr, y_tr, n_iter, random_state, train_loop_order_labels)

    train_auc, test_auc = evaluate_model_safe(search.best_estimator_, X_tr, y_tr, X_te, y_te, context=f"pair {l_train}->{l_test}")
    return {
        "train_loops": str(l_train),
        "test_loops": str(l_test),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "train_auc": train_auc,
        "test_auc": test_auc,
        "best_params": json.dumps(search.best_params_),
    }


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
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Get loop order labels for this fold if available
        fold_loop_order_labels = None
        if loop_order_col is not None:
            fold_loop_order_labels = loop_order_col[train_idx]
        
        # Build model and do minimal hyperparameter search
        pipe = build_model(random_state + fold_idx)  # Different seed per fold
        search = randomized_search(pipe, X_train_fold, y_train_fold, n_iter, random_state + fold_idx, fold_loop_order_labels)
        
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
    
    logger.info(f"Mixed CV {l1},{l2}: 5-fold CV train_auc = {mean_train_auc:.4f} ± {std_train_auc:.4f}, test_auc = {mean_test_auc:.4f} ± {std_test_auc:.4f}")
    
    return {
        "train_loops": ",".join(map(str, train_loops)),
        "test_loops": ",".join(map(str, train_loops)),
        "n_train": int(len(y)),
        "n_test": int(len(y)),  # Same as train since it's CV
        "train_auc": mean_train_auc,
        "test_auc": mean_test_auc,
        "train_auc_std": std_train_auc,
        "test_auc_std": std_test_auc,
        "best_params": json.dumps({}),  # No single best params for CV
    }


def run_directional_chain(loop_map: Dict[int, pd.DataFrame], target_col: str,
                          n_iter: int, random_state: int):
    loops = sorted(loop_map.keys())
    if len(loops) < 3:
        logger.info("Not enough loops for directional chain (need ≥3). Skipping.")
        return []

    results = []
    train_loops = [loops[0], loops[1]]
    for target_loop in loops[2:]:
        train_frames = [loop_map[l] for l in train_loops]
        train_df = pd.concat(train_frames, axis=0, ignore_index=True)

        # Extra safety: if concatenation introduces all-NaN columns, drop them
        train_df = drop_all_nan_features(train_df, target_col)

        test_df = loop_map[target_loop]

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

        pipe = build_model(random_state)
        search = randomized_search(pipe, X_tr, y_tr, n_iter, random_state, train_loop_order_labels)

        train_auc, test_auc = evaluate_model_safe(search.best_estimator_, X_tr, y_tr, X_te, y_te, context=f"chain {','.join(map(str,train_loops))}->{target_loop}")
        results.append({
            "train_loops": ",".join(map(str, train_loops)),
            "test_loops": str(target_loop),
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "train_auc": train_auc,
            "test_auc": test_auc,
            "best_params": json.dumps(search.best_params_),
        })

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
        
        # Skip experiments involving loop 5 due to insufficient samples
        if l_train == 5 or l_test == 5:
            logger.info(f"Skipping pair {l_train}->{l_test} (involves loop 5 with insufficient samples)")
            continue
            
        try:
            res = run_pair_mixed(loop_map[l_train], loop_map[l_test], l_train, l_test,
                                 args.target_col, args.n_iter, args.random_state)
            rows.append(res)
            logger.info(f"Pair {l_train}->{l_test}: test_auc={res['test_auc']:.4f}")
        except Exception as e:
            logger.exception(f"Pair {l_train}->{l_test} failed: {e}")

    # 2b) Additional pair mixed loops (6->8, 6->9, 7->9)
    logger.info("Running additional pair mixed loop experiments...")
    additional_pairs = [(6, 8), (6, 9), (7, 9)]
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

    # 2c) Mixed loop CV experiments (train on mix, test on same mix via CV)
    logger.info("Running mixed loop CV experiments...")
    mixed_pairs = [(6, 7), (7, 8), (8, 9), (6, 8), (6, 9), (7, 9)]
    for l1, l2 in mixed_pairs:
        if l1 in loop_map and l2 in loop_map:
            try:
                res = run_mixed_loop_cv([l1, l2], loop_map, args.target_col, args.n_iter, args.random_state)
                rows.append(res)
                logger.info(f"Mixed CV {l1},{l2}: train_auc={res['train_auc']:.4f}")
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
