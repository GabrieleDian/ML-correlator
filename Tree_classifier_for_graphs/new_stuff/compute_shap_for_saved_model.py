#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

"""
Compute SHAP values for a saved model from modelling_function_12_split.py
This script replicates the SHAP computation from modelling_function_instance_faster.py
but uses a pre-trained model instead of training a new one.

To run this script:
  Option 1 (Recommended): Use base Python where shap is installed:
    ~/miniconda3/bin/python compute_shap_for_saved_model.py
  
  Option 2: Use the conda environment (if shap is installed there):
    ~/miniconda3/envs/ml-correlator/bin/python compute_shap_for_saved_model.py
    OR
    conda activate ml-correlator
    python compute_shap_for_saved_model.py

Note: Python 3.14 has compatibility issues with shap. If you encounter errors,
use the base Python interpreter (Option 1).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import logging
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import resample

try:
    import shap
except ImportError:
    # Try to use shap from base environment if available
    import sys
    base_python_path = str(Path.home() / "miniconda3" / "lib" / "python3.13" / "site-packages")
    if Path(base_python_path).exists() and base_python_path not in sys.path:
        sys.path.insert(0, base_python_path)
    try:
        import shap
        print(f"Using shap from base environment")
    except ImportError:
        print("ERROR: shap is not available in this environment.")
        print("Python 3.14 is not compatible with shap's dependencies (numba).")
        print("Please run this script using the base Python interpreter:")
        print("  ~/miniconda3/bin/python compute_shap_for_saved_model.py")
        sys.exit(1)

warnings.filterwarnings("ignore")


def setup_logging(log_file: Path):
    """Set up logging to both console and file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []
    
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    logging.info("=" * 80)
    logging.info(f"SHAP computation started - Log file: {log_file}")
    logging.info("=" * 80)


def load_and_align_parquet_dataset_lean(
    dataset_dir: str | Path,
    loops: list[int] = list(range(5, 12)),
    target_col: str = "COEFFICIENTS",
    loop_col: str = "loops",
) -> pd.DataFrame:
    """
    Two-pass loader: discover common features, then read only needed columns.
    Copied from modelling_function_12_split.py for consistency.
    """
    import pyarrow.parquet as pq
    import re
    
    dataset_dir = Path(dataset_dir)
    files = sorted(dataset_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {dataset_dir}")
    
    def _infer_loop_from_name(p: Path) -> int | None:
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else None
    
    # PASS 1: find common features
    feature_sets = []
    present_files = []
    
    for p in files:
        try:
            schema_names = set(pq.ParquetFile(p).schema.names)
        except Exception as e:
            raise RuntimeError(f"Failed to read schema for {p}: {e}")
        
        if target_col not in schema_names:
            raise ValueError(f"Target column '{target_col}' missing in {p.name}")
        
        minimal_cols = [c for c in (loop_col, target_col) if c in schema_names]
        df_head = pd.read_parquet(p, columns=minimal_cols, engine="pyarrow")
        
        if loop_col not in df_head.columns:
            lg = _infer_loop_from_name(p)
            if lg is None:
                raise ValueError(f"Cannot infer loop id from filename: {p.name}")
            df_head[loop_col] = lg
        
        df_head = df_head[df_head[loop_col].isin(loops)]
        if df_head.empty:
            del df_head
            continue
        
        feats = list(schema_names - {target_col, loop_col})
        feature_sets.append(set(feats))
        present_files.append(p)
        del df_head
    
    if not present_files:
        raise ValueError(f"No rows found for loops {loops} in {dataset_dir}")
    
    common_feats = sorted(list(set.intersection(*feature_sets)))
    if not common_feats:
        raise ValueError("No common feature columns across parquet files.")
    
    # PASS 2: read only needed columns, downcast dtypes
    wanted_cols = common_feats + [loop_col, target_col]
    chunks = []
    
    for p in present_files:
        try:
            schema_names = set(pq.ParquetFile(p).schema.names)
        except Exception as e:
            raise RuntimeError(f"Failed to read schema for {p}: {e}")
        
        cols_to_read = [c for c in wanted_cols if c in schema_names]
        df = pd.read_parquet(p, columns=cols_to_read, engine="pyarrow")
        
        if loop_col not in df.columns:
            df[loop_col] = _infer_loop_from_name(p)
        
        df = df[df[loop_col].isin(loops)]
        if df.empty:
            del df
            continue
        
        for c in common_feats:
            if c in df.columns and df[c].dtype != np.float32:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        
        if target_col in df.columns:
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(np.int8)
        df[loop_col] = pd.to_numeric(df[loop_col], errors="coerce").astype(np.int16)
        
        chunks.append(df)
    
    if not chunks:
        raise ValueError("After filtering and downcasting, no data remained to merge.")
    
    merged = pd.concat(chunks, ignore_index=True)
    return merged


def calculate_shap_values_for_saved_model(
    model_path: Path,
    feature_names: list[str],
    X: np.ndarray,
    shap_sample_size: int = 500,
    random_state: int = 42
):
    """
    Calculate SHAP values for a pre-trained model.
    Adapted from calculate_global_shap_values in modelling_function_instance_faster.py
    """
    logging.info(f"Loading model from: {model_path}")
    
    # Load the saved model
    try:
        import joblib
        model = joblib.load(model_path)
    except Exception as e:
        logging.warning(f"joblib load failed: {e}, trying pickle...")
        import pickle
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    
    logging.info(f"Model loaded successfully. Type: {type(model)}")
    
    # Try to reload the booster from JSON if available (more compatible)
    model_dir = model_path.parent
    booster_json_path = model_dir / "final_model_booster.json"
    if booster_json_path.exists():
        logging.info("Found booster JSON file, reloading booster for better compatibility...")
        try:
            import xgboost as xgb
            booster = xgb.Booster()
            booster.load_model(str(booster_json_path))
            # Create a temporary model wrapper or use booster directly
            # We'll use the booster for SHAP instead of the sklearn wrapper
            model_booster = booster
            use_booster = True
        except Exception as e:
            logging.warning(f"Failed to load booster from JSON: {e}, using sklearn model...")
            model_booster = None
            use_booster = False
    else:
        model_booster = None
        use_booster = False
    
    # Ensure X is float32
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    
    logging.info(f"Data shape: {X.shape}")
    logging.info(f"Number of features: {len(feature_names)}")
    
    # Create SHAP explainer
    # For XGBoost models, try different initialization methods based on version compatibility
    logging.info("Creating SHAP TreeExplainer...")
    
    # Prefer using booster from JSON if available (avoids version compatibility issues)
    if use_booster and model_booster is not None:
        try:
            explainer = shap.TreeExplainer(model_booster, model_output='probability')
            logging.info("Using booster from JSON file with model_output='probability'")
        except Exception as e:
            logging.warning(f"TreeExplainer with JSON booster failed: {e}, trying sklearn model...")
            use_booster = False
    
    if not use_booster:
        try:
            # Try using booster from sklearn model
            if hasattr(model, 'get_booster'):
                booster = model.get_booster()
                # Try without model_output first (sometimes causes issues)
                explainer = shap.TreeExplainer(booster)
                logging.info("Using XGBoost booster from sklearn model")
            else:
                raise AttributeError("Model doesn't have get_booster method")
        except Exception as e1:
            logging.warning(f"TreeExplainer with booster failed: {e1}, trying direct model...")
            try:
                # Direct model initialization
                explainer = shap.TreeExplainer(model)
                logging.info("Using sklearn model directly")
            except Exception as e2:
                logging.error(f"All TreeExplainer initialization methods failed. Last error: {e2}")
                raise RuntimeError(f"Failed to create SHAP TreeExplainer: {e2}") from e2
    
    # Sample data if needed
    if shap_sample_size is not None and X.shape[0] > shap_sample_size:
        logging.info(f"Sampling {shap_sample_size} rows from {X.shape[0]} total rows...")
        print(f"Sampling {shap_sample_size} rows from {X.shape[0]} total rows...")
        np.random.seed(random_state)
        sample_indices = np.random.choice(X.shape[0], size=shap_sample_size, replace=False)
        X_sample = X[sample_indices]
        logging.info(f"Computing SHAP values on {X_sample.shape[0]} samples (this may take several minutes)...")
        print(f"Computing SHAP values on {X_sample.shape[0]} samples (this may take several minutes)...")
        print("Progress: Starting SHAP computation...")
        
        # For large samples, compute in batches with progress updates
        if X_sample.shape[0] > 1000:
            batch_size = 1000
            n_batches = (X_sample.shape[0] + batch_size - 1) // batch_size
            logging.info(f"Computing SHAP in {n_batches} batches of ~{batch_size} samples each...")
            print(f"Computing SHAP in {n_batches} batches of ~{batch_size} samples each...")
            
            shap_batches = []
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, X_sample.shape[0])
                batch = X_sample[start_idx:end_idx]
                
                logging.info(f"Processing batch {i+1}/{n_batches} (samples {start_idx+1}-{end_idx})...")
                print(f"Progress: Batch {i+1}/{n_batches} (samples {start_idx+1}-{end_idx})...", flush=True)
                
                batch_shap = explainer.shap_values(batch)
                shap_batches.append(batch_shap)
                
                # Progress update
                if (i + 1) % max(1, n_batches // 10) == 0 or i == n_batches - 1:
                    progress_pct = ((i + 1) / n_batches) * 100
                    logging.info(f"Progress: {progress_pct:.1f}% complete ({i+1}/{n_batches} batches)")
                    print(f"Progress: {progress_pct:.1f}% complete ({i+1}/{n_batches} batches)", flush=True)
            
            # Concatenate batches
            shap_values = np.concatenate(shap_batches, axis=0)
            logging.info("All batches processed, concatenating results...")
            print("All batches processed, concatenating results...", flush=True)
        else:
            shap_values = explainer.shap_values(X_sample)
            print("SHAP computation completed!", flush=True)
        
        X_used = X_sample
    else:
        logging.info(f"Computing SHAP values on full dataset ({X.shape[0]} samples)...")
        print(f"Computing SHAP values on full dataset ({X.shape[0]} samples)...")
        print("Progress: Starting SHAP computation (this may take a while)...", flush=True)
        shap_values = explainer.shap_values(X)
        print("SHAP computation completed!", flush=True)
        X_used = X
    
    logging.info(f"SHAP values computed. Shape: {np.array(shap_values).shape}")
    print(f"SHAP values computed. Shape: {np.array(shap_values).shape}", flush=True)
    
    return shap_values, explainer, model, X_used


def get_global_feature_importance(shap_values, feature_names):
    """Get global feature importance from SHAP values."""
    global_importance = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': global_importance
    }).sort_values('importance', ascending=False)
    return importance_df


def get_global_feature_direction(shap_values, feature_names):
    """Get mean signed SHAP values (feature direction)."""
    mean_shap = np.mean(shap_values, axis=0)
    df = pd.DataFrame({
        'feature': feature_names,
        'mean_signed_shap': mean_shap
    }).sort_values('mean_signed_shap', ascending=False)
    return df


def compute_shap_for_saved_model(
    results_dir: Path,
    dataset_dir: str | Path,
    loops: list[int] = list(range(5, 12)),
    shap_sample_size: int = 10000,
    random_state: int = 42
):
    """
    Main function to compute SHAP values for a saved model.
    
    Parameters:
    -----------
    results_dir : Path
        Directory containing the saved model artifacts (from modelling_function_12_split.py)
    dataset_dir : str | Path
        Directory containing the parquet dataset files
    loops : list[int]
        Loop values to include in the data
    shap_sample_size : int
        Number of samples to use for SHAP computation (None for all)
    random_state : int
        Random seed for sampling
    """
    results_dir = Path(results_dir)
    
    # Setup logging
    log_file = results_dir / "shap_computation.log"
    setup_logging(log_file)
    
    logging.info(f"Results directory: {results_dir}")
    logging.info(f"Dataset directory: {dataset_dir}")
    logging.info(f"Loops: {loops}")
    logging.info(f"SHAP sample size: {shap_sample_size}")
    
    # Load model artifacts
    model_path = results_dir / "final_model_sklearn.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    feature_names_path = results_dir / "feature_names.json"
    if not feature_names_path.exists():
        raise FileNotFoundError(f"Feature names file not found: {feature_names_path}")
    
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    
    logging.info(f"Loaded {len(feature_names)} feature names")
    
    # Load data
    logging.info(f"Loading data from: {dataset_dir}")
    merged = load_and_align_parquet_dataset_lean(
        dataset_dir=dataset_dir,
        loops=loops,
        target_col="COEFFICIENTS",
        loop_col="loops"
    )
    logging.info(f"Merged data shape: {merged.shape}")
    
    # Prepare X (all loops combined, matching the training data)
    X = merged[feature_names].to_numpy(dtype=np.float32, copy=False)
    logging.info(f"Feature matrix shape: {X.shape}")
    
    # Compute SHAP values
    logging.info("\n" + "="*60)
    logging.info("Computing SHAP values")
    logging.info("="*60)
    
    shap_values, explainer, model, X_used = calculate_shap_values_for_saved_model(
        model_path=model_path,
        feature_names=feature_names,
        X=X,
        shap_sample_size=shap_sample_size,
        random_state=random_state
    )
    
    # Get feature importance and direction
    logging.info("Computing feature importance...")
    print("Computing feature importance...", flush=True)
    importance_df = get_global_feature_importance(shap_values, feature_names)
    direction_df = get_global_feature_direction(shap_values, feature_names)
    
    # Merge importance and direction
    shap_full = importance_df.merge(direction_df, on="feature")
    shap_full_path = results_dir / "shap_full.csv"
    shap_full.to_csv(shap_full_path, index=False)
    logging.info(f"Saved SHAP full results to: {shap_full_path}")
    
    # Create SHAP summary plot
    logging.info("Creating SHAP summary plot...")
    print("Creating SHAP summary plot...", flush=True)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_used, feature_names=feature_names, max_display=20, show=False)
    plt.title("SHAP Summary (Merged Model)")
    shap_png_path = results_dir / "shap_summary.png"
    plt.savefig(shap_png_path, bbox_inches='tight', dpi=200)
    plt.close()
    logging.info(f"Saved SHAP summary plot to: {shap_png_path}")
    print(f"Saved SHAP summary plot to: {shap_png_path}", flush=True)
    
    # Save SHAP sample CSV
    logging.info("Saving SHAP sample CSV...")
    print("Saving SHAP sample CSV...", flush=True)
    if shap_sample_size is not None and len(shap_values) <= shap_sample_size:
        shap_sample = pd.DataFrame(shap_values, columns=feature_names)
        shap_sample["sample_index"] = np.arange(len(shap_sample))
    else:
        n_samples = min(shap_sample_size, len(shap_values)) if shap_sample_size else len(shap_values)
        shap_sample = resample(
            pd.DataFrame(shap_values, columns=feature_names),
            n_samples=n_samples,
            random_state=random_state
        )
        shap_sample["sample_index"] = np.arange(len(shap_sample))
    
    shap_sample_path = results_dir / "shap_sample.csv"
    shap_sample.to_csv(shap_sample_path, index=False)
    logging.info(f"Saved SHAP sample CSV to: {shap_sample_path}")
    
    # Log top features
    logging.info("\n" + "="*60)
    logging.info("Top 20 Features by SHAP Importance")
    logging.info("="*60)
    logging.info(shap_full.head(20).to_string(index=False))
    
    logging.info("\n" + "="*60)
    logging.info("SHAP computation completed successfully!")
    logging.info(f"All outputs saved to: {results_dir}")
    logging.info("="*60)
    
    return {
        'shap_values': shap_values,
        'explainer': explainer,
        'model': model,
        'X_used': X_used,
        'shap_full_df': shap_full,
        'shap_full_path': str(shap_full_path),
        'shap_summary_png': str(shap_png_path),
        'shap_sample_path': str(shap_sample_path)
    }


if __name__ == "__main__":
    # Configuration
    results_dir = Path(
        "/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/"
        "bayes_results/bayes_results_merged/merged_20251102-215239"
    )
    
    dataset_dir = Path(
        "/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/"
        "features/merged/dataset/dataset"
    )
    
    loops = list(range(5, 12))
    shap_sample_size = 10000
    random_state = 42
    
    print(f"Computing SHAP values for model in: {results_dir}")
    print(f"Using dataset from: {dataset_dir}")
    print(f"Loops: {loops}")
    print(f"SHAP sample size: {shap_sample_size}\n")
    
    results = compute_shap_for_saved_model(
        results_dir=results_dir,
        dataset_dir=dataset_dir,
        loops=loops,
        shap_sample_size=shap_sample_size,
        random_state=random_state
    )
    
    print("\n=== SHAP Computation Complete ===")
    print(f"SHAP full CSV: {results['shap_full_path']}")
    print(f"SHAP summary PNG: {results['shap_summary_png']}")
    print(f"SHAP sample CSV: {results['shap_sample_path']}")

