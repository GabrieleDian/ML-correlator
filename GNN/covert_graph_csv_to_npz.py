#!/usr/bin/env python3
"""
Convert slow CSV graph files into fast .npz binaries for later use.
Supports rational coefficients like '-1/2', '3/4', etc.
"""

import pandas as pd
import numpy as np
import ast
from pathlib import Path
from tqdm import tqdm


def convert_csv_to_npz(loop_order, base_dir="../Graph_Edge_Data"):
    base_dir = Path(base_dir)
    csv_path = base_dir / f"den_graph_data_{loop_order}.csv"
    npz_path = base_dir / f"den_graph_edges_{loop_order}.npz"

    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        return
    if npz_path.exists():
        print(f"⚠️  {npz_path.name} already exists, skipping.")
        return

    print(f"📂 Converting {csv_path.name} → {npz_path.name}")

    df = pd.read_csv(csv_path)

    denom_edges = [ast.literal_eval(e) for e in tqdm(df["DEN_EDGES"], desc="Parsing DEN_EDGES")]
    numer_edges = [ast.literal_eval(e) for e in tqdm(df["NUM_EDGES"], desc="Parsing NUM_EDGES")]

    # Convert coefficients robustly (handles fractions and floats)
    coeffs = np.array([(x) for x in df["COEFFICIENTS"]], dtype=float)

    np.savez_compressed(
        npz_path,
        denom_edges=np.array(denom_edges, dtype=object),
        numer_edges=np.array(numer_edges, dtype=object),
        coefficients=coeffs,
    )

    size = npz_path.stat().st_size / 1e6
    print(f"✅ Saved binary version: {npz_path} ({size:.2f} MB)")

def convert_all_csv_to_npz(base_dir="../Graph_Edge_Data"):
    base_dir = Path(base_dir)
    csv_files = sorted(base_dir.glob("den_graph_data_*.csv"))
    for f in csv_files:
        ext = f.stem.split("_")[-1]
        convert_csv_to_npz(ext, base_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loops", nargs="+", required=True, help="Loop orders to convert (e.g. 7 8 9 10 11 12)")
    parser.add_argument("--data-dir", default="../Graph_Edge_Data", help="Base directory with CSVs")
    args = parser.parse_args()

    for loop in args.loops:
        convert_csv_to_npz(loop, args.data_dir)
    