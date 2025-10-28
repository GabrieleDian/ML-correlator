import pandas as pd
import numpy as np
import ast
from pathlib import Path
from tqdm import tqdm
import argparse

def convert_csv_to_npz(file_ext, base_dir="../Graph_Edge_Data"):
    """
    Convert den_graph_data_{file_ext}.csv → den_graph_data_{file_ext}.npz
    Automatically detects whether edges are in 'EDGES' or 'DEN_EDGES' column.
    Saves edges + coefficients arrays.
    """
    base_dir = Path(base_dir)
    csv_path = base_dir / f"den_graph_data_{file_ext}.csv"
    npz_path = base_dir / f"den_graph_data_{file_ext}.npz"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"📂 Converting {csv_path.name} → {npz_path.name}")

    df = pd.read_csv(csv_path)

    # --- Auto-detect the correct edge column ---
    edge_col = None
    for candidate in ["DEN_EDGES", "EDGES", "NUM_EDGES"]:
        if candidate in df.columns:
            edge_col = candidate
            break
    if edge_col is None:
        raise ValueError(f"No edge column found in {csv_path} (expected 'EDGES' or 'DEN_EDGES').")

    if "COEFFICIENTS" not in df.columns:
        raise ValueError(f"Missing 'COEFFICIENTS' column in {csv_path}")

    # --- Parse data ---
    edges_list = [ast.literal_eval(e) for e in tqdm(df[edge_col], desc=f"Parsing {edge_col}")]
    coeffs = df["COEFFICIENTS"].to_numpy()

    # --- Save NPZ ---
    np.savez_compressed(npz_path,
                        edges=np.array(edges_list, dtype=object),
                        coefficients=coeffs)
    print(f"✅ Saved {npz_path.name} with {len(edges_list)} graphs and {len(coeffs)} coefficients.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loops", nargs="+", required=True,
                        help="Loop orders to convert, e.g. --loops 6 7 8")
    parser.add_argument("--data_dir", type=str, default="../Graph_Edge_Data",
                        help="Base directory containing den_graph_data_*.csv")
    args = parser.parse_args()

    for loop in args.loops:
        convert_csv_to_npz(loop, args.data_dir)
