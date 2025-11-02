#!/usr/bin/env bash
set -euo pipefail

outdir="/Users/rezadoobary/Documents/MLCORRELATORS/ML-correlator/Tree_classifier_for_graphs/new_stuff/features/12loops"
mkdir -p "$outdir"

for i in {1..20}; do
  echo ">> Processing i=$i"
  python "/Users/rezadoobary/Documents/MLCORRELATORS/ML-correlator/Tree_classifier_for_graphs/new_stuff/fgraph_features_cli2.py" \
    --input "/Users/rezadoobary/Downloads/den_graph_data_12_${i}.csv" \
    --output "$outdir/${i}part.csv" \
    --groups centrality
done

echo "✓ All runs complete."
