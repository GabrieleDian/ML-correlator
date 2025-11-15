#!/usr/bin/env bash
set -euo pipefail

outdir="/Users/rezadoobary/Documents/MLCORRELATORS/ML-correlator/Tree_classifier_for_graphs/new_stuff/features/fgraphs2"
mkdir -p "$outdir"

for i in 5; do
  echo ">> Processing i=$i"
  conda run -n torch-env python "/Users/rezadoobary/Documents/MLCORRELATORS/ML-correlator/Tree_classifier_for_graphs/new_stuff/fgraph_features_cli3.py" \
    --input "/Users/rezadoobary/Downloads/graph_data_${i}.csv" \
    --groups all \
    --output "$outdir/${i}_fgraph_feats.csv" \
    --out-format "csv" \
    --workers 8 \
    --intra-workers 4

done

echo "✓ All runs complete."