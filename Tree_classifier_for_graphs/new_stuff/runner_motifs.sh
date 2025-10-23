#!/usr/bin/env bash
set -euo pipefail

for i in 5 6 7 8 9 10; do
  echo ">> Processing i=$i"
  python "/Users/rezadoobary/Documents/MLCorrelator2/ML-correlator/Tree_classifier_for_graphs/new_stuff/motif_features_cli.py" \
    --input "/Users/rezadoobary/Documents/MLCorrelator2/ML-correlator/Graph_Edge_Data/den_graph_data_${i}.csv" \
    --output "/Users/rezadoobary/Documents/MLCorrelator2/ML-correlator/Tree_classifier_for_graphs/new_stuff/features/motif_features_den/${i}loops.csv" \
    --groups spectral,motifs34,motifs5,induced4,induced5 \
    --workers 8 
done

echo "✓ All runs complete."
