#!/usr/bin/env bash
set -euo pipefail

for i in 8 9; do
  echo ">> Processing i=$i"
  python "/Users/rezadoobary/Documents/MLCorrelator2/ML-correlator/Tree_classifier_for_graphs/new_stuff/fgraph_features_cli2.py" \
    --input "/Users/rezadoobary/Documents/MLCorrelator2/ML-correlator/Graph_Edge_Data/graph_data_${i}.csv" \
    --output "/Users/rezadoobary/Documents/MLCorrelator2/ML-correlator/Tree_classifier_for_graphs/new_stuff/features/fgraphs/${i}loops.csv" \
    --groups connectivity,centrality,spectral_laplacian,motifs34,motifs5,induced4,induced5
done

echo "✓ All runs complete."
