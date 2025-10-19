#!/usr/bin/env bash
set -euo pipefail

for i in 10; do
  echo ">> Processing i=$i"
  python "/Users/rezadoobary/Documents/MLCorrelator2/ML-correlator/Tree_classifier_for_graphs/new_stuff/motif_features_cli.py" \
    --input "/Users/rezadoobary/Documents/MLCorrelator2/ML-correlator/Graph_Edge_Data/den_graph_data_${i}.csv" \
    --output "/Users/rezadoobary/Documents/MLCorrelator2/ML-correlator/Tree_classifier_for_graphs/new_stuff/features/fgraphs/${i}loops_test.csv" \
    --groups motifs34,motifs5,induced4,induced5 \
    --workers 1
done

echo "✓ All runs complete."
