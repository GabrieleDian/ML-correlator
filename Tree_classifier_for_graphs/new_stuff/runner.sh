#!/usr/bin/env bash
set -euo pipefail

outdir="/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/features/merged/dataset/dataset/fgraphs/fgraphs/graf/features"
mkdir -p "$outdir"

for i in 11; do
  echo ">> Processing i=$i"
  PYTHONWARNINGS=ignore::UserWarning python -u -W ignore::UserWarning "/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/fgraph_features_cli3_faster.py" \
    --input "/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/features/merged/dataset/dataset/fgraphs/fgraphs/graf/graph_data_${i}.csv" \
    --groups spectral_laplacian,motifs34,motifs5,induced4,induced5,centrality \
    --batch-size 100000 \
    --batch-dir "$outdir/batches" \
    --from-batch 0 \
    --output "$outdir/${i}_fgraph_feats.csv" \
    --out-format "csv" \
    --workers 8 \
    --s3-uri s3://physicsml/fgraphs_gnn/fgraphs/features/11loops/batches \
    --stream-to-s3


done

echo "✓ All runs complete."