#!/usr/bin/env bash

# Usage:
#   ./upload.sh /local/folder s3://my-bucket/path

#LOCAL_DIR="$1"
#S3_URI="$2"

LOCAL_DIR="/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/features/merged/dataset/dataset/dataset"
S3_URI="s3://physicsml/fgraphs_gnn/den_graphs/features/"

if [ -z "$LOCAL_DIR" ] || [ -z "$S3_URI" ]; then
  echo "Usage: $0 <local-folder> <s3-uri>"
  exit 1
fi

aws s3 sync "$LOCAL_DIR" "$S3_URI"
