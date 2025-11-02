#!/bin/bash
# Usage: ./copy_to_academic.sh <local_path> <remote_path>
# Example: ./copy_to_academic.sh ./myfolder /home/ec2-user/

LOCAL_PATH=/Users/rezadoobary/Documents/MLCORRELATORS/ML-correlator/Tree_classifier_for_graphs/new_stuff/features/merged/new2_merged
REMOTE_PATH=/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/features/new_features

if [ -z "$LOCAL_PATH" ] || [ -z "$REMOTE_PATH" ]; then
  echo "Usage: $0 <local_path> <remote_path>"
  exit 1
fi

echo "Copying $LOCAL_PATH to Academic:$REMOTE_PATH ..."
scp -i /Users/rezadoobary/Downloads/academic.pem -r "$LOCAL_PATH" ec2-user@3.89.223.63:"$REMOTE_PATH"
echo "✅ Transfer complete."