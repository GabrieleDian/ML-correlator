#!/bin/bash
# Usage: ./copy_to_academic.sh <local_path> <remote_path>
# Example: ./copy_to_academic.sh ./myfolder /home/ec2-user/

LOCAL_PATH=/Users/rezadoobary/Documents/MLCORRELATORS/ML-correlator/Tree_classifier_for_graphs/new_stuff/features/merged/dataset
REMOTE_PATH=/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/features/merged/dataset

if [ -z "$LOCAL_PATH" ] || [ -z "$REMOTE_PATH" ]; then
  echo "Usage: $0 <local_path> <remote_path>"
  exit 1
fi

echo "Copying $LOCAL_PATH to Academic:$REMOTE_PATH ..."
scp -i /Users/rezadoobary/Downloads/Academic2.pem -r "$LOCAL_PATH" ec2-user@54.234.21.255:"$REMOTE_PATH"
echo "✅ Transfer complete."