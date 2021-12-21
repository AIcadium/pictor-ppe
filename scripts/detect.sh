#!/bin/bash
set -eou pipefail

for filename in extras/*.mp4; do
    python score.py "$filename"
done

aws s3 cp output "s3://${TEMP_BUCKET:-span-staging-temp-data}/workplace-safety" --recursive
