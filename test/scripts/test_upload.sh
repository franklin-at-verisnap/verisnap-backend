#!/usr/bin/env bash
set -euo pipefail

# Navigate to project root (assumes this script in test/scripts)
cd "$(dirname "$0")/../.."

JSON_FILE=test/test.json
if [ ! -f "$JSON_FILE" ]; then
  echo "ERROR: JSON payload not found at $JSON_FILE"
  exit 1
fi

echo "### Testing /upload with payload $JSON_FILE ###"
curl -i -X POST \
     -H "Content-Type: application/json" \
     -d @"$JSON_FILE" \
     http://localhost:9000/upload
echo