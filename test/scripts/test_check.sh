#!/usr/bin/env bash
set -euo pipefail

# Navigate to project root (assumes this script in test/scripts)
cd "$(dirname "$0")/../.."

IMAGE_PATH=uploaded_image.jpg
if [ ! -f "$IMAGE_PATH" ]; then
  echo "ERROR: image file not found at $IMAGE_PATH"
  exit 1
fi
IMG_BASE64=$(base64 -w 0 "$IMAGE_PATH")

echo "### Testing /check with $IMAGE_PATH ###"
curl -i -X POST \
     -H "Content-Type: application/json" \
     -d "{\"userid\":\"franklin@verisnap.com\",\"image\":\"$IMG_BASE64\"}" \
     http://localhost:9000/check
echo