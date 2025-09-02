#!/usr/bin/env bash
set -euo pipefail

# Navigate to project root (assumes this script in test/scripts)
cd "$(dirname "$0")/../.."

echo "### Testing /ping ###"
curl -i http://localhost:9000/ping
echo