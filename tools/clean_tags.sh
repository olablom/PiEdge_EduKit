#!/usr/bin/env bash
set -euo pipefail
pattern="${1:-v0.1.*}"
echo "Cleaning local tags matching: $pattern"
git tag -l "$pattern" | xargs -r -n1 git tag -d
echo "Fetching fresh tags (forced)…"
git fetch --tags --force
echo "Done."
