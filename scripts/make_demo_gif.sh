#!/usr/bin/env bash
# Build assets/demo_compressed.gif from output/frame*.jpg (after tracking visualization).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
FPS="${1:-12}"
OUT="${2:-assets/demo_compressed.gif}"
ffmpeg -y -framerate "$FPS" -pattern_type glob -i 'output/frame*.jpg' \
  -vf "scale=1280:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
  -loop 0 "$OUT"
echo "Wrote $OUT"
