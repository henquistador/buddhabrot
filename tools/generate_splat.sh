#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
build_dir="$repo_dir/outputs/buddhabrot"
mkdir -p "$build_dir" "$repo_dir/public"

clang++ -O3 -std=c++20 -pthread \
  "$repo_dir/tools/buddhabrot_splat.cpp" \
  -o "$build_dir/buddhabrot_splat"

"$build_dir/buddhabrot_splat" \
  --samples "${BUDDHABROT_SAMPLES:-10000000}" \
  --iterations "${BUDDHABROT_ITERATIONS:-1048576}" \
  --resolution "${BUDDHABROT_RESOLUTION:-800}" \
  --depth "${BUDDHABROT_DEPTH:-48}" \
  --max-splats "${BUDDHABROT_MAX_SPLATS:-600000}" \
  --output "$build_dir/splat.ply" \
  --stats "$repo_dir/public/buddhabrot.json"

npx --yes @playcanvas/splat-transform \
  "$build_dir/splat.ply" \
  --filter-nan \
  "$repo_dir/public/buddhabrot.spz" \
  --spz-version 3 \
  -w

du -h "$build_dir/splat.ply" "$repo_dir/public/buddhabrot.spz"
