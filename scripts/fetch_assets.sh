#!/usr/bin/env bash
set -euo pipefail
dest="$(dirname "$0")/../assets/mujoco_menagerie"
if [ ! -d "$dest" ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git "$dest"
fi
git -C "$dest" sparse-checkout set universal_robots_ur5e
