#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "$0")" && pwd)
for task in task1 task2 task3; do
  (
    cd "$root_dir/$task"
    tectonic -Z search-path="$root_dir/common" \
      --keep-logs --keep-intermediates main.tex
  )
done
