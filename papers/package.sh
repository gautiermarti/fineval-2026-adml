#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "$0")" && pwd)
dist_dir="$root_dir/dist"
mkdir -p "$dist_dir"

for task in task1 task2 task3; do
  archive="$dist_dir/adml_fineval2026_${task}.zip"
  rm -f "$archive"
  (
    cd "$root_dir"
    zip -q -j -X "$archive" \
      "$task/main.tex" \
      "$task/main.pdf" \
      "common/references.bib" \
      "common/acl.sty" \
      "common/acl_natbib.bst" \
      "common/ACL_FORMATTING.md" \
      "README.md"
  )
done

(
  cd "$dist_dir"
  sha256sum adml_fineval2026_task1.zip \
    adml_fineval2026_task2.zip \
    adml_fineval2026_task3.zip > SHA256SUMS
)
