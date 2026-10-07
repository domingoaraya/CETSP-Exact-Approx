#!/usr/bin/env bash
# Merges the per-triple CSVs into one file per n (Results/n<n>/experiment_results_n<n>.csv)
# and one global file (Results/experiment_results_all.csv), with a single header.
cd "$(dirname "$0")"
all="Results/experiment_results_all.csv"; : > "$all"; hdr_done=0
for d in Results/n*/; do
  [ -d "$d" ] || continue
  n=$(basename "$d"); out="${d%/}/experiment_results_${n}.csv"
  files=$(find "$d" -mindepth 2 -name '*.csv' | sort)
  [ -z "$files" ] && continue
  first=$(echo "$files" | head -n 1)
  head -n 1 "$first" > "$out"
  [ "$hdr_done" -eq 0 ] && { head -n 1 "$first" > "$all"; hdr_done=1; }
  for f in $files; do tail -n +2 "$f" >> "$out"; tail -n +2 "$f" >> "$all"; done
  echo "$(( $(wc -l < "$out") - 1 )) rows -> $out"
done
echo "$(( $(wc -l < "$all") - 1 )) rows -> $all"
