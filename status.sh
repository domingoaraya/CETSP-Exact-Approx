#!/usr/bin/env bash
# Progress of the launch.sh experiment: completed jobs (triples) per n and per solver.
cd "$(dirname "$0")"
INSTANCES=5
for d in Results/n*/; do
  [ -d "$d" ] || continue
  n=$(basename "$d"); done_j=0; total_j=0; rows=0
  printf "== %s ==\n" "$n"
  for ld in "$d"*/; do
    [ -d "$ld" ] || continue
    label=$(basename "$ld"); c=0; t=0
    for f in "$ld"*.csv; do
      [ -e "$f" ] || continue
      r=$(( $(wc -l < "$f") - 1 )); rows=$((rows+r)); t=$((t+1)); [ "$r" -ge "$INSTANCES" ] && c=$((c+1))
    done
    printf "  %-20s %2d / 9 triples complete\n" "$label" "$c"
    done_j=$((done_j+c)); total_j=$((total_j+t))
  done
  printf "  -> %d triples complete, %d rows written\n" "$done_j" "$rows"
done
echo "queued, not started: $([ -f Results/.queue ] && wc -l < Results/.queue || echo 0)"
