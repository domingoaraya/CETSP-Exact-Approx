#!/usr/bin/env bash
# Full experiment: for each n in N_LIST, each instance configuration
# (r_mean, sigma) and each solver configuration, solve INSTANCES repetitions
# and write one CSV per triple to
#     Results/n<n>/<label>/r<r_mean>_s<sigma>.csv        (INSTANCES rows)
# with the Gurobi log of each instance in logs/gurobi/n<n>/ and the process
# output in logs/n<n>/<label>__r<r_mean>_s<sigma>.log.
#
# Jobs are handed out from a dynamic queue to the CPU sets in CPUSETS
# (taskset), in order: all n=10 jobs first, then n=15, then n=20; within each
# n, by solver configuration in CONFIGS order. A CSV with INSTANCES rows counts
# as complete, so rerunning the script resumes only what is pending. All
# parameters live in this file; the script takes no arguments.
set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------- parameters
N_LIST=(10 15 20)
INSTANCES=5
TIME_LIMIT=1800
CPUSETS=(
  "0,1,16,17"
  "2,3,18,19"
  "4,5,20,21"
  "6,7,22,23"
  "8,9,24,25"
  "10,11,26,27"
  "12,13,28,29"
  "14,15,30,31"
)
THREADS=4          # Gurobi threads per job (= CPUs in each set)

# Instance configurations: "r_mean sigma" (the paper's grid).
INSTANCE_CONFIGS=(
  "0.25 0"
  "0.25 0.2"
  "0.25 0.5"
  "0.5 0"
  "0.5 0.2"
  "0.5 0.5"
  "1 0"
  "1 0.2"
  "1 0.5"
)

# Solver configurations: "label|flags", in alphabetical order by label. Comment
# out the ones you don't want to run. Lists every combination run_experiment.py
# can generate (98):
# model x {A} x {D, cut type} x {S} x {OC} x {B}, with -S only for ABF/PBF,
# -OC only for PBF-D with dual cuts, and -B (symmetry breaking, no reverse-tour
# cuts) for all of them; each entry is followed by its -B variant. Each label
# matches the one run_experiment.py prints for those flags (check with
# --amount_of_instances 0).
CONFIGS=(
  "ABF|--model_type arc --decomposition False --extended False --strengthen False --optimize_coefficients False --symmetry_breaking False"
  "ABF-A|--model_type arc --decomposition False --extended True --strengthen False --optimize_coefficients False --symmetry_breaking False"
  "ABF-A-B|--model_type arc --decomposition False --extended True --strengthen False --optimize_coefficients False --symmetry_breaking True"
  "ABF-A-D-DE|--model_type arc --decomposition True --extended True --strengthen False --optimize_coefficients False --abf_cut_type dual+enumerative --symmetry_breaking False"
  "ABF-A-D-DE-B|--model_type arc --decomposition True --extended True --strengthen False --optimize_coefficients False --abf_cut_type dual+enumerative --symmetry_breaking True"
  "ABF-A-D-DE-S|--model_type arc --decomposition True --extended True --strengthen True --optimize_coefficients False --abf_cut_type dual+enumerative --symmetry_breaking False"
  "ABF-A-D-DE-S-B|--model_type arc --decomposition True --extended True --strengthen True --optimize_coefficients False --abf_cut_type dual+enumerative --symmetry_breaking True"
  "ABF-A-D-dual|--model_type arc --decomposition True --extended True --strengthen False --optimize_coefficients False --abf_cut_type dual --symmetry_breaking False"
  "ABF-A-D-dual-B|--model_type arc --decomposition True --extended True --strengthen False --optimize_coefficients False --abf_cut_type dual --symmetry_breaking True"
  "ABF-A-D-dual-S|--model_type arc --decomposition True --extended True --strengthen True --optimize_coefficients False --abf_cut_type dual --symmetry_breaking False"
  "ABF-A-D-dual-S-B|--model_type arc --decomposition True --extended True --strengthen True --optimize_coefficients False --abf_cut_type dual --symmetry_breaking True"
  "ABF-A-D-enum|--model_type arc --decomposition True --extended True --strengthen False --optimize_coefficients False --abf_cut_type enumerative --symmetry_breaking False"
  "ABF-A-D-enum-B|--model_type arc --decomposition True --extended True --strengthen False --optimize_coefficients False --abf_cut_type enumerative --symmetry_breaking True"
  "ABF-A-D-enum-S|--model_type arc --decomposition True --extended True --strengthen True --optimize_coefficients False --abf_cut_type enumerative --symmetry_breaking False"
  "ABF-A-D-enum-S-B|--model_type arc --decomposition True --extended True --strengthen True --optimize_coefficients False --abf_cut_type enumerative --symmetry_breaking True"
  "ABF-A-S|--model_type arc --decomposition False --extended True --strengthen True --optimize_coefficients False --symmetry_breaking False"
  "ABF-A-S-B|--model_type arc --decomposition False --extended True --strengthen True --optimize_coefficients False --symmetry_breaking True"
  "ABF-B|--model_type arc --decomposition False --extended False --strengthen False --optimize_coefficients False --symmetry_breaking True"
  "ABF-D-DE|--model_type arc --decomposition True --extended False --strengthen False --optimize_coefficients False --abf_cut_type dual+enumerative --symmetry_breaking False"
  "ABF-D-DE-B|--model_type arc --decomposition True --extended False --strengthen False --optimize_coefficients False --abf_cut_type dual+enumerative --symmetry_breaking True"
  "ABF-D-DE-S|--model_type arc --decomposition True --extended False --strengthen True --optimize_coefficients False --abf_cut_type dual+enumerative --symmetry_breaking False"
  "ABF-D-DE-S-B|--model_type arc --decomposition True --extended False --strengthen True --optimize_coefficients False --abf_cut_type dual+enumerative --symmetry_breaking True"
  "ABF-D-dual|--model_type arc --decomposition True --extended False --strengthen False --optimize_coefficients False --abf_cut_type dual --symmetry_breaking False"
  "ABF-D-dual-B|--model_type arc --decomposition True --extended False --strengthen False --optimize_coefficients False --abf_cut_type dual --symmetry_breaking True"
  "ABF-D-dual-S|--model_type arc --decomposition True --extended False --strengthen True --optimize_coefficients False --abf_cut_type dual --symmetry_breaking False"
  "ABF-D-dual-S-B|--model_type arc --decomposition True --extended False --strengthen True --optimize_coefficients False --abf_cut_type dual --symmetry_breaking True"
  "ABF-D-enum|--model_type arc --decomposition True --extended False --strengthen False --optimize_coefficients False --abf_cut_type enumerative --symmetry_breaking False"
  "ABF-D-enum-B|--model_type arc --decomposition True --extended False --strengthen False --optimize_coefficients False --abf_cut_type enumerative --symmetry_breaking True"
  "ABF-D-enum-S|--model_type arc --decomposition True --extended False --strengthen True --optimize_coefficients False --abf_cut_type enumerative --symmetry_breaking False"
  "ABF-D-enum-S-B|--model_type arc --decomposition True --extended False --strengthen True --optimize_coefficients False --abf_cut_type enumerative --symmetry_breaking True"
  "ABF-S|--model_type arc --decomposition False --extended False --strengthen True --optimize_coefficients False --symmetry_breaking False"
  "ABF-S-B|--model_type arc --decomposition False --extended False --strengthen True --optimize_coefficients False --symmetry_breaking True"
  "BS|--model_type BS --symmetry_breaking False"
  "BS-B|--model_type BS --symmetry_breaking True"
  "PBF|--model_type perspective --decomposition False --extended False --strengthen False --optimize_coefficients False --symmetry_breaking False"
  "PBF-A|--model_type perspective --decomposition False --extended True --strengthen False --optimize_coefficients False --symmetry_breaking False"
  "PBF-A-B|--model_type perspective --decomposition False --extended True --strengthen False --optimize_coefficients False --symmetry_breaking True"
  "PBF-A-D-DE|--model_type perspective --decomposition True --extended True --strengthen False --optimize_coefficients False --pbf_cut_type dual+enumerative --symmetry_breaking False"
  "PBF-A-D-DE-B|--model_type perspective --decomposition True --extended True --strengthen False --optimize_coefficients False --pbf_cut_type dual+enumerative --symmetry_breaking True"
  "PBF-A-D-DE-OC|--model_type perspective --decomposition True --extended True --strengthen False --optimize_coefficients True --pbf_cut_type dual+enumerative --symmetry_breaking False"
  "PBF-A-D-DE-OC-B|--model_type perspective --decomposition True --extended True --strengthen False --optimize_coefficients True --pbf_cut_type dual+enumerative --symmetry_breaking True"
  "PBF-A-D-DE-S|--model_type perspective --decomposition True --extended True --strengthen True --optimize_coefficients False --pbf_cut_type dual+enumerative --symmetry_breaking False"
  "PBF-A-D-DE-S-B|--model_type perspective --decomposition True --extended True --strengthen True --optimize_coefficients False --pbf_cut_type dual+enumerative --symmetry_breaking True"
  "PBF-A-D-DE-S-OC|--model_type perspective --decomposition True --extended True --strengthen True --optimize_coefficients True --pbf_cut_type dual+enumerative --symmetry_breaking False"
  "PBF-A-D-DE-S-OC-B|--model_type perspective --decomposition True --extended True --strengthen True --optimize_coefficients True --pbf_cut_type dual+enumerative --symmetry_breaking True"
  "PBF-A-D-dual|--model_type perspective --decomposition True --extended True --strengthen False --optimize_coefficients False --pbf_cut_type dual --symmetry_breaking False"
  "PBF-A-D-dual-B|--model_type perspective --decomposition True --extended True --strengthen False --optimize_coefficients False --pbf_cut_type dual --symmetry_breaking True"
  "PBF-A-D-dual-OC|--model_type perspective --decomposition True --extended True --strengthen False --optimize_coefficients True --pbf_cut_type dual --symmetry_breaking False"
  "PBF-A-D-dual-OC-B|--model_type perspective --decomposition True --extended True --strengthen False --optimize_coefficients True --pbf_cut_type dual --symmetry_breaking True"
  "PBF-A-D-dual-S|--model_type perspective --decomposition True --extended True --strengthen True --optimize_coefficients False --pbf_cut_type dual --symmetry_breaking False"
  "PBF-A-D-dual-S-B|--model_type perspective --decomposition True --extended True --strengthen True --optimize_coefficients False --pbf_cut_type dual --symmetry_breaking True"
  "PBF-A-D-dual-S-OC|--model_type perspective --decomposition True --extended True --strengthen True --optimize_coefficients True --pbf_cut_type dual --symmetry_breaking False"
  "PBF-A-D-dual-S-OC-B|--model_type perspective --decomposition True --extended True --strengthen True --optimize_coefficients True --pbf_cut_type dual --symmetry_breaking True"
  "PBF-A-D-enum|--model_type perspective --decomposition True --extended True --strengthen False --optimize_coefficients False --pbf_cut_type enumerative --symmetry_breaking False"
  "PBF-A-D-enum-B|--model_type perspective --decomposition True --extended True --strengthen False --optimize_coefficients False --pbf_cut_type enumerative --symmetry_breaking True"
  "PBF-A-D-enum-S|--model_type perspective --decomposition True --extended True --strengthen True --optimize_coefficients False --pbf_cut_type enumerative --symmetry_breaking False"
  "PBF-A-D-enum-S-B|--model_type perspective --decomposition True --extended True --strengthen True --optimize_coefficients False --pbf_cut_type enumerative --symmetry_breaking True"
  "PBF-A-S|--model_type perspective --decomposition False --extended True --strengthen True --optimize_coefficients False --symmetry_breaking False"
  "PBF-A-S-B|--model_type perspective --decomposition False --extended True --strengthen True --optimize_coefficients False --symmetry_breaking True"
  "PBF-B|--model_type perspective --decomposition False --extended False --strengthen False --optimize_coefficients False --symmetry_breaking True"
  "PBF-D-DE|--model_type perspective --decomposition True --extended False --strengthen False --optimize_coefficients False --pbf_cut_type dual+enumerative --symmetry_breaking False"
  "PBF-D-DE-B|--model_type perspective --decomposition True --extended False --strengthen False --optimize_coefficients False --pbf_cut_type dual+enumerative --symmetry_breaking True"
  "PBF-D-DE-OC|--model_type perspective --decomposition True --extended False --strengthen False --optimize_coefficients True --pbf_cut_type dual+enumerative --symmetry_breaking False"
  "PBF-D-DE-OC-B|--model_type perspective --decomposition True --extended False --strengthen False --optimize_coefficients True --pbf_cut_type dual+enumerative --symmetry_breaking True"
  "PBF-D-DE-S|--model_type perspective --decomposition True --extended False --strengthen True --optimize_coefficients False --pbf_cut_type dual+enumerative --symmetry_breaking False"
  "PBF-D-DE-S-B|--model_type perspective --decomposition True --extended False --strengthen True --optimize_coefficients False --pbf_cut_type dual+enumerative --symmetry_breaking True"
  "PBF-D-DE-S-OC|--model_type perspective --decomposition True --extended False --strengthen True --optimize_coefficients True --pbf_cut_type dual+enumerative --symmetry_breaking False"
  "PBF-D-DE-S-OC-B|--model_type perspective --decomposition True --extended False --strengthen True --optimize_coefficients True --pbf_cut_type dual+enumerative --symmetry_breaking True"
  "PBF-D-dual|--model_type perspective --decomposition True --extended False --strengthen False --optimize_coefficients False --pbf_cut_type dual --symmetry_breaking False"
  "PBF-D-dual-B|--model_type perspective --decomposition True --extended False --strengthen False --optimize_coefficients False --pbf_cut_type dual --symmetry_breaking True"
  "PBF-D-dual-OC|--model_type perspective --decomposition True --extended False --strengthen False --optimize_coefficients True --pbf_cut_type dual --symmetry_breaking False"
  "PBF-D-dual-OC-B|--model_type perspective --decomposition True --extended False --strengthen False --optimize_coefficients True --pbf_cut_type dual --symmetry_breaking True"
  "PBF-D-dual-S|--model_type perspective --decomposition True --extended False --strengthen True --optimize_coefficients False --pbf_cut_type dual --symmetry_breaking False"
  "PBF-D-dual-S-B|--model_type perspective --decomposition True --extended False --strengthen True --optimize_coefficients False --pbf_cut_type dual --symmetry_breaking True"
  "PBF-D-dual-S-OC|--model_type perspective --decomposition True --extended False --strengthen True --optimize_coefficients True --pbf_cut_type dual --symmetry_breaking False"
  "PBF-D-dual-S-OC-B|--model_type perspective --decomposition True --extended False --strengthen True --optimize_coefficients True --pbf_cut_type dual --symmetry_breaking True"
  "PBF-D-enum|--model_type perspective --decomposition True --extended False --strengthen False --optimize_coefficients False --pbf_cut_type enumerative --symmetry_breaking False"
  "PBF-D-enum-B|--model_type perspective --decomposition True --extended False --strengthen False --optimize_coefficients False --pbf_cut_type enumerative --symmetry_breaking True"
  "PBF-D-enum-S|--model_type perspective --decomposition True --extended False --strengthen True --optimize_coefficients False --pbf_cut_type enumerative --symmetry_breaking False"
  "PBF-D-enum-S-B|--model_type perspective --decomposition True --extended False --strengthen True --optimize_coefficients False --pbf_cut_type enumerative --symmetry_breaking True"
  "PBF-S|--model_type perspective --decomposition False --extended False --strengthen True --optimize_coefficients False --symmetry_breaking False"
  "PBF-S-B|--model_type perspective --decomposition False --extended False --strengthen True --optimize_coefficients False --symmetry_breaking True"
  "SBF|--model_type seq --decomposition False --extended False --strengthen False --optimize_coefficients False --symmetry_breaking False"
  "SBF-A|--model_type seq --decomposition False --extended True --strengthen False --optimize_coefficients False --symmetry_breaking False"
  "SBF-A-B|--model_type seq --decomposition False --extended True --strengthen False --optimize_coefficients False --symmetry_breaking True"
  "SBF-A-D-DE|--model_type seq --decomposition True --extended True --strengthen False --optimize_coefficients False --sbf_cut_type dual+enumerative --symmetry_breaking False"
  "SBF-A-D-DE-B|--model_type seq --decomposition True --extended True --strengthen False --optimize_coefficients False --sbf_cut_type dual+enumerative --symmetry_breaking True"
  "SBF-A-D-dual|--model_type seq --decomposition True --extended True --strengthen False --optimize_coefficients False --sbf_cut_type dual --symmetry_breaking False"
  "SBF-A-D-dual-B|--model_type seq --decomposition True --extended True --strengthen False --optimize_coefficients False --sbf_cut_type dual --symmetry_breaking True"
  "SBF-A-D-enum|--model_type seq --decomposition True --extended True --strengthen False --optimize_coefficients False --sbf_cut_type enumerative --symmetry_breaking False"
  "SBF-A-D-enum-B|--model_type seq --decomposition True --extended True --strengthen False --optimize_coefficients False --sbf_cut_type enumerative --symmetry_breaking True"
  "SBF-B|--model_type seq --decomposition False --extended False --strengthen False --optimize_coefficients False --symmetry_breaking True"
  "SBF-D-DE|--model_type seq --decomposition True --extended False --strengthen False --optimize_coefficients False --sbf_cut_type dual+enumerative --symmetry_breaking False"
  "SBF-D-DE-B|--model_type seq --decomposition True --extended False --strengthen False --optimize_coefficients False --sbf_cut_type dual+enumerative --symmetry_breaking True"
  "SBF-D-dual|--model_type seq --decomposition True --extended False --strengthen False --optimize_coefficients False --sbf_cut_type dual --symmetry_breaking False"
  "SBF-D-dual-B|--model_type seq --decomposition True --extended False --strengthen False --optimize_coefficients False --sbf_cut_type dual --symmetry_breaking True"
  "SBF-D-enum|--model_type seq --decomposition True --extended False --strengthen False --optimize_coefficients False --sbf_cut_type enumerative --symmetry_breaking False"
  "SBF-D-enum-B|--model_type seq --decomposition True --extended False --strengthen False --optimize_coefficients False --sbf_cut_type enumerative --symmetry_breaking True"
)

# ------------------------------------------------------------------ queue
QUEUE="Results/.queue"
LOCK="Results/.queue.lock"
mkdir -p Results logs

csv_path() {  # csv_path <n> <label> <r_mean> <sigma>
  echo "Results/n$1/$2/r$3_s$4.csv"
}

: > "$QUEUE"
total=0
for n in "${N_LIST[@]}"; do
  pend=0
  for entry in "${CONFIGS[@]}"; do
    label="${entry%%|*}"
    for ic in "${INSTANCE_CONFIGS[@]}"; do
      read -r rm sg <<< "$ic"
      f=$(csv_path "$n" "$label" "$rm" "$sg")
      rows=0; [ -f "$f" ] && rows=$(( $(wc -l < "$f") - 1 ))
      if [ "$rows" -lt "$INSTANCES" ]; then echo "$n|$rm|$sg|$entry" >> "$QUEUE"; pend=$((pend+1)); fi
      total=$((total+1))
    done
  done
  echo "n=$n: $pend of $(( ${#CONFIGS[@]} * ${#INSTANCE_CONFIGS[@]} )) jobs pending"
done
echo "total: $(wc -l < "$QUEUE") of $total jobs pending  (each job = $INSTANCES instances, $TIME_LIMIT s limit)"

# Generate every instance before launching, so that several processes never
# write the same file at once. Same formulas as run_experiment.py.
python - "$INSTANCES" "${N_LIST[@]}" -- "${INSTANCE_CONFIGS[@]}" <<'PY'
import os, sys
from utils.instance_handler import create_and_save_instance
k = int(sys.argv[1]); sep = sys.argv.index('--')
ns = list(map(int, sys.argv[2:sep])); ics = [tuple(map(float, s.split())) for s in sys.argv[sep+1:]]
for n in ns:
    for r, s in ics:
        r_min, r_max = r*(1-s), r*(1+s)
        folder = f"Instances/{n}_{r_min}_{r_max}"
        os.makedirs(folder, exist_ok=True)
        for i in range(k):
            if not os.path.exists(f"{folder}/instance_{i}.txt"):
                create_and_save_instance(n, r_min, r_max, i)
PY

pop() {  # atomically removes and returns the first line of the queue; empty when nothing is left
  (
    flock 9
    head -n 1 "$QUEUE"
    tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
  ) 9>"$LOCK"
}

worker() {  # worker <cpuset>
  local cpuset=$1 tag="cpus_${1//,/_}"
  while true; do
    local entry; entry=$(pop)
    [ -z "$entry" ] && break
    IFS='|' read -r n rm sg label flags <<< "$entry"
    local out; out=$(csv_path "$n" "$label" "$rm" "$sg")
    local log="logs/n${n}/${label}__r${rm}_s${sg}.log"
    mkdir -p "$(dirname "$out")" "logs/n${n}" "logs/gurobi/n${n}"
    echo "[$(date '+%F %T')] $tag -> n=$n $label r=$rm sigma=$sg"
    taskset -c "$cpuset" python -u run_experiment.py \
        --n_nodes "$n" --r_mean "$rm" --sigma "$sg" --amount_of_instances "$INSTANCES" \
        --time_limit "$TIME_LIMIT" --threads "$THREADS" --verbosity low \
        --gurobi_log_dir "logs/gurobi/n${n}" --outfile "$out" $flags > "$log" 2>&1 \
      || echo "[$(date '+%F %T')] $tag !! n=$n $label r=$rm sigma=$sg exited with an error (see $log)"
  done
  echo "[$(date '+%F %T')] $tag no work left, exiting"
}

# Ctrl-C or kill on the launcher also stops the workers and Gurobi.
trap 'echo "interrupted, stopping workers"; trap - INT TERM; kill 0' INT TERM

for cs in "${CPUSETS[@]}"; do
  worker "$cs" &
done
echo "Launched ${#CPUSETS[@]} workers. Progress: ./status.sh   |   merge: ./merge.sh"
wait
echo "All done."
