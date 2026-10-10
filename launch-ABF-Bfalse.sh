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
N_LIST=(10 15 20 25)
INSTANCES=5
TIME_LIMIT=1800
CPUSETS=(
  "0,1,2,3"
  "4,5,6,7"
  "8,9,10,11"
  "12,13,14,15"
  "16,17,18,19"
  "20,21,22,23"
  "24,25,26,27"
  "28,29,30,31"
  "32,33,34,35"
  "36,37,38,39"
  "40,41,42,43"
  "44,45,46,47"
  "48,49,50,51"
  "52,53,54,55"
  "56,57,58,59"
  "60,61,62,63"
  "64,65,66,67"
  "68,69,70,71"
  "72,73,74,75"
  "76,77,78,79"
  "80,81,82,83"
  "84,85,86,87"
  "88,89,90,91"
  "92,93,94,95"
  "96,97,98,99"
  "100,101,102,103"
  "104,105,106,107"
  "108,109,110,111"
  "112,113,114,115"
  "116,117,118,119"
  "120,121,122,123"
  "124,125,126,127"
  "128,129,130,131"
  "132,133,134,135"
  "136,137,138,139"
  "140,141,142,143"
  "144,145,146,147"
  "148,149,150,151"
  "152,153,154,155"
  "156,157,158,159"
  "160,161,162,163"
  "164,165,166,167"
  "168,169,170,171"
  "172,173,174,175"
  "176,177,178,179"
  "180,181,182,183"
  "184,185,186,187"
  "188,189,190,191"
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
  "ABF-A-D-DE|--model_type arc --decomposition True --extended True --strengthen False --optimize_coefficients False --abf_cut_type dual+enumerative --symmetry_breaking False"
  "ABF-A-D-DE-S|--model_type arc --decomposition True --extended True --strengthen True --optimize_coefficients False --abf_cut_type dual+enumerative --symmetry_breaking False"
  "ABF-A-D-dual|--model_type arc --decomposition True --extended True --strengthen False --optimize_coefficients False --abf_cut_type dual --symmetry_breaking False"
  "ABF-A-D-dual-S|--model_type arc --decomposition True --extended True --strengthen True --optimize_coefficients False --abf_cut_type dual --symmetry_breaking False"
  "ABF-A-D-enum|--model_type arc --decomposition True --extended True --strengthen False --optimize_coefficients False --abf_cut_type enumerative --symmetry_breaking False"
  "ABF-A-D-enum-S|--model_type arc --decomposition True --extended True --strengthen True --optimize_coefficients False --abf_cut_type enumerative --symmetry_breaking False"
  "ABF-A-S|--model_type arc --decomposition False --extended True --strengthen True --optimize_coefficients False --symmetry_breaking False"
  "ABF-S|--model_type arc --decomposition False --extended False --strengthen True --optimize_coefficients False --symmetry_breaking False"
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
python3 - "$INSTANCES" "${N_LIST[@]}" -- "${INSTANCE_CONFIGS[@]}" <<'PY'
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
    taskset -c "$cpuset" python3 -u run_experiment.py \
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
