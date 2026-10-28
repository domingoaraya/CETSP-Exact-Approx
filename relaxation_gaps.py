"""
Continuous relaxation gaps of the formulations of an experiment.

For every row of the merged results file (Results/experiment_results_all.csv,
written by merge.sh) whose formulation is a full model (ABF, SBF, PBF and
their -A approximations, in any -S/-B variant), builds the model on the
row's instance, solves its continuous relaxation (integrality dropped; the
SOCPs are solved as such, not through the outer approximation Gurobi uses
inside the branch-and-cut) and writes

    Formulation, n, r_min, r_max, Instance, LB_relax, UB_best, Gap_relax

with Gap_relax = (UB_best - LB_relax) / UB_best to
Results/n<n>/relaxation_gaps_n<n>.csv and Results/relaxation_gaps_all.csv.

UB_best is the best upper bound of the instance over all methods in the
input file, decompositions and BS included; their rows are otherwise
skipped. Missing instance files are generated as run_experiment.py does.

Usage: python relaxation_gaps.py [--results Results] [--input Results/experiment_results_all.csv]
                                 [--instances Instances] [--nu 3]
"""
import argparse
import csv
import math
import os
import sys

from gurobipy import GRB

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classes.data_handling import CETSPData          # noqa: E402
from classes.models import CETSPModel                # noqa: E402
from utils.instance_handler import create_and_save_instance   # noqa: E402
from utils.results import parse_formulation          # noqa: E402

KEY_DECIMALS = 8
OUT_COLUMNS = ["Formulation", "n", "r_min", "r_max", "Instance", "LB_relax", "UB_best", "Gap_relax"]
MODEL_TYPE = {'ABF': 'arc', 'SBF': 'seq', 'PBF': 'perspective', 'BS': 'BS'}


def instance_key(n, r_min, r_max, inst):
    return (int(n), round(float(r_min), KEY_DECIMALS), round(float(r_max), KEY_DECIMALS), int(inst))


def is_full_model(label):
    """True for ABF, SBF, PBF and their -A approximations; False for decompositions and BS."""
    p = parse_formulation(label)
    return p['base'] != 'BS' and not p['decomposition']


def relaxation_value(instance_path, label, nu):
    """Optimal value of the continuous relaxation of a label's model, or None if it is not solved."""
    p = parse_formulation(label)
    cetsp = CETSPModel(CETSPData.from_file(instance_path), model_type=MODEL_TYPE[p['base']],
                       decomposition=p['decomposition'], extended=p['extended'], nu=nu,
                       symmetry_breaking=p['symmetry'])
    cetsp.build()
    # build() leaves its additions pending; relax() only sees the updated model
    cetsp.model.update()
    if cetsp.model.NumVars == 0:
        raise RuntimeError(f"empty model for {label}")
    relaxed = cetsp.model.relax()
    relaxed.Params.OutputFlag = 0
    relaxed.Params.BarHomogeneous = 1
    relaxed.Params.NumericFocus = 3
    relaxed.optimize()
    if relaxed.Status == GRB.OPTIMAL:
        return relaxed.ObjVal
    return max(relaxed.ObjBound, 0)


def read_rows(path):
    """Rows of the merged results file, tagged with their n directory."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["_ndir"] = f"n{int(row['n'])}"
    return rows


def main():
    ap = argparse.ArgumentParser(description="Continuous relaxation gaps per instance and formulation.")
    ap.add_argument("--results", default="Results", help="output directory (one file per n plus a global one)")
    ap.add_argument("--input", default=os.path.join("Results", "experiment_results_all.csv"),
                    help="merged results file that defines the rows and UB_best")
    ap.add_argument("--instances", default="Instances")
    ap.add_argument("--nu", type=int, default=3)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"{args.input} not found; run merge.sh first")
    rows = read_rows(args.input)
    if not rows:
        raise SystemExit(f"no rows in {args.input}")

    ub_best = {}
    for row in rows:
        key = instance_key(row["n"], row["r_min"], row["r_max"], row["Instance"])
        try:
            ub = float(row["UB"])
        except (TypeError, ValueError):
            continue
        if math.isfinite(ub):
            ub_best[key] = min(ub, ub_best.get(key, math.inf))

    skipped = sorted({row["Formulation"] for row in rows if not is_full_model(row["Formulation"])})
    rows = [row for row in rows if is_full_model(row["Formulation"])]
    if skipped:
        print("skipped (decompositions and BS): " + ", ".join(skipped))
    if not rows:
        raise SystemExit("no rows of full models in the input")

    # One relaxation per (label, instance)
    needed = {}
    for row in rows:
        key = instance_key(row["n"], row["r_min"], row["r_max"], row["Instance"])
        needed.setdefault((row["Formulation"], key), (row["n"], row["r_min"], row["r_max"], row["Instance"]))
    values = {}
    for i, ((label, key), (n, r_min, r_max, inst)) in enumerate(sorted(needed.items(), key=lambda kv: (kv[0][1], kv[0][0])), 1):
        folder = os.path.join(args.instances, f"{n}_{r_min}_{r_max}")
        path = os.path.join(folder, f"instance_{inst}.txt")
        if not os.path.exists(path):
            # Same generator and seed as run_experiment.py
            os.makedirs(folder, exist_ok=True)
            create_and_save_instance(int(n), float(r_min), float(r_max), int(inst))
        values[(label, key)] = relaxation_value(path, label, args.nu)
        print(f"[{i}/{len(needed)}] {label} n={n} r=[{r_min}, {r_max}] inst {inst}: {values[(label, key)]}", flush=True)

    out_rows = []
    for row in rows:
        key = instance_key(row["n"], row["r_min"], row["r_max"], row["Instance"])
        lb = values.get((row["Formulation"], key))
        ub = ub_best.get(key, math.inf)
        gap = (ub - lb) / ub if (lb is not None and math.isfinite(ub) and ub > 0) else None
        fmt = lambda v: "" if v is None or not math.isfinite(v) else repr(v)
        out_rows.append({"Formulation": row["Formulation"], "n": row["n"], "r_min": row["r_min"],
                         "r_max": row["r_max"], "Instance": row["Instance"],
                         "LB_relax": fmt(lb), "UB_best": fmt(ub), "Gap_relax": fmt(gap), "_ndir": row["_ndir"]})

    for ndir in sorted({r["_ndir"] for r in out_rows}):
        os.makedirs(os.path.join(args.results, ndir), exist_ok=True)
        path = os.path.join(args.results, ndir, f"relaxation_gaps_{ndir}.csv")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=OUT_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(r for r in out_rows if r["_ndir"] == ndir)
        print(f"{sum(r['_ndir'] == ndir for r in out_rows)} rows -> {path}")
    path = os.path.join(args.results, "relaxation_gaps_all.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    print(f"{len(out_rows)} rows -> {path}")

    sums = {}
    for r in out_rows:
        if r["Gap_relax"] != "":
            s, c = sums.get(r["Formulation"], (0.0, 0))
            sums[r["Formulation"]] = (s + float(r["Gap_relax"]), c + 1)
    if sums:
        print("\nmean Gap_relax by formulation:")
        for label, (s, c) in sorted(sums.items()):
            print(f"  {label:24s} {s / c:8.4f}  ({c} rows)")


if __name__ == "__main__":
    main()
