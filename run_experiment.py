import os
import tqdm
from itertools import product
import argparse
import csv
from gurobipy import GRB

from classes.models import CETSPModel
from classes.data_handling import CETSPData
from utils.instance_handler import create_and_save_instance

RESULT_COLUMNS = [
    "Formulation", "n", "r_min", "r_max", "Instance",
    "UB", "LB", "Root Bound", "Nodes", "Status", "Gap", "Time",
    "Cuts", "DFJ_Cuts", "Sub_Failures",
]

# Number of decimals used when comparing radii keys between the CSV and the
# values computed at runtime (guards against float repr noise such as
# 0.12500000000000003 vs 0.125).
KEY_DECIMALS = 8


def result_key(formulation, n, r_min, r_max, instance):
    """Canonical key identifying one (configuration, instance) run."""
    return (str(formulation), int(n), round(float(r_min), KEY_DECIMALS),
            round(float(r_max), KEY_DECIMALS), int(instance))


def load_completed_runs(outfile):
    """
    Reads an existing results CSV and returns the set of keys already solved.
    Returns an empty set if the file does not exist or is empty.
    Raises ValueError if the header does not match RESULT_COLUMNS.
    """
    if not os.path.exists(outfile) or os.path.getsize(outfile) == 0:
        return set()

    with open(outfile, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return set()
        if list(reader.fieldnames) != RESULT_COLUMNS:
            raise ValueError(
                f"{outfile} exists but its header does not match the expected "
                f"columns.\n  found:    {reader.fieldnames}\n  expected: {RESULT_COLUMNS}\n"
                "Use a different --outfile or remove the file to start over."
            )
        completed = set()
        for row in reader:
            try:
                completed.add(result_key(row["Formulation"], row["n"], row["r_min"],
                                         row["r_max"], row["Instance"]))
            except (KeyError, ValueError):
                # Skip malformed / partially written trailing rows
                continue
    return completed


class ResultsWriter:
    """
    Appends result rows to a CSV as they are produced, flushing after each
    write so that an interrupted run leaves a usable file behind.
    """
    def __init__(self, outfile):
        self.outfile = outfile
        os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
        write_header = (not os.path.exists(outfile)) or os.path.getsize(outfile) == 0
        self._fh = open(outfile, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=RESULT_COLUMNS)
        if write_header:
            self._writer.writeheader()
            self._fh.flush()

    def write(self, row):
        self._writer.writerow(row)
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def solve_instance(instance_path, model_type, time_limit, extended, decomposition, nu,
                   cut_type, strengthen, optimize_coefficients, threads):
    """
    Solves a single instance and returns a dict with the metric columns
    (everything in RESULT_COLUMNS except the identifying key).
    """
    cetsp_data = CETSPData.from_file(instance_path)

    model = CETSPModel(
        cetsp_data,
        model_type=model_type,
        decomposition=decomposition,
        extended=extended,
        nu=nu,
        cut_type=cut_type,
        strengthen=strengthen,
        optimize_coefficients=optimize_coefficients,
        threads=threads
    )
    model.build()
    model.optimize(time_limit)

    summary = model.get_solution_summary()

    if isinstance(summary, dict):
        if model.model.Status == GRB.OPTIMAL:
            status = "Optimal"
        elif model.model.Status == GRB.SUBOPTIMAL:
            status = "Suboptimal"
        elif model.model.Status == GRB.TIME_LIMIT:
            status = "Time_Limit"
        else:
            status = f"Other_{model.model.Status}"
        return {
            "UB": summary.get('upper_bound', float('inf')),
            "LB": summary.get('lower_bound', 0.0),
            "Root Bound": summary.get('root_bound', float('inf')),
            "Nodes": summary.get('node_count', 0),
            "Status": status,
            "Gap": summary.get('gap', 0),
            "Time": summary.get('runtime', 0),
            "Cuts": summary.get('cuts_added', 0),
            "DFJ_Cuts": summary.get('dfj_cuts', 0),
            "Sub_Failures": summary.get('subproblem_failures', 0),
        }
    else:
        return {
            "UB": float('inf'),
            "LB": 0.0,
            "Root Bound": float('inf'),
            "Nodes": 0,
            "Status": "Failed",
            "Gap": float('inf'),
            "Time": time_limit,
            "Cuts": 0,
            "DFJ_Cuts": 0,
            "Sub_Failures": getattr(model, 'subproblem_failures', 0),
        }


def run_test(repetitions, n, r_min, r_max, model_type, formulation_name, writer,
             completed, time_limit=600, extended=False, decomposition=False, nu=None,
             cut_type=None, verbosity='high', strengthen=False,
             optimize_coefficients=False, threads=0):
    """
    Runs a test for a given configuration on a set of instances.

    Instances whose key is already in `completed` are skipped. Each newly
    solved instance is written to `writer` immediately and added to
    `completed`. Returns the number of instances solved in this call.
    """
    folder = f"Instances/{n}_{r_min}_{r_max}"
    os.makedirs(folder, exist_ok=True)

    # Generate instances if they don't exist
    current_instances = os.listdir(folder)
    if len(current_instances) < repetitions:
        for i in range(len(current_instances), repetitions):
            create_and_save_instance(n, r_min, r_max, i)

    pending = [i for i in range(repetitions)
               if result_key(formulation_name, n, r_min, r_max, i) not in completed]

    if verbosity == 'high':
        print(f"\nRunning {repetitions} instances of {n} cities with radii between {r_min} and {r_max}")
        print(f"Using {model_type}-based formulation, time limit of {time_limit} seconds")
        if extended:
            print(f"Using extended formulation with parameter nu = {nu}")
        if decomposition:
            print("Using decomposition")
        if cut_type:
            print(f"Using {cut_type} cuts for sequence decomposition")
        if strengthen:
            print(f"Using DFJ strengthening")
        if optimize_coefficients:
            print("Optimising the dual cut coefficients")
        skipped = repetitions - len(pending)
        if skipped:
            print(f"Skipping {skipped} instance(s) already present in {writer.outfile}")

    if not pending:
        return 0

    pbar = tqdm.tqdm(total=len(pending), desc="Solving instances") if verbosity == 'high' else None

    for i in pending:
        instance_path = os.path.join(folder, f"instance_{i}.txt")

        metrics = solve_instance(
            instance_path, model_type, time_limit, extended, decomposition, nu,
            cut_type, strengthen, optimize_coefficients, threads
        )

        row = {
            "Formulation": formulation_name,
            "n": n,
            "r_min": r_min,
            "r_max": r_max,
            "Instance": i,
        }
        row.update(metrics)
        writer.write(row)
        completed.add(result_key(formulation_name, n, r_min, r_max, i))

        if pbar:
            pbar.update(1)

    if pbar:
        pbar.close()

    return len(pending)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CETSP experiments with various configurations.")
    parser.add_argument("--time_limit", type=int, default=600, help="Time limit for Gurobi solver in seconds.")
    parser.add_argument("--amount_of_instances", type=int, default=5, help="Number of instances to generate/test per configuration.")
    parser.add_argument("--nu", type=int, default=3, help="Parameter for the extended formulation.")
    parser.add_argument("--n_nodes", type=int, nargs='+', default=[10, 15, 20], help="List of number of nodes (n) to test.")
    parser.add_argument("--r_mean", type=float, nargs='+', default=[0.25, 0.5, 1], help="List of mean radii (r) to test.")
    parser.add_argument("--sigma", type=float, nargs='+', default=[0, 0.2, 0.5], help="List of sigma values for radii variation.")
    parser.add_argument("--model_type", type=str, nargs='+', default=['arc', 'seq', 'perspective'], choices=['arc', 'seq', 'perspective', 'B&S'], help="List of model types to test.")
    parser.add_argument("--decomposition", type=str, nargs='+', default=['False', 'True'], choices=['False', 'True'], help="Use decomposition (True/False).")
    parser.add_argument("--extended", type=str, nargs='+', default=['False', 'True'], choices=['False', 'True'], help="Use extended formulation (True/False).")
    parser.add_argument("--sbf_cut_type", type=str, nargs='+', default=None, choices=['dual', 'enumerative', 'dual+enumerative'], help="Cut type for sequence model decomposition.")
    parser.add_argument("--pbf_cut_type", type=str, nargs='+', default=None, choices=['dual', 'enumerative', 'dual+enumerative'], help="Cut type for perspective model decomposition.")
    parser.add_argument("--strengthen", type=str, nargs='+', default=['False'], choices=['False', 'True'], help="Use DFJ cut strengthening at root node (True/False).")
    parser.add_argument("--optimize_coefficients", type=str, nargs='+', default=['False'], choices=['False', 'True'], help="Optimise the PBF dual cut coefficients (True/False).")
    parser.add_argument("--threads", type=int, default=0, help="Gurobi threads per model. 0 lets Gurobi choose, 1 forces a single thread.")
    parser.add_argument("--verbosity", type=str, default='high', choices=['high', 'low'], help="Verbosity level for experiment output ('high' or 'low').")
    parser.add_argument("--outfile", type=str, default="Results/experiment_results.csv",
                        help="CSV file where results are appended as each instance finishes. "
                             "If the file already exists, runs already recorded in it are skipped, "
                             "so re-running the same command resumes an interrupted experiment.")

    args = parser.parse_args()

    # Convert string arguments to boolean where necessary
    decomposition_options = [True if d == 'True' else False for d in args.decomposition]
    extended_options = [True if e == 'True' else False for e in args.extended]
    strengthen_options = [True if s == 'True' else False for s in args.strengthen]
    oc_options = [True if o == 'True' else False for o in args.optimize_coefficients]

    configurations = []
    for model_type_val in args.model_type:
        if model_type_val == 'B&S':
            # B&S carries its own decomposition and refinement loop; the
            # extended, cut-type and strengthening switches do not apply.
            configurations.append({
                'model_type': 'B&S',
                'decomposition': False,
                'extended': False,
                'cut_type': None,
                'nu': None,
                'strengthen': False,
                'optimize_coefficients': False
            })
            continue
        for decomposition_val in decomposition_options:
            for extended_val in extended_options:
                for strengthen_val, oc_val in product(strengthen_options, oc_options):
                    # DFJ strengthening is only valid for arc and cont formulations
                    if strengthen_val and model_type_val not in ['arc', 'perspective']:
                        continue

                    if decomposition_val and model_type_val in ['seq', 'perspective']:
                        # Determine which cuts to use based on the model type
                        if model_type_val == 'seq':
                            if args.sbf_cut_type is None:
                                cuts_to_use = ['enumerative'] if extended_val else ['dual']
                            else:
                                cuts_to_use = args.sbf_cut_type
                        else:  # perspective
                            if args.pbf_cut_type is None:
                                cuts_to_use = ['enumerative'] if extended_val else ['dual'] # Por determinar estos defaults
                            else:
                                cuts_to_use = args.pbf_cut_type

                        # Build the configuration using the determined cuts
                        for cut_type_val in cuts_to_use:
                            # Coefficient optimisation only changes the perspective
                            # dual cut; the False variant already covers the rest.
                            if oc_val and not (model_type_val == 'perspective'
                                               and 'dual' in cut_type_val):
                                continue
                            configurations.append({
                                'model_type': model_type_val,
                                'decomposition': decomposition_val,
                                'extended': extended_val,
                                'cut_type': cut_type_val,
                                'nu': args.nu if extended_val else None,
                                'strengthen': strengthen_val,
                                'optimize_coefficients': oc_val
                            })
                    else:
                        if oc_val:
                            continue
                        configurations.append({
                            'model_type': model_type_val,
                            'decomposition': decomposition_val,
                            'extended': extended_val,
                            'cut_type': None,
                            'nu': args.nu if extended_val else None,
                            'strengthen': strengthen_val,
                            'optimize_coefficients': False
                        })

    completed = load_completed_runs(args.outfile)
    if completed:
        print(f"Resuming: {len(completed)} run(s) already recorded in {args.outfile}")

    total_solved = 0
    with ResultsWriter(args.outfile) as writer:
        for config in configurations:
            model_type_str = config['model_type']

            if model_type_str == 'seq':
                formulation_name = "SBF"
            elif model_type_str == 'perspective':
                formulation_name = "PBF"
            elif model_type_str == 'B&S':
                formulation_name = "B&S"
            else:
                formulation_name = "ABF"

            if config['extended']:
                formulation_name += "-A"
            if config['decomposition']:
                formulation_name += "-D"
            if config['cut_type']:
                if config['cut_type'] == 'dual+enumerative':
                    formulation_name += "-DE"
                else:
                    formulation_name += f"-{config['cut_type'][:4]}"
            if config['strengthen']:
                formulation_name += "-S"
            if config['optimize_coefficients']:
                formulation_name += "-OC"

            for n_nodes_val in args.n_nodes:
                for r_mean_val in args.r_mean:
                    for sigma_val in args.sigma:
                        r_min_val = r_mean_val * (1 - sigma_val)
                        r_max_val = r_mean_val * (1 + sigma_val)

                        total_solved += run_test(
                            repetitions=args.amount_of_instances,
                            n=n_nodes_val,
                            r_min=r_min_val,
                            r_max=r_max_val,
                            model_type=config['model_type'],
                            formulation_name=formulation_name,
                            writer=writer,
                            completed=completed,
                            time_limit=args.time_limit,
                            extended=config['extended'],
                            decomposition=config['decomposition'],
                            nu=config['nu'],
                            cut_type=config['cut_type'],
                            verbosity=args.verbosity,
                            strengthen=config['strengthen'],
                            optimize_coefficients=config['optimize_coefficients'],
                            threads=args.threads
                        )

            if args.verbosity == 'low':
                print(f"Finished {formulation_name}")

    print(f"\nExperiment finished. {total_solved} new run(s) written to {args.outfile}")
