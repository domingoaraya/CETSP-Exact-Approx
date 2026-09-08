import pandas as pd
import os
import tqdm
import numpy as np
from itertools import product
import argparse
from gurobipy import GRB

from classes.models import CETSPModel
from classes.data_handling import CETSPData
from utils.instance_handler import create_and_save_instance

def run_test(repetitions, n, r_min, r_max, model_type, time_limit=600, extended=False, decomposition=False, nu=None, cut_type=None, verbosity='high', strengthen=False):
    """
    Runs a test for a given configuration on a set of instances.
    """
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

    folder = f"Instances/{n}_{r_min}_{r_max}"
    os.makedirs(folder, exist_ok=True)
    
    # Generate instances if they don't exist
    current_instances = os.listdir(folder)
    if len(current_instances) < repetitions:
        for i in range(len(current_instances), repetitions):
            create_and_save_instance(n, r_min, r_max, i)

    instances_to_run = [f"instance_{i}.txt" for i in range(repetitions)]

    results = {
        'ub': [],
        'lb': [],
        'root_bound': [],
        'node_count': [],
        'status': [],
        'gap': [],
        'runtime': [],
        'cuts': [],
        'dfj_cuts': [],
        'sub_failures': []
    }

    # Derived from r_min/r_max for the subproblem telemetry log only
    # (Diagnostics/subproblem_log.csv); does not affect solving.
    r_mean_val = (r_min + r_max) / 2.0
    r_denom = r_min + r_max
    sigma_val = (r_max - r_min) / r_denom if r_denom > 0 else 0.0

    if verbosity == 'high':
        pbar = tqdm.tqdm(total=repetitions, desc="Solving instances")
    else:
        pbar = None

    for instance_file in instances_to_run:
        instance_path = os.path.join(folder, instance_file)
        
        cetsp_data = CETSPData.from_file(instance_path)
        # Telemetry-only context, consumed by CETSP_L2_Solver's subproblem log.
        cetsp_data.r_mean = r_mean_val
        cetsp_data.sigma = sigma_val
        cetsp_data.instance_id = instance_file

        model = CETSPModel(
            cetsp_data,
            model_type=model_type,
            decomposition=decomposition,
            extended=extended,
            nu=nu,
            cut_type=cut_type,
            strengthen=strengthen
        )
        model.build()
        model.optimize(time_limit)

        summary = model.get_solution_summary()

        if isinstance(summary, dict):
            if model.model.Status == GRB.OPTIMAL:
                results['status'].append("Optimal")
            elif model.model.Status == GRB.SUBOPTIMAL:
                results['status'].append("Suboptimal")
            elif model.model.Status == GRB.TIME_LIMIT:
                results['status'].append("Time_Limit")
            else:
                # Runs that trip the subproblem-failure path are exactly the ones
                # likely to end in a non-standard status; don't drop them silently.
                results['status'].append(f"Other_{model.model.Status}")
            results['ub'].append(summary.get('upper_bound', float('inf')))
            results['lb'].append(summary.get('lower_bound', 0.0))
            results['root_bound'].append(summary.get('root_bound', float('inf')))
            results['node_count'].append(summary.get('node_count', 0))
            results['gap'].append(summary.get('gap', 0))
            results['runtime'].append(summary.get('runtime', 0))
            results['cuts'].append(summary.get('cuts_added', 0))
            results['dfj_cuts'].append(summary.get('dfj_cuts', 0))
            results['sub_failures'].append(summary.get('subproblem_failures', 0))
        else:
            results['status'].append("Failed")
            results['ub'].append(float('inf'))
            results['lb'].append(0.0)
            results['root_bound'].append(float('inf'))
            results['node_count'].append(0)
            results['gap'].append(float('inf'))
            results['runtime'].append(time_limit)
            results['cuts'].append(0)
            results['dfj_cuts'].append(0)
            results['sub_failures'].append(getattr(model, 'subproblem_failures', 0))

        if pbar:
            pbar.update(1)

    if pbar:
        pbar.close()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CETSP experiments with various configurations.")
    parser.add_argument("--time_limit", type=int, default=600, help="Time limit for Gurobi solver in seconds.")
    parser.add_argument("--amount_of_instances", type=int, default=5, help="Number of instances to generate/test per configuration.")
    parser.add_argument("--nu", type=int, default=3, help="Parameter for the extended formulation.")
    parser.add_argument("--n_nodes", type=int, nargs='+', default=[10, 15, 20], help="List of number of nodes (n) to test.")
    parser.add_argument("--r_mean", type=float, nargs='+', default=[0.25, 0.5, 1], help="List of mean radii (r) to test.")
    parser.add_argument("--sigma", type=float, nargs='+', default=[0, 0.2, 0.5], help="List of sigma values for radii variation.")
    parser.add_argument("--model_type", type=str, nargs='+', default=['arc', 'seq', 'perspective'], choices=['arc', 'seq', 'perspective'], help="List of model types to test.") # Conectar B&S
    parser.add_argument("--decomposition", type=str, nargs='+', default=['False', 'True'], choices=['False', 'True'], help="Use decomposition (True/False).")
    parser.add_argument("--extended", type=str, nargs='+', default=['False', 'True'], choices=['False', 'True'], help="Use extended formulation (True/False).")
    parser.add_argument("--sbf_cut_type", type=str, nargs='+', default=None, choices=['dual', 'enumerative', 'dual+enumerative'], help="Cut type for sequence model decomposition.")
    parser.add_argument("--pbf_cut_type", type=str, nargs='+', default=None, choices=['dual', 'enumerative', 'dual+enumerative'], help="Cut type for perspective model decomposition.")
    parser.add_argument("--strengthen", type=str, nargs='+', default=['False'], choices=['False', 'True'], help="Use DFJ cut strengthening at root node (True/False).")
    parser.add_argument("--verbosity", type=str, default='high', choices=['high', 'low'], help="Verbosity level for experiment output ('high' or 'low').")

    args = parser.parse_args()

    # Convert string arguments to boolean where necessary
    decomposition_options = [True if d == 'True' else False for d in args.decomposition]
    extended_options = [True if e == 'True' else False for e in args.extended]
    strengthen_options = [True if s == 'True' else False for s in args.strengthen]

    final_results = []
    
    configurations = []
    for model_type_val in args.model_type:
        for decomposition_val in decomposition_options:
            for extended_val in extended_options:
                for strengthen_val in strengthen_options:
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
                            configurations.append({
                                'model_type': model_type_val,
                                'decomposition': decomposition_val,
                                'extended': extended_val,
                                'cut_type': cut_type_val,
                                'nu': args.nu if extended_val else None,
                                'strengthen': strengthen_val
                            })
                    else:
                        configurations.append({
                            'model_type': model_type_val,
                            'decomposition': decomposition_val,
                            'extended': extended_val,
                            'cut_type': None,
                            'nu': args.nu if extended_val else None,
                            'strengthen': strengthen_val
                        })

    for config in configurations:
        model_type_str = config['model_type']
        
        if model_type_str == 'seq':
            formulation_name = "SBF"
        elif model_type_str == 'perspective':
            formulation_name = "PBF"
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

        for n_nodes_val in args.n_nodes:
            for r_mean_val in args.r_mean:
                for sigma_val in args.sigma:
                    r_min_val = r_mean_val * (1 - sigma_val)
                    r_max_val = r_mean_val * (1 + sigma_val)

                    instance_results = run_test(
                        repetitions=args.amount_of_instances,
                        n=n_nodes_val,
                        r_min=r_min_val,
                        r_max=r_max_val,
                        model_type=config['model_type'],
                        time_limit=args.time_limit,
                        extended=config['extended'],
                        decomposition=config['decomposition'],
                        nu=config['nu'],
                        cut_type=config['cut_type'],
                        verbosity=args.verbosity,
                        strengthen=config['strengthen']
                    )

                    for i in range(args.amount_of_instances):
                        row_data = {
                            "Formulation": formulation_name,
                            "n": n_nodes_val,
                            "r_min": r_min_val,
                            "r_max": r_max_val,
                            "Instance": i,
                            "UB": instance_results['ub'][i],
                            "LB": instance_results['lb'][i],
                            "Root Bound": instance_results['root_bound'][i],
                            "Nodes": instance_results['node_count'][i],
                            "Status": instance_results['status'][i],
                            "Gap": instance_results['gap'][i],
                            "Time": instance_results['runtime'][i],
                            "Cuts": instance_results['cuts'][i],
                            "DFJ_Cuts": instance_results['dfj_cuts'][i],
                            "Sub_Failures": instance_results['sub_failures'][i],
                        }
                        final_results.append(row_data)
                    
        if args.verbosity == 'low':
            print(f"Finished {formulation_name}")

    results_df = pd.DataFrame(final_results)
    
    # Ensure 'Results' directory exists
    os.makedirs("Results", exist_ok=True)
    results_df.to_csv("Results/experiment_results.csv", index=False)
    
    print("\nExperiment finished. Results saved to Results/experiment_results.csv")