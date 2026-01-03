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

def run_test(repetitions, n, r_min, r_max, model_type, time_limit=600, extended=False, decomposition=False, nu=None, seq_cut_type=None, verbosity='high'):
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
        if seq_cut_type:
            print(f"Using {seq_cut_type} cuts for sequence decomposition")

    folder = f"Instances/{n}_{r_min}_{r_max}"
    os.makedirs(folder, exist_ok=True)
    
    # Generate instances if they don't exist
    current_instances = os.listdir(folder)
    if len(current_instances) < repetitions:
        for i in range(len(current_instances), repetitions):
            create_and_save_instance(n, r_min, r_max, i)

    instances_to_run = [f"instance_{i}.txt" for i in range(repetitions)]

    results = {
        'status': [],
        'gap': [],
        'runtime': [],
        'cuts': []
    }

    if verbosity == 'high':
        pbar = tqdm.tqdm(total=repetitions, desc="Solving instances")
    else:
        pbar = None

    for instance_file in instances_to_run:
        instance_path = os.path.join(folder, instance_file)
        
        cetsp_data = CETSPData.from_file(instance_path)

        model = CETSPModel(
            cetsp_data,
            model_type=model_type,
            decomposition=decomposition,
            extended=extended,
            nu=nu,
            seq_cut_type=seq_cut_type
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
            results['gap'].append(summary.get('gap', 0))
            results['runtime'].append(summary.get('runtime', 0))
            results['cuts'].append(summary.get('cuts_added', 0))
        else:
            results['status'].append("Failed")
            results['gap'].append(float('inf'))
            results['runtime'].append(time_limit)
            results['cuts'].append(0)

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
    parser.add_argument("--model_type", type=str, nargs='+', default=['arc', 'seq'], choices=['arc', 'seq'], help="List of model types to test.")
    parser.add_argument("--decomposition", type=str, nargs='+', default=['False', 'True'], choices=['False', 'True'], help="Use decomposition (True/False).")
    parser.add_argument("--extended", type=str, nargs='+', default=['False', 'True'], choices=['False', 'True'], help="Use extended formulation (True/False).")
    parser.add_argument("--seq_cut_type", type=str, nargs='+', default=['enumerative'], choices=['dual', 'enumerative', 'dual+enumerative'], help="Cut type for sequence model decomposition.")
    parser.add_argument("--verbosity", type=str, default='high', choices=['high', 'low'], help="Verbosity level for experiment output ('high' or 'low').")

    args = parser.parse_args()

    # Convert string arguments to boolean where necessary
    decomposition_options = [True if d == 'True' else False for d in args.decomposition]
    extended_options = [True if e == 'True' else False for e in args.extended]

    final_results = []
    
    configurations = []
    for model_type_val in args.model_type:
        for decomposition_val in decomposition_options:
            for extended_val in extended_options:
                if model_type_val == 'seq' and decomposition_val:
                    for seq_cut_type_val in args.seq_cut_type:
                        configurations.append({
                            'model_type': model_type_val,
                            'decomposition': decomposition_val,
                            'extended': extended_val,
                            'seq_cut_type': seq_cut_type_val,
                            'nu': args.nu if extended_val else None
                        })
                else:
                    configurations.append({
                        'model_type': model_type_val,
                        'decomposition': decomposition_val,
                        'extended': extended_val,
                        'seq_cut_type': None,
                        'nu': args.nu if extended_val else None
                    })

    for config in configurations:
        model_type_str = config['model_type']
        
        formulation_name = f"SBF" if model_type_str == 'seq' else "ABF"
        if config['extended']:
            formulation_name += "-A"
        if config['decomposition']:
            formulation_name += "-D"
        if config['seq_cut_type']:
            if config['seq_cut_type'] == 'dual+enumerative':
                formulation_name += "-DE"
            else:
                formulation_name += f"-{config['seq_cut_type'][:4]}"

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
                        seq_cut_type=config['seq_cut_type'],
                        verbosity=args.verbosity
                    )

                    for i in range(args.amount_of_instances):
                        row_data = {
                            "Formulation": formulation_name,
                            "n": n_nodes_val,
                            "r_min": r_min_val,
                            "r_max": r_max_val,
                            "Instance": i,
                            "Status": instance_results['status'][i],
                            "Gap": instance_results['gap'][i],
                            "Time": instance_results['runtime'][i],
                            "Cuts": instance_results['cuts'][i],
                        }
                        final_results.append(row_data)
                    
        if args.verbosity == 'low':
            print(f"Finished {formulation_name}")

    results_df = pd.DataFrame(final_results)
    
    # Ensure 'Results' directory exists
    os.makedirs("Results", exist_ok=True)
    results_df.to_csv("Results/experiment_results.csv", index=False)
    
    print("\nExperiment finished. Results saved to Results/experiment_results.csv")