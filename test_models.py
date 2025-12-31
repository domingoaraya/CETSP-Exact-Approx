import numpy as np
from classes.data_handling import CETSPData
from classes.models import CETSPModel
from classes.plotting import Plotter
from gurobipy import GRB

def main():
    """
    Main function to run the CETSP solver and visualize the solution.
    """
    # --- 1. Define the CETSP instance ---
    np.random.seed(42)  # for reproducibility
    n = 6  # number of regions
    centers = {i: (np.random.uniform(0, 10), np.random.uniform(0, 10)) for i in range(n)}
    radii = {i: np.random.uniform(0.5, 1.5) for i in range(n)}
    
    # --- 2. Create the data object ---
    cetsp_data = CETSPData(n, centers, radii)

    # --- 3. Define experiment configurations ---
    configurations = []
    for model_type in ['arc', 'seq']:
        for decomposition in [False, True]:
            for extended in [False, True]:
                if model_type == 'seq' and decomposition:
                    for seq_cut_type in ['dual', 'enumerative']:
                        configurations.append({
                            'model_type': model_type,
                            'decomposition': decomposition,
                            'extended': extended,
                            'seq_cut_type': seq_cut_type
                        })
                else:
                    configurations.append({
                        'model_type': model_type,
                        'decomposition': decomposition,
                        'extended': extended,
                        'seq_cut_type': None
                    })

    # --- 4. Run experiment ---
    for config in configurations:
        model_type = config['model_type']
        use_decomposition = config['decomposition']
        use_extended = config['extended']
        seq_cut_type = config['seq_cut_type']
        nu = 5 if use_extended else None

        print("-" * 50)
        print(f"Running configuration:")
        print(f"  - Model Type: {model_type}")
        print(f"  - Decomposition: {use_decomposition}")
        print(f"  - Extended: {use_extended}")
        if seq_cut_type:
            print(f"  - Seq Cut Type: {seq_cut_type}")
        
        cetsp_model = CETSPModel(
            cetsp_data, 
            model_type=model_type,
            decomposition=use_decomposition,
            extended=use_extended,
            nu=nu,
            seq_cut_type=seq_cut_type
        )
        cetsp_model.build()

        print("Starting optimization...")
        cetsp_model.optimize(time_limit=60)
        print("Optimization finished.")

        solution_summary = cetsp_model.get_solution_summary()
        print("\nSolution Summary:")
        if isinstance(solution_summary, dict):
            print(f"  - Objective Value: {solution_summary['objective_value']:.4f}")
            print(f"  - Runtime: {solution_summary['runtime']:.4f} seconds")
            print(f"  - GAP: {solution_summary['gap']:.4f}")
            print(f"  - Number of Cuts Added: {solution_summary['cuts_added']}")
            print(f"  - Status: {solution_summary['status']}")

            # Plot the solution
            plotter = Plotter(cetsp_model)
            plotter.plot_solution(file_path=None)
            print(f"Solution plotted successfully.")
        else:
            print(f"  - {solution_summary}")
        print("-" * 50)

if __name__ == "__main__":
    main()