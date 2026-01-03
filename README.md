# Exact and Approximate Formulations for the Close-Enough TSP

This repository contains the research code for the paper "Exact and approximate formulations for the close-enough TSP" by Domingo Araya, Gustavo Angulo, and Margarita Castro.

The CETSP is a generalization of the Euclidean TSP where each node is represented by a disk in the plane, and the objective is to find the shortest possible tour that visits at least one point within each disk.

## Research Overview

This research implements and compares multiple approaches to solve the CETSP:

- **SOCP Formulations**: Exact formulations using second-order cone programming. Two variants are proposed:
    - **Arc-Based Formulation (ABF)**: Generalizes the classic TSP formulation.
    - **Sequence-Based Formulation (SBF)**: Focuses on the sequential aspect of the tour.
- **MILP Formulations**: Approximations of the SOCP models using mixed-integer linear programming, derived from polyhedral approximations of the second-order cone.
- **Decomposition Schemes**: Decomposition schemes are proposed for both SOCP and MILP formulations. The master problem determines the visit sequence, while the subproblem computes the optimal visit points within the disks.

## How to Run

The experiments can be executed using the `run_experiment.py` script. The script allows for a wide range of configurations via command-line arguments.

### Basic execution
```bash
python run_experiment.py
```
This command will run the experiments with default parameters, i.e., those used for the results presented in the paper.

### Command-line parameters
Experiments can be customized using the following parameters:

- `--time_limit`: Time limit for Gurobi solver in seconds (default: 600).
- `--amount_of_instances`: Number of instances to test per configuration (default: 5).
- `--nu`: Parameter for the extended formulation (default: 3).
- `--n_nodes`: List of number of nodes (n) to test (default: 10 15 20).
- `--r_mean`: List of mean radii (r) to test (default: 0.25 0.5 1).
- `--sigma`: List of sigma values for radii variation (default: 0 0.2 0.5).
- `--model_type`: List of model types to test. Choices: `arc`, `seq` (default: `arc` `seq`).
- `--decomposition`: Use decomposition. Choices: `True`, `False` (default: `False` `True`).
- `--extended`: Use extended formulation. Choices: `True`, `False` (default: `False` `True`).
- `--seq_cut_type`: Cut type for sequence model decomposition. Choices: `dual`, `enumerative`, `dual+enumerative` (default: `enumerative`).
- `--verbosity`: Verbosity level for experiment output. Choices: `high`, `low` (default: `high`).

### Example
To run a specific experiment with a time limit of 300 seconds on instances with 10 nodes, using only the arc-based model with decomposition:
```bash
python run_experiment.py --time_limit 300 --n_nodes 10 --model_type arc --decomposition True --extended False
```

## Repository Structure

```
E&AF-CETSP/
├───classes/
│   ├───data_handling.py
│   ├───models.py
│   ├───plotting.py
│   └───solver.py
├───Instances/
│   ├───10_0.125_0.375/
│   └───...
├───Results/
│   └───paper_results.csv
├───utils/
│   ├───analysis_functions.py
│   └───instance_handler.py
├───results_analysis.ipynb
└───run_experiment.py
```

### Core Implementation

#### `classes/`
- **`data_handling.py`**: Contains the `CETSPData` class, which is used to store and manage the data for a given CETSP instance.
- **`models.py`**: Defines the `CETSPModel` class, which builds and encapsulates the Gurobi models for the different formulations (ABF, SBF, and their variants).
- **`plotting.py`**: Contains the `Plotter` class, which provides methods for visualizing CETSP instances and their solutions.
- **`solver.py`**: Implements the solution procedures within the `CETSP_L2_Solver` class, including the decomposition schemes and cut generation logic.

#### `utils/`
- **`analysis_functions.py`**: Contains functions for analyzing the results of the experiments.
- **`instance_handler.py`**: Helper functions for creating, saving, and loading CETSP instances.

### Main Scripts and Notebooks
- **`run_experiment.py`**: The main script to run the computational experiments. It allows for flexible configuration of the tests via command-line arguments.
- **`results_analysis.ipynb`**: A Jupyter notebook for analyzing and visualizing the results from the experiments.

## Dependencies

### Core Dependencies:
-   **Python**: Version 3.10 or higher.
-   **Gurobi Optimizer**: A valid license (e.g., academic license) is required to run the optimization models.
-   **pandas**: For data manipulation and analysis.
-   **numpy**: For numerical operations.
-   **matplotlib**: For plotting and data visualization.
-   **tqdm**: For displaying progress bars during experiment execution.

### Built-in Libraries used:
- `os`
- `itertools`
- `argparse`
- `copy`
- `colorsys`
- `warnings`


