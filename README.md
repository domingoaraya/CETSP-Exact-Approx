# Exact and Approximate Formulations for the Close-Enough TSP

This repository contains the research code for the paper "Exact and approximate formulations for the close-enough TSP" by Domingo Araya, Gustavo Angulo, and Margarita Castro.

The CETSP is a generalization of the Euclidean TSP where each node is represented by a disk in the plane, and the objective is to find the shortest possible tour that visits at least one point within each disk.

## Research Overview

This research implements and compares multiple approaches to solve the CETSP:

- **SOCP Formulations**: Exact formulations using second-order cone programming. Three variants are proposed:
    - **Arc-Based Formulation (ABF)**: Generalizes the classic TSP formulation.
    - **Sequence-Based Formulation (SBF)**: Focuses on the sequential aspect of the tour.
    - **Perspective-Based Formulation (PBF)**: Prices each arc through a perspective reformulation of the neighborhood cones.
- **MILP Formulations**: Approximations of the SOCP models using mixed-integer linear programming, derived from polyhedral approximations of the second-order cone.
- **Decomposition Schemes**: Decomposition schemes are proposed for both SOCP and MILP formulations. The master problem determines the visit sequence, while the subproblem computes the optimal visit points within the disks.
    - **Cut coefficient optimization**: For the PBF dual cut, the coefficients of arcs outside the support of the incumbent are unconstrained by the subproblem. Minimizing over them yields a stronger cut at no cost to validity.
- **Benchmark**: The discretization-based formulation of Behdani & Smith (2014) (`B&S`) is implemented for comparison. It is inexact by construction: its bound is a relaxation and its tour is feasible but not necessarily optimal.

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
- `--model_type`: List of model types to test. Choices: `arc`, `seq`, `perspective`, `B&S` (default: `arc` `seq` `perspective`).
- `--decomposition`: Use decomposition. Choices: `True`, `False` (default: `False` `True`).
- `--extended`: Use extended formulation. Choices: `True`, `False` (default: `False` `True`).
- `--sbf_cut_type`: Cut type for sequence model decomposition. Choices: `dual`, `enumerative`, `dual+enumerative` (default: `enumerative` for SBF-A-D and `dual` for SBF-D).
- `--pbf_cut_type`: Cut type for perspective model decomposition. Same choices and defaults as above.
- `--strengthen`: Use DFJ cut strengthening at the root node. Only available for ABF and PBF. Choices: `True`, `False` (default: `False`).
- `--optimize_coefficients`: Optimize the PBF dual cut coefficients. Only applies to perspective decompositions using dual cuts; other configurations are skipped. Choices: `True`, `False` (default: `False`).
- `--threads`: Gurobi threads per model, applied to the master, the subproblems and the upper bound solve alike. 0 lets Gurobi choose, 1 forces a single thread (default: 0).
- `--verbosity`: Verbosity level for experiment output. Choices: `high`, `low` (default: `high`).

`B&S` ignores the decomposition, extended, cut type and strengthening switches, since it carries its own decomposition and cell refinement loop. It always yields a single configuration.

### Example
To run a specific experiment with a time limit of 300 seconds on instances with 10 nodes, using only the arc-based model with decomposition:
```bash
python run_experiment.py --time_limit 300 --n_nodes 10 --model_type arc --decomposition True --extended False
```

### Formulation labels

Results are keyed by a label built from the configuration, e.g. `PBF-A-D-dual-OC`:

| Element | Meaning |
|---|---|
| `ABF`, `SBF`, `PBF`, `B&S` | base formulation |
| `-A` | extended (polyhedral approximation of the cone) |
| `-D` | Benders decomposition |
| `-dual`, `-enum`, `-DE` | cut type: dual, enumerative, or both |
| `-S` | DFJ cut strengthening at the root |
| `-OC` | optimized dual cut coefficients |

## Verification

Two scripts check that the implementation is sound. Both exit non-zero on failure, so either can gate a commit.

```bash
python utils/check_formulations.py
```
Solves the same instances with every formulation variant and verifies they agree: the reported bound matches the tour actually returned, exact formulations reach the same optimum, and the relaxations bracket them. Takes a few minutes on the default grid.

```bash
python utils/check_cut_coefficients.py
```
Checks the PBF cut coefficient kernel in isolation, without solving any instance: feasibility, strength against the trivial completion, and agreement with an exact conic solve.

## Repository Structure

```
E&AF-CETSP/
├───classes/
│   ├───cut_coefficients.py
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
│   ├───check_cut_coefficients.py
│   ├───check_formulations.py
│   └───instance_handler.py
├───requirements.txt
├───results_analysis.ipynb
└───run_experiment.py
```

### Core Implementation

#### `classes/`
- **`cut_coefficients.py`**: Minimizes the PBF dual cut coefficients on arcs outside the support of the incumbent, using a projected Weiszfeld iteration. Compiled with numba and warmed up at import.
- **`data_handling.py`**: Contains the `CETSPData` class, which is used to store and manage the data for a given CETSP instance, including the cell discretization used by `B&S`.
- **`models.py`**: Defines the `CETSPModel` class, which builds and encapsulates the Gurobi models for the different formulations (ABF, SBF, PBF, B&S and their variants), and drives the callbacks and refinement loop.
- **`plotting.py`**: Contains the `Plotter` class, which provides methods for visualizing CETSP instances and their solutions.
- **`solver.py`**: Implements the solution procedures within the `CETSP_L2_Solver` class, including the decomposition schemes and cut generation logic.

#### `utils/`
- **`analysis_functions.py`**: Contains functions for analyzing the results of the experiments.
- **`check_cut_coefficients.py`**: Sanity check for the cut coefficient kernel.
- **`check_formulations.py`**: Cross-formulation sanity check.
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
-   **numba**: JIT compilation of the PBF cut coefficient kernel.
-   **scipy**: Convex hulls for the `B&S` cell discretization.
-   **networkx**: Graph handling for DFJ cut separation.
-   **matplotlib**: For plotting and data visualization.
-   **tqdm**: For displaying progress bars during experiment execution.

Pinned versions are in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Built-in Libraries used:
- `os`
- `itertools`
- `argparse`
- `copy`
- `colorsys`
- `warnings`
