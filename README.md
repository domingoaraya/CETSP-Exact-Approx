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
- **Benchmark**: The discretization-based formulation of Behdani & Smith (2014) (`BS`) is implemented for comparison. It is inexact by construction: its bound is a relaxation and its tour is feasible but not necessarily optimal.

## How to Run

The experiments can be executed using the `run_experiment.py` script. The script allows for a wide range of configurations via command-line arguments.

### Basic execution
```bash
python run_experiment.py
```
This runs 12 configurations — the three formulations, each monolithic, approximated, decomposed, and both — over 135 instances (`n` of 10, 15 and 20, nine radius regimes, five each), at up to 600 s per run. It is not the configuration behind the paper results, which added `BS`, the `-S`, `-OC` and `-B` modifiers, `n` up to 30, and a 1800 s limit.

### Reproducing the reported formulations

```bash
python run_experiment.py --paper --n_nodes 15 20 25 30 --time_limit 1800 
```
`--paper` leaves only the 15 formulations used for the paper's results and ignores the formulation flags; the instance flags still apply. No single combination of the formulation flags produces exactly that set, which is why the flag exists. The remaining defaults already match the reported runs.

This runs everything sequentially. Splitting the grid across processes is recommended, each with its own `--outfile`. Each file is restartable: re-running the same command skips what is already recorded.

### Command-line parameters
Experiments can be customized using the following parameters:

- `--paper`: Run only the 15 formulations reported in the paper, ignoring the formulation flags below. The instance parameters still apply.
- `--time_limit`: Time limit for Gurobi solver in seconds (default: 600).
- `--amount_of_instances`: Number of instances to test per configuration (default: 5).
- `--nu`: Parameter for the extended formulation (default: 3).
- `--n_nodes`: List of number of nodes (n) to test (default: 10 15 20).
- `--r_mean`: List of mean radii (r) to test (default: 0.25 0.5 1).
- `--sigma`: List of sigma values for radii variation (default: 0 0.2 0.5).
- `--model_type`: List of model types to test. Choices: `arc`, `seq`, `perspective`, `BS` (default: `arc` `seq` `perspective`).
- `--decomposition`: Use decomposition. Choices: `True`, `False` (default: `False` `True`).
- `--extended`: Use extended formulation. Choices: `True`, `False` (default: `False` `True`).
- `--sbf_cut_type`: Cut type for sequence model decomposition. Choices: `dual`, `enumerative`, `dual+enumerative` (default: `enumerative` for SBF-A-D and `dual` for SBF-D).
- `--pbf_cut_type`: Cut type for perspective model decomposition. Same choices and defaults as above.
- `--strengthen`: Use DFJ cut strengthening at the root node. Only available for ABF and PBF. Choices: `True`, `False` (default: `False`).
- `--optimize_coefficients`: Optimize the PBF dual cut coefficients. Only applies to perspective decompositions using dual cuts; other configurations are skipped. Choices: `True`, `False` (default: `False`).
- `--symmetry_breaking`: Add the symmetry-breaking inequalities, fixing the tour's orientation. Choices: `True`, `False` (default: `False`).
- `--threads`: Gurobi threads per model, applied to the master, the subproblems and the upper bound solve alike. 0 lets Gurobi choose, 1 forces a single thread (default: 0).
- `--verbosity`: Verbosity level for experiment output. Choices: `high`, `low` (default: `high`).
- `--outfile`: CSV file where results are written (default: `Results/experiment_results.csv`). Each row is appended as soon as its instance finishes. If the file already exists, every `(Formulation, n, r_min, r_max, Instance)` combination already recorded in it is skipped.

`BS` ignores the decomposition, extended, cut type and strengthening switches, since it carries its own decomposition and cell refinement loop. It always yields a single configuration.

### Example
To run a specific experiment with a time limit of 300 seconds on instances with 10 nodes, using only the arc-based model with decomposition:
```bash
python run_experiment.py --time_limit 300 --n_nodes 10 --model_type arc --decomposition True --extended False
```

### Resuming an interrupted experiment
Results are appended to `--outfile` after each instance is solved. If the run is killed (Ctrl-C, job time limit, crash), simply launch the same command again with the same `--outfile`; completed runs are detected from the file and skipped. Delete the file (or point `--outfile` elsewhere) to start from scratch.

### Formulation labels

Results are keyed by a label built from the configuration, e.g. `PBF-A-D-dual-OC`:

| Element | Meaning |
|---|---|
| `ABF`, `SBF`, `PBF`, `BS` | base formulation |
| `-A` | extended (polyhedral approximation of the cone) |
| `-D` | Benders decomposition |
| `-dual`, `-enum`, `-DE` | cut type: dual, enumerative, or both |
| `-S` | DFJ cut strengthening at the root |
| `-OC` | optimized dual cut coefficients |
| `-B` | symmetry-breaking inequalities |

## Verification

Three checks, each exiting non-zero on failure. The first two are development checks: they exercise the implementation on small throwaway instances and say nothing about the runs in `Results/`. Only `check_bounds.py` reads reported results.

```bash
python utils/check_formulations.py
```
Solves small instances with all 82 configurations `run_experiment.py` can build and verifies they agree: the reported bound matches the tour actually returned, exact formulations reach the same optimum, and the relaxations bracket them. Run it after changing model building or cut generation. About 10 minutes on the default grid.

```bash
python utils/check_cut_coefficients.py
```
Checks the PBF cut coefficient kernel on random draws, without solving any instance: validity, strength against the trivial completion, and agreement with an exact conic solve. Run it after changing `classes/cut_coefficients.py`.

```bash
python utils/check_bounds.py
```
Checks a results file for bound crossovers: no run may report a lower bound above its own upper bound, and no instance may have its best lower bound above its best upper bound. Reads `Results/experiment_results_all.csv` unless given another file, and runs in a second. Run it on any results file before using its numbers.

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
│   ├───15_0.125_0.375/
│   └───...
├───Results/
│   ├───experiment_results_all.csv
│   ├───paper_results.csv
│   └───relaxation_gaps_all.csv
├───utils/
│   ├───check_bounds.py
│   ├───check_cut_coefficients.py
│   ├───check_formulations.py
│   ├───instance_handler.py
│   ├───plots.py
│   └───results.py
├───paper_results.ipynb
├───relaxation_gaps.py
├───requirements.txt
└───run_experiment.py
```

### Core Implementation

#### `classes/`
- **`cut_coefficients.py`**: Minimizes the PBF dual cut coefficients on arcs outside the support of the incumbent, using a projected Weiszfeld iteration. Compiled with numba and warmed up at import.
- **`data_handling.py`**: Contains the `CETSPData` class, which is used to store and manage the data for a given CETSP instance, including the cell discretization used by `BS`.
- **`models.py`**: Defines the `CETSPModel` class, which builds and encapsulates the Gurobi models for the different formulations (ABF, SBF, PBF, BS and their variants), and drives the callbacks and refinement loop.
- **`plotting.py`**: Contains the `Plotter` class, which provides methods for visualizing CETSP instances and their solutions.
- **`solver.py`**: Implements the solution procedures within the `CETSP_L2_Solver` class, including the decomposition schemes and cut generation logic.

#### `utils/`
- **`results.py`**: Loads a results file into a tidy frame and selects from it.
  Formulation labels are parsed into their modifiers, so a set of procedures is
  chosen by predicate (`select(df, base='PBF', decomposition=True)`) rather than
  by literal label; the radius regime is recovered from `r_min` and `r_max`, and
  the time limit inferred from the runs that hit it. Also builds the summary and
  head-to-head tables.
- **`plots.py`**: Performance profiles and head-to-head scatter plots over any
  set of procedures, with axis extents read off the data and a color per
  formulation that does not change when a figure is filtered.
- **`check_bounds.py`**: Bound-consistency check over a results file.
- **`check_cut_coefficients.py`**: Sanity check for the cut coefficient kernel.
- **`check_formulations.py`**: Cross-formulation sanity check.
- **`instance_handler.py`**: Helper functions for creating, saving, and loading CETSP instances.

### Main Scripts and Notebooks
- **`run_experiment.py`**: The main script to run the computational experiments. It allows for flexible configuration of the tests via command-line arguments.
- **`relaxation_gaps.py`**: Computes the continuous relaxation gap of each formulation on each instance, writing `Results/relaxation_gaps_all.csv`.
- **`paper_results.ipynb`**: Produces the figures and tables of the paper's results section.

## Dependencies

### Core Dependencies:
-   **Python**: Version 3.10 or higher.
-   **Gurobi Optimizer**: A valid license (e.g., academic license) is required to run the optimization models.
-   **pandas**: For data manipulation and analysis.
-   **numpy**: For numerical operations.
-   **numba**: JIT compilation of the PBF cut coefficient kernel.
-   **scipy**: Convex hulls for the `BS` cell discretization.
-   **networkx**: Graph handling for DFJ cut separation.
-   **matplotlib**: For plotting and data visualization.
-   **tqdm**: For displaying progress bars during experiment execution.
-   **openpyxl**: Read results stored as `.xlsx` rather than CSV.

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
