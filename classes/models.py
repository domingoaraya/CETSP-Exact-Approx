import sys
import time
from gurobipy import Model, GRB, quicksum
import networkx as nx
from classes.data_handling import CETSPData
from classes.solver import CETSP_L2_Solver

class CETSPModel:
    """
    A class to represent and solve a Close Enough Traveling Salesperson Problem (CETSP).
    This class orchestrates the data handling, model building, and optimization process.
    """

    def __init__(self, data: CETSPData, model_type: str, decomposition: bool = False, extended: bool = False, nu: int = 3, cut_type: str = 'enumerative', strengthen: bool = False):
        """
        Initializes the CETSPModel.

        Args:
            data (CETSPData): The data for the CETSP instance.
            model_type (str): The type of model to build ('arc', 'seq', 'perspective', or 'B&S').
            decomposition (bool): Whether to use Benders decomposition. Defaults to False.
            extended (bool): Whether to use the extended formulation for the L2 norm. Defaults to False.
            nu (int, optional): The parameter for the extended formulation. Defaults to 3.
            cut_type (str, optional): The type of cut for decompositions ('dual' or 'enumerative'). Defaults to 'enumerative'.
            strengthen (bool): Whether to use DFJ cut separation at the root node. Defaults to False.
        """
        self.data = data
        self.model_type = model_type
        self.decomposition = decomposition
        self.extended = extended
        self.nu = nu
        self.cut_type = cut_type if cut_type is not None else 'enumerative'
        self.strengthen = strengthen
        self.model = Model("CETSP")
        self.solver = None
        self.upper_bound = None
        self.lower_bound = None
        self.root_bound = None
        self.node_count = None
        self.runtime = None
        self.gap = None
        self.arcs = None
        self.points = None
        self.cuts = 0
        self.dfj_cuts = 0
        self.status = None
        self.bs_history = []
        self.G = None

        self.subproblem_failures = 0
        self._sub_failure_warned = False

    def build(self):
        """
        Builds the optimization model.
        """
        self.data.eliminate_redundancies()
        self.solver = CETSP_L2_Solver(self.model, self.data, self.model_type, self.decomposition, self.extended, self.nu)
        self.solver.build()

        # Initialize NetworkX graph for DFJ separation
        if self.strengthen and self.model_type in ['arc', 'perspective']:
            n = self.data.n
            self.G = nx.DiGraph()
            self.G.add_nodes_from(range(n))
            self.G.add_edges_from(
                (i, j) for i in range(n) for j in range(n) if i != j
            )
            nx.set_edge_attributes(self.G, 0.0, 'capacity')

    def optimize(self, time_limit: int = 600, max_iterations: int = 5):
        """
        Solves the optimization model.

        Args:
            time_limit (int, optional): The time limit for the solver in seconds. Defaults to 600.
            max_iterations (int, optional): Maximum cell refinement iterations for B&S. Defaults to 5.
        """
        if self.model_type == 'B&S':
            self.optimize_bs(time_limit=time_limit, max_iterations=max_iterations)
            return

        try:
            self.model.setParam('TimeLimit', time_limit)
            # PreCrush is required for cbCut() to work with presolved model
            if self.strengthen and self.model_type in ['arc', 'perspective']:
                self.model.Params.PreCrush = 1

            if self.decomposition:
                # LazyConstraints is only needed for cbLazy (decomposition/Benders cuts)
                self.model.Params.LazyConstraints = 1

            self.model.optimize(self._unified_callback)
            
            self.runtime = self.model.Runtime
            self._retrieve_solution()

        except Exception as e:
            print(f"An error occurred during optimization: {e}")

    def optimize_bs(self, time_limit: int = 600, max_iterations: int = 5):
        """
        Solves the CETSP using the Behdani & Smith (B&S) cell refinement outer loop.

        Args:
            time_limit (int, optional): Total time limit in seconds. Defaults to 600.
            max_iterations (int, optional): Maximum cell refinement iterations. Defaults to 5.
        """
        start_time = time.time()
        self.data.initialize_bs_cells()
        self.bs_history = []

        # Build Master Problem outside loop so SECs and Benders cuts persist
        self.build()
        self.model.Params.LazyConstraints = 1

        for iteration in range(max_iterations):
            time_remaining = time_limit - (time.time() - start_time)
            if time_remaining <= 0:
                break

            cuts_before = self.cuts
            self.model.setParam('TimeLimit', max(time_remaining, 0.1))
            self.model.optimize(self._unified_callback)

            self.status = self.model.status
            if self.model.status == GRB.TIME_LIMIT or self.model.solCount == 0:
                break

            x_sol = self.model.getAttr('X', self.solver.x)
            
            # Invariant Check: Subtours must be eliminated by lazy constraints in callback
            subtours = self._find_subtours(x_sol)
            assert len(subtours) == 1, "Numerical error: Invalid sequence bypassed the Benders callback."

            # Re-solve subproblem to extract active cell flow paths f_sol
            current_estimation = self.model.objVal - self.solver.theta.X
            sub_obj, duals = self.solver._solve_subproblem(x_sol, current_estimation)


            if sub_obj is None:
                # Without a solved subproblem there is no f_sol to refine cells from.
                self._record_subproblem_failure("optimize_bs initial subproblem")
                break

            f_sol = duals['f_sol']
            active_cell_edges = [edge for edge, flow in f_sol.items() if flow > 0.01]

            active_cells = set()
            for delta, sigma in active_cell_edges:
                active_cells.add(delta)
                active_cells.add(sigma)

            # Telemetry recording
            self.bs_history.append({
                'iteration': iteration,
                'runtime': time.time() - start_time,
                'lower_bound': self.model.objVal,
                'active_cells': len(active_cells),
                'cuts_added': self.cuts - cuts_before
            })

            if not active_cells:
                break

            # Refine active cells
            self.data.refine_bs_cells(list(active_cells))

            # Re-solve subproblem using the newly tightened geometry
            refined_sub_obj, refined_duals = self.solver._solve_subproblem(x_sol, current_estimation)

            if refined_sub_obj is None:
                self._record_subproblem_failure("optimize_bs refined subproblem")
            elif refined_sub_obj > current_estimation + 1e-5:
                lhs, rhs = self.solver._generate_bs_cut_expr(x_sol, refined_duals)
                if lhs is not None:
                    self.model.addConstr(lhs >= rhs, name=f"refined_cut_iter_{iteration}")
                    self.cuts += 1

        self.runtime = time.time() - start_time
        self.lower_bound = self.model.ObjBound

        if self.model.solCount > 0:
            x_sol = self.model.getAttr('X', self.solver.x)
            ub_obj_val, ub_arcs, ub_points = self.compute_upper_bound(x_sol)

            if ub_obj_val != float('inf'):
                self.upper_bound = ub_obj_val
                self.arcs = ub_arcs
                self.points = ub_points
                if self.upper_bound > 0:
                    self.gap = (self.upper_bound - self.lower_bound) / self.upper_bound
                else:
                    self.gap = 0.0
            else:
                # self.solution = self.model.objVal
                # self.gap = self.model.MIPGap
                # self._extract_arcs_and_points()
                raise Exception("The upper bound model could not be solved.")

    def _find_subtours(self, x_sol):
        """
        Finds all subtours (connected components) in the binary solution x_sol.
        
        Args:
            x_sol (dict): Binary arc solution mapping (i, j) to 0 or 1.

        Returns:
            list: List of sets, where each set contains node indices of a subtour.
        """
        adj = {i: [] for i in range(self.data.n)}
        for i in range(self.data.n):
            for j in range(self.data.n):
                if x_sol.get((i, j), 0) > 0.5:
                    adj[i].append(j)

        unvisited = set(range(self.data.n))
        subtours = []

        while unvisited:
            node = next(iter(unvisited))
            component = set()
            curr = node
            while curr in unvisited:
                unvisited.remove(curr)
                component.add(curr)
                next_nodes = adj[curr]
                if next_nodes:
                    curr = next_nodes[0]
                else:
                    break
            if component:
                subtours.append(component)

        return subtours

    def _record_subproblem_failure(self, context):
        """
        Permanent. Counts a subproblem failure and prints a one-time (per
        model instance) console warning.
        """
        self.subproblem_failures += 1
        if not self._sub_failure_warned:
            print(
                f"[CETSP] subproblem failed to solve to the required status "
                f"({context}, model_type={self.model_type}). "
                f"Further occurrences this run are counted but not printed.",
                file=sys.stderr,
            )
            self._sub_failure_warned = True


    def _unified_callback(self, model, where):
        """
        Gurobi callback for lazy constraints (MIPSOL) and DFJ user cuts (MIPNODE).
        """
        if where == GRB.Callback.MIPSOL:
            # Only run subproblem / Benders logic for decomposition or B&S models
            if self.decomposition or self.model_type == 'B&S':
                x_sol = model.cbGetSolution(self.solver.x)

                # 1. Subtour Check FIRST for B&S model
                if self.model_type == 'B&S':
                    subtours = self._find_subtours(x_sol)
                    if len(subtours) > 1:
                        for S in subtours:
                            model.cbLazy(quicksum(self.solver.x[i, j] for i in S for j in S) <= len(S) - 1)
                            self.dfj_cuts += 1
                        return  # EXIT IMMEDIATELY - Do not run subproblem on disconnected tours

                # 2. Subproblem Execution (Only reached if tour is fully connected)
                current_objective = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                current_estimation = current_objective - model.cbGetSolution(self.solver.theta)
                
                # Extract d_sol for PBF extended decomposition
                d_sol = model.cbGetSolution(self.solver.d) if self.model_type == 'perspective' and self.extended else None

                # Solve the subproblem
                sub_obj, duals = self.solver._solve_subproblem(x_sol, current_estimation, d_sol)

                if sub_obj is None:
                    self._record_subproblem_failure("MIPSOL callback")
                    return

                if sub_obj < -1e-6:
                    self._record_subproblem_failure(
                        f"negative Q(x_hat)={sub_obj:.6g}.")
                    return

                if model.cbGetSolution(self.solver.theta) < sub_obj - 1e-6:
                    self.solver._add_decomposition_cuts(x_sol, sub_obj, duals, self.cut_type)
                    if self.model_type == 'B&S':
                        self.cuts += 1
                    elif self.cut_type == 'dual' and self.model_type in ['seq', 'perspective']:
                        self.cuts += 1
                    elif self.cut_type == 'dual+enumerative' and self.model_type in ['seq', 'perspective']:
                        # Both dual and enumerative cuts are added
                        self.cuts += 3
                    else:
                        # Enumerative cuts are added in pairs
                        self.cuts += 2

        elif where == GRB.Callback.MIPNODE:
            if model.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0:
                self.root_bound = model.cbGet(GRB.Callback.MIPNODE_OBJBND)

            # DFJ fractional separation at the root node
            if not (self.strengthen and self.model_type in ['arc', 'perspective']):
                return

            # Only separate at the root node with an optimal relaxation
            if (model.cbGet(GRB.Callback.MIPNODE_NODCNT) != 0 or
                    model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL):
                return

            DFJ_TOP_K = 1  # Number of most-violated DFJ cuts to inject per callback

            n = self.data.n
            x_frac = model.cbGetNodeRel(self.solver.x)

            # Bulk-update edge capacities from fractional solution
            capacities = {
                (i, j): x_frac[i, j]
                for i in range(n) for j in range(n) if i != j
            }
            nx.set_edge_attributes(self.G, capacities, 'capacity')

            # Find all violated subsets via min s-t cuts from depot (s=0)
            violated = []
            for t in range(1, n):
                cut_value, partition = nx.minimum_cut(self.G, 0, t, capacity='capacity')
                if cut_value < 1 - 1e-4:
                    S = frozenset(partition[1])  # sink side (contains t)
                    violation = 1.0 - cut_value
                    violated.append((S, violation))

            if not violated:
                return

            # Deduplicate identical subsets, keeping max violation
            unique = {}
            for S, viol in violated:
                if S not in unique or viol > unique[S]:
                    unique[S] = viol

            # Top-K filtering: pick the most violated unique subsets
            top_k = sorted(unique.items(), key=lambda item: item[1], reverse=True)[:DFJ_TOP_K]

            # Inject DFJ cut-set inequalities: sum_{i in S, j not in S} x_{ij} >= 1
            N = set(range(n))
            for S, _ in top_k:
                S_complement = N - S
                model.cbCut(
                    quicksum(
                        self.solver.x[i, j]
                        for i in S for j in S_complement
                    ) >= 1
                )
                self.dfj_cuts += 1

    def compute_upper_bound(self, x_sol):
        """
        Computes an upper bound for the solution by solving the model with fixed integer variables.

        Args:
            x_sol (dict): A dictionary mapping the integer variables to their solution values.
        
        Returns:
            tuple: A tuple containing (upper_bound_value, arcs, points), or (float('inf'), None, None) if no solution is found.
        """
        ub_model = Model("UB_Model")
        ub_model.setParam('OutputFlag', 0)
        model_type_ub = 'perspective' if self.model_type == 'perspective' else ('arc' if self.model_type in ['arc', 'B&S'] else 'seq')
        ub_solver = CETSP_L2_Solver(ub_model, self.data, model_type_ub)
        ub_solver.build()

        # Fix integer variables
        for i in range(self.data.n):
            for j in range(self.data.n):
                ub_solver.x[i, j].lb = x_sol[i, j]
                ub_solver.x[i, j].ub = x_sol[i, j]
        
        ub_model.optimize()

        if ub_model.status == GRB.OPTIMAL:
            # Extract arcs and points from the upper bound model
            arcs = []
            points = {}

            # Retrieve x solution from ub_model
            ub_x_sol = ub_model.getAttr('X', ub_solver.x)

            if model_type_ub in ['arc', 'perspective']:
                for i in range(self.data.n):
                    for j in range(self.data.n):
                        if ub_x_sol[i, j] > 0.5:
                            arcs.append((i, j))
            elif model_type_ub == 'seq':
                tour_sequence = [0] * self.data.n
                for k in range(self.data.n):
                    for i in range(self.data.n):
                        if ub_x_sol[i, k] > 0.5:
                            tour_sequence[k] = i
                            break
                
                for k in range(self.data.n - 1):
                    arcs.append((tour_sequence[k], tour_sequence[k+1]))
                arcs.append((tour_sequence[self.data.n - 1], tour_sequence[0]))

            # Extract points if they exist in the model
            if hasattr(ub_solver, 'p_x') and hasattr(ub_solver, 'p_y'):
                ub_p_x_sol = ub_model.getAttr('X', ub_solver.p_x)
                ub_p_y_sol = ub_model.getAttr('X', ub_solver.p_y)
                if model_type_ub == 'arc':
                    for i in range(self.data.n):
                        points[i] = (ub_p_x_sol[i], ub_p_y_sol[i])
                elif model_type_ub == 'seq':
                    for k in range(self.data.n):
                        i = tour_sequence[k]
                        points[i] = (ub_p_x_sol[k], ub_p_y_sol[k])
            elif model_type_ub == 'perspective' and hasattr(ub_solver, 'p_i'):
                ub_p_i_sol = ub_model.getAttr('X', ub_solver.p_i)
                for i in range(self.data.n):
                    for j in range(self.data.n):
                        if ub_x_sol[i, j] > 0.5:
                            points[i] = (ub_p_i_sol[i, j, 0], ub_p_i_sol[i, j, 1])
            
            return ub_model.objVal, arcs, points
        else:
            return float('inf'), None, None

    def _retrieve_solution(self):
        """
        Retrieves the solution from the solved model.
        """
        if self.model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:

            self.status = self.model.status
            self.lower_bound = self.model.ObjBound

            if self.model.solCount > 0:
                x_sol = self.model.getAttr('X', self.solver.x)

                if self.extended and not self.decomposition:
                    # For extended formulations without decomposition, solve an exact SOCP to get the feasible solution
                    ub_obj_val, ub_arcs, ub_points = self.compute_upper_bound(x_sol)
                    self.upper_bound = ub_obj_val
                    self.arcs = ub_arcs
                    self.points = ub_points
                else:
                    # For all other formulations the model objective is the true UB.
                    self.upper_bound = self.model.ObjVal
                    self._extract_arcs_and_points()

                    if self.decomposition and not self.extended:
                        # For non-extended decompositions, extract points for plotting without overriding the upper bound.
                        _, _, plot_points = self.compute_upper_bound(x_sol)
                        if plot_points:
                            self.points = plot_points

                # Recalculate gap robustly
                if self.upper_bound is not None and self.upper_bound > 0 and self.upper_bound != float('inf'):
                    self.gap = (self.upper_bound - self.lower_bound) / self.upper_bound
                else:
                    self.gap = float('inf')
            else:
                self.upper_bound = float('inf')
                self.gap = float('inf')
        else:
            print(f"Optimization ended with status: {self.model.status}")

    def _extract_arcs_and_points(self):
        """
        Extracts the tour arcs and the coordinates of the points from the solution.
        """
        if not self.solver or not hasattr(self.solver, 'x'):
            return

        self.arcs = []
        self.points = {}

        x_sol = self.model.getAttr('X', self.solver.x)

        if self.model_type in ['arc', 'B&S']:
            for i in range(self.data.n):
                for j in range(self.data.n):
                    if x_sol[i, j] > 0.5:
                        self.arcs.append((i, j))
        elif self.model_type == 'seq':
            tour_sequence = [0] * self.data.n
            for k in range(self.data.n):
                for i in range(self.data.n):
                    if x_sol[i, k] > 0.5:
                        tour_sequence[k] = i
                        break
            
            for k in range(self.data.n - 1):
                self.arcs.append((tour_sequence[k], tour_sequence[k+1]))
            self.arcs.append((tour_sequence[self.data.n - 1], tour_sequence[0]))
        elif self.model_type == 'perspective':
            p_i_sol = self.model.getAttr('X', self.solver.p_i)
            for i in range(self.data.n):
                for j in range(self.data.n):
                    if i != j and x_sol[i, j] > 0.5:
                        self.arcs.append((i, j))
                        self.points[i] = (p_i_sol[i, j, 0], p_i_sol[i, j, 1])

        # Extract points if they exist in the model
        if hasattr(self.solver, 'p_x') and hasattr(self.solver, 'p_y'):
            p_x_sol = self.model.getAttr('X', self.solver.p_x)
            p_y_sol = self.model.getAttr('X', self.solver.p_y)
            if self.model_type in ['arc', 'B&S']:
                for i in range(self.data.n):
                    self.points[i] = (p_x_sol[i], p_y_sol[i])
            elif self.model_type == 'seq':
                for k in range(self.data.n):
                    i = tour_sequence[k]
                    self.points[i] = (p_x_sol[k], p_y_sol[k])

    def get_solution_summary(self):
        """
        Returns a summary of the solution.
        """
        if self.upper_bound is not None:
            return {
                "upper_bound": self.upper_bound,
                "lower_bound": self.lower_bound,
                "root_bound": self.root_bound,
                "node_count": self.node_count,
                "runtime": self.runtime,
                "gap": self.gap,
                "status": self.status,
                "cuts_added": self.cuts,
                "dfj_cuts": self.dfj_cuts,
                "subproblem_failures": self.subproblem_failures,
                "arcs": self.arcs,
                "points": self.points
            }
        else:
            return "No solution available."