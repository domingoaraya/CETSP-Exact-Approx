import time
from gurobipy import Model, GRB, quicksum
from classes.data_handling import CETSPData
from classes.solver import CETSP_L2_Solver

class CETSPModel:
    """
    A class to represent and solve a Close Enough Traveling Salesperson Problem (CETSP).
    This class orchestrates the data handling, model building, and optimization process.
    """

    def __init__(self, data: CETSPData, model_type: str, decomposition: bool = False, extended: bool = False, nu: int = 3, seq_cut_type: str = 'enumerative'):
        """
        Initializes the CETSPModel.

        Args:
            data (CETSPData): The data for the CETSP instance.
            model_type (str): The type of model to build ('arc', 'seq', or 'B&S').
            decomposition (bool): Whether to use Benders decomposition. Defaults to False.
            extended (bool): Whether to use the extended formulation for the L2 norm. Defaults to False.
            nu (int, optional): The parameter for the extended formulation. Defaults to 3.
            seq_cut_type (str, optional): The type of cut for the sequence model ('dual' or 'enumerative'). Defaults to 'enumerative'.
        """
        self.data = data
        self.model_type = model_type
        self.decomposition = decomposition
        self.extended = extended
        self.nu = nu
        self.seq_cut_type = seq_cut_type
        self.model = Model("CETSP")
        self.solver = None
        self.solution = None
        self.runtime = None
        self.gap = None
        self.arcs = None
        self.points = None
        self.cuts = 0
        self.status = None
        self.bs_history = []

    def build(self):
        """
        Builds the optimization model.
        """
        self.data.eliminate_redundancies()
        self.solver = CETSP_L2_Solver(self.model, self.data, self.model_type, self.decomposition, self.extended, self.nu)
        self.solver.build()

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
            if self.decomposition:
                # The Benders loop is managed via a callback
                self.model.Params.LazyConstraints = 1
                self.model.optimize(self._benders_callback)
            else:
                self.model.optimize()
            
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
            self.model.optimize(self._benders_callback)

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

            f_sol = duals.get('f_sol', {})
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
            if refined_sub_obj > current_estimation + 1e-5:
                lhs, rhs = self.solver._generate_bs_cut_expr(x_sol, refined_duals)
                if lhs is not None:
                    self.model.addConstr(lhs >= rhs, name=f"refined_cut_iter_{iteration}")
                    self.cuts += 1

        self.runtime = time.time() - start_time

        if self.model.solCount > 0:
            x_sol = self.model.getAttr('X', self.solver.x)
            ub_obj_val, ub_arcs, ub_points = self.compute_upper_bound(x_sol)

            if ub_obj_val != float('inf'):
                self.solution = ub_obj_val
                self.arcs = ub_arcs
                self.points = ub_points
                if self.model.objVal > 0:
                    self.gap = (ub_obj_val - self.model.objVal) / ub_obj_val
                else:
                    self.gap = 0.0
            else:
                self.solution = self.model.objVal
                self.gap = self.model.MIPGap
                self._extract_arcs_and_points()

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

    def _benders_callback(self, model, where):
        """
        Gurobi callback to add Subtour Elimination Constraints and Benders cuts.
        """
        if where == GRB.Callback.MIPSOL:
            x_sol = model.cbGetSolution(self.solver.x)

            # 1. Subtour Check FIRST for B&S model
            if self.model_type == 'B&S':
                subtours = self._find_subtours(x_sol)
                if len(subtours) > 1:
                    for S in subtours:
                        model.cbLazy(quicksum(self.solver.x[i, j] for i in S for j in S) <= len(S) - 1)
                    return # EXIT IMMEDIATELY - Do not run Benders Subproblem on disconnected tours!

            # 2. Benders Subproblem Execution (Only reached if tour is fully connected)
            current_objective = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            current_estimation = current_objective - model.cbGetSolution(self.solver.theta)
            
            # Solve the subproblem
            sub_obj, duals = self.solver._solve_subproblem(x_sol, current_estimation)

            if model.cbGetSolution(self.solver.theta) < sub_obj - 1e-6:
                self.solver._add_benders_cut(x_sol, sub_obj, duals, self.seq_cut_type, current_estimation)
                if self.model_type == 'B&S':
                    self.cuts += 1
                elif self.seq_cut_type == 'dual' and self.model_type == 'seq':
                    self.cuts += 1
                elif self.seq_cut_type == 'dual+enumerative' and self.model_type == 'seq':
                    # Both dual and enumerative cuts are added
                    self.cuts += 3
                else:
                    # Enumerative cuts are added in pairs
                    self.cuts += 2

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
        model_type_ub = 'arc' if self.model_type in ['arc', 'B&S'] else 'seq'
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

            if model_type_ub == 'arc':
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
            
            return ub_model.objVal, arcs, points
        else:
            return float('inf'), None, None

    def _retrieve_solution(self):
        """
        Retrieves the solution from the solved model.
        """
        if self.model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:

            self.status = self.model.status

            if self.model.solCount > 0:
                self.solution = self.model.objVal
                self.gap = self.model.MIPGap

                x_sol = self.model.getAttr('X', self.solver.x)

                if self.extended and not self.decomposition:
                    # For extended formulations without decomposition, solve an exact SOCP to get the feasible solution
                    ub_obj_val, ub_arcs, ub_points = self.compute_upper_bound(x_sol)
                    
                    self.solution = ub_obj_val
                    self.arcs = ub_arcs
                    self.points = ub_points
                    
                    # Recalculate gap
                    if ub_obj_val != float('inf') and self.model.objVal > 0:
                        self.gap = (ub_obj_val - self.model.objVal) / ub_obj_val
                    else:
                        self.gap = float('inf')
                
                elif self.decomposition and not self.extended:
                    # For non-extended decompositions, we have the binary variables but not the points.
                    # We solve the subproblem one last time to get the points for plotting.
                    _, self.arcs, self.points = self.compute_upper_bound(x_sol)

                else:
                    # For other cases (non-extended non-decomposed, or extended-decomposed), extract from the main model.
                    self._extract_arcs_and_points()
            else:
                print("No feasible solution found.")
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
        if self.solution is not None:
            return {
                "objective_value": self.solution,
                "runtime": self.runtime,
                "gap": self.gap,
                "status": self.status,
                "cuts_added": self.cuts,
                "arcs": self.arcs,
                "points": self.points
            }
        else:
            return "No solution available."