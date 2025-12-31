from gurobipy import Model, GRB
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
            model_type (str): The type of model to build ('arc' or 'seq').
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

    def build(self):
        """
        Builds the optimization model.
        """
        self.data.eliminate_redundancies()
        self.solver = CETSP_L2_Solver(self.model, self.data, self.model_type, self.decomposition, self.extended, self.nu)
        self.solver.build()

    def optimize(self, time_limit: int = 600):
        """
        Solves the optimization model.

        Args:
            time_limit (int, optional): The time limit for the solver in seconds. Defaults to 3600.
        """
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
            

    def _benders_callback(self, model, where):
        """
        Gurobi callback to add Benders cuts.
        """
        if where == GRB.Callback.MIPSOL:
            x_sol = model.cbGetSolution(self.solver.x)

            # Retrieve current distance estimations 
            current_objective = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            current_estimation = current_objective - model.cbGetSolution(self.solver.theta)
            
            # Solve the subproblem
            sub_obj, duals = self.solver._solve_subproblem(x_sol, current_estimation)

            if model.cbGetSolution(self.solver.theta) < sub_obj - 1e-6:
                self.solver._add_benders_cut(x_sol, sub_obj, duals, self.seq_cut_type, current_estimation)
                if self.seq_cut_type == 'dual' and self.model_type == 'seq':
                    self.cuts += 1
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
        ub_solver = CETSP_L2_Solver(ub_model, self.data, self.model_type) # Note: decomposition and extended are False by default
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

            if self.model_type == 'arc':
                for i in range(self.data.n):
                    for j in range(self.data.n):
                        if ub_x_sol[i, j] > 0.5:
                            arcs.append((i, j))
            elif self.model_type == 'seq':
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
                if self.model_type == 'arc':
                    for i in range(self.data.n):
                        points[i] = (ub_p_x_sol[i], ub_p_y_sol[i])
                elif self.model_type == 'seq':
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

        if self.model_type == 'arc':
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
            if self.model_type == 'arc':
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