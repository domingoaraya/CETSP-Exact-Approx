from gurobipy import Model, GRB, LinExpr, quicksum
import numpy as np

class CETSP_L2_Solver:
    """
    A solver for the Close Enough Traveling Salesperson Problem (CETSP) with L2 norm.
    This class is responsible for building the mathematical model in Gurobi.
    """

    def __init__(self, model, data, model_type, decomposition=False, extended=False, nu=None):
        """
        Initializes the CETSP_L2_Solver.

        Args:
            model (gurobipy.Model): The Gurobi model object.
            data (CETSPData): The data for the CETSP instance.
            model_type (str): The type of model to build ('arc' or 'seq').
            decomposition (bool): Whether to use decomposition.
            extended (bool): Whether to use the extended formulation.
            nu (int): The parameter for the extended formulation.
        """
        self.model = model
        self.data = data
        self.model_type = model_type
        self.decomposition = decomposition
        self.extended = extended
        self.nu = nu
        self.n = data.n

    def build(self):
        """
        Builds the CETSP model by creating variables, constraints, and the objective function.
        """
        self.model.setParam('OutputFlag', 0)

        if self.model_type == "B&S":
            self._build_bs_master_model()
        elif self.decomposition:
            self._build_decomposition_model()
        else:
            self._create_variables()
            self._create_tour_constraints()

            if self.model_type == "arc":
                self._create_arc_formulation_specific_parts()
            elif self.model_type == "seq":
                self._create_seq_formulation_specific_parts()
            else:
                raise ValueError("Invalid model type specified.")

            self._set_objective()

    def _build_bs_master_model(self):
        """
        Builds the master problem for the Behdani & Smith (B&S) formulation.
        """
        self._create_variables()
        self._create_tour_constraints()
        self._set_objective()

    def _create_variables(self):
        """
        Creates the decision variables for the model.
        """
        # Binary variables indicating the sequence of visited regions
        self.x = self.model.addVars(self.n, self.n, vtype=GRB.BINARY, name="x")

        if self.model_type == "B&S":
            self.theta = self.model.addVar(vtype=GRB.CONTINUOUS, name="theta", lb=0)
            return

        if not (self.decomposition and not self.extended):
            # Continuous variables for the coordinates of the points in each region
            self.p_x = self.model.addVars(self.n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="p_x")
            self.p_y = self.model.addVars(self.n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="p_y")

        if self.model_type == "arc":
            # Continuous variables for the distance between points
            if not (self.decomposition and not self.extended):
                self.d = self.model.addVars(self.n, self.n, vtype=GRB.CONTINUOUS, name="d")
            # Auxiliary variables for the linearized objective function
            self.d_aux = self.model.addVars(self.n, self.n, vtype=GRB.CONTINUOUS, name="d_aux")
        elif self.model_type == "seq":
            # Continuous variables for the distance between points
            if not (self.decomposition and not self.extended):
                self.d = self.model.addVars(self.n, vtype=GRB.CONTINUOUS, name="d")
        
        if self.decomposition:
            self.theta = self.model.addVar(vtype=GRB.CONTINUOUS, name="theta", lb = 0)


    def _create_tour_constraints(self):
        """
        Creates the constraints to ensure a valid tour is formed.
        """
        # Each region must be visited exactly once
        self.model.addConstrs((self.x.sum(i, '*') == 1 for i in range(self.n)), name="visit_once")
        # Each position in the tour must be occupied by exactly one region
        self.model.addConstrs((self.x.sum('*', j) == 1 for j in range(self.n)), name="occupy_once")
        # No self-loops for arc-based and B&S formulations
        if self.model_type in ['arc', 'B&S']:
            self.model.addConstrs((self.x[i, i] == 0 for i in range(self.n)), name="no_self_loops")
        
    def _create_neighborhood_constraints(self):
        """
        Creates the neighborhood constraints for the L2 norm.
        These constraints ensure that the chosen points lie within their respective circular regions.
        """
        if not self.extended:
            if self.model_type == 'arc':
                self.model.addConstrs(((self.p_x[i] - self.data.centers[i][0])**2 + (self.p_y[i] - self.data.centers[i][1])**2 <= self.data.radii[i]**2 for i in range(self.n)), name="neighborhood")
            elif self.model_type == 'seq':
                self.model.addConstrs(((self.p_x[k] - quicksum(self.x[i, k] * self.data.centers[i][0] for i in range(self.n)))**2 +
                                      (self.p_y[k] - quicksum(self.x[i, k] * self.data.centers[i][1] for i in range(self.n)))**2 <=
                                      quicksum(self.x[i, k] * self.data.radii[i] for i in range(self.n))**2
                                      for k in range(self.n)), name="neighborhood")
        else:
            self._create_approximated_socp_constraints('neighborhood')

    def _create_distance_constraints(self):
        """
        Creates the distance constraints for the L2 norm.
        These constraints define the distance between consecutive points in the tour.
        """
        if not self.extended:
            if self.model_type == 'arc':
                self.model.addConstrs((self.d[i,j]**2 >= (self.p_x[i] - self.p_x[j])**2 + (self.p_y[i] - self.p_y[j])**2 for i in range(self.n) for j in range(self.n)), name="distance")
            elif self.model_type == 'seq':
                self.model.addConstrs((self.d[k]**2 >= (self.p_x[k] - self.p_x[k+1])**2 + (self.p_y[k] - self.p_y[k+1])**2 for k in range(self.n - 1)), name="distance")
                self.model.addConstr((self.d[self.n - 1]**2 >= (self.p_x[self.n - 1] - self.p_x[0])**2 + (self.p_y[self.n - 1] - self.p_y[0])**2), name="distance_wrap_around")
        else:
            self._create_approximated_socp_constraints('distance')

    def _create_approximated_socp_constraints(self, constraint_type):
        """
        Creates polyhedral approximations of the second-order cone constraints.

        Args:
            constraint_type (str): The type of constraint to approximate ('neighborhood' or 'distance').
        """
        if constraint_type == 'neighborhood':
            xi_c = self.model.addVars(self.n, self.nu + 1, vtype=GRB.CONTINUOUS, name="xi_c")
            eta_c = self.model.addVars(self.n, self.nu + 1, vtype=GRB.CONTINUOUS, name="eta_c")

            self.model.addConstrs(xi_c[k,j] == np.cos(np.pi * 2**(-(j+1))) * xi_c[k,j-1] + np.sin(np.pi * 2**(-(j+1))) * eta_c[k,j-1] for k in range(self.n) for j in range(1, self.nu + 1))
            self.model.addConstrs(eta_c[k,j] >= -np.sin(np.pi * 2**(-(j+1))) * xi_c[k,j-1] + np.cos(np.pi * 2**(-(j+1))) * eta_c[k,j-1] for k in range(self.n) for j in range(1, self.nu + 1))
            self.model.addConstrs(eta_c[k,j] >= np.sin(np.pi * 2**(-(j+1))) * xi_c[k,j-1] - np.cos(np.pi * 2**(-(j+1))) * eta_c[k,j-1] for k in range(self.n) for j in range(1, self.nu + 1))

            self.model.addConstrs(eta_c[k,self.nu] <= np.tan(np.pi * 2**(-(self.nu + 1))) * xi_c[k,self.nu] for k in range(self.n))

            if self.model_type == 'arc':
                self.model.addConstrs(xi_c[i,0] >= self.p_x[i] - self.data.centers[i][0] for i in range(self.n))
                self.model.addConstrs(xi_c[i,0] >= -self.p_x[i] + self.data.centers[i][0] for i in range(self.n))
                self.model.addConstrs(eta_c[i,0] >= self.p_y[i] - self.data.centers[i][1] for i in range(self.n))
                self.model.addConstrs(eta_c[i,0] >= -self.p_y[i] + self.data.centers[i][1] for i in range(self.n))

                self.model.addConstrs(xi_c[i,self.nu] <= self.data.radii[i] for i in range(self.n))

            elif self.model_type == 'seq':
                self.model.addConstrs(xi_c[k,0] >= sum(self.x[i,k]*self.data.centers[i][0] for i in range(self.n)) - self.p_x[k] for k in range(self.n))
                self.model.addConstrs(xi_c[k,0] >= -sum(self.x[i,k]*self.data.centers[i][0] for i in range(self.n)) + self.p_x[k] for k in range(self.n))
                self.model.addConstrs(eta_c[k,0] >= sum(self.x[i,k]*self.data.centers[i][1] for i in range(self.n)) - self.p_y[k] for k in range(self.n))
                self.model.addConstrs(eta_c[k,0] >= -sum(self.x[i,k]*self.data.centers[i][1] for i in range(self.n)) + self.p_y[k] for k in range(self.n))

                self.model.addConstrs(xi_c[k,self.nu] <= sum(self.x[i,k]*self.data.radii[i] for i in range(self.n)) for k in range(self.n))
        
        elif constraint_type == 'distance':
            if self.model_type == 'arc':
                xi_d = self.model.addVars(self.n, self.n, self.nu + 1, vtype=GRB.CONTINUOUS, name="xi_d")
                eta_d = self.model.addVars(self.n, self.n, self.nu + 1, vtype=GRB.CONTINUOUS, name="eta_d")

                self.model.addConstrs(xi_d[i,j,0] >= self.p_x[i] - self.p_x[j] for i in range(self.n) for j in range(self.n))
                self.model.addConstrs(xi_d[i,j,0] >= -self.p_x[i] + self.p_x[j] for i in range(self.n) for j in range(self.n))
                self.model.addConstrs(eta_d[i,j,0] >= self.p_y[i] - self.p_y[j] for i in range(self.n) for j in range(self.n))
                self.model.addConstrs(eta_d[i,j,0] >= -self.p_y[i] + self.p_y[j] for i in range(self.n) for j in range(self.n))

                self.model.addConstrs(xi_d[i,j,k] == np.cos(np.pi * 2**(-(k+1))) * xi_d[i,j,k-1] + np.sin(np.pi * 2**(-(k+1))) * eta_d[i,j,k-1] for i in range(self.n) for j in range(self.n) for k in range(1, self.nu + 1))
                self.model.addConstrs(eta_d[i,j,k] >= -np.sin(np.pi * 2**(-(k+1))) * xi_d[i,j,k-1] + np.cos(np.pi * 2**(-(k+1))) * eta_d[i,j,k-1] for i in range(self.n) for j in range(self.n) for k in range(1, self.nu + 1))
                self.model.addConstrs(eta_d[i,j,k] >= np.sin(np.pi * 2**(-(k+1))) * xi_d[i,j,k-1] - np.cos(np.pi * 2**(-(k+1))) * eta_d[i,j,k-1] for i in range(self.n) for j in range(self.n) for k in range(1, self.nu + 1))

                self.model.addConstrs(xi_d [i,j,self.nu] <= self.d[i,j] for i in range(self.n) for j in range(self.n))
                self.model.addConstrs(eta_d[i,j,self.nu] <= np.tan(np.pi * 2**(-(self.nu + 1))) * xi_d[i,j,self.nu] for i in range(self.n) for j in range(self.n))

            elif self.model_type == 'seq':
                xi_d = self.model.addVars(self.n, self.nu + 1, vtype=GRB.CONTINUOUS, name="xi_d")
                eta_d = self.model.addVars(self.n, self.nu + 1, vtype=GRB.CONTINUOUS, name="eta_d")

                self.model.addConstrs(xi_d[k,0] >= self.p_x[k] - self.p_x[k + 1] for k in range(self.n - 1))
                self.model.addConstr(xi_d[self.n - 1,0] >= self.p_x[self.n - 1] - self.p_x[0])
                self.model.addConstrs(xi_d[k,0] >= -self.p_x[k] + self.p_x[k + 1] for k in range(self.n - 1))
                self.model.addConstr(xi_d[self.n - 1,0] >= -self.p_x[self.n - 1] + self.p_x[0])
                self.model.addConstrs(eta_d[k,0] >= self.p_y[k] - self.p_y[k + 1] for k in range(self.n - 1))
                self.model.addConstr(eta_d[self.n - 1,0] >= self.p_y[self.n - 1] - self.p_y[0])
                self.model.addConstrs(eta_d[k,0] >= -self.p_y[k] + self.p_y[k + 1] for k in range(self.n - 1))
                self.model.addConstr(eta_d[self.n - 1,0] >= -self.p_y[self.n - 1] + self.p_y[0])

                self.model.addConstrs(xi_d[k,j] == np.cos(np.pi * 2**(-(j+1))) * xi_d[k,j-1] + np.sin(np.pi * 2**(-(j+1))) * eta_d[k,j-1] for k in range(self.n) for j in range(1, self.nu + 1))
                self.model.addConstrs(eta_d[k,j] >= -np.sin(np.pi * 2**(-(j+1))) * xi_d[k,j-1] + np.cos(np.pi * 2**(-(j+1))) * eta_d[k,j-1] for k in range(self.n) for j in range(1, self.nu + 1))
                self.model.addConstrs(eta_d[k,j] >= np.sin(np.pi * 2**(-(j+1))) * xi_d[k,j-1] - np.cos(np.pi * 2**(-(j+1))) * eta_d[k,j-1] for k in range(self.n) for j in range(1, self.nu + 1))

                self.model.addConstrs(xi_d[k,self.nu] <= self.d[k] for k in range(self.n))
                self.model.addConstrs(eta_d[k,self.nu] <= np.tan(np.pi * 2**(-(self.nu + 1))) * xi_d[k,self.nu] for k in range(self.n))

    def _build_decomposition_model(self):
        """
        Builds the master problem for the Benders decomposition.
        """
        self._create_variables()
        self._create_tour_constraints()
        if self.model_type == "arc":
            self._create_subtour_elimination_constraints()
        elif self.model_type == "seq":
            self.model.addConstr(self.x[0, 0] == 1, name="fix_start")
        
        if self.extended:
            if self.model_type == "arc":
                self._create_arc_formulation_specific_parts()
            elif self.model_type == "seq":
                self._create_seq_formulation_specific_parts()

        self._set_objective()
        
    def _solve_subproblem(self, x_sol, current_estimation):
        """
        Solves the subproblem for a given integer solution.

        Args:
            x_sol (dict): A dictionary with the values of the x variables.
            current_estimation (float): The current distance estimation from the master problem.

        Returns:
            A tuple containing the subproblem objective value and the dual variables.
        """
        sub_model = Model("subproblem")
        sub_model.setParam('OutputFlag', 0)

        if self.model_type == 'arc':
            p_x = sub_model.addVars(self.n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="p_x")
            p_y = sub_model.addVars(self.n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="p_y")
            tour_arcs = []
            for i in range(self.n):
                for j in range(self.n):
                    if x_sol[i,j] > 0.5:
                        tour_arcs.append((i,j))
            
            d = sub_model.addVars(tour_arcs, vtype=GRB.CONTINUOUS, name="d")
            
            sub_model.addConstrs(((p_x[i] - self.data.centers[i][0])**2 + (p_y[i] - self.data.centers[i][1])**2 <= self.data.radii[i]**2 for i in range(self.n)), name="neighborhood")
            sub_model.addConstrs((d[i,j]**2 >= (p_x[i] - p_x[j])**2 + (p_y[i] - p_y[j])**2 for i,j in tour_arcs), name="distance")

            if not self.extended:
                sub_model.setObjective(quicksum(d[i,j] - self.estimation[i,j] for i,j in tour_arcs), GRB.MINIMIZE)
            else:
                sub_model.setObjective(quicksum(d[i,j] for i,j in tour_arcs) - current_estimation, GRB.MINIMIZE)
            sub_model.optimize()
            if sub_model.status == GRB.OPTIMAL or sub_model.status == GRB.SUBOPTIMAL:
                return sub_model.objVal, {}
            else:
                return float('inf'), {}


        elif self.model_type == 'seq':
            lambda_n = sub_model.addVars(self.n, vtype=GRB.CONTINUOUS, name="lambda_n")
            lambda_d = sub_model.addVars(self.n, vtype=GRB.CONTINUOUS, name="lambda_d")
            lambda_c = sub_model.addVars(self.n, vtype=GRB.CONTINUOUS, name="lambda_c")

            mu = sub_model.addVars(self.n, self.n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="mu")
            rho_c = sub_model.addVars(self.n, 2, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="rho_c")
            rho_d = sub_model.addVars(self.n, 2, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="rho_d")

            sub_model.addConstrs(1 - lambda_n[k] - lambda_d[k] == 0 for k in range(self.n))
            sub_model.addConstrs(mu[i,k] + self.data.centers[i][0]*rho_c[k,0] + self.data.centers[i][1]*rho_c[k,1] + self.data.radii[i]*lambda_c[k] == 0 for i in range(self.n) for k in range(self.n))
            sub_model.addConstrs(rho_d[k,0] - rho_d[k-1,0] + rho_c[k,0] == 0 for k in range(1, self.n))
            sub_model.addConstr(rho_d[0,0] - rho_d[self.n-1,0] + rho_c[0,0] == 0)
            sub_model.addConstrs(rho_d[k,1] - rho_d[k-1,1] + rho_c[k,1] == 0 for k in range(1, self.n))
            sub_model.addConstr(rho_d[0,1] - rho_d[self.n-1,1] + rho_c[0,1] == 0)

            sub_model.addConstrs(rho_d[k,0]**2 + rho_d[k,1]**2 <= lambda_d[k]**2 for k in range(self.n))
            sub_model.addConstrs(rho_c[k,0]**2 + rho_c[k,1]**2 <= lambda_c[k]**2 for k in range(self.n))

            if not self.extended:
                sub_model.setObjective(quicksum(mu[i,k]*x_sol[i,k] for i in range(self.n) for k in range(self.n)), GRB.MAXIMIZE)
            else:
                sub_model.setObjective(quicksum(mu[i,k]*x_sol[i,k] for i in range(self.n) for k in range(self.n)) - current_estimation, GRB.MAXIMIZE)
            sub_model.optimize()

            if sub_model.status == GRB.OPTIMAL:
                return sub_model.objVal, sub_model.getAttr('X', mu)
            else:
                return float('-inf'), {}

        elif self.model_type == 'B&S':
            sub_model = Model("bs_subproblem")
            sub_model.setParam('OutputFlag', 0)

            tour_arcs = [(i, j) for i in range(self.n) for j in range(self.n) if x_sol[i, j] > 0.5 and i != j]

            cells_by_target = {}
            for i in range(self.n):
                cells_by_target[i] = [key for key in self.data.bs_cells.keys() if key[0] == i]

            f_vars = {}
            c_cap = {}

            for i, j in tour_arcs:
                m_ij = self.estimation[i, j]
                for delta in cells_by_target[i]:
                    for sigma in cells_by_target[j]:
                        d_ds = self.data.bs_distances[(delta, sigma)]
                        cost = max(d_ds - m_ij, 0.0)

                        f_var = sub_model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, obj=cost, name=f"f_{delta}_{sigma}")
                        f_vars[(delta, sigma)] = f_var
                        c_cap[(delta, sigma)] = sub_model.addConstr(f_var <= 1.0, name=f"cap_{delta}_{sigma}")

            for i in range(1, self.n):
                in_arcs = [(k, i_arc) for (k, i_arc) in tour_arcs if i_arc == i]
                out_arcs = [(i_arc, j) for (i_arc, j) in tour_arcs if i_arc == i]

                for delta in cells_by_target[i]:
                    in_flow = quicksum(f_vars[(sigma, delta)] for k, _ in in_arcs for sigma in cells_by_target[k] if (sigma, delta) in f_vars)
                    out_flow = quicksum(f_vars[(delta, sigma)] for _, j in out_arcs for sigma in cells_by_target[j] if (delta, sigma) in f_vars)
                    sub_model.addConstr(in_flow - out_flow == 0, name=f"flow_cons_{delta}")

            depot_out_arcs = [(i_arc, j) for (i_arc, j) in tour_arcs if i_arc == 0]
            depot_in_arcs = [(k, j_arc) for (k, j_arc) in tour_arcs if j_arc == 0]

            source_flow = quicksum(f_vars[(delta, sigma)] for _, j in depot_out_arcs for delta in cells_by_target[0] for sigma in cells_by_target[j] if (delta, sigma) in f_vars)
            sub_model.addConstr(source_flow == 1.0, name="source_flow")

            sink_flow = quicksum(f_vars[(delta, sigma)] for k, _ in depot_in_arcs for delta in cells_by_target[k] for sigma in cells_by_target[0] if (delta, sigma) in f_vars)
            sink_constr = sub_model.addConstr(sink_flow == 1.0, name="sink_flow")

            sub_model.optimize()

            if sub_model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
                sub_obj = sub_model.objVal
                pi_tau = sink_constr.Pi
                gamma = {}
                f_sol = {}
                for key, cap_c in c_cap.items():
                    gamma[key] = max(-cap_c.Pi, 0.0)
                    f_sol[key] = f_vars[key].X

                duals = {
                    'pi_tau': pi_tau,
                    'gamma': gamma,
                    'f_sol': f_sol
                }
                return sub_obj, duals
            else:
                return float('inf'), {}

    def _add_benders_cut(self, x_sol, sub_obj, duals, cut_type='enumerative', current_estimation=None):
        """
        Adds a Benders cut to the master problem.

        Args:
            x_sol (dict): A dictionary with the values of the x variables.
            sub_obj (float): The objective value of the subproblem.
            duals (dict): The dual variables from the subproblem.
            cut_type (str): The type of cut to add for the 'seq' model ('dual' or 'enumerative').
            current_estimation (float): The current distance estimation from the master problem.
        """
        if self.model_type == 'arc':

            tour_arcs = []
            for i in range(self.n):
                for j in range(self.n):
                    if x_sol[i,j] > 0.5:
                        tour_arcs.append((i,j))
            
            delta = LinExpr()
            delta_rev = LinExpr()
            for i,j in tour_arcs:
                delta += (1 - self.x[i,j])
                delta_rev += (1 - self.x[j,i])

            self.model.cbLazy(-(sub_obj/2)*delta + sub_obj <= self.theta)
            self.model.cbLazy(-(sub_obj/2)*delta_rev + sub_obj <= self.theta)

        elif self.model_type == 'seq':
            if 'dual' in cut_type:
                self.model.cbLazy(quicksum(duals[i,k]*self.x[i,k] for i in range(self.n) for k in range(self.n)) - current_estimation <= self.theta)
            if 'enumerative' in cut_type:
                tour_seq = []
                for i in range(self.n):
                    for k in range(self.n):
                        if x_sol[i,k] > 0.5:
                            tour_seq.append((i,k))

                delta = LinExpr()
                delta_rev = LinExpr()
                for i,k in tour_seq:
                    delta += (1 - self.x[i,k])
                    if k != 0:
                        delta_rev += (1 - self.x[i, self.n - k])
                    else:
                        delta_rev += (1 - self.x[i,k])
                
                self.model.cbLazy(-(sub_obj/2)*delta + sub_obj <= self.theta)
                self.model.cbLazy(-(sub_obj/2)*delta_rev + sub_obj <= self.theta)

        elif self.model_type == 'B&S':
            lhs, rhs = self._generate_bs_cut_expr(x_sol, duals)
            if lhs is not None:
                self.model.cbLazy(lhs >= rhs)

    def _generate_bs_cut_expr(self, x_sol, duals):
        """
        Generates the linear expression (LHS) and bound (RHS) for a Behdani & Smith (B&S) cut.
        Does not interact directly with the Gurobi model or callbacks.

        Args:
            x_sol (dict): Binary arc solution mapping (i, j) to 0 or 1.
            duals (dict): The dual variables from the subproblem.

        Returns:
            tuple: (lhs, rhs) where lhs is a Gurobi LinExpr and rhs is a float, or (None, None) if invalid.
        """
        if not duals or 'pi_tau' not in duals:
            return None, None

        pi_tau = duals.get('pi_tau', 0.0)
        gamma = duals.get('gamma', {})

        inactive_arcs = [(i, j) for i in range(self.n) for j in range(self.n) if i != j and x_sol.get((i, j), 0) <= 0.5]

        eta = {}
        eta_bar = {}
        for i, j in inactive_arcs:
            cells_i = [key for key in self.data.bs_cells.keys() if key[0] == i]
            cells_j = [key for key in self.data.bs_cells.keys() if key[0] == j]
            eta_ij = sum(gamma.get((d, s), 0.0) for d in cells_i for s in cells_j)
            eta[(i, j)] = eta_ij
            eta_bar[(i, j)] = min(eta_ij, pi_tau)

        rho = min(eta_bar.values()) if eta_bar else 0.0

        if 2.0 * rho <= pi_tau:
            # Case A
            u01 = [arc for arc in inactive_arcs if eta_bar[arc] >= pi_tau - rho]
            u02 = [arc for arc in inactive_arcs if eta_bar[arc] < pi_tau - rho]

            lhs = self.theta + quicksum((pi_tau - rho) * self.x[i, j] for i, j in u01) + quicksum(eta_bar[i, j] * self.x[i, j] for i, j in u02)
            rhs = pi_tau - rho
        else:
            # Case B
            lhs = self.theta + quicksum((pi_tau / 2.0) * self.x[i, j] for i, j in inactive_arcs)
            rhs = pi_tau / 2.0

        return lhs, rhs


    def _create_arc_formulation_specific_parts(self):
        """
        Creates the constraints and variables specific to the arc formulation.
        """
        self._create_subtour_elimination_constraints()
        self._create_neighborhood_constraints()
        self._create_distance_constraints()
        self._create_big_m_constraints()

    def _create_seq_formulation_specific_parts(self):
        """
        Creates the constraints and variables specific to the sequence formulation.
        """
        # Fix x[0,0] to 1 to break symmetry
        self.model.addConstr(self.x[0, 0] == 1, name="fix_start")
        self._create_tour_constraints()
        self._create_neighborhood_constraints()
        self._create_distance_constraints()

    def _create_subtour_elimination_constraints(self):
        """
        Creates subtour elimination constraints (SCF formulation).
        """
        u = self.model.addVars(self.n, self.n, vtype=GRB.CONTINUOUS, name="u")
        self.model.addConstrs((u.sum(i, '*') - u.sum('*', i) == -1 for i in range(1, self.n)), name="subtour_elim_flow")
        self.model.addConstr((u.sum(0, '*') - u.sum('*', 0) == self.n - 1), name="subtour_elim_source")
        self.model.addConstrs((u[i, j] <= (self.n - 1) * self.x[i, j] for i in range(self.n) for j in range(self.n)), name="subtour_elim_capacity")

    def _create_big_m_constraints(self):
        """
        Creates the Big-M constraints to linearize the objective function.
        """
        M = self._compute_big_m()
        self.model.addConstrs((self.d[i,j] - M[i,j]*(1 - self.x[i,j]) <= self.d_aux[i,j] for i in range(self.n) for j in range(self.n)), name="big_m")

    def _compute_big_m(self):
        """
        Computes the Big-M values for the linearization.
        """
        M = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                dist_centers = np.sqrt((self.data.centers[i][0] - self.data.centers[j][0])**2 + (self.data.centers[i][1] - self.data.centers[j][1])**2)
                M[i,j] = self.data.radii[i] + dist_centers + self.data.radii[j]
        return M

    def _compute_distance_estimations(self):
        """
        Computes the minimum possible distance between two neighborhoods for the L2 norm.
        """
        self.estimation = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                dist_centers = np.sqrt((self.data.centers[i][0] - self.data.centers[j][0])**2 + (self.data.centers[i][1] - self.data.centers[j][1])**2)
                self.estimation[i,j] = max(dist_centers - self.data.radii[i] - self.data.radii[j], 0)

    def _set_objective(self):
        """
        Sets the objective function for the model.
        """
        if self.model_type == 'B&S':
            self._compute_distance_estimations()
            self.model.setObjective(quicksum(self.x[i,j]*self.estimation[i,j] for i in range(self.n) for j in range(self.n)) + self.theta, GRB.MINIMIZE)
            return

        if not self.decomposition:
            if self.model_type == 'arc':
                self.model.setObjective(self.d_aux.sum(), GRB.MINIMIZE)
            elif self.model_type == 'seq':
                self.model.setObjective(self.d.sum(), GRB.MINIMIZE)
        else:
            if not self.extended:
                if self.model_type == 'arc':
                    self._compute_distance_estimations()
                    self.model.setObjective(quicksum(self.x[i,j]*self.estimation[i,j] for i in range(self.n) for j in range(self.n)) + self.theta, GRB.MINIMIZE)
                elif self.model_type == 'seq':
                    self.model.setObjective(self.theta, GRB.MINIMIZE)
            else:
                if self.model_type == 'arc':
                    self.model.setObjective(self.d_aux.sum() + self.theta, GRB.MINIMIZE)
                elif self.model_type == 'seq':
                    self.model.setObjective(self.d.sum() + self.theta, GRB.MINIMIZE)

