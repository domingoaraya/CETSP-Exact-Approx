from gurobipy import Model, GRB, LinExpr, quicksum
import numpy as np
import sys

from classes.cut_coefficients import MAX_ITER, fill_coefficients, pack_instance

# Flag to warn only once when Gurobi ObjVal disagrees with the stored objective.
_objval_mismatch_warned = False

# Numerical focus for arc SOCPs to avoid numerical issues. Value selected after careful testing. 
_ARC_NUMERIC_FOCUS = 2


class CETSP_L2_Solver:
    """
    A solver for the Close Enough Traveling Salesperson Problem (CETSP) with L2 norm.
    This class is responsible for building the mathematical model in Gurobi.
    """

    def __init__(self, model, data, model_type, decomposition=False, extended=False, nu=None,
                 optimize_coefficients=False, threads=0):
        """
        Initializes the CETSP_L2_Solver.

        Args:
            model (gurobipy.Model): The Gurobi model object.
            data (CETSPData): The data for the CETSP instance.
            model_type (str): The type of model to build ('arc' or 'seq').
            decomposition (bool): Whether to use decomposition.
            extended (bool): Whether to use the extended formulation.
            nu (int): The parameter for the extended formulation.
            optimize_coefficients (bool): Minimise the perspective dual cut's
                coefficients on arcs outside the support instead of using the
                trivial gamma = 0 completion.
            threads (int): Gurobi Threads for every model built here. 0 lets
                Gurobi choose.
        """
        self.model = model
        self.data = data
        self.model_type = model_type
        self.decomposition = decomposition
        self.extended = extended
        self.nu = nu
        self.n = data.n
        self.threads = threads
        self.optimize_coefficients = optimize_coefficients and model_type == 'perspective'
        if self.optimize_coefficients:
            self._centers, self._radii = pack_instance(data)

    def build(self):
        """
        Builds the CETSP model by creating variables, constraints, and the objective function.
        """
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('Threads', self.threads)

        if self.model_type == 'arc' and not self.extended and not self.decomposition:
            # Avoid numerical issues in arc-based SOCPs
            self.model.setParam('NumericFocus', _ARC_NUMERIC_FOCUS)

        if self.model_type == "B&S":
            self._build_bs_master_model()
        elif self.decomposition:
            self._build_decomposition_model()
            if self.model_type == 'perspective':
                self._prepare_dual_cut_buffers()
        else:
            self._create_variables()
            self._create_tour_constraints()

            if self.model_type == "arc":
                self._create_arc_formulation_specific_parts()
            elif self.model_type == "seq":
                self._create_seq_formulation_specific_parts()
            elif self.model_type == "perspective":
                self._create_perspective_formulation_specific_parts()
            else:
                raise ValueError("Invalid model type specified.")

            self._set_objective()

    def _prepare_dual_cut_buffers(self):
        """
        Preallocates the perspective dual cut's variable list and C array. Only
        the coefficients change between callbacks.
        """
        pairs = [(i, j) for i in range(self.n) for j in range(self.n) if i != j]
        self._cut_rows = np.array([i for i, j in pairs], dtype=np.intp)
        self._cut_cols = np.array([j for i, j in pairs], dtype=np.intp)
        self._cut_vars = [self.x[i, j] for i, j in pairs]
        if self.extended:
            self._cut_vars = self._cut_vars + [self.d[i, j] for i, j in pairs]
            self._cut_tail = [-1.0] * len(pairs)
        else:
            self._cut_tail = []
        self._C_buf = np.zeros((self.n, self.n))

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

        if self.model_type == "perspective":
            self.p_i = self.model.addVars(self.n, self.n, 2, lb=-GRB.INFINITY, name="p_i")
            self.p_j = self.model.addVars(self.n, self.n, 2, lb=-GRB.INFINITY, name="p_j")
            self.d = self.model.addVars(self.n, self.n, vtype=GRB.CONTINUOUS, name="d")
        else:
            if not (self.decomposition and not self.extended):
                # CONTINUOUS variables for the coordinates of the points in each region
                self.p_x = self.model.addVars(self.n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="p_x")
                self.p_y = self.model.addVars(self.n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="p_y")
                if self.model_type == 'arc':
                    self._apply_arc_point_bounds(self.p_x, self.p_y)

            if self.model_type == "arc":
                # CONTINUOUS variables for the distance between points
                if not (self.decomposition and not self.extended):
                    self.d = self.model.addVars(self.n, self.n, vtype=GRB.CONTINUOUS, name="d")
                # Auxiliary variables for the linearized objective function
                self.d_aux = self.model.addVars(self.n, self.n, vtype=GRB.CONTINUOUS, name="d_aux")
            elif self.model_type == "seq":
                # CONTINUOUS variables for the distance between points
                if not (self.decomposition and not self.extended):
                    self.d = self.model.addVars(self.n, vtype=GRB.CONTINUOUS, name="d")
        
        if self.decomposition:
            self.theta = self.model.addVar(vtype=GRB.CONTINUOUS, name="theta", lb = 0)


    def _apply_arc_point_bounds(self, p_x, p_y):
        """
        Bound each arc point variable to the box c_i +/- r_i to improve conditioning 
        and avoid problems with empty interior cones (depot).
        """
        for i in range(self.n):
            cx, cy = self.data.centers[i]
            r = self.data.radii[i]
            p_x[i].LB, p_x[i].UB = cx - r, cx + r
            p_y[i].LB, p_y[i].UB = cy - r, cy + r

    def _create_tour_constraints(self):
        """
        Creates the constraints to ensure a valid tour is formed.
        """
        # Each region must be visited exactly once
        self.model.addConstrs((self.x.sum(i, '*') == 1 for i in range(self.n)), name="visit_once")
        # Each position in the tour must be occupied by exactly one region
        self.model.addConstrs((self.x.sum('*', j) == 1 for j in range(self.n)), name="occupy_once")
        # No self-loops for arc-based, B&S, and perspective formulations
        if self.model_type in ['arc', 'B&S', 'perspective']:
            self.model.addConstrs((self.x[i, i] == 0 for i in range(self.n)), name="no_self_loops")
        
    def _create_neighborhood_constraints(self):
        """
        Creates the neighborhood constraints for the L2 norm.
        These constraints ensure that the chosen points lie within their respective circular regions.
        """
        if not self.extended:
            if self.model_type == 'arc':
                # radii[i] == 0 targets are already pinned to their center by
                # _apply_arc_point_bounds
                self.model.addConstrs(((self.p_x[i] - self.data.centers[i][0])**2 + (self.p_y[i] - self.data.centers[i][1])**2 <= self.data.radii[i]**2 for i in range(self.n) if self.data.radii[i] > 0.0), name="neighborhood")
            elif self.model_type == 'seq':
                # x[0,0] == 1 is fixed, if depot has radius 0, cone can be skipped
                depot_start = self.data.radii[0] == 0.0
                positions = range(1, self.n) if depot_start else range(self.n)
                if depot_start:
                    self.p_x[0].LB = self.p_x[0].UB = self.data.centers[0][0]
                    self.p_y[0].LB = self.p_y[0].UB = self.data.centers[0][1]
                self.model.addConstrs(((self.p_x[k] - quicksum(self.x[i, k] * self.data.centers[i][0] for i in range(self.n)))**2 +
                                      (self.p_y[k] - quicksum(self.x[i, k] * self.data.centers[i][1] for i in range(self.n)))**2 <=
                                      quicksum(self.x[i, k] * self.data.radii[i] for i in range(self.n))**2
                                      for k in positions), name="neighborhood")
            elif self.model_type == 'perspective':
                # A zero-radius target (the depot) turns its cone into
                # ||p - x c|| <= 0, an empty-interior constraint with no strictly
                # feasible point for the barrier. Those points can be fixed
                self.model.addConstrs(((self.p_i[i, j, 0] - self.x[i, j] * self.data.centers[i][0])**2 + (self.p_i[i, j, 1] - self.x[i, j] * self.data.centers[i][1])**2 <= (self.data.radii[i] * self.x[i, j])**2 for i in range(self.n) for j in range(self.n) if i != j and self.data.radii[i] > 0.0), name="neighborhood_p_i_perspective")
                self.model.addConstrs(((self.p_j[i, j, 0] - self.x[i, j] * self.data.centers[j][0])**2 + (self.p_j[i, j, 1] - self.x[i, j] * self.data.centers[j][1])**2 <= (self.data.radii[j] * self.x[i, j])**2 for i in range(self.n) for j in range(self.n) if i != j and self.data.radii[j] > 0.0), name="neighborhood_p_j_perspective")
                
                self.model.addConstrs((self.p_i[i, j, dim] == self.x[i, j] * self.data.centers[i][dim] for i in range(self.n) for j in range(self.n) for dim in range(2) if i != j and self.data.radii[i] == 0.0), name="neighborhood_p_i_degenerate")
                self.model.addConstrs((self.p_j[i, j, dim] == self.x[i, j] * self.data.centers[j][dim] for i in range(self.n) for j in range(self.n) for dim in range(2) if i != j and self.data.radii[j] == 0.0), name="neighborhood_p_j_degenerate")
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
            elif self.model_type == 'perspective':
                self.model.addConstrs((self.d[i, j]**2 >= (self.p_i[i, j, 0] - self.p_j[i, j, 0])**2 + (self.p_i[i, j, 1] - self.p_j[i, j, 1])**2 for i in range(self.n) for j in range(self.n) if i != j), name="distance_perspective")
        else:
            self._create_approximated_socp_constraints('distance')

    def _create_approximated_socp_constraints(self, constraint_type):
        """
        Creates polyhedral approximations of the second-order cone constraints.

        Args:
            constraint_type (str): The type of constraint to approximate ('neighborhood' or 'distance').
        """
        if constraint_type == 'neighborhood':
            if self.model_type in ['arc', 'seq']:
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

            elif self.model_type == 'perspective':
                # Neighborhood cone for node i: ||p_i^{ij} - x_{ij} c_i|| <= x_{ij} r_i
                xi_c_i = self.model.addVars(self.n, self.n, self.nu + 1, vtype=GRB.CONTINUOUS, name="xi_c_i")
                eta_c_i = self.model.addVars(self.n, self.n, self.nu + 1, vtype=GRB.CONTINUOUS, name="eta_c_i")
                # Neighborhood cone for node j: ||p_j^{ij} - x_{ij} c_j|| <= x_{ij} r_j
                xi_c_j = self.model.addVars(self.n, self.n, self.nu + 1, vtype=GRB.CONTINUOUS, name="xi_c_j")
                eta_c_j = self.model.addVars(self.n, self.n, self.nu + 1, vtype=GRB.CONTINUOUS, name="eta_c_j")

                # Rotation layers for xi_c_i and eta_c_i
                self.model.addConstrs(xi_c_i[i,j,k] == np.cos(np.pi * 2**(-(k+1))) * xi_c_i[i,j,k-1] + np.sin(np.pi * 2**(-(k+1))) * eta_c_i[i,j,k-1] for i in range(self.n) for j in range(self.n) if i != j for k in range(1, self.nu + 1))
                self.model.addConstrs(eta_c_i[i,j,k] >= -np.sin(np.pi * 2**(-(k+1))) * xi_c_i[i,j,k-1] + np.cos(np.pi * 2**(-(k+1))) * eta_c_i[i,j,k-1] for i in range(self.n) for j in range(self.n) if i != j for k in range(1, self.nu + 1))
                self.model.addConstrs(eta_c_i[i,j,k] >= np.sin(np.pi * 2**(-(k+1))) * xi_c_i[i,j,k-1] - np.cos(np.pi * 2**(-(k+1))) * eta_c_i[i,j,k-1] for i in range(self.n) for j in range(self.n) if i != j for k in range(1, self.nu + 1))

                # Rotation layers for xi_c_j and eta_c_j
                self.model.addConstrs(xi_c_j[i,j,k] == np.cos(np.pi * 2**(-(k+1))) * xi_c_j[i,j,k-1] + np.sin(np.pi * 2**(-(k+1))) * eta_c_j[i,j,k-1] for i in range(self.n) for j in range(self.n) if i != j for k in range(1, self.nu + 1))
                self.model.addConstrs(eta_c_j[i,j,k] >= -np.sin(np.pi * 2**(-(k+1))) * xi_c_j[i,j,k-1] + np.cos(np.pi * 2**(-(k+1))) * eta_c_j[i,j,k-1] for i in range(self.n) for j in range(self.n) if i != j for k in range(1, self.nu + 1))
                self.model.addConstrs(eta_c_j[i,j,k] >= np.sin(np.pi * 2**(-(k+1))) * xi_c_j[i,j,k-1] - np.cos(np.pi * 2**(-(k+1))) * eta_c_j[i,j,k-1] for i in range(self.n) for j in range(self.n) if i != j for k in range(1, self.nu + 1))

                # Base layer (k=0) for node i
                self.model.addConstrs(xi_c_i[i,j,0] >= self.p_i[i,j,0] - self.x[i,j] * self.data.centers[i][0] for i in range(self.n) for j in range(self.n) if i != j)
                self.model.addConstrs(xi_c_i[i,j,0] >= -(self.p_i[i,j,0] - self.x[i,j] * self.data.centers[i][0]) for i in range(self.n) for j in range(self.n) if i != j)
                self.model.addConstrs(eta_c_i[i,j,0] >= self.p_i[i,j,1] - self.x[i,j] * self.data.centers[i][1] for i in range(self.n) for j in range(self.n) if i != j)
                self.model.addConstrs(eta_c_i[i,j,0] >= -(self.p_i[i,j,1] - self.x[i,j] * self.data.centers[i][1]) for i in range(self.n) for j in range(self.n) if i != j)

                # Base layer (k=0) for node j
                self.model.addConstrs(xi_c_j[i,j,0] >= self.p_j[i,j,0] - self.x[i,j] * self.data.centers[j][0] for i in range(self.n) for j in range(self.n) if i != j)
                self.model.addConstrs(xi_c_j[i,j,0] >= -(self.p_j[i,j,0] - self.x[i,j] * self.data.centers[j][0]) for i in range(self.n) for j in range(self.n) if i != j)
                self.model.addConstrs(eta_c_j[i,j,0] >= self.p_j[i,j,1] - self.x[i,j] * self.data.centers[j][1] for i in range(self.n) for j in range(self.n) if i != j)
                self.model.addConstrs(eta_c_j[i,j,0] >= -(self.p_j[i,j,1] - self.x[i,j] * self.data.centers[j][1]) for i in range(self.n) for j in range(self.n) if i != j)

                # Bounding layer (k=nu) for node i
                self.model.addConstrs(xi_c_i[i,j,self.nu] <= self.data.radii[i] * self.x[i,j] for i in range(self.n) for j in range(self.n) if i != j)
                self.model.addConstrs(eta_c_i[i,j,self.nu] <= np.tan(np.pi * 2**(-(self.nu + 1))) * xi_c_i[i,j,self.nu] for i in range(self.n) for j in range(self.n) if i != j)

                # Bounding layer (k=nu) for node j
                self.model.addConstrs(xi_c_j[i,j,self.nu] <= self.data.radii[j] * self.x[i,j] for i in range(self.n) for j in range(self.n) if i != j)
                self.model.addConstrs(eta_c_j[i,j,self.nu] <= np.tan(np.pi * 2**(-(self.nu + 1))) * xi_c_j[i,j,self.nu] for i in range(self.n) for j in range(self.n) if i != j)
        
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

            elif self.model_type == 'perspective':
                # Distance cone: ||p_i^{ij} - p_j^{ij}|| <= d_{ij}
                xi_d = self.model.addVars(self.n, self.n, self.nu + 1, vtype=GRB.CONTINUOUS, name="xi_d")
                eta_d = self.model.addVars(self.n, self.n, self.nu + 1, vtype=GRB.CONTINUOUS, name="eta_d")

                # Rotation layers
                self.model.addConstrs(xi_d[i,j,k] == np.cos(np.pi * 2**(-(k+1))) * xi_d[i,j,k-1] + np.sin(np.pi * 2**(-(k+1))) * eta_d[i,j,k-1] for i in range(self.n) for j in range(self.n) if i != j for k in range(1, self.nu + 1))
                self.model.addConstrs(eta_d[i,j,k] >= -np.sin(np.pi * 2**(-(k+1))) * xi_d[i,j,k-1] + np.cos(np.pi * 2**(-(k+1))) * eta_d[i,j,k-1] for i in range(self.n) for j in range(self.n) if i != j for k in range(1, self.nu + 1))
                self.model.addConstrs(eta_d[i,j,k] >= np.sin(np.pi * 2**(-(k+1))) * xi_d[i,j,k-1] - np.cos(np.pi * 2**(-(k+1))) * eta_d[i,j,k-1] for i in range(self.n) for j in range(self.n) if i != j for k in range(1, self.nu + 1))

                # Base layer (k=0)
                self.model.addConstrs(xi_d[i,j,0] >= self.p_i[i,j,0] - self.p_j[i,j,0] for i in range(self.n) for j in range(self.n) if i != j)
                self.model.addConstrs(xi_d[i,j,0] >= -(self.p_i[i,j,0] - self.p_j[i,j,0]) for i in range(self.n) for j in range(self.n) if i != j)
                self.model.addConstrs(eta_d[i,j,0] >= self.p_i[i,j,1] - self.p_j[i,j,1] for i in range(self.n) for j in range(self.n) if i != j)
                self.model.addConstrs(eta_d[i,j,0] >= -(self.p_i[i,j,1] - self.p_j[i,j,1]) for i in range(self.n) for j in range(self.n) if i != j)

                # Bounding layer (k=nu)
                self.model.addConstrs(xi_d[i,j,self.nu] <= self.d[i,j] for i in range(self.n) for j in range(self.n) if i != j)
                self.model.addConstrs(eta_d[i,j,self.nu] <= np.tan(np.pi * 2**(-(self.nu + 1))) * xi_d[i,j,self.nu] for i in range(self.n) for j in range(self.n) if i != j)

    def _build_decomposition_model(self):
        """
        Builds the master problem for the Benders decomposition.
        """
        self._create_variables()
        self._create_tour_constraints()
        if self.model_type in ['arc', 'perspective']:
            self._create_subtour_elimination_constraints()
        elif self.model_type == "seq":
            self.model.addConstr(self.x[0, 0] == 1, name="fix_start")
        
        if self.extended:
            if self.model_type == "arc":
                self._create_arc_formulation_specific_parts()
            elif self.model_type == "seq":
                self._create_seq_formulation_specific_parts()
            elif self.model_type == "perspective":
                self._create_perspective_formulation_specific_parts()

        self._set_objective()
        
    def _subproblem_objective(self, sub_model):
        """
        Read the subproblem objective without trusting Model.ObjVal.

        Gurobi 12.0.2 has been observed to return ObjVal with the wrong sign on
        the maximization QCP subproblems while the returned point is itself
        correct and optimal. getObjective().getValue() re-evaluates the stored
        objective at that point, so it cannot disagree with the model. ObjVal is
        still read, purely to report the discrepancy if it appears.
        """
        val = sub_model.getObjective().getValue()
        try:
            reported = sub_model.ObjVal
            if abs(val - reported) > 1e-6 * max(1.0, abs(val)):
                global _objval_mismatch_warned
                if not _objval_mismatch_warned:
                    _objval_mismatch_warned = True
                    print('[CETSP] Gurobi ObjVal disagrees with the stored objective on the '
                          '%s subproblem (ObjVal=%.9g, recomputed=%.9g); using the recomputed '
                          'value. Not printed again this run.'
                          % (self.model_type, reported, val), file=sys.stderr)
        except Exception:
            pass
        return val

    def _solve_subproblem(self, x_sol, current_estimation, d_sol=None):
        """
        Solves the subproblem for a given integer solution.

        Args:
            x_sol (dict): A dictionary with the values of the x variables.
            current_estimation (float): The current distance estimation from the master problem.
            d_sol (dict): The distance variable values (used by the CBF extended decomposition).

        Returns:
            A tuple containing the subproblem objective value and the dual variables.
        """
        sub_model = Model("subproblem")
        sub_model.setParam('OutputFlag', 0)
        sub_model.setParam('Threads', self.threads)

        if self.model_type == 'arc':
            sub_model.setParam('NumericFocus', _ARC_NUMERIC_FOCUS)
            p_x = sub_model.addVars(self.n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="p_x")
            p_y = sub_model.addVars(self.n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="p_y")
            self._apply_arc_point_bounds(p_x, p_y)
            tour_arcs = []
            for i in range(self.n):
                for j in range(self.n):
                    if x_sol[i,j] > 0.5:
                        tour_arcs.append((i,j))
            
            d = sub_model.addVars(tour_arcs, vtype=GRB.CONTINUOUS, name="d")
            
            sub_model.addConstrs(((p_x[i] - self.data.centers[i][0])**2 + (p_y[i] - self.data.centers[i][1])**2 <= self.data.radii[i]**2 for i in range(self.n) if self.data.radii[i] > 0.0), name="neighborhood")
            sub_model.addConstrs((d[i,j]**2 >= (p_x[i] - p_x[j])**2 + (p_y[i] - p_y[j])**2 for i,j in tour_arcs), name="distance")

            if not self.extended:
                sub_model.setObjective(quicksum(d[i,j] - self.estimation[i,j] for i,j in tour_arcs), GRB.MINIMIZE)
            else:
                sub_model.setObjective(quicksum(d[i,j] for i,j in tour_arcs) - current_estimation, GRB.MINIMIZE)
            sub_model.optimize()
            # Primal minimization: a suboptimal solve returns Q >= Q*, and the
            # enumerative cut built from it removes a feasible point.
            if sub_model.status == GRB.OPTIMAL:
                return self._subproblem_objective(sub_model), {}
            else:
                return None, None


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
                return self._subproblem_objective(sub_model), sub_model.getAttr('X', mu)
            else:
                return None, None

        elif self.model_type == 'perspective':
            tour_arcs = [(i, j) for i in range(self.n) for j in range(self.n) if x_sol[i, j] > 0.5 and i != j]

            # Map previous and next nodes for reduced dual constraints
            nxt = {}
            prv = {}
            for i, j in tour_arcs:
                nxt[i] = j
                prv[j] = i

            gamma = sub_model.addVars(tour_arcs, 2, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="gamma")
            tau = sub_model.addVars(self.n, vtype=GRB.CONTINUOUS, lb=0, name="tau")

            sub_model.addConstrs((gamma[i,j,0]**2 + gamma[i,j,1]**2 <= 1.0 for (i,j) in tour_arcs), name="gamma_norm")
            
            # ||gamma_{i,next(i)} - gamma_{prev(i),i}|| <= tau_i
            sub_model.addConstrs(
                ((gamma[i, nxt[i], 0] - gamma[prv[i], i, 0])**2 + 
                 (gamma[i, nxt[i], 1] - gamma[prv[i], i, 1])**2 <= tau[i]**2 for i in range(self.n)), 
                name="tau_bound"
            )
            
            obj_terms = []
            for i, j in tour_arcs:
                E_ij = d_sol[i,j] if self.extended else self.estimation[i,j]
                obj_terms.append(
                    (self.data.centers[i][0] - self.data.centers[j][0]) * gamma[i,j,0] +
                    (self.data.centers[i][1] - self.data.centers[j][1]) * gamma[i,j,1] - E_ij
                )
            for i in range(self.n):
                obj_terms.append(-self.data.radii[i] * tau[i])
            
            sub_model.setObjective(quicksum(obj_terms), GRB.MAXIMIZE)
            sub_model.optimize()

            if sub_model.status == GRB.OPTIMAL:
                # Extract gamma
                gamma_vals = {(i, j, dim): gamma[i, j, dim].X for i, j in tour_arcs for dim in range(2)}
                
                # Reconstruct eta
                eta_vals = np.zeros((self.n, 2))
                for i in range(self.n):
                    for dim in range(2):
                        eta_vals[i, dim] = 0.5 * (gamma_vals[i, nxt[i], dim] + gamma_vals[prv[i], i, dim])
                
                # Reconstruct alpha and lambda
                alpha_i_vals = {}
                alpha_j_vals = {}
                lambda_i_vals = {}
                lambda_j_vals = {}
                
                for i, j in tour_arcs:
                    for dim in range(2):
                        alpha_i_vals[i, j, dim] = eta_vals[i, dim] - gamma_vals[i, j, dim]
                        alpha_j_vals[i, j, dim] = -eta_vals[j, dim] + gamma_vals[i, j, dim]
                    
                    lambda_i_vals[i, j] = float(np.sqrt(alpha_i_vals[i, j, 0]**2 + alpha_i_vals[i, j, 1]**2))
                    lambda_j_vals[i, j] = float(np.sqrt(alpha_j_vals[i, j, 0]**2 + alpha_j_vals[i, j, 1]**2))

                succ = np.full(self.n, -1, dtype=np.int64)
                for i, j in tour_arcs:
                    succ[i] = j

                duals = {
                    'eta': eta_vals,
                    'succ': succ,
                    'alpha_i': alpha_i_vals,
                    'alpha_j': alpha_j_vals,
                    'lambda_i': lambda_i_vals,
                    'lambda_j': lambda_j_vals
                }

                return self._subproblem_objective(sub_model), duals
            else:
                return None, None

        elif self.model_type == 'B&S':
            sub_model = Model("bs_subproblem")
            sub_model.setParam('OutputFlag', 0)
            sub_model.setParam('Threads', self.threads)

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

            flow_c = {}
            for i in range(1, self.n):
                in_arcs = [(k, i_arc) for (k, i_arc) in tour_arcs if i_arc == i]
                out_arcs = [(i_arc, j) for (i_arc, j) in tour_arcs if i_arc == i]

                for delta in cells_by_target[i]:
                    in_flow = quicksum(f_vars[(sigma, delta)] for k, _ in in_arcs for sigma in cells_by_target[k] if (sigma, delta) in f_vars)
                    out_flow = quicksum(f_vars[(delta, sigma)] for _, j in out_arcs for sigma in cells_by_target[j] if (delta, sigma) in f_vars)
                    flow_c[delta] = sub_model.addConstr(in_flow - out_flow == 0, name=f"flow_cons_{delta}")

            depot_out_arcs = [(i_arc, j) for (i_arc, j) in tour_arcs if i_arc == 0]
            depot_in_arcs = [(k, j_arc) for (k, j_arc) in tour_arcs if j_arc == 0]

            source_flow = quicksum(f_vars[(delta, sigma)] for _, j in depot_out_arcs for delta in cells_by_target[0] for sigma in cells_by_target[j] if (delta, sigma) in f_vars)
            sub_model.addConstr(source_flow == 1.0, name="source_flow")

            sink_flow = quicksum(f_vars[(delta, sigma)] for k, _ in depot_in_arcs for delta in cells_by_target[k] for sigma in cells_by_target[0] if (delta, sigma) in f_vars)
            sink_constr = sub_model.addConstr(sink_flow == 1.0, name="sink_flow")

            sub_model.optimize()

            if sub_model.status == GRB.OPTIMAL:
                sub_obj = self._subproblem_objective(sub_model)
                pi_tau = sink_constr.Pi
                f_sol = {key: f_vars[key].X for key in c_cap}

                # Node potentials. The depot's cell is the source when an arc
                # leaves it and the sink when an arc enters it, so it is not in
                # flow_c; the cut builder supplies 0 and pi_tau for those.
                pi = {delta: c.Pi for delta, c in flow_c.items()}

                duals = {
                    'pi_tau': pi_tau,
                    'pi': pi,
                    'f_sol': f_sol
                }
                return sub_obj, duals
            else:
                return None, None

    def _add_decomposition_cuts(self, x_sol, sub_obj, duals, cut_type='enumerative'):
        """
        Adds decomposition cuts (Benders and/or integer optimality cuts) to the master problem.

        Args:
            x_sol (dict): A dictionary with the values of the x variables.
            sub_obj (float): The objective value of the subproblem.
            duals (dict): The dual variables from the subproblem.
            cut_type (str): The type of cut to add ('dual', 'enumerative', or 'dual+enumerative').
        """

        # Avoid numerical issues
        sub_obj = max(0.0, sub_obj)

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

            self.model.cbLazy(-(sub_obj/3)*delta + sub_obj <= self.theta)
            self.model.cbLazy(-(sub_obj/3)*delta_rev + sub_obj <= self.theta)

        elif self.model_type == 'seq':
            if 'dual' in cut_type:
                if not self.extended:
                    self.model.cbLazy(quicksum(duals[i,k]*self.x[i,k] for i in range(self.n) for k in range(self.n)) <= self.theta)
                else:
                    self.model.cbLazy(quicksum(duals[i,k]*self.x[i,k] for i in range(self.n) for k in range(self.n)) - quicksum(self.d[k] for k in range(self.n)) <= self.theta)
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

        elif self.model_type == 'perspective':
            tour_arcs = [(i, j) for i in range(self.n) for j in range(self.n) if x_sol[i, j] > 0.5 and i != j]
            tour_arcs_set = set(tour_arcs)

            if 'dual' in cut_type:
                # Compute C_ij for all i != j
                eta_vals = duals['eta']
                alpha_i_vals = duals['alpha_i']
                alpha_j_vals = duals['alpha_j']
                lambda_i_vals = duals['lambda_i']
                lambda_j_vals = duals['lambda_j']

                C = self._C_buf
                if self.optimize_coefficients:
                    fill_coefficients(self._centers, self._radii, eta_vals,
                                      duals['succ'], C, MAX_ITER)
                for i in range(self.n):
                    for j in range(self.n):
                        if i == j:
                            continue
                        if (i, j) in tour_arcs_set:
                            C[i, j] = (self.data.centers[i][0] * alpha_i_vals[i, j, 0]
                                       + self.data.centers[i][1] * alpha_i_vals[i, j, 1]
                                       + self.data.radii[i] * lambda_i_vals[i, j]
                                       + self.data.centers[j][0] * alpha_j_vals[i, j, 0]
                                       + self.data.centers[j][1] * alpha_j_vals[i, j, 1]
                                       + self.data.radii[j] * lambda_j_vals[i, j])
                        elif not self.optimize_coefficients:
                            eta_i_0 = eta_vals[i, 0]
                            eta_i_1 = eta_vals[i, 1]
                            eta_j_0 = eta_vals[j, 0]
                            eta_j_1 = eta_vals[j, 1]
                            norm_eta_i = np.sqrt(eta_i_0**2 + eta_i_1**2)
                            norm_eta_j = np.sqrt(eta_j_0**2 + eta_j_1**2)
                            C[i, j] = (self.data.centers[i][0] * eta_i_0
                                       + self.data.centers[i][1] * eta_i_1
                                       + norm_eta_i * self.data.radii[i]
                                       - self.data.centers[j][0] * eta_j_0
                                       - self.data.centers[j][1] * eta_j_1
                                       + norm_eta_j * self.data.radii[j])

                if not self.extended:
                    coeffs = (-(self.estimation + C))[self._cut_rows, self._cut_cols].tolist()
                else:
                    coeffs = (-C)[self._cut_rows, self._cut_cols].tolist() + self._cut_tail
                self.model.cbLazy(LinExpr(coeffs, self._cut_vars) <= self.theta)

            if 'enumerative' in cut_type:
                delta = LinExpr()
                delta_rev = LinExpr()
                for i, j in tour_arcs:
                    delta += (1 - self.x[i, j])
                    delta_rev += (1 - self.x[j, i])

                self.model.cbLazy(-(sub_obj/3)*delta + sub_obj <= self.theta)
                self.model.cbLazy(-(sub_obj/3)*delta_rev + sub_obj <= self.theta)

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
        pi = duals.get('pi', {})

        inactive_arcs = [(i, j) for i in range(self.n) for j in range(self.n) if i != j and x_sol.get((i, j), 0) <= 0.5]

        cells_by_target = {}
        for key in self.data.bs_cells:
            cells_by_target.setdefault(key[0], []).append(key)

        eta = {}
        eta_bar = {}
        for i, j in inactive_arcs:
            m_ij = self.estimation[i, j]
            eta_ij = 0.0
            for delta in cells_by_target.get(i, []):
                pi_d = 0.0 if i == 0 else pi.get(delta, 0.0)
                for sigma in cells_by_target.get(j, []):
                    d_ds = self.data.bs_distances.get((delta, sigma))
                    if d_ds is None:
                        continue
                    pi_s = pi_tau if j == 0 else pi.get(sigma, 0.0)
                    eta_ij += max(0.0, pi_s - (d_ds - m_ij) - pi_d)
            eta[(i, j)] = eta_ij
            eta_bar[(i, j)] = min(eta_ij, pi_tau)

        rho = min(eta_bar.values()) if eta_bar else 0.0

        if 2.0 * rho <= pi_tau:
            # Case A
            u01 = [arc for arc in inactive_arcs if eta_bar[arc] >= pi_tau - rho]
            u02 = [arc for arc in inactive_arcs if eta_bar[arc] < pi_tau - rho]

            lhs = self.theta + quicksum((pi_tau - rho) * self.x[i, j] for i, j in u01) + quicksum(eta_bar[i, j] * self.x[i, j] for i, j in u02)
            rhs = pi_tau
        else:
            # Case B
            lhs = self.theta + quicksum((pi_tau / 2.0) * self.x[i, j] for i, j in inactive_arcs)
            rhs = pi_tau

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

    def _create_perspective_formulation_specific_parts(self):
        """
        Creates the constraints and variables specific to the perspective-based formulation (PBF).
        """
        self._create_subtour_elimination_constraints()
        self._create_perspective_continuity_constraints()
        self._create_neighborhood_constraints()
        self._create_distance_constraints()

    def _create_perspective_continuity_constraints(self):
        """
        Creates position continuity constraints for the PBF model.
        """
        self.model.addConstrs((quicksum(self.p_j[i, j, dim] for i in range(self.n) if i != j) == quicksum(self.p_i[j, k, dim] for k in range(self.n) if k != j) for j in range(self.n) for dim in range(2)), name="continuity_perspective")

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

        scale = 1.0
        if self.extended:
            scale = 1.0 / np.cos(np.pi / (2 ** (self.nu + 1)))

        for i in range(self.n):
            for j in range(self.n):
                dist_centers = np.sqrt((self.data.centers[i][0] - self.data.centers[j][0])**2 + (self.data.centers[i][1] - self.data.centers[j][1])**2)
                M[i,j] = self.data.radii[i] * scale + dist_centers + self.data.radii[j] * scale
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
            elif self.model_type == 'perspective':
                self.model.setObjective(self.d.sum(), GRB.MINIMIZE)
        else:
            if not self.extended:
                if self.model_type in ['arc', 'perspective']:
                    self._compute_distance_estimations()
                    self.model.setObjective(quicksum(self.x[i,j]*self.estimation[i,j] for i in range(self.n) for j in range(self.n)) + self.theta, GRB.MINIMIZE)
                elif self.model_type == 'seq':
                    self.model.setObjective(self.theta, GRB.MINIMIZE)
            else:
                if self.model_type == 'arc':
                    self.model.setObjective(self.d_aux.sum() + self.theta, GRB.MINIMIZE)
                elif self.model_type in ['seq', 'perspective']:
                    self.model.setObjective(self.d.sum() + self.theta, GRB.MINIMIZE)

