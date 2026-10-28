from gurobipy import Model, GRB, LinExpr, quicksum
import numpy as np
import sys

from classes.cut_coefficients import MAX_ITER, fill_coefficients, pack_instance

# Warn only once per run when Gurobi's ObjVal disagrees with the stored objective.
_objval_mismatch_warned = False

# NumericFocus for the arc SOCPs; disabled after testing, kept for reference.
#_ARC_NUMERIC_FOCUS = 2


class CETSP_L2_Solver:
    """
    Builds the Gurobi models of the CETSP formulations (ABF, SBF, PBF, BS),
    their decomposition masters and subproblems, and the decomposition cuts.
    """

    def __init__(self, model, data, model_type, decomposition=False, extended=False, nu=None,
                 optimize_coefficients=False, threads=0, symmetry_breaking=True):
        """
        Args:
            model (gurobipy.Model): Gurobi model to build into.
            data (CETSPData): instance data.
            model_type (str): 'arc' (ABF), 'seq' (SBF), 'perspective' (PBF) or 'BS'.
            decomposition (bool): build the decomposition master instead of the full model.
            extended (bool): use the polyhedral approximation of the cones.
            nu (int): approximation parameter of the extended models.
            optimize_coefficients (bool): optimize the PBF dual cut coefficients on
                arcs outside the support (PBF with dual cuts only).
            threads (int): Gurobi threads for every model built here; 0 lets Gurobi choose.
            symmetry_breaking (bool): fix the tour orientation (see
                _create_symmetry_breaking_constraints). Must be False when x is
                fixed to a given tour.
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
        self.symmetry_breaking = symmetry_breaking
        # Gurobi status of the last subproblem that did not solve to optimality (None otherwise)
        self.last_subproblem_status = None
        if self.optimize_coefficients:
            self._centers, self._radii = pack_instance(data)

    def build(self):
        """
        Builds the full formulation, the decomposition master or the BS master.
        """
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('Threads', self.threads)

        #if self.model_type == 'arc' and not self.extended and not self.decomposition:
        #    self.model.setParam('NumericFocus', _ARC_NUMERIC_FOCUS)

        if self.model_type == "BS":
            self._build_bs_master_model()
        elif self.decomposition:
            self._build_decomposition_model()
            if self.model_type == 'perspective':
                self._prepare_dual_cut_buffers()
        else:
            self._create_variables()
            self._create_degree_constraints()

            if self.model_type == "arc":
                self._create_arc_formulation_specific_parts()
            elif self.model_type == "seq":
                self._create_seq_formulation_specific_parts()
            elif self.model_type == "perspective":
                self._create_perspective_formulation_specific_parts()
            else:
                raise ValueError("Invalid model type specified.")

            self._set_objective()

        if self.symmetry_breaking:
            self._create_symmetry_breaking_constraints()

    def build_tour_socp(self, tour_arcs):
        """
        Builds the SOCP of a fixed tour: one point per target, one distance per
        tour arc, minimize the sum. Used to price incumbents of the arc family
        and BS instead of fixing x in the full ABF, which is larger and whose
        big-M amplifies the conic tolerances (its objective has been observed
        up to 4e-4 below the true tour length).

        Args:
            tour_arcs (list): arcs (i, j) of the tour, in either orientation.
        """
        self.p_x = self.model.addVars(self.n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="p_x")
        self.p_y = self.model.addVars(self.n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="p_y")
        self._apply_arc_point_bounds(self.p_x, self.p_y)
        self.tour_arcs = list(tour_arcs)
        self.d = self.model.addVars(self.tour_arcs, vtype=GRB.CONTINUOUS, name="d")
        # radii[i] == 0 targets are pinned to their center by _apply_arc_point_bounds
        self.model.addConstrs(((self.p_x[i] - self.data.centers[i][0])**2 + (self.p_y[i] - self.data.centers[i][1])**2 <= self.data.radii[i]**2
                               for i in range(self.n) if self.data.radii[i] > 0.0), name="neighborhood")
        self.model.addConstrs((self.d[i, j]**2 >= (self.p_x[i] - self.p_x[j])**2 + (self.p_y[i] - self.p_y[j])**2
                               for (i, j) in self.tour_arcs), name="distance")
        self.model.setObjective(self.d.sum(), GRB.MINIMIZE)

    def _create_symmetry_breaking_constraints(self):
        """
        Fixes the tour orientation: the depot's successor has a smaller label
        than its predecessor, which keeps exactly one of each tour and its
        reverse (n >= 3). Arc-based models: sum_j j x_0j + 1 <= sum_i i x_i0;
        sequence model: the same on positions 1 and n-1. The implied fixings
        are set as bounds. Not for models whose x is fixed to a given tour.
        """
        n = self.n
        if n < 3:
            return
        if self.model_type == 'seq':
            succ = [self.x[i, 1] for i in range(1, n)]
            pred = [self.x[i, n - 1] for i in range(1, n)]
            self.x[n - 1, 1].ub = 0.0
            self.x[1, n - 1].ub = 0.0
        else:
            succ = [self.x[0, i] for i in range(1, n)]
            pred = [self.x[i, 0] for i in range(1, n)]
            self.x[0, n - 1].ub = 0.0
            self.x[1, 0].ub = 0.0
        self.model.addConstr(
            quicksum(i * succ[i - 1] for i in range(1, n)) + 1 <= quicksum(i * pred[i - 1] for i in range(1, n)),
            name="orientation")

    def _prepare_dual_cut_buffers(self):
        """
        Preallocates the variable list and coefficient array of the PBF dual
        cut; only the coefficients change between callbacks.
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
        Builds the master problem of the BS decomposition.
        """
        self._create_variables()
        self._create_degree_constraints()
        self._set_objective()

    def _create_variables(self):
        """
        Creates the decision variables.
        """
        # x[i, j]: arc (i, j) is used (arc-based models) or target i is at position j (seq)
        self.x = self.model.addVars(self.n, self.n, vtype=GRB.BINARY, name="x")

        if self.model_type == "BS":
            self.theta = self.model.addVar(vtype=GRB.CONTINUOUS, name="theta", lb=0)
            return

        if self.model_type == "perspective":
            self.p_i = self.model.addVars(self.n, self.n, 2, lb=-GRB.INFINITY, name="p_i")
            self.p_j = self.model.addVars(self.n, self.n, 2, lb=-GRB.INFINITY, name="p_j")
            self.d = self.model.addVars(self.n, self.n, vtype=GRB.CONTINUOUS, name="d")
        else:
            if not (self.decomposition and not self.extended):
                # Visit point of each target (arc) or of each position (seq)
                self.p_x = self.model.addVars(self.n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="p_x")
                self.p_y = self.model.addVars(self.n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="p_y")
                if self.model_type == 'arc':
                    self._apply_arc_point_bounds(self.p_x, self.p_y)

            if self.model_type == "arc":
                # Arc distance, 0 on unused arcs (activated in _create_distance_constraints)
                if not (self.decomposition and not self.extended):
                    self.d = self.model.addVars(self.n, self.n, vtype=GRB.CONTINUOUS, name="d")
            elif self.model_type == "seq":
                # Distance between consecutive positions
                if not (self.decomposition and not self.extended):
                    self.d = self.model.addVars(self.n, vtype=GRB.CONTINUOUS, name="d")
        
        if self.decomposition:
            self.theta = self.model.addVar(vtype=GRB.CONTINUOUS, name="theta", lb = 0)


    def _apply_arc_point_bounds(self, p_x, p_y):
        """
        Bounds each point to the box c_i +/- r_i; a target with r_i = 0 is
        thereby fixed to its center, which avoids an empty-interior cone.
        """
        for i in range(self.n):
            cx, cy = self.data.centers[i]
            r = self.data.radii[i]
            p_x[i].LB, p_x[i].UB = cx - r, cx + r
            p_y[i].LB, p_y[i].UB = cy - r, cy + r

    def _create_degree_constraints(self):
        """
        Creates the degree (assignment) constraints and forbids self-loops in
        the arc-based models.
        """
        self.model.addConstrs((self.x.sum(i, '*') == 1 for i in range(self.n)), name="visit_once")
        self.model.addConstrs((self.x.sum('*', j) == 1 for j in range(self.n)), name="occupy_once")
        if self.model_type in ['arc', 'BS', 'perspective']:
            self.model.addConstrs((self.x[i, i] == 0 for i in range(self.n)), name="no_self_loops")
        
    def _create_neighborhood_constraints(self):
        """
        Creates the neighborhood constraints (each visit point inside its
        disk), exact or polyhedrally approximated.
        """
        if not self.extended:
            if self.model_type == 'arc':
                # Targets with r_i = 0 are fixed by _apply_arc_point_bounds
                self.model.addConstrs(((self.p_x[i] - self.data.centers[i][0])**2 + (self.p_y[i] - self.data.centers[i][1])**2 <= self.data.radii[i]**2 for i in range(self.n) if self.data.radii[i] > 0.0), name="neighborhood")
            elif self.model_type == 'seq':
                # The depot is fixed at position 0; with r_0 = 0 its cone reduces to fixing the point
                depot_start = self.data.radii[0] == 0.0
                positions = range(1, self.n) if depot_start else range(self.n)
                if depot_start:
                    self.p_x[0].LB = self.p_x[0].UB = self.data.centers[0][0]
                    self.p_y[0].LB = self.p_y[0].UB = self.data.centers[0][1]
                # Stated on the auxiliaries q_k = p_k - sum_i x_ik c_i and
                # rho_k = sum_i x_ik r_i: the expanded form has bilinear x*p and
                # x*x terms that Gurobi does not recognize as a second-order cone.
                self.q_seq = self.model.addVars(positions, 2, lb=-GRB.INFINITY, name="q_nbhd")
                self.rho_seq = self.model.addVars(positions, lb=0.0, name="rho_nbhd")
                self.model.addConstrs((self.q_seq[k, 0] == self.p_x[k] - quicksum(self.x[i, k] * self.data.centers[i][0] for i in range(self.n))
                                       for k in positions), name="q_nbhd_x")
                self.model.addConstrs((self.q_seq[k, 1] == self.p_y[k] - quicksum(self.x[i, k] * self.data.centers[i][1] for i in range(self.n))
                                       for k in positions), name="q_nbhd_y")
                self.model.addConstrs((self.rho_seq[k] == quicksum(self.x[i, k] * self.data.radii[i] for i in range(self.n))
                                       for k in positions), name="rho_nbhd")
                self.model.addConstrs((self.q_seq[k, 0]**2 + self.q_seq[k, 1]**2 <= self.rho_seq[k]**2
                                       for k in positions), name="neighborhood")
            elif self.model_type == 'perspective':
                # Same reformulation on q = p - x c. Targets with r = 0 (the
                # depot) get the linear constraint p = x c instead of a cone.
                arcs_i = [(i, j) for i in range(self.n) for j in range(self.n) if i != j and self.data.radii[i] > 0.0]
                arcs_j = [(i, j) for i in range(self.n) for j in range(self.n) if i != j and self.data.radii[j] > 0.0]
                self.q_i = self.model.addVars(arcs_i, 2, lb=-GRB.INFINITY, name="q_i")
                self.q_j = self.model.addVars(arcs_j, 2, lb=-GRB.INFINITY, name="q_j")
                self.model.addConstrs((self.q_i[i, j, dim] == self.p_i[i, j, dim] - self.x[i, j] * self.data.centers[i][dim]
                                       for (i, j) in arcs_i for dim in range(2)), name="q_i_def")
                self.model.addConstrs((self.q_j[i, j, dim] == self.p_j[i, j, dim] - self.x[i, j] * self.data.centers[j][dim]
                                       for (i, j) in arcs_j for dim in range(2)), name="q_j_def")
                self.model.addConstrs((self.q_i[i, j, 0]**2 + self.q_i[i, j, 1]**2 <= (self.data.radii[i] * self.x[i, j])**2
                                       for (i, j) in arcs_i), name="neighborhood_p_i_perspective")
                self.model.addConstrs((self.q_j[i, j, 0]**2 + self.q_j[i, j, 1]**2 <= (self.data.radii[j] * self.x[i, j])**2
                                       for (i, j) in arcs_j), name="neighborhood_p_j_perspective")
                
                self.model.addConstrs((self.p_i[i, j, dim] == self.x[i, j] * self.data.centers[i][dim] for i in range(self.n) for j in range(self.n) for dim in range(2) if i != j and self.data.radii[i] == 0.0), name="neighborhood_p_i_degenerate")
                self.model.addConstrs((self.p_j[i, j, dim] == self.x[i, j] * self.data.centers[j][dim] for i in range(self.n) for j in range(self.n) for dim in range(2) if i != j and self.data.radii[j] == 0.0), name="neighborhood_p_j_degenerate")
        else:
            self._create_approximated_socp_constraints('neighborhood')

    def _create_distance_constraints(self):
        """
        Creates the distance constraints (distance variables at least the
        distance between consecutive visit points), exact or approximated.
        """
        if not self.extended:
            if self.model_type == 'arc':
                # ||p_i - p_j|| <= d_ij + M_ij (1 - x_ij), stated on the auxiliary
                # t_ij = d_ij + M_ij (1 - x_ij) so that the cone has no d*x terms
                M = self._compute_big_m()
                self.t_arc = self.model.addVars(self.n, self.n, lb=0.0, name="t_dist")
                self.model.addConstrs((self.t_arc[i, j] == self.d[i, j] + M[i, j] * (1 - self.x[i, j])
                                       for i in range(self.n) for j in range(self.n)), name="t_dist_def")
                self.model.addConstrs((self.t_arc[i,j]**2 >= (self.p_x[i] - self.p_x[j])**2 + (self.p_y[i] - self.p_y[j])**2 for i in range(self.n) for j in range(self.n)), name="distance")
            elif self.model_type == 'seq':
                self.model.addConstrs((self.d[k]**2 >= (self.p_x[k] - self.p_x[k+1])**2 + (self.p_y[k] - self.p_y[k+1])**2 for k in range(self.n - 1)), name="distance")
                self.model.addConstr((self.d[self.n - 1]**2 >= (self.p_x[self.n - 1] - self.p_x[0])**2 + (self.p_y[self.n - 1] - self.p_y[0])**2), name="distance_wrap_around")
            elif self.model_type == 'perspective':
                self.model.addConstrs((self.d[i, j]**2 >= (self.p_i[i, j, 0] - self.p_j[i, j, 0])**2 + (self.p_i[i, j, 1] - self.p_j[i, j, 1])**2 for i in range(self.n) for j in range(self.n) if i != j), name="distance_perspective")
        else:
            self._create_approximated_socp_constraints('distance')

    def _create_approximated_socp_constraints(self, constraint_type):
        """
        Creates the polyhedral approximation of the second-order cone
        constraints (Ben-Tal and Nemirovski, parameter nu).

        Args:
            constraint_type (str): 'neighborhood' or 'distance'.
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

                # The big-M activation enters the last layer directly (it is linear)
                M = self._compute_big_m()
                self.model.addConstrs(xi_d [i,j,self.nu] <= self.d[i,j] + M[i,j] * (1 - self.x[i,j]) for i in range(self.n) for j in range(self.n))
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
        Builds the decomposition master problem.
        """
        self._create_variables()
        self._create_degree_constraints()

        if self.extended:
            # The formulation-specific parts include subtour elimination (arc,
            # perspective) or the depot fixing (seq).
            if self.model_type == "arc":
                self._create_arc_formulation_specific_parts()
            elif self.model_type == "seq":
                self._create_seq_formulation_specific_parts()
            elif self.model_type == "perspective":
                self._create_perspective_formulation_specific_parts()
        else:
            if self.model_type in ['arc', 'perspective']:
                self._create_subtour_elimination_constraints()
            elif self.model_type == "seq":
                self.x[0, 0].lb = 1

        self._set_objective()
        
    def _subproblem_objective(self, sub_model):
        """
        Objective value of a solved subproblem, re-evaluated from the stored
        objective. Gurobi 12.0.2 has returned ObjVal with the wrong sign on the
        maximization QCP subproblems; a mismatch is reported once per run.
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
                          % (self.model_type, reported, val), file=sys.stderr, flush=True)
        except Exception:
            pass
        return val

    def _solve_subproblem(self, x_sol, current_estimation):
        """
        Solves the decomposition subproblem for an integer master solution.

        Args:
            x_sol (dict): values of the x variables.
            current_estimation (float): master objective at x_sol without theta,
                so that the subproblem value is Q = (tour length) - current_estimation
                and theta >= Q is the optimality condition.

        Returns:
            (sub_obj, duals): sub_obj = Q if the subproblem solved to optimality;
            a lower bound on Q, with last_subproblem_status set, if only a
            suboptimal solution is available (the cut built from it is still
            valid); (None, None) if there is no usable information.
        """
        self.last_subproblem_status = None
        sub_model = Model("subproblem")
        sub_model.setParam('OutputFlag', 0)
        sub_model.setParam('Threads', self.threads)

        if self.model_type == 'arc':
            #sub_model.setParam('NumericFocus', _ARC_NUMERIC_FOCUS)
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

            sub_model.setObjective(quicksum(d[i,j] for i,j in tour_arcs) - current_estimation, GRB.MINIMIZE)
            sub_model.optimize()
            # A suboptimal primal overestimates Q and would make the enumerative
            # cut invalid; its ObjBound is a safe lower bound.
            if sub_model.status == GRB.OPTIMAL:
                return self._subproblem_objective(sub_model), {}
            self.last_subproblem_status = sub_model.status
            if sub_model.status == GRB.SUBOPTIMAL:
                try:
                    q_lb = float(sub_model.ObjBound)
                except Exception:
                    q_lb = None
                if q_lb is not None and np.isfinite(q_lb):
                    return q_lb, {}
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

            # Dual of the fixed-sequence subproblem (see the paper), shifted by -current_estimation
            sub_model.setObjective(quicksum(mu[i,k]*x_sol[i,k] for i in range(self.n) for k in range(self.n)) - current_estimation, GRB.MAXIMIZE)
            sub_model.optimize()

            # Any feasible dual point gives a valid (weaker) cut
            if sub_model.status != GRB.OPTIMAL:
                self.last_subproblem_status = sub_model.status
            if sub_model.status == GRB.OPTIMAL or (sub_model.status == GRB.SUBOPTIMAL and sub_model.SolCount > 0):
                return self._subproblem_objective(sub_model), sub_model.getAttr('X', mu)
            return None, None

        elif self.model_type == 'perspective':
            tour_arcs = [(i, j) for i in range(self.n) for j in range(self.n) if x_sol[i, j] > 0.5 and i != j]

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
            
            # Reduced dual of the fixed-sequence subproblem (see the paper),
            # shifted by -current_estimation
            obj = LinExpr()
            for i, j in tour_arcs:
                obj.addTerms([self.data.centers[i][0] - self.data.centers[j][0],
                              self.data.centers[i][1] - self.data.centers[j][1]],
                             [gamma[i, j, 0], gamma[i, j, 1]])
            for i in range(self.n):
                obj.addTerms([-self.data.radii[i]], [tau[i]])
            obj.addConstant(-current_estimation)

            sub_model.setObjective(obj, GRB.MAXIMIZE)
            sub_model.optimize()

            # Any feasible dual point gives a valid (weaker) cut
            if sub_model.status != GRB.OPTIMAL:
                self.last_subproblem_status = sub_model.status
            if sub_model.status == GRB.OPTIMAL or (sub_model.status == GRB.SUBOPTIMAL and sub_model.SolCount > 0):
                gamma_vals = {(i, j, dim): gamma[i, j, dim].X for i, j in tour_arcs for dim in range(2)}
                
                # Recover eta, alpha and lambda from gamma
                eta_vals = np.zeros((self.n, 2))
                for i in range(self.n):
                    for dim in range(2):
                        eta_vals[i, dim] = 0.5 * (gamma_vals[i, nxt[i], dim] + gamma_vals[prv[i], i, dim])
                
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

        elif self.model_type == 'BS':
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

            depot_in_arcs = [(k, j_arc) for (k, j_arc) in tour_arcs if j_arc == 0]

            sink_flow = quicksum(f_vars[(delta, sigma)] for k, _ in depot_in_arcs for delta in cells_by_target[k] for sigma in cells_by_target[0] if (delta, sigma) in f_vars)
            sink_constr = sub_model.addConstr(sink_flow == 1.0, name="sink_flow")

            sub_model.optimize()

            # The BS cut needs the LP duals, available only on an optimal solve
            if sub_model.status != GRB.OPTIMAL:
                self.last_subproblem_status = sub_model.status
            if sub_model.status == GRB.OPTIMAL:
                sub_obj = self._subproblem_objective(sub_model)
                pi_tau = sink_constr.Pi
                f_sol = {key: f_vars[key].X for key in c_cap}

                # Node potentials; the depot's cell (source and sink) is not in
                # flow_c, and the cut builder uses 0 and pi_tau for it.
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
        Adds the decomposition cuts for an integer master solution as lazy
        constraints. Unless symmetry_breaking is set, each cut is also added
        for the reverse tour.

        Args:
            x_sol (dict): values of the x variables.
            sub_obj (float): subproblem value Q at x_sol (or a lower bound on it).
            duals (dict): subproblem duals, as returned by _solve_subproblem.
            cut_type (str): 'dual', 'enumerative' or 'dual+enumerative'.

        Returns:
            int: number of cuts added.
        """
        add_reverse = not self.symmetry_breaking
        n_added = 0

        # Q >= 0 on tours; clip rounding noise
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

            # Coefficient Q/3: two distinct tours differ in at least three arcs of the support
            self.model.cbLazy(-(sub_obj/3)*delta + sub_obj <= self.theta)
            n_added += 1
            if add_reverse:
                self.model.cbLazy(-(sub_obj/3)*delta_rev + sub_obj <= self.theta)
                n_added += 1

        elif self.model_type == 'seq':
            if 'dual' in cut_type:
                # The reverse sequence maps position k to (n - k) mod n (depot
                # fixed at 0); its dual is mu with the columns permuted accordingly
                positions = [lambda k: k]
                if add_reverse:
                    positions.append(lambda k: (self.n - k) % self.n)
                for pos in positions:
                    lhs = quicksum(duals[i, k] * self.x[i, pos(k)] for i in range(self.n) for k in range(self.n))
                    if self.extended:
                        lhs -= quicksum(self.d[k] for k in range(self.n))
                    self.model.cbLazy(lhs <= self.theta)
                    n_added += 1
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
                
                # Coefficient Q/2: two distinct sequences differ in at least two assignments of the support
                self.model.cbLazy(-(sub_obj/2)*delta + sub_obj <= self.theta)
                n_added += 1
                if add_reverse:
                    self.model.cbLazy(-(sub_obj/2)*delta_rev + sub_obj <= self.theta)
                    n_added += 1

        elif self.model_type == 'perspective':
            tour_arcs = [(i, j) for i in range(self.n) for j in range(self.n) if x_sol[i, j] > 0.5 and i != j]
            tour_arcs_set = set(tour_arcs)

            if 'dual' in cut_type:
                # C_ij on the support from the duals, elsewhere from the
                # optimized or the plain (gamma = 0) completion
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

                # The reverse tour's dual is (-eta, -gamma) with gamma'_ji = -gamma_ij,
                # so its coefficient matrix is the transpose of C
                for coef_mat in ((C, C.T) if add_reverse else (C,)):
                    if not self.extended:
                        coeffs = (-(self.estimation + coef_mat))[self._cut_rows, self._cut_cols].tolist()
                    else:
                        coeffs = (-coef_mat)[self._cut_rows, self._cut_cols].tolist() + self._cut_tail
                    self.model.cbLazy(LinExpr(coeffs, self._cut_vars) <= self.theta)
                    n_added += 1

            if 'enumerative' in cut_type:
                delta = LinExpr()
                delta_rev = LinExpr()
                for i, j in tour_arcs:
                    delta += (1 - self.x[i, j])
                    delta_rev += (1 - self.x[j, i])

                self.model.cbLazy(-(sub_obj/3)*delta + sub_obj <= self.theta)
                n_added += 1
                if add_reverse:
                    self.model.cbLazy(-(sub_obj/3)*delta_rev + sub_obj <= self.theta)
                    n_added += 1

        elif self.model_type == 'BS':
            lhs, rhs = self._generate_bs_cut_expr(x_sol, duals)
            if lhs is not None:
                self.model.cbLazy(lhs >= rhs)
                n_added += 1
        return n_added

    def _generate_bs_cut_expr(self, x_sol, duals):
        """
        Builds the BS cut lhs >= rhs from the subproblem duals, without
        touching the model.

        Args:
            x_sol (dict): values of the x variables.
            duals (dict): subproblem duals, as returned by _solve_subproblem.

        Returns:
            (lhs, rhs): a LinExpr and a float, or (None, None) if duals are missing.
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

    def _create_seq_formulation_specific_parts(self):
        """
        Creates the constraints and variables specific to the sequence formulation.
        """
        self.x[0, 0].lb = 1
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
        Creates the PBF continuity constraints: each target is entered and left at the same point.
        """
        self.model.addConstrs((quicksum(self.p_j[i, j, dim] for i in range(self.n) if i != j) == quicksum(self.p_i[j, k, dim] for k in range(self.n) if k != j) for j in range(self.n) for dim in range(2)), name="continuity_perspective")

    def _create_subtour_elimination_constraints(self):
        """
        Creates the single-commodity-flow subtour elimination constraints.
        """
        u = self.model.addVars(self.n, self.n, vtype=GRB.CONTINUOUS, name="u")
        self.model.addConstrs((u.sum(i, '*') - u.sum('*', i) == -1 for i in range(1, self.n)), name="subtour_elim_flow")
        self.model.addConstr((u.sum(0, '*') - u.sum('*', 0) == self.n - 1), name="subtour_elim_source")
        self.model.addConstrs((u[i, j] <= (self.n - 1) * self.x[i, j] for i in range(self.n) for j in range(self.n)), name="subtour_elim_capacity")

    def _compute_big_m(self):
        """
        Big-M of the arc distance activation: the largest distance between two
        points of disks i and j, scaled by the approximation factor in the
        extended models.
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
        Distance estimate m_ij of the master problem: the minimum distance between disks i and j.
        """
        self.estimation = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                dist_centers = np.sqrt((self.data.centers[i][0] - self.data.centers[j][0])**2 + (self.data.centers[i][1] - self.data.centers[j][1])**2)
                self.estimation[i,j] = max(dist_centers - self.data.radii[i] - self.data.radii[j], 0)

    def _set_objective(self):
        """
        Sets the objective: total distance, plus theta in the decomposition masters.
        """
        if self.model_type == 'BS':
            self._compute_distance_estimations()
            self.model.setObjective(quicksum(self.x[i,j]*self.estimation[i,j] for i in range(self.n) for j in range(self.n)) + self.theta, GRB.MINIMIZE)
            return

        if not self.decomposition:
            self.model.setObjective(self.d.sum(), GRB.MINIMIZE)
        else:
            if not self.extended:
                if self.model_type in ['arc', 'perspective']:
                    self._compute_distance_estimations()
                    self.model.setObjective(quicksum(self.x[i,j]*self.estimation[i,j] for i in range(self.n) for j in range(self.n)) + self.theta, GRB.MINIMIZE)
                elif self.model_type == 'seq':
                    self.model.setObjective(self.theta, GRB.MINIMIZE)
            else:
                self.model.setObjective(self.d.sum() + self.theta, GRB.MINIMIZE)

