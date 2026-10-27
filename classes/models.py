import sys
import traceback
import time
from gurobipy import Model, GRB, quicksum
import networkx as nx
from classes.data_handling import CETSPData
from classes.solver import CETSP_L2_Solver

class CETSPModel:
    """
    Solves a CETSP instance with one of the formulations: builds the model,
    runs the branch-and-cut (with the decomposition callbacks) or the BS
    refinement loop, and reports bounds, gap and solution.
    """
    # Number of subproblem failures reported per instance before going quiet
    SUBPROBLEM_WARNINGS = 3

    def __init__(self, data: CETSPData, model_type: str, decomposition: bool = False, extended: bool = False, nu: int = 3, cut_type: str = 'enumerative', strengthen: bool = False,
                 optimize_coefficients: bool = False, threads: int = 0, symmetry_breaking: bool = True):
        """
        Args:
            data (CETSPData): instance data.
            model_type (str): 'arc' (ABF), 'seq' (SBF), 'perspective' (PBF) or 'BS'.
            decomposition (bool): use the decomposition instead of the full model.
            extended (bool): use the polyhedral approximation of the cones.
            nu (int): approximation parameter of the extended models.
            cut_type (str): decomposition cuts: 'dual', 'enumerative' or 'dual+enumerative'.
            strengthen (bool): separate DFJ cuts at the root node (arc and perspective).
            optimize_coefficients (bool): optimize the PBF dual cut coefficients (PBF with dual cuts only).
            threads (int): Gurobi threads for every model of the run; 0 lets Gurobi choose.
            symmetry_breaking (bool): fix the tour orientation and skip the reverse-tour cuts.
        """
        self.data = data
        self.model_type = model_type
        self.decomposition = decomposition
        self.extended = extended
        self.nu = nu
        self.cut_type = cut_type if cut_type is not None else 'enumerative'
        self.strengthen = strengthen
        self.optimize_coefficients = optimize_coefficients
        self.symmetry_breaking = symmetry_breaking
        self.threads = threads
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
        self.unverified_incumbent = False
        self.bs_history = []
        self.G = None

        self.subproblem_failures = 0

    def build(self):
        """
        Removes redundant targets and builds the model.
        """
        self.data.eliminate_redundancies()
        self.solver = CETSP_L2_Solver(self.model, self.data, self.model_type, self.decomposition, self.extended, self.nu, self.optimize_coefficients,
                                      threads=self.threads, symmetry_breaking=self.symmetry_breaking)
        self.solver.build()

        # Flow network for the DFJ separation
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
        Solves the model.

        Args:
            time_limit (int): time limit in seconds.
            max_iterations (int): cell refinement iterations of BS.
        """
        # Wall-clock time of the whole solve, including callbacks, the BS
        # refinement loop and the upper-bound SOCP
        start_time = time.time()
        try:
            if self.model_type == 'BS':
                self.optimize_bs(time_limit=time_limit, max_iterations=max_iterations)
            else:
                self.model.setParam('TimeLimit', time_limit)
                # cbCut needs PreCrush; cbLazy needs LazyConstraints
                if self.strengthen and self.model_type in ['arc', 'perspective']:
                    self.model.Params.PreCrush = 1
                if self.decomposition:
                    self.model.Params.LazyConstraints = 1

                self.model.optimize(self._unified_callback)
                self._retrieve_solution()

        except Exception as e:
            # stderr with flush: stdout is block-buffered when redirected to a
            # file and is lost if the process crashes
            print(f"An error occurred during optimization: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
        finally:
            self.runtime = time.time() - start_time

    def optimize_bs(self, time_limit: int = 600, max_iterations: int = 5):
        """
        Solves the instance with the BS decomposition and its cell refinement loop.

        Args:
            time_limit (int): total time limit in seconds.
            max_iterations (int): cell refinement iterations.
        """
        start_time = time.time()
        self.bs_history = []

        # Build the master once. The cells must be discretized after build(),
        # which renumbers the targets when it removes redundant ones.
        if self.solver is None:
            self.build()
        self.data.initialize_bs_cells()
        self.model.Params.LazyConstraints = 1

        # Best bound over the iterations; an iteration cut short by the time limit reports a weaker one
        best_bound = -float('inf')

        for iteration in range(max_iterations):
            time_remaining = time_limit - (time.time() - start_time)
            if time_remaining <= 0:
                break

            cuts_before = self.cuts
            self.model.setParam('TimeLimit', max(time_remaining, 0.1))
            self.model.optimize(self._unified_callback)

            self.status = self.model.status
            try:
                best_bound = max(best_bound, self.model.ObjBound)
            except Exception:
                pass
            if self.model.status == GRB.TIME_LIMIT or self.model.solCount == 0:
                break

            x_sol = self.model.getAttr('X', self.solver.x)
            
            # Subtours are eliminated by the lazy constraints of the callback
            subtours = self._find_subtours(x_sol)
            assert len(subtours) == 1, "Numerical error: Invalid sequence bypassed the Benders callback."

            # Re-solve the subproblem to get the active cell flow f_sol
            current_estimation = self.model.objVal - self.solver.theta.X
            sub_obj, duals = self.solver._solve_subproblem(x_sol, current_estimation)


            if sub_obj is None:
                # No f_sol to refine the cells from
                self._record_subproblem_failure("optimize_bs initial subproblem")
                break

            f_sol = duals['f_sol']
            active_cell_edges = [edge for edge, flow in f_sol.items() if flow > 0.01]

            active_cells = set()
            for delta, sigma in active_cell_edges:
                active_cells.add(delta)
                active_cells.add(sigma)

            self.bs_history.append({
                'iteration': iteration,
                'runtime': time.time() - start_time,
                'lower_bound': self.model.objVal,
                'active_cells': len(active_cells),
                'cuts_added': self.cuts - cuts_before
            })

            if not active_cells:
                break

            self.data.refine_bs_cells(list(active_cells))

            # Re-solve on the refined cells; a violated cut is kept in the model
            refined_sub_obj, refined_duals = self.solver._solve_subproblem(x_sol, current_estimation)

            if refined_sub_obj is None:
                self._record_subproblem_failure("optimize_bs refined subproblem")
            elif refined_sub_obj > self.solver.theta.X + 1e-5:
                lhs, rhs = self.solver._generate_bs_cut_expr(x_sol, refined_duals)
                if lhs is not None:
                    self.model.addConstr(lhs >= rhs, name=f"refined_cut_iter_{iteration}")
                    self.cuts += 1

        if best_bound > -float('inf'):
            self.lower_bound = best_bound
        else:
            try:
                self.lower_bound = self.model.ObjBound
            except Exception:
                self.lower_bound = 0.0
        self.node_count = self.model.NodeCount

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
        Returns the cycles of the arc solution x_sol as a list of node sets.
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
        Counts a subproblem failure and reports the first SUBPROBLEM_WARNINGS
        of them on stderr.
        """
        self.subproblem_failures += 1
        status = getattr(self.solver, 'last_subproblem_status', None)
        if self.subproblem_failures <= self.SUBPROBLEM_WARNINGS:
            print(
                f"[CETSP] subproblem failed to solve to optimality "
                f"({context}, model_type={self.model_type}, gurobi_status={status}, "
                f"failure #{self.subproblem_failures}). "
                + ("Further occurrences this run are counted but not printed."
                   if self.subproblem_failures == self.SUBPROBLEM_WARNINGS else ""),
                file=sys.stderr, flush=True,
            )


    def _unified_callback(self, model, where):
        """
        Gurobi callback: lazy cuts at MIPSOL, root bound and DFJ user cuts at
        MIPNODE. gurobipy swallows exceptions raised inside a callback, so an
        exception while separating an incumbent is recorded as a subproblem
        failure instead of silently letting the incumbent through.
        """
        if where == GRB.Callback.MIPSOL:
            try:
                self._mipsol_callback(model)
            except Exception as e:
                self._record_subproblem_failure(
                    f"MIPSOL callback raised {type(e).__name__}: {e}; incumbent accepted unverified")
        elif where == GRB.Callback.MIPNODE:
            self._mipnode_callback(model)

    def _mipsol_callback(self, model):
        """Lazy separation of an integer incumbent (decompositions and BS)."""
        if self.decomposition or self.model_type == 'BS':
            x_sol = model.cbGetSolution(self.solver.x)

            # BS: subtour elimination first; the subproblem needs a single cycle
            if self.model_type == 'BS':
                subtours = self._find_subtours(x_sol)
                if len(subtours) > 1:
                    for S in subtours:
                        model.cbLazy(quicksum(self.solver.x[i, j] for i in S for j in S) <= len(S) - 1)
                        self.dfj_cuts += 1
                    return

            # The incumbent's objective is a valid upper bound iff theta_hat >= Q,
            # with Q = tour length - (objective without theta)
            theta_hat = model.cbGetSolution(self.solver.theta)
            current_estimation = model.cbGet(GRB.Callback.MIPSOL_OBJ) - theta_hat

            sub_obj, duals = self.solver._solve_subproblem(x_sol, current_estimation)
            # weak: sub_obj is only a lower bound on Q (suboptimal subproblem)
            weak = self.solver.last_subproblem_status is not None

            # An incumbent is accepted only when theta_hat >= Q is certified.
            # Otherwise it is accepted unverified and counted as a failure; the
            # reported upper bound is recomputed from the tour afterwards.
            if sub_obj is None:
                self._record_subproblem_failure(
                    "MIPSOL callback, no subproblem solution, incumbent accepted unverified")
                return
            # Q >= 0 on the plain masters; on the extended ones sum d can
            # exceed the tour length, so a negative Q is legitimate there
            if not self.extended and sub_obj < -1e-6 * max(1.0, abs(current_estimation)):
                self._record_subproblem_failure(
                    f"MIPSOL callback, negative Q(x_hat)={sub_obj:.6g}, incumbent accepted unverified")
                return

            if theta_hat < sub_obj - 1e-6:
                # The cut is violated by the incumbent whether Q is exact or weak
                self.cuts += self.solver._add_decomposition_cuts(x_sol, sub_obj, duals, self.cut_type)
            elif weak:
                self._record_subproblem_failure(
                    "MIPSOL callback, weak cut not violated, incumbent accepted unverified")

    def _mipnode_callback(self, model):
        """Records the root bound and separates DFJ user cuts at the root node."""
        if model.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0:
            self.root_bound = model.cbGet(GRB.Callback.MIPNODE_OBJBND)

        if not (self.strengthen and self.model_type in ['arc', 'perspective']):
            return

        # Root node with an optimal relaxation only
        if (model.cbGet(GRB.Callback.MIPNODE_NODCNT) != 0 or
                model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL):
            return

        DFJ_TOP_K = 1  # most violated cuts added per callback

        n = self.data.n
        x_frac = model.cbGetNodeRel(self.solver.x)

        capacities = {
            (i, j): x_frac[i, j]
            for i in range(n) for j in range(n) if i != j
        }
        nx.set_edge_attributes(self.G, capacities, 'capacity')

        # Violated cut-set inequalities via min s-t cuts from the depot
        violated = []
        for t in range(1, n):
            cut_value, partition = nx.minimum_cut(self.G, 0, t, capacity='capacity')
            if cut_value < 1 - 1e-4:
                S = frozenset(partition[1])  # sink side (contains t)
                violation = 1.0 - cut_value
                violated.append((S, violation))

        if not violated:
            return

        unique = {}
        for S, viol in violated:
            if S not in unique or viol > unique[S]:
                unique[S] = viol

        top_k = sorted(unique.items(), key=lambda item: item[1], reverse=True)[:DFJ_TOP_K]

        # sum_{i in S, j not in S} x_ij >= 1
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
        Prices the tour of an integer solution with the exact SOCP: the tour
        SOCP for the arc family and BS, the full model with x fixed otherwise.

        Args:
            x_sol (dict): values of the x variables.

        Returns:
            (value, arcs, points), or (inf, None, None) if the SOCP fails.
        """
        ub_model = Model("UB_Model")
        ub_model.setParam('OutputFlag', 0)
        model_type_ub = 'perspective' if self.model_type == 'perspective' else ('arc' if self.model_type in ['arc', 'BS'] else 'seq')
        # The tour may have either orientation: no symmetry breaking here
        ub_solver = CETSP_L2_Solver(ub_model, self.data, model_type_ub, threads=self.threads, symmetry_breaking=False)

        if model_type_ub == 'arc':
            # Rounded incumbent: MIP values can sit up to IntFeasTol from 0/1
            tour_arcs = [(i, j) for i in range(self.data.n) for j in range(self.data.n)
                         if i != j and x_sol[i, j] > 0.5]
            ub_solver.build_tour_socp(tour_arcs)
        else:
            ub_solver.build()

            # Fix x to the rounded incumbent
            for i in range(self.data.n):
                for j in range(self.data.n):
                    v = 1.0 if x_sol[i, j] > 0.5 else 0.0
                    ub_solver.x[i, j].lb = v
                    ub_solver.x[i, j].ub = v

        ub_model.optimize()

        # Any feasible point is a valid upper bound, so SUBOPTIMAL with a solution is accepted
        if ub_model.status == GRB.OPTIMAL or (ub_model.status == GRB.SUBOPTIMAL and ub_model.SolCount > 0):
            arcs = []
            points = {}

            if model_type_ub == 'arc':
                arcs = list(ub_solver.tour_arcs)
            else:
                ub_x_sol = ub_model.getAttr('X', ub_solver.x)

            if model_type_ub == 'perspective':
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
        Reads bounds, gap and solution from the solved model. The upper bound
        of the decompositions and of ABF is recomputed from the tour with the
        exact SOCP.
        """

        self.status = self.model.status
        try:
            self.lower_bound = self.model.ObjBound
        except Exception:
            self.lower_bound = 0.0
        try:
            self.node_count = self.model.NodeCount
        except Exception:
            self.node_count = None

        if self.model.solCount > 0:
            x_sol = self.model.getAttr('X', self.solver.x)

            if self.extended and not self.decomposition:
                # Approximate models: the feasible solution comes from the exact SOCP
                ub_obj_val, ub_arcs, ub_points = self.compute_upper_bound(x_sol)
                self.upper_bound = ub_obj_val
                self.arcs = ub_arcs
                self.points = ub_points
            elif self.decomposition:
                # The master objective is the tour length only if every incumbent
                # was verified by the subproblem; report the recomputed length and
                # flag a mismatch when a subproblem failed
                self.upper_bound = self.model.ObjVal
                self._extract_arcs_and_points()
                ub_true, _, plot_points = self.compute_upper_bound(x_sol)
                if plot_points:
                    self.points = plot_points
                if ub_true != float('inf'):
                    if self.subproblem_failures > 0 and ub_true > self.upper_bound + 1e-6 * max(1.0, abs(ub_true)):
                        self.unverified_incumbent = True
                        print(f"[CETSP] master objective {self.upper_bound:.9g} is below the true "
                              f"length {ub_true:.9g} of the incumbent tour (a subproblem failure "
                              f"let it through); reporting the true length.",
                              file=sys.stderr, flush=True)
                    self.upper_bound = ub_true
            else:
                # Full SOCP models: the objective is the tour length up to solver
                # tolerances, which the big-M of ABF amplifies (see build_tour_socp)
                self.upper_bound = self.model.ObjVal
                self._extract_arcs_and_points()
                if self.model_type == 'arc':
                    ub_true, _, plot_points = self.compute_upper_bound(x_sol)
                    if ub_true != float('inf'):
                        self.upper_bound = ub_true
                        if plot_points:
                            self.points = plot_points

            if self.upper_bound is not None and self.upper_bound > 0 and self.upper_bound != float('inf'):
                self.gap = (self.upper_bound - self.lower_bound) / self.upper_bound
            else:
                self.gap = float('inf')
        else:
            self.upper_bound = float('inf')
            self.gap = float('inf')

    def _extract_arcs_and_points(self):
        """
        Reads the tour arcs and the visit points from the solved model.
        """
        if not self.solver or not hasattr(self.solver, 'x'):
            return

        self.arcs = []
        self.points = {}

        x_sol = self.model.getAttr('X', self.solver.x)

        if self.model_type in ['arc', 'BS']:
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

        if hasattr(self.solver, 'p_x') and hasattr(self.solver, 'p_y'):
            p_x_sol = self.model.getAttr('X', self.solver.p_x)
            p_y_sol = self.model.getAttr('X', self.solver.p_y)
            if self.model_type in ['arc', 'BS']:
                for i in range(self.data.n):
                    self.points[i] = (p_x_sol[i], p_y_sol[i])
            elif self.model_type == 'seq':
                for k in range(self.data.n):
                    i = tour_sequence[k]
                    self.points[i] = (p_x_sol[k], p_y_sol[k])

    def get_solution_summary(self):
        """
        Returns the run's metrics and solution as a dict, or a message if there is no solution.
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
                "unverified_incumbent": self.unverified_incumbent,
                "subproblem_failures": self.subproblem_failures,
                "arcs": self.arcs,
                "points": self.points
            }
        else:
            return "No solution available."