import numpy as np
import copy
from scipy.spatial import ConvexHull
from utils.instance_handler import read_instance

class CETSPData:
    """
    A CETSP instance: disk centers and radii, with the preprocessing and the
    boundary-cell discretization used by BS.
    """

    def __init__(self, n, centers, radii):
        """
        Args:
            n (int): number of targets.
            centers (dict): target index -> (x, y).
            radii (dict): target index -> radius.
        """
        self.n = n

        if isinstance(centers, (list, np.ndarray)):
            self.centers = {i: tuple(centers[i]) for i in range(n)}
        else:
            self.centers = centers
        if isinstance(radii, (list, np.ndarray)):
            self.radii = {i: float(radii[i]) for i in range(n)}
        else:
            self.radii = radii

        self.redundancies = []
        self.min_coord_x = None
        self.max_coord_x = None
        self.min_coord_y = None
        self.max_coord_y = None
        self.bs_cells = {}
        self.bs_distances = {}

    @classmethod
    def from_file(cls, file_path):
        """
        Reads an instance file (see utils.instance_handler).
        """
        n, centers, radii = read_instance(file_path)
        return cls(n, centers, radii)


    def eliminate_redundancies(self):
        """
        Removes every disk that contains another disk (visiting the inner one
        visits it too) and renumbers the remaining targets consecutively.
        """

        self.redundancies = []
        
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    distance_sq = (self.centers[i][0] - self.centers[j][0])**2 + (self.centers[i][1] - self.centers[j][1])**2

                    # Identical disks: keep the one with the smaller index
                    if distance_sq == 0 and self.radii[j] == self.radii[i] and i < j:
                        continue
                    if np.sqrt(distance_sq) + self.radii[j] <= self.radii[i]:
                        self.redundancies.append(i)

        if not self.redundancies:
            return

        self.redundancies = list(set(self.redundancies))

        non_redundant_centers = {k: v for k, v in self.centers.items() if k not in self.redundancies}
        
        non_redundant_radii = {k: v for k, v in self.radii.items() if k not in self.redundancies}

        self.centers = {new_key: value for new_key, value in enumerate(non_redundant_centers.values())}
        self.radii = {new_key: value for new_key, value in enumerate(non_redundant_radii.values())}

        self.n = len(self.centers)


    def compute_min_max_coords(self):
        """
        Bounding box of the disks (used for plot limits).
        """
        if not self.centers:
            self.min_coord_x, self.max_coord_x, self.min_coord_y, self.max_coord_y = 0, 0, 0, 0
            return

        self.min_coord_x = min(c[0] - self.radii[k] for k, c in self.centers.items())
        self.max_coord_x = max(c[0] + self.radii[k] for k, c in self.centers.items())
        self.min_coord_y = min(c[1] - self.radii[k] for k, c in self.centers.items())
        self.max_coord_y = max(c[1] + self.radii[k] for k, c in self.centers.items())

    @staticmethod
    def _get_endpoints(cell):
        """Endpoints of a boundary cell (an arc of a disk)."""
        c = cell['center']
        r = cell['radius']
        s = cell['start_angle']
        e = cell['end_angle']
        p1 = np.array([c[0] + r * np.cos(s), c[1] + r * np.sin(s)], dtype=float)
        p2 = np.array([c[0] + r * np.cos(e), c[1] + r * np.sin(e)], dtype=float)
        return p1, p2

    @staticmethod
    def _is_angle_in_arc(theta, start, end):
        """Whether angle theta lies in the arc [start, end] (end may exceed 2 pi)."""
        norm_theta = (theta - start) % (2 * np.pi) + start
        return norm_theta <= end + 1e-9

    def initialize_bs_cells(self, N_prime=4):
        """
        Discretizes the part of each disk boundary that lies inside the convex
        hull of the centers into N_prime cells (arcs), as in BS, and computes
        the cell-to-cell distances.

        Args:
            N_prime (int): cells per boundary arc.
        """
        self.bs_cells = {}
        self.bs_distances = {}

        coords = np.array([self.centers[i] for i in range(self.n)])
        
        hull = None
        if self.n >= 3:
            try:
                hull = ConvexHull(coords)
            except Exception:
                hull = None

        for i in range(self.n):
            radius = self.radii[i]
            center = np.array(self.centers[i], dtype=float)
            
            if radius == 0.0:
                self.bs_cells[(i, 0)] = {
                    'center': self.centers[i],
                    'radius': 0.0,
                    'start_angle': 0.0,
                    'end_angle': 0.0
                }
                continue

            if hull is None:
                arcs = [(0.0, 2 * np.pi)]
            else:
                hull_indices = list(hull.vertices)
                hull_vertices = [coords[j] for j in hull_indices]
                n_v = len(hull_vertices)

                intersections = []
                for j in range(n_v):
                    A = hull_vertices[j]
                    B = hull_vertices[(j + 1) % n_v]
                    
                    U = A - center
                    V = B - A
                    
                    a = np.dot(V, V)
                    if a == 0:
                        continue
                    b = 2.0 * np.dot(U, V)
                    c = np.dot(U, U) - radius**2
                    
                    discriminant = b**2 - 4.0 * a * c
                    if discriminant >= 0:
                        sqrt_disc = np.sqrt(discriminant)
                        t1 = (-b + sqrt_disc) / (2.0 * a)
                        t2 = (-b - sqrt_disc) / (2.0 * a)
                        
                        for t in [t1, t2]:
                            if 0.0 <= t <= 1.0:
                                pt = A + t * V
                                angle = np.arctan2(pt[1] - center[1], pt[0] - center[0])
                                angle = angle if angle >= 0 else angle + 2 * np.pi
                                intersections.append(angle)

                intersections.extend([0.0, 2 * np.pi])
                intersections = np.unique(np.sort(intersections))

                valid_arcs = []
                for k in range(len(intersections) - 1):
                    a1, a2 = intersections[k], intersections[k+1]
                    if a2 - a1 < 1e-6:
                        continue
                    
                    mid_angle = (a1 + a2) / 2.0
                    mid_pt = center + radius * np.array([np.cos(mid_angle), np.sin(mid_angle)])
                    
                    # Midpoint inside the hull
                    if np.all(np.dot(hull.equations[:, :-1], mid_pt) + hull.equations[:, -1] <= 1e-6):
                        valid_arcs.append((a1, a2))

                if len(valid_arcs) == 0:
                    arcs = [(0.0, 2 * np.pi)]
                else:
                    # Keep every arc inside the hull; the two pieces wrapping through 0 are one arc
                    arcs = list(valid_arcs)
                    if (len(arcs) >= 2 and np.isclose(arcs[0][0], 0.0)
                            and np.isclose(arcs[-1][1], 2 * np.pi)):
                        first = arcs.pop(0)
                        last = arcs.pop()
                        arcs.append((last[0], first[1] + 2 * np.pi))

            cell_idx = 0
            for start_angle, end_angle in arcs:
                angle_step = (end_angle - start_angle) / N_prime
                for step in range(N_prime):
                    a1 = start_angle + step * angle_step
                    a2 = start_angle + (step + 1) * angle_step
                    self.bs_cells[(i, cell_idx)] = {
                        'center': self.centers[i],
                        'radius': radius,
                        'start_angle': a1,
                        'end_angle': a2
                    }
                    cell_idx += 1

        if self.bs_cells:
            self._compute_arc_distances(list(self.bs_cells.keys()))

    def _calc_single_pair_dist(self, delta, sigma):
        """
        Minimum distance between two boundary cells, in closed form: the
        candidates are the arc intersections, the pairs collinear with the
        centers, each endpoint against the other arc, and the endpoint pairs.
        """
        i, c1 = delta
        j, c2 = sigma

        if i == j:
            return 0.0

        cell1 = self.bs_cells[delta]
        cell2 = self.bs_cells[sigma]

        r_i = cell1['radius']
        r_j = cell2['radius']
        C_i = np.array(cell1['center'], dtype=float)
        C_j = np.array(cell2['center'], dtype=float)

        # Point targets
        if r_i == 0.0 and r_j == 0.0:
            return float(np.linalg.norm(C_i - C_j))

        if r_i == 0.0 or r_j == 0.0:
            if r_i == 0.0:
                point = C_i
                arc_center = C_j
                arc_radius = r_j
                start_ang = cell2['start_angle']
                end_ang = cell2['end_angle']
                target_cell = cell2
            else:
                point = C_j
                arc_center = C_i
                arc_radius = r_i
                start_ang = cell1['start_angle']
                end_ang = cell1['end_angle']
                target_cell = cell1

            center_dist = np.linalg.norm(point - arc_center)
            theta = np.arctan2(point[1] - arc_center[1], point[0] - arc_center[0])

            if self._is_angle_in_arc(theta, start_ang, end_ang):
                return float(max(0.0, center_dist - arc_radius))
            else:
                ep1, ep2 = self._get_endpoints(target_cell)
                d1 = np.linalg.norm(point - ep1)
                d2 = np.linalg.norm(point - ep2)
                return float(min(d1, d2))

        # Two arcs
        D = np.linalg.norm(C_j - C_i)

        # Arc intersections
        if D > 0 and abs(r_i - r_j) <= D <= r_i + r_j:
            a = (r_i**2 - r_j**2 + D**2) / (2.0 * D)
            h_sq = r_i**2 - a**2
            h = np.sqrt(max(0.0, h_sq))
            P0 = C_i + (a / D) * (C_j - C_i)
            
            N_perp = np.array([-(C_j[1] - C_i[1]) / D, (C_j[0] - C_i[0]) / D])

            intersections = [P0 + h * N_perp]
            if h > 1e-12:
                intersections.append(P0 - h * N_perp)

            for P_int in intersections:
                theta_i = np.arctan2(P_int[1] - C_i[1], P_int[0] - C_i[0])
                theta_j = np.arctan2(P_int[1] - C_j[1], P_int[0] - C_j[0])
                if self._is_angle_in_arc(theta_i, cell1['start_angle'], cell1['end_angle']) and \
                   self._is_angle_in_arc(theta_j, cell2['start_angle'], cell2['end_angle']):
                    return 0.0

        min_dist = float('inf')

        # a) Pairs collinear with the centers
        if D > 0:
            theta_ij = np.arctan2(C_j[1] - C_i[1], C_j[0] - C_i[0])
            theta_ji = np.arctan2(C_i[1] - C_j[1], C_i[0] - C_j[0])
            # Each point may face the other center or face away from it; the
            # same-side pairs are the minimizers when one disk lies inside the other
            for s_i, s_j, val in ((0.0, 0.0, abs(D - r_i - r_j)),
                                  (0.0, np.pi, abs(D - r_i + r_j)),
                                  (np.pi, 0.0, abs(D + r_i - r_j))):
                if self._is_angle_in_arc(theta_ij + s_i, cell1['start_angle'], cell1['end_angle']) and \
                   self._is_angle_in_arc(theta_ji + s_j, cell2['start_angle'], cell2['end_angle']):
                    min_dist = min(min_dist, val)

        # b) Endpoints against the other arc
        ep1_i, ep2_i = self._get_endpoints(cell1)
        for E in [ep1_i, ep2_i]:
            theta_j = np.arctan2(E[1] - C_j[1], E[0] - C_j[0])
            if self._is_angle_in_arc(theta_j, cell2['start_angle'], cell2['end_angle']):
                dist = abs(np.linalg.norm(E - C_j) - r_j)
                min_dist = min(min_dist, dist)

        ep1_j, ep2_j = self._get_endpoints(cell2)
        for E in [ep1_j, ep2_j]:
            theta_i = np.arctan2(E[1] - C_i[1], E[0] - C_i[0])
            if self._is_angle_in_arc(theta_i, cell1['start_angle'], cell1['end_angle']):
                dist = abs(np.linalg.norm(E - C_i) - r_i)
                min_dist = min(min_dist, dist)

        # c) Endpoint pairs
        for p_i in [ep1_i, ep2_i]:
            for p_j in [ep1_j, ep2_j]:
                dist = np.linalg.norm(p_i - p_j)
                min_dist = min(min_dist, dist)

        return float(min_dist)

    def _compute_arc_distances(self, new_cells=None):
        """
        Computes the cell-to-cell distances for new_cells (all cells if None)
        against every other cell.

        Args:
            new_cells (list): cell keys (target_idx, cell_idx).
        """
        if new_cells is None:
            all_keys = list(self.bs_cells.keys())
            n_keys = len(all_keys)
            for idx1 in range(n_keys):
                delta = all_keys[idx1]
                for idx2 in range(idx1, n_keys):
                    sigma = all_keys[idx2]
                    dist = self._calc_single_pair_dist(delta, sigma)
                    self.bs_distances[(delta, sigma)] = dist
                    self.bs_distances[(sigma, delta)] = dist
        else:
            all_keys = list(self.bs_cells.keys())
            for delta in new_cells:
                if delta not in self.bs_cells:
                    continue
                for sigma in all_keys:
                    dist = self._calc_single_pair_dist(delta, sigma)
                    self.bs_distances[(delta, sigma)] = dist
                    self.bs_distances[(sigma, delta)] = dist

    def refine_bs_cells(self, active_cells):
        """
        Bisects the given cells; the first child keeps the parent's key.

        Args:
            active_cells (list): cell keys (target_idx, cell_idx).
        """
        new_cell_keys = []
        for target_idx, cell_idx in active_cells:
            cell_key = (target_idx, cell_idx)
            if cell_key not in self.bs_cells:
                continue

            cell = self.bs_cells[cell_key]
            if cell['radius'] == 0.0 or cell['start_angle'] == cell['end_angle']:
                continue

            mid_angle = (cell['start_angle'] + cell['end_angle']) / 2.0

            existing_indices = [c_idx for t_idx, c_idx in self.bs_cells.keys() if t_idx == target_idx]
            max_idx = max(existing_indices) if existing_indices else cell_idx

            child1_key = cell_key
            child2_key = (target_idx, max_idx + 1)

            child1 = {
                'center': cell['center'],
                'radius': cell['radius'],
                'start_angle': cell['start_angle'],
                'end_angle': mid_angle
            }
            child2 = {
                'center': cell['center'],
                'radius': cell['radius'],
                'start_angle': mid_angle,
                'end_angle': cell['end_angle']
            }

            self.bs_cells[child1_key] = child1
            self.bs_cells[child2_key] = child2

            new_cell_keys.extend([child1_key, child2_key])

        if new_cell_keys:
            try:
                self._compute_arc_distances(new_cell_keys)
            except NotImplementedError:
                pass

    def __repr__(self):
        return f"CETSPData(n={self.n}, centers={self.centers}, radii={self.radii})"


