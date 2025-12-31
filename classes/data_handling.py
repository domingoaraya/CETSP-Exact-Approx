import numpy as np
import copy
from utils.instance_handler import read_instance

class CETSPData:
    """
    Represents the data for a Close Enough Traveling Salesperson Problem (CETSP) instance.
    This class stores the problem's parameters and provides methods for data preprocessing.
    """

    def __init__(self, n, centers, radii):
        """
        Initializes the CETSPData object.

        Args:
            n (int): The number of regions (neighborhoods).
            centers (dict): A dictionary mapping region indices to their center coordinates (e.g., {0: (x0, y0), 1: (x1, y1), ...}).
            radii (dict): A dictionary mapping region indices to their radii (e.g., {0: r0, 1: r1, ...}).
        """
        self.n = n
        self.centers = centers
        self.radii = radii
        self.redundancies = []
        self.min_coord_x = None
        self.max_coord_x = None
        self.min_coord_y = None
        self.max_coord_y = None

    @classmethod
    def from_file(cls, file_path):
        """
        Creates a CETSPData object from an instance file.
        """
        n, centers, radii = read_instance(file_path)
        return cls(n, centers, radii)


    def eliminate_redundancies(self):
        """
        Identifies and handles redundant regions for the L2 norm.
        A region is considered redundant if it is completely contained within another region.
        This method updates the number of regions, centers, and radii to reflect the non-redundant set.
        """
        
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    # Condition for L2 norm: Check if region j is inside region i
                    distance_sq = (self.centers[i][0] - self.centers[j][0])**2 + (self.centers[i][1] - self.centers[j][1])**2
                    if np.sqrt(distance_sq) + self.radii[j] <= self.radii[i]:
                        self.redundancies.append(j)

        if not self.redundancies:
            return

        # Remove duplicates from redundancies list
        self.redundancies = list(set(self.redundancies))

        # Separate non-redundant and redundant elements
        non_redundant_centers = {k: v for k, v in self.centers.items() if k not in self.redundancies}
        
        non_redundant_radii = {k: v for k, v in self.radii.items() if k not in self.redundancies}

        # Reassign keys for centers and radii to be consecutive
        self.centers = {new_key: value for new_key, value in enumerate(non_redundant_centers.values())}
        self.radii = {new_key: value for new_key, value in enumerate(non_redundant_radii.values())}

        # Update the number of regions
        self.n = len(self.centers)


    def compute_min_max_coords(self):
        """
        Computes the minimum and maximum x and y coordinates that bound all regions.
        This is useful for setting plot limits.
        """
        if not self.centers:
            self.min_coord_x, self.max_coord_x, self.min_coord_y, self.max_coord_y = 0, 0, 0, 0
            return

        self.min_coord_x = min(c[0] - self.radii[k] for k, c in self.centers.items())
        self.max_coord_x = max(c[0] + self.radii[k] for k, c in self.centers.items())
        self.min_coord_y = min(c[1] - self.radii[k] for k, c in self.centers.items())
        self.max_coord_y = max(c[1] + self.radii[k] for k, c in self.centers.items())

    def __repr__(self):
        return f"CETSPData(n={self.n}, centers={self.centers}, radii={self.radii})"

