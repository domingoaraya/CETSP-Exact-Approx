import matplotlib.pyplot as plt

class Plotter:
    """
    Plots the disks and the tour of a solved CETSPModel.
    """

    def __init__(self, model):
        """
        Args:
            model (CETSPModel): solved model.
        """
        self.model = model
        self.data = model.data

    def plot_solution(self, file_path=None):
        """
        Draws the solution and shows it; also saves it to file_path if given.

        Args:
            file_path (str): output image path.
        """
        fig, ax = plt.subplots()

        # Plot the regions (circles for L2 norm)
        for i in range(self.data.n):
            circle = plt.Circle((self.data.centers[i][0], self.data.centers[i][1]), self.data.radii[i], fill=False, color='black')
            ax.add_patch(circle)
            plt.plot(self.data.centers[i][0], self.data.centers[i][1], 'k.', markersize=3)
            plt.text(self.data.centers[i][0], self.data.centers[i][1], str(i), fontsize=8, ha='right', va='bottom')

        # Set plot limits
        self.data.compute_min_max_coords()
        ax.set_xlim(self.data.min_coord_x - 1, self.data.max_coord_x + 1)
        ax.set_ylim(self.data.min_coord_y - 1, self.data.max_coord_y + 1)

        # Plot the tour
        if self.model.arcs and self.model.points:
            for i, j in self.model.arcs:
                p_i = self.model.points[i]
                p_j = self.model.points[j]
                plt.plot([p_i[0], p_j[0]], [p_i[1], p_j[1]], 'r-')

        # Plot the selected points
        if self.model.points:
            for i, point in self.model.points.items():
                plt.plot(point[0], point[1], 'r.', markersize=5)

        # Final adjustments
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        if file_path:
            plt.savefig(file_path, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.show()
