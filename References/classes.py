from gurobipy import *
import numpy as np
import matplotlib.pyplot as plt
import copy

class ModelCETSP:

    def __init__(self, data):
        
        self.n = data[0]
        self.centros = data[1]
        self.radios = data[2]

        self.model = Model("CETSP")

    def print_instance(self):

        for i in range(self.n):
            print(f"Centro {i}: ({self.centros[i][0]}, {self.centros[i][1]})")
            print(f"Radio {i}: {self.radios[i]}")

    def compute_overlap_ratio(self):

        suma_radios = sum(self.radios.values())
        media_radios = suma_radios / len(self.radios)

        self.__compute_min_max_coords()

        largo_lado_mayor = max(self.max_coord_x - self.min_coord_x, self.max_coord_y - self.min_coord_y)

        self.overlap = media_radios / largo_lado_mayor
    
    def build_model(self, norm_type, model_type, decomposition = False, extended = False, nu = None):

        self.modelo = Model("CETSP")
        self.modelo.setParam('OutputFlag', 0)
        # self.modelo.setParam('MIQCPMethod', 1)

        self.__detect_model_type(model_type)
        self.__detect_norm(norm_type)
        self.__eliminate_redundancies()
        self.__compute_big_M()

        self.decomposition = decomposition
        if self.decomposition:
            self.L = 0
            self.theta = self.modelo.addVar(vtype=GRB.CONTINUOUS, lb=self.L, name="theta")
            self.modelo.Params.LazyConstraints = 1
            self.p = {}
            self.arcs = []
            self.ub = GRB.INFINITY
            self.cut_count = 0

        self.extended = extended
        if self.extended:
            self.nu = nu
            if self.norm != 2:
                raise ValueError("Extended formulation only available for L_2 norm")
            if self.decomposition:
                #raise ValueError("Extended formulation and decomposition not compatible for the moment")
                pass

        # Variables comunes

        self.x = self.modelo.addVars(self.n, self.n, vtype=GRB.BINARY, name="x")

        if not self.decomposition or (self.decomposition and self.extended):      
            self.p_x = self.modelo.addVars(self.n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="p_x")
            self.p_y = self.modelo.addVars(self.n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="p_y")

        # Restricciones comunes

        self.__create_tour_constraints()

        if getattr(self, 'arc_model', False):
            self.__create_arc_form_specific_parts()

        elif getattr(self, 'seq_model', False):
            self.__create_seq_form_specific_parts()

        # Añadir función objetivo

        self.__set_objective()

    def optimize(self, time_limit):

        try:
            self.modelo.setParam('TimeLimit', time_limit)
            if not self.decomposition:
                self.modelo.optimize()
            elif self.decomposition:
                if getattr(self, 'arc_model', False):
                    self.modelo.optimize(self.__optcut_arc)
                elif getattr(self, 'seq_model', False):
                    self.modelo.optimize(self.__optcut_seq)
        except GurobiError as e:
            print(f"Error code {str(e.errno)}: {str(e)}")

    def get_solution(self):

        self.runtime = self.modelo.Runtime

        if not self.extended:
            self.gap = self.modelo.MIPGap

        if not self.decomposition:
            self.__retrieve_arcs()

            if not self.extended:
                self.p = {}

                for i in range(self.n):
                    self.p[i] = (self.p_x[i].x, self.p_y[i].x)

            elif self.extended:
                self.__compute_optimal_points()
                self.lb = self.modelo.ObjBound
                self.gap = (self.ub - self.lb) / self.ub
                self.runtime = self.runtime + self.aux_runtime

        elif self.decomposition and self.extended:
            self.lb = self.modelo.ObjBound
            # self.gap = (self.ub - self.lb) / self.ub
            self.gap = self.modelo.MIPGap

        elif self.decomposition and not self.extended and getattr(self, 'seq_model', False):
            self.__seq_subproblem()

        elif self.decomposition and not self.extended and getattr(self, 'arc_model', False) and self.norm == 2:
            self.__compute_optimal_points()

    def plot_solution(self):

        fig, ax = plt.subplots()

        for i in range(self.n + len(self.redundancies)):

            if self.norm == 2:
                circle = plt.Circle((self.centros[i][0], self.centros[i][1]), self.radios[i], fill=False, color='black')
                ax.add_patch(circle)
            elif self.norm == 1:
                diamond = plt.Polygon(
                    [[self.centros[i][0], self.centros[i][1] + self.radios[i]],
                    [self.centros[i][0] + self.radios[i], self.centros[i][1]],
                    [self.centros[i][0], self.centros[i][1] - self.radios[i]],
                    [self.centros[i][0] - self.radios[i], self.centros[i][1]]],
                    closed=True,
                    fill=False,
                    edgecolor='black'
                )
                ax.add_patch(diamond)
            elif self.norm == GRB.INFINITY:
                square = plt.Rectangle(
                    (self.centros[i][0] - self.radios[i], self.centros[i][1] - self.radios[i]),  # Bottom-left corner
                    2 * self.radios[i],  # Width
                    2 * self.radios[i],  # Height
                    fill=False,
                    edgecolor='black'
                )
                ax.add_patch(square)

            if i == 0:
                plt.plot(self.centros[i][0], self.centros[i][1], 'k.', markersize=7)
            else:
                plt.plot(self.centros[i][0], self.centros[i][1], 'k.', markersize=3)

            plt.text(self.centros[i][0], self.centros[i][1], str(i), fontsize=8, ha='right', va='bottom')

        # check if self.coordinates_flag is True
        if not getattr(self, 'coordinates_flag', False):
            self.__compute_min_max_coords()

        ax.set_xlim(self.min_coord_x - 1, self.max_coord_x + 1)
        ax.set_ylim(self.min_coord_y - 1, self.max_coord_y + 1)

        # ax.set_xlabel('X Coordinate')
        # ax.set_ylabel('Y Coordinate')

        plt.gca().set_aspect('equal', adjustable='box')
        # plt.grid()

        if getattr(self, 'arc_model', False) or self.extended:
            for i, j in self.arcs:
                plt.plot([self.p[i][0], self.p[j][0]], [self.p[i][1], self.p[j][1]], 'r-')

        elif getattr(self, 'seq_model', False):
            for k in range(self.n - 1):
                plt.plot([self.p[k][0], self.p[k + 1][0]], [self.p[k][1], self.p[k + 1][1]], 'r-')
            plt.plot([self.p[self.n - 1][0], self.p[0][0]], [self.p[self.n - 1][1], self.p[0][1]], 'r-')

        # Hide axes
        ax.axis('off')

        # Save the figure with transparent background and no extra space
        plt.savefig('plot2.png', transparent=True, bbox_inches='tight', pad_inches=0, dpi=1000)

        plt.show()

#############################################################################################################

    def __compute_min_max_coords(self):

        self.min_coord_x = min(c[0] - self.radios[k] for k, c in self.centros.items())
        self.max_coord_x = max(c[0] + self.radios[k] for k, c in self.centros.items())
        self.min_coord_y = min(c[1] - self.radios[k] for k, c in self.centros.items())
        self.max_coord_y = max(c[1] + self.radios[k] for k, c in self.centros.items())

        self.coordinates_flag = True

    def __detect_model_type(self, model_type):

        if model_type == "arc":
            self.arc_model = True
        elif model_type == "seq":
            self.seq_model = True
        else:
            raise ValueError("Invalid model type")


    def __detect_norm(self, norm_type):

        if norm_type == "L_1":
            self.norm = 1
        
        elif norm_type == "L_2":
            self.norm = 2

        elif norm_type == "L_inf":
            self.norm = GRB.INFINITY
        
        else:           
            raise ValueError("Invalid norm type")
           
    def __eliminate_redundancies(self):
        
        self.redundancies = []

        if self.norm == 2:
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        if np.sqrt((self.centros[i][0] - self.centros[j][0])**2 + (self.centros[i][1] - self.centros[j][1])**2) + self.radios[j] <= self.radios[i]:
                            #print(f"Neighborhood {i} contains neighborhood {j}")
                            self.redundancies.append(i)
                            break
        elif self.norm == 1:
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        if abs(self.centros[i][0] - self.centros[j][0]) + abs(self.centros[i][1] - self.centros[j][1]) + self.radios[j] <= self.radios[i]:
                            #print(f"Neighborhood {i} contains neighborhood {j}")
                            self.redundancies.append(i)
                            break
        elif self.norm == GRB.INFINITY:
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        if max(abs(self.centros[i][0] - self.centros[j][0]), abs(self.centros[i][1] - self.centros[j][1])) + self.radios[j] <= self.radios[i]:
                            #print(f"Neighborhood {i} contains neighborhood {j}")
                            self.redundancies.append(i)
                            break
    
        if len(self.redundancies) == 0:
            #print("No redundancies detected")
            return
        
        else:
            # Separate non-redundant and redundant elements
            non_redundant_centros = {k: v for k, v in self.centros.items() if k not in self.redundancies}
            redundant_centros = {k: v for k, v in self.centros.items() if k in self.redundancies}
            
            non_redundant_radios = {k: v for k, v in self.radios.items() if k not in self.redundancies}
            redundant_radios = {k: v for k, v in self.radios.items() if k in self.redundancies}

            # Reassign keys for centros
            reassigned_centros = {new_key: value for new_key, value in enumerate(non_redundant_centros.values())}
            reassigned_centros.update({len(reassigned_centros) + i: value for i, value in enumerate(redundant_centros.values())})

            # Reassign keys for radios
            reassigned_radios = {new_key: value for new_key, value in enumerate(non_redundant_radios.values())}
            reassigned_radios.update({len(reassigned_radios) + i: value for i, value in enumerate(redundant_radios.values())})

            # Update the dictionaries and self.n
            self.centros = reassigned_centros
            self.radios = reassigned_radios
            self.n -= len(self.redundancies)
            self.funtional_n = copy.deepcopy(self.n)

    def __create_tour_constraints(self):

        self.modelo.addConstrs(sum(self.x[i, j] for j in range(self.n)) == 1 for i in range(self.n))
        self.modelo.addConstrs(sum(self.x[i, j] for i in range(self.n)) == 1 for j in range(self.n))

    def __create_arc_form_specific_parts(self):

        self.__create_subtour_constraints()

        if not self.decomposition or (self.decomposition and self.extended):

            self.__create_neigh_constraints()

            self.d = self.modelo.addVars(self.n, self.n, vtype=GRB.CONTINUOUS, name="d")

            self.__create_distance_constraints()

            self.d_aux = self.modelo.addVars(self.n, self.n, vtype=GRB.CONTINUOUS, name="d_aux")

            self.__create_big_M_constraints()

    def __create_seq_form_specific_parts(self):

        self.modelo.addConstr(self.x[0,0] == 1)

        if not self.decomposition or (self.decomposition and self.extended):

            self.__create_neigh_constraints()

            self.d = self.modelo.addVars(self.n, vtype=GRB.CONTINUOUS, name="d")

            self.__create_distance_constraints()
    
    def __create_neigh_constraints(self):
            
        if self.norm == GRB.INFINITY:

            if not self.decomposition:
                if getattr(self, 'arc_model', False):
                    self.modelo.addConstrs(self.p_x[i] >= self.centros[i][0] - self.radios[i] for i in range(self.n))
                    self.modelo.addConstrs(self.p_x[i] <= self.centros[i][0] + self.radios[i] for i in range(self.n))
                    self.modelo.addConstrs(self.p_y[i] >= self.centros[i][1] - self.radios[i] for i in range(self.n))
                    self.modelo.addConstrs(self.p_y[i] <= self.centros[i][1] + self.radios[i] for i in range(self.n))

                elif getattr(self, 'seq_model', False):
                    self.modelo.addConstrs(self.p_x[k] >= sum(self.x[i,k]*(self.centros[i][0] - self.radios[i]) for i in range(self.n)) for k in range(self.n))
                    self.modelo.addConstrs(self.p_x[k] <= sum(self.x[i,k]*(self.centros[i][0] + self.radios[i]) for i in range(self.n)) for k in range(self.n))
                    self.modelo.addConstrs(self.p_y[k] >= sum(self.x[i,k]*(self.centros[i][1] - self.radios[i]) for i in range(self.n)) for k in range(self.n))
                    self.modelo.addConstrs(self.p_y[k] <= sum(self.x[i,k]*(self.centros[i][1] + self.radios[i]) for i in range(self.n)) for k in range(self.n))
            
            elif self.decomposition:
                if getattr(self, 'arc_model', False):
                    self.neigh_con_1 = self.sub.addConstrs(self.p_x[i] >= self.centros[i][0] - self.radios[i] for i in range(self.n))
                    self.neigh_con_2 = self.sub.addConstrs(self.p_x[i] <= self.centros[i][0] + self.radios[i] for i in range(self.n))
                    self.neigh_con_3 = self.sub.addConstrs(self.p_y[i] >= self.centros[i][1] - self.radios[i] for i in range(self.n))
                    self.neigh_con_4 = self.sub.addConstrs(self.p_y[i] <= self.centros[i][1] + self.radios[i] for i in range(self.n))

                elif getattr(self, 'seq_model', False):
                    self.neigh_con_1 = self.sub.addConstrs(self.p_x[k] >= sum(self.x_sol[i,k]*(self.centros[i][0] - self.radios[i]) for i in range(self.n)) for k in range(self.n))
                    self.neigh_con_2 = self.sub.addConstrs(self.p_x[k] <= sum(self.x_sol[i,k]*(self.centros[i][0] + self.radios[i]) for i in range(self.n)) for k in range(self.n))
                    self.neigh_con_3 = self.sub.addConstrs(self.p_y[k] >= sum(self.x_sol[i,k]*(self.centros[i][1] - self.radios[i]) for i in range(self.n)) for k in range(self.n))
                    self.neigh_con_4 = self.sub.addConstrs(self.p_y[k] <= sum(self.x_sol[i,k]*(self.centros[i][1] + self.radios[i]) for i in range(self.n)) for k in range(self.n))

        elif self.norm == 1:

            if not self.decomposition:
                if getattr(self, 'arc_model', False):
                    self.modelo.addConstrs((self.p_x[i] - self.centros[i][0]) + (self.p_y[i] - self.centros[i][1]) <= self.radios[i] for i in range(self.n))
                    self.modelo.addConstrs((self.p_x[i] - self.centros[i][0]) - (self.p_y[i] - self.centros[i][1]) <= self.radios[i] for i in range(self.n))
                    self.modelo.addConstrs(-(self.p_x[i] - self.centros[i][0]) + (self.p_y[i] - self.centros[i][1]) <= self.radios[i] for i in range(self.n))
                    self.modelo.addConstrs(-(self.p_x[i] - self.centros[i][0]) - (self.p_y[i] - self.centros[i][1]) <= self.radios[i] for i in range(self.n))

                elif getattr(self, 'seq_model', False):
                    self.modelo.addConstrs((self.p_x[k] - sum(self.x[i,k]*self.centros[i][0] for i in range(self.n))) + (self.p_y[k] - sum(self.x[i,k]*self.centros[i][1] for i in range(self.n))) <= sum(self.x[i,k]*self.radios[i] for i in range(self.n)) for k in range(self.n))
                    self.modelo.addConstrs((self.p_x[k] - sum(self.x[i,k]*self.centros[i][0] for i in range(self.n))) - (self.p_y[k] - sum(self.x[i,k]*self.centros[i][1] for i in range(self.n))) <= sum(self.x[i,k]*self.radios[i] for i in range(self.n)) for k in range(self.n))
                    self.modelo.addConstrs(-(self.p_x[k] - sum(self.x[i,k]*self.centros[i][0] for i in range(self.n))) + (self.p_y[k] - sum(self.x[i,k]*self.centros[i][1] for i in range(self.n))) <= sum(self.x[i,k]*self.radios[i] for i in range(self.n)) for k in range(self.n))
                    self.modelo.addConstrs(-(self.p_x[k] - sum(self.x[i,k]*self.centros[i][0] for i in range(self.n))) - (self.p_y[k] - sum(self.x[i,k]*self.centros[i][1] for i in range(self.n))) <= sum(self.x[i,k]*self.radios[i] for i in range(self.n)) for k in range(self.n))

            elif self.decomposition:
                if getattr(self, 'arc_model', False):
                    self.sub.addConstrs((self.p_x[i] - self.centros[i][0]) + (self.p_y[i] - self.centros[i][1]) <= self.radios[i] for i in range(self.n))
                    self.sub.addConstrs((self.p_x[i] - self.centros[i][0]) - (self.p_y[i] - self.centros[i][1]) <= self.radios[i] for i in range(self.n))
                    self.sub.addConstrs(-(self.p_x[i] - self.centros[i][0]) + (self.p_y[i] - self.centros[i][1]) <= self.radios[i] for i in range(self.n))
                    self.sub.addConstrs(-(self.p_x[i] - self.centros[i][0]) - (self.p_y[i] - self.centros[i][1]) <= self.radios[i] for i in range(self.n))

                elif getattr(self, 'seq_model', False):
                    self.neigh_con_1 = self.sub.addConstrs((self.p_x[k] - sum(self.x_sol[i,k]*self.centros[i][0] for i in range(self.n))) + (self.p_y[k] - sum(self.x_sol[i,k]*self.centros[i][1] for i in range(self.n))) <= sum(self.x_sol[i,k]*self.radios[i] for i in range(self.n)) for k in range(self.n))
                    self.neigh_con_2 = self.sub.addConstrs((self.p_x[k] - sum(self.x_sol[i,k]*self.centros[i][0] for i in range(self.n))) - (self.p_y[k] - sum(self.x_sol[i,k]*self.centros[i][1] for i in range(self.n))) <= sum(self.x_sol[i,k]*self.radios[i] for i in range(self.n)) for k in range(self.n))
                    self.neigh_con_3 = self.sub.addConstrs(-(self.p_x[k] - sum(self.x_sol[i,k]*self.centros[i][0] for i in range(self.n))) + (self.p_y[k] - sum(self.x_sol[i,k]*self.centros[i][1] for i in range(self.n))) <= sum(self.x_sol[i,k]*self.radios[i] for i in range(self.n)) for k in range(self.n))
                    self.neigh_con_4 = self.sub.addConstrs(-(self.p_x[k] - sum(self.x_sol[i,k]*self.centros[i][0] for i in range(self.n))) - (self.p_y[k] - sum(self.x_sol[i,k]*self.centros[i][1] for i in range(self.n))) <= sum(self.x_sol[i,k]*self.radios[i] for i in range(self.n)) for k in range(self.n))

        elif self.norm == 2:

            if not self.extended or getattr(self, 'sub_flag', False):
                if getattr(self, 'arc_model', False):
                    if not self.decomposition:
                        self.modelo.addConstrs((self.p_x[i] - self.centros[i][0])**2 + (self.p_y[i] - self.centros[i][1])**2 <= self.radios[i]**2 for i in range(self.n))
                    elif self.decomposition:
                        self.sub.addConstrs((self.p_x[i] - self.centros[i][0])**2 + (self.p_y[i] - self.centros[i][1])**2 <= self.radios[i]**2 for i in range(self.n))

                elif getattr(self, 'seq_model', False):
                    if not self.decomposition:
                        self.modelo.addConstrs((self.p_x[k] - sum(self.x[i,k]*self.centros[i][0] for i in range(self.n)))**2 + (self.p_y[k] - sum(self.x[i,k]*self.centros[i][1] for i in range(self.n)))**2 <= sum(self.x[i,k]*self.radios[i] for i in range(self.n))**2 for k in range(self.n))
                    elif self.decomposition:
                        self.sub.addConstrs((self.p_x[k] - sum(self.x_sol[i,k]*self.centros[i][0] for i in range(self.n)))**2 + (self.p_y[k] - sum(self.x_sol[i,k]*self.centros[i][1] for i in range(self.n)))**2 <= sum(self.x_sol[i,k]*self.radios[i] for i in range(self.n))**2 for k in range(self.n))

            elif self.extended:
                xi_c = self.modelo.addVars(self.n, self.nu + 1, vtype=GRB.CONTINUOUS, name="xi_c")
                eta_c = self.modelo.addVars(self.n, self.nu + 1, vtype=GRB.CONTINUOUS, name="eta_c")

                self.modelo.addConstrs(xi_c[k,j] == np.cos(np.pi * 2**(-(j+1))) * xi_c[k,j-1] + np.sin(np.pi * 2**(-(j+1))) * eta_c[k,j-1] for k in range(self.n) for j in range(1, self.nu + 1))
                self.modelo.addConstrs(eta_c[k,j] >= -np.sin(np.pi * 2**(-(j+1))) * xi_c[k,j-1] + np.cos(np.pi * 2**(-(j+1))) * eta_c[k,j-1] for k in range(self.n) for j in range(1, self.nu + 1))
                self.modelo.addConstrs(eta_c[k,j] >= np.sin(np.pi * 2**(-(j+1))) * xi_c[k,j-1] - np.cos(np.pi * 2**(-(j+1))) * eta_c[k,j-1] for k in range(self.n) for j in range(1, self.nu + 1))

                self.modelo.addConstrs(eta_c[k,self.nu] <= np.tan(np.pi * 2**(-(self.nu + 1))) * xi_c[k,self.nu] for k in range(self.n))

                if getattr(self, 'arc_model', False):
                    self.modelo.addConstrs(xi_c[i,0] >= self.p_x[i] - self.centros[i][0] for i in range(self.n))
                    self.modelo.addConstrs(xi_c[i,0] >= -self.p_x[i] + self.centros[i][0] for i in range(self.n))
                    self.modelo.addConstrs(eta_c[i,0] >= self.p_y[i] - self.centros[i][1] for i in range(self.n))
                    self.modelo.addConstrs(eta_c[i,0] >= -self.p_y[i] + self.centros[i][1] for i in range(self.n))

                    self.modelo.addConstrs(xi_c[i,self.nu] <= self.radios[i] for i in range(self.n))

                elif getattr(self, 'seq_model', False):
                    self.modelo.addConstrs(xi_c[k,0] >= sum(self.x[i,k]*self.centros[i][0] for i in range(self.n)) - self.p_x[k] for k in range(self.n))
                    self.modelo.addConstrs(xi_c[k,0] >= -sum(self.x[i,k]*self.centros[i][0] for i in range(self.n)) + self.p_x[k] for k in range(self.n))
                    self.modelo.addConstrs(eta_c[k,0] >= sum(self.x[i,k]*self.centros[i][1] for i in range(self.n)) - self.p_y[k] for k in range(self.n))
                    self.modelo.addConstrs(eta_c[k,0] >= -sum(self.x[i,k]*self.centros[i][1] for i in range(self.n)) + self.p_y[k] for k in range(self.n))

                    self.modelo.addConstrs(xi_c[k,self.nu] <= sum(self.x[i,k]*self.radios[i] for i in range(self.n)) for k in range(self.n))


    def __create_subtour_constraints(self):

        u = self.modelo.addVars(self.n, self.n, vtype=GRB.CONTINUOUS, name="u")

        self.modelo.addConstrs(sum(u[i, j] for j in range(self.n)) - sum(u[j, i] for j in range(self.n)) == -1 for i in range(1, self.n))
        self.modelo.addConstr(sum(u[0, j] for j in range(1, self.n)) - sum(u[j, 0] for j in range(1, self.n)) == self.n - 1)
        self.modelo.addConstrs(u[i, j] <= (self.n - 1) * self.x[i, j] for i in range(self.n) for j in range(self.n))

    def __create_distance_constraints(self):
            
        if self.norm == GRB.INFINITY:

            if not self.decomposition:
                if getattr(self, 'arc_model', False):
                    self.modelo.addConstrs(self.d[i,j] >= self.p_x[i] - self.p_x[j] for i in range(self.n) for j in range(self.n))
                    self.modelo.addConstrs(self.d[i,j] >= -self.p_x[i] + self.p_x[j] for i in range(self.n) for j in range(self.n))
                    self.modelo.addConstrs(self.d[i,j] >= self.p_y[i] - self.p_y[j] for i in range(self.n) for j in range(self.n))
                    self.modelo.addConstrs(self.d[i,j] >= -self.p_y[i] + self.p_y[j] for i in range(self.n) for j in range(self.n))

                
                elif getattr(self, 'seq_model', False):
                    self.modelo.addConstrs(self.d[k] >= self.p_x[k] - self.p_x[k+1] for k in range(self.n - 1))
                    self.modelo.addConstr(self.d[self.n - 1] >= self.p_x[self.n - 1] - self.p_x[0])
                    self.modelo.addConstrs(self.d[k] >= self.p_y[k] - self.p_y[k+1] for k in range(self.n - 1))
                    self.modelo.addConstr(self.d[self.n - 1] >= self.p_y[self.n - 1] - self.p_y[0])
                    self.modelo.addConstrs(self.d[k] >= -self.p_x[k] + self.p_x[k+1] for k in range(self.n - 1))
                    self.modelo.addConstr(self.d[self.n - 1] >= -self.p_x[self.n - 1] + self.p_x[0])
                    self.modelo.addConstrs(self.d[k] >= -self.p_y[k] + self.p_y[k+1] for k in range(self.n - 1))
                    self.modelo.addConstr(self.d[self.n - 1] >= -self.p_y[self.n - 1] + self.p_y[0])

            elif self.decomposition:
                if getattr(self, 'arc_model', False):
                    self.dist_con_1 = self.sub.addConstrs(self.d[i,j] >= self.p_x[i] - self.p_x[j] for i,j in self.soporte)
                    self.dist_con_2 = self.sub.addConstrs(self.d[i,j] >= -self.p_x[i] + self.p_x[j] for i,j in self.soporte)
                    self.dist_con_3 = self.sub.addConstrs(self.d[i,j] >= self.p_y[i] - self.p_y[j] for i,j in self.soporte)
                    self.dist_con_4 = self.sub.addConstrs(self.d[i,j] >= -self.p_y[i] + self.p_y[j] for i,j in self.soporte)
                
                elif getattr(self, 'seq_model', False):
                    self.dist_con_1 = self.sub.addConstrs(self.d[k] >= self.p_x[k] - self.p_x[k+1] for k in range(self.n - 1))
                    self.dist_con_1[self.n-1] = self.sub.addConstr(self.d[self.n - 1] >= self.p_x[self.n - 1] - self.p_x[0])

                    self.dist_con_2 = self.sub.addConstrs(self.d[k] >= self.p_y[k] - self.p_y[k+1] for k in range(self.n - 1))
                    self.dist_con_2[self.n-1] = self.sub.addConstr(self.d[self.n - 1] >= self.p_y[self.n - 1] - self.p_y[0])

                    self.dist_con_3 = self.sub.addConstrs(self.d[k] >= -self.p_x[k] + self.p_x[k+1] for k in range(self.n - 1))
                    self.dist_con_3[self.n-1] = self.sub.addConstr(self.d[self.n - 1] >= -self.p_x[self.n - 1] + self.p_x[0])

                    self.dist_con_4 = self.sub.addConstrs(self.d[k] >= -self.p_y[k] + self.p_y[k+1] for k in range(self.n - 1))
                    self.dist_con_4[self.n-1] = self.sub.addConstr(self.d[self.n - 1] >= -self.p_y[self.n - 1] + self.p_y[0])

        elif self.norm == 1:

            if not self.decomposition:
                if getattr(self, 'arc_model', False):
                    self.modelo.addConstrs(self.d[i,j] >= (self.p_x[i] - self.p_x[j]) + (self.p_y[i] - self.p_y[j]) for i in range(self.n) for j in range(self.n))
                    self.modelo.addConstrs(self.d[i,j] >= (self.p_x[i] - self.p_x[j]) - (self.p_y[i] - self.p_y[j]) for i in range(self.n) for j in range(self.n))
                    self.modelo.addConstrs(self.d[i,j] >= -(self.p_x[i] - self.p_x[j]) + (self.p_y[i] - self.p_y[j]) for i in range(self.n) for j in range(self.n))
                    self.modelo.addConstrs(self.d[i,j] >= -(self.p_x[i] - self.p_x[j]) - (self.p_y[i] - self.p_y[j]) for i in range(self.n) for j in range(self.n))

                elif getattr(self, 'seq_model', False):
                    self.modelo.addConstrs(self.d[k] >= (self.p_x[k] - self.p_x[k+1]) + (self.p_y[k] - self.p_y[k+1]) for k in range(self.n - 1))
                    self.modelo.addConstr(self.d[self.n - 1] >= (self.p_x[self.n - 1] - self.p_x[0]) + (self.p_y[self.n - 1] - self.p_y[0]))
                    self.modelo.addConstrs(self.d[k] >= (self.p_x[k] - self.p_x[k+1]) - (self.p_y[k] - self.p_y[k+1]) for k in range(self.n - 1))
                    self.modelo.addConstr(self.d[self.n - 1] >= (self.p_x[self.n - 1] - self.p_x[0]) - (self.p_y[self.n - 1] - self.p_y[0]))
                    self.modelo.addConstrs(self.d[k] >= -(self.p_x[k] - self.p_x[k+1]) + (self.p_y[k] - self.p_y[k+1]) for k in range(self.n - 1))
                    self.modelo.addConstr(self.d[self.n - 1] >= -(self.p_x[self.n - 1] - self.p_x[0]) + (self.p_y[self.n - 1] - self.p_y[0]))
                    self.modelo.addConstrs(self.d[k] >= -(self.p_x[k] - self.p_x[k+1]) - (self.p_y[k] - self.p_y[k+1]) for k in range(self.n - 1))
                    self.modelo.addConstr(self.d[self.n - 1] >= -(self.p_x[self.n - 1] - self.p_x[0]) - (self.p_y[self.n - 1] - self.p_y[0]))

            elif self.decomposition:
                if getattr(self, 'arc_model', False):
                    self.sub.addConstrs(self.d[i,j] >= (self.p_x[i] - self.p_x[j]) + (self.p_y[i] - self.p_y[j]) for i,j in self.soporte)
                    self.sub.addConstrs(self.d[i,j] >= (self.p_x[i] - self.p_x[j]) - (self.p_y[i] - self.p_y[j]) for i,j in self.soporte)
                    self.sub.addConstrs(self.d[i,j] >= -(self.p_x[i] - self.p_x[j]) + (self.p_y[i] - self.p_y[j]) for i,j in self.soporte)
                    self.sub.addConstrs(self.d[i,j] >= -(self.p_x[i] - self.p_x[j]) - (self.p_y[i] - self.p_y[j]) for i,j in self.soporte)

                elif getattr(self, 'seq_model', False):
                    self.sub.addConstrs(self.d[k] >= (self.p_x[k] - self.p_x[k+1]) + (self.p_y[k] - self.p_y[k+1]) for k in range(self.n - 1))
                    self.sub.addConstr(self.d[self.n - 1] >= (self.p_x[self.n - 1] - self.p_x[0]) + (self.p_y[self.n - 1] - self.p_y[0]))
                    self.sub.addConstrs(self.d[k] >= (self.p_x[k] - self.p_x[k+1]) - (self.p_y[k] - self.p_y[k+1]) for k in range(self.n - 1))
                    self.sub.addConstr(self.d[self.n - 1] >= (self.p_x[self.n - 1] - self.p_x[0]) - (self.p_y[self.n - 1] - self.p_y[0]))
                    self.sub.addConstrs(self.d[k] >= -(self.p_x[k] - self.p_x[k+1]) + (self.p_y[k] - self.p_y[k+1]) for k in range(self.n - 1))
                    self.sub.addConstr(self.d[self.n - 1] >= -(self.p_x[self.n - 1] - self.p_x[0]) + (self.p_y[self.n - 1] - self.p_y[0]))
                    self.sub.addConstrs(self.d[k] >= -(self.p_x[k] - self.p_x[k+1]) - (self.p_y[k] - self.p_y[k+1]) for k in range(self.n - 1))
                    self.sub.addConstr(self.d[self.n - 1] >= -(self.p_x[self.n - 1] - self.p_x[0]) - (self.p_y[self.n - 1] - self.p_y[0]))

        elif self.norm == 2:

            if not self.extended or getattr(self, 'sub_flag', False):
                
                if not self.decomposition:
                    if getattr(self, 'arc_model', False):
                        self.modelo.addConstrs(self.d[i,j]**2 >= (self.p_x[i] - self.p_x[j])**2 + (self.p_y[i] - self.p_y[j])**2 for i in range(self.n) for j in range(self.n))

                    elif getattr(self, 'seq_model', False):
                        self.modelo.addConstrs(self.d[k]**2 >= (self.p_x[k] - self.p_x[k+1])**2 + (self.p_y[k] - self.p_y[k+1])**2 for k in range(self.n - 1))
                        self.modelo.addConstr(self.d[self.n - 1]**2 >= (self.p_x[self.n - 1] - self.p_x[0])**2 + (self.p_y[self.n - 1] - self.p_y[0])**2)

                elif self.decomposition:
                    if getattr(self, 'arc_model', False):
                        self.sub.addConstrs(self.d[i,j]**2 >= (self.p_x[i] - self.p_x[j])**2 + (self.p_y[i] - self.p_y[j])**2 for i,j in self.soporte)

                    elif getattr(self, 'seq_model', False):
                        self.sub.addConstrs(self.d[k]**2 >= (self.p_x[k] - self.p_x[k+1])**2 + (self.p_y[k] - self.p_y[k+1])**2 for k in range(self.n - 1))
                        self.sub.addConstr(self.d[self.n - 1]**2 >= (self.p_x[self.n - 1] - self.p_x[0])**2 + (self.p_y[self.n - 1] - self.p_y[0])**2)
            
            elif self.extended:
                if getattr(self, 'arc_model', False):

                    xi_d = self.modelo.addVars(self.n, self.n, self.nu + 1, vtype=GRB.CONTINUOUS, name="xi_d")
                    eta_d = self.modelo.addVars(self.n, self.n, self.nu + 1, vtype=GRB.CONTINUOUS, name="eta_d")

                    self.modelo.addConstrs(xi_d[i,j,0] >= self.p_x[i] - self.p_x[j] for i in range(self.n) for j in range(self.n))
                    self.modelo.addConstrs(xi_d[i,j,0] >= -self.p_x[i] + self.p_x[j] for i in range(self.n) for j in range(self.n))
                    self.modelo.addConstrs(eta_d[i,j,0] >= self.p_y[i] - self.p_y[j] for i in range(self.n) for j in range(self.n))
                    self.modelo.addConstrs(eta_d[i,j,0] >= -self.p_y[i] + self.p_y[j] for i in range(self.n) for j in range(self.n))

                    self.modelo.addConstrs(xi_d[i,j,k] == np.cos(np.pi * 2**(-(k+1))) * xi_d[i,j,k-1] + np.sin(np.pi * 2**(-(k+1))) * eta_d[i,j,k-1] for i in range(self.n) for j in range(self.n) for k in range(1, self.nu + 1))
                    self.modelo.addConstrs(eta_d[i,j,k] >= -np.sin(np.pi * 2**(-(k+1))) * xi_d[i,j,k-1] + np.cos(np.pi * 2**(-(k+1))) * eta_d[i,j,k-1] for i in range(self.n) for j in range(self.n) for k in range(1, self.nu + 1))
                    self.modelo.addConstrs(eta_d[i,j,k] >= np.sin(np.pi * 2**(-(k+1))) * xi_d[i,j,k-1] - np.cos(np.pi * 2**(-(k+1))) * eta_d[i,j,k-1] for i in range(self.n) for j in range(self.n) for k in range(1, self.nu + 1))

                    self.modelo.addConstrs(xi_d [i,j,self.nu] <= self.d[i,j] for i in range(self.n) for j in range(self.n))
                    self.modelo.addConstrs(eta_d[i,j,self.nu] <= np.tan(np.pi * 2**(-(self.nu + 1))) * xi_d[i,j,self.nu] for i in range(self.n) for j in range(self.n))

                elif getattr(self, 'seq_model', False):

                    xi_d = self.modelo.addVars(self.n, self.nu + 1, vtype=GRB.CONTINUOUS, name="xi_d")
                    eta_d = self.modelo.addVars(self.n, self.nu + 1, vtype=GRB.CONTINUOUS, name="eta_d")

                    self.modelo.addConstrs(xi_d[k,0] >= self.p_x[k] - self.p_x[k + 1] for k in range(self.n - 1))
                    self.modelo.addConstr(xi_d[self.n - 1,0] >= self.p_x[self.n - 1] - self.p_x[0])
                    self.modelo.addConstrs(xi_d[k,0] >= -self.p_x[k] + self.p_x[k + 1] for k in range(self.n - 1))
                    self.modelo.addConstr(xi_d[self.n - 1,0] >= -self.p_x[self.n - 1] + self.p_x[0])
                    self.modelo.addConstrs(eta_d[k,0] >= self.p_y[k] - self.p_y[k + 1] for k in range(self.n - 1))
                    self.modelo.addConstr(eta_d[self.n - 1,0] >= self.p_y[self.n - 1] - self.p_y[0])
                    self.modelo.addConstrs(eta_d[k,0] >= -self.p_y[k] + self.p_y[k + 1] for k in range(self.n - 1))
                    self.modelo.addConstr(eta_d[self.n - 1,0] >= -self.p_y[self.n - 1] + self.p_y[0])

                    self.modelo.addConstrs(xi_d[k,j] == np.cos(np.pi * 2**(-(j+1))) * xi_d[k,j-1] + np.sin(np.pi * 2**(-(j+1))) * eta_d[k,j-1] for k in range(self.n) for j in range(1, self.nu + 1))
                    self.modelo.addConstrs(eta_d[k,j] >= -np.sin(np.pi * 2**(-(j+1))) * xi_d[k,j-1] + np.cos(np.pi * 2**(-(j+1))) * eta_d[k,j-1] for k in range(self.n) for j in range(1, self.nu + 1))
                    self.modelo.addConstrs(eta_d[k,j] >= np.sin(np.pi * 2**(-(j+1))) * xi_d[k,j-1] - np.cos(np.pi * 2**(-(j+1))) * eta_d[k,j-1] for k in range(self.n) for j in range(1, self.nu + 1))

                    self.modelo.addConstrs(xi_d[k,self.nu] <= self.d[k] for k in range(self.n))
                    self.modelo.addConstrs(eta_d[k,self.nu] <= np.tan(np.pi * 2**(-(self.nu + 1))) * xi_d[k,self.nu] for k in range(self.n))

    def __create_big_M_constraints(self):

        M = self.__compute_big_M()

        self.modelo.addConstrs(self.d[i,j] - self.M[i,j]*(1 - self.x[i,j]) <= self.d_aux[i,j] for i in range(self.n) for j in range(self.n))
        
    def __compute_big_M(self):

        self.M = np.zeros((self.n, self.n))

        if self.norm == 2:
            for i in range(self.n):
                for j in range(self.n):
                    self.M[i,j] = self.radios[i] + np.sqrt((self.centros[i][0] - self.centros[j][0])**2 + (self.centros[i][1] - self.centros[j][1])**2) + self.radios[j]
        
        elif self.norm == 1:
            for i in range(self.n):
                for j in range(self.n):
                    self.M[i,j] = self.radios[i] + abs(self.centros[i][0] - self.centros[j][0]) + abs(self.centros[i][1] - self.centros[j][1]) + self.radios[j]

        elif self.norm == GRB.INFINITY:
            for i in range(self.n):
                for j in range(self.n):
                    self.M[i,j] = self.radios[i] + max(abs(self.centros[i][0] - self.centros[j][0]), abs(self.centros[i][1] - self.centros[j][1])) + self.radios[j]

    def __set_objective(self):

        if not self.decomposition:
            
            if getattr(self, 'arc_model', False):
                self.modelo.setObjective(sum(self.d_aux[i,j] for i in range(self.n) for j in range(self.n)), GRB.MINIMIZE)

            elif getattr(self, 'seq_model', False):
                self.modelo.setObjective(sum(self.d[k] for k in range(self.n)), GRB.MINIMIZE)

        elif self.decomposition:

            if not self.extended:

                if getattr(self, 'arc_model', False):
                    self.__compute_distance_estimations()
                    self.modelo.setObjective(sum(self.x[i,j]*self.estimation[i,j] for i in range(self.n) for j in range(self.n)) + self.theta, GRB.MINIMIZE)

                elif getattr(self, 'seq_model', False):
                    self.modelo.setObjective(self.theta, GRB.MINIMIZE)

            elif self.extended:

                if getattr(self, 'arc_model', False):
                    self.modelo.setObjective(sum(self.d_aux[i,j] for i in range(self.n) for j in range(self.n)) + self.theta, GRB.MINIMIZE)
                
                elif getattr(self, 'seq_model', False):
                    self.modelo.setObjective(sum(self.d[k] for k in range(self.n)) + self.theta, GRB.MINIMIZE)

    def __retrieve_arcs(self):
        if getattr(self, 'arc_model', False):
            self.arcs = []

            for i in range(self.n):
                for j in range(self.n):
                    if self.x[i,j].x > 0.5:
                        self.arcs.append((i, j))

        elif getattr(self, 'seq_model', False):
            self.arcs = []

            for k in range(self.n-1):
                for i in range(self.n):
                    for j in range(self.n):
                        if self.x[i,k].x > 0.5 and self.x[j,k+1].x > 0.5:
                            self.arcs.append((i, j))
            for i in range(self.n):
                for j in range(self.n):
                    if self.x[i,self.n-1].x > 0.5 and self.x[j,0].x > 0.5:
                        self.arcs.append((i, j))


    def __compute_optimal_points(self):

        aux_model = Model()
        aux_model.Params.OutputFlag = 0

        p = aux_model.addVars(self.n, 2, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="p")

        if not self.decomposition or (self.decomposition and not self.extended and self.norm == 2):
            d = aux_model.addVars(self.arcs, vtype=GRB.CONTINUOUS, name="d")
        elif self.decomposition and self.extended:
            d = aux_model.addVars(self.aux_arcs, vtype=GRB.CONTINUOUS, name="d")

        aux_model.addConstrs((p[i,0]-self.centros[i][0])**2 + (p[i,1]-self.centros[i][1])**2 <= self.radios[i]**2 for i in range(self.n))

        if not self.decomposition or (self.decomposition and not self.extended and self.norm == 2):            
            aux_model.addConstrs(d[i,j]**2 >= (p[i,0] - p[j,0])**2 + (p[i,1] - p[j,1])**2 for i, j in self.arcs)
            aux_model.setObjective(sum(d[i,j] for i, j in self.arcs), GRB.MINIMIZE)

        elif self.decomposition and self.extended:
            aux_model.addConstrs(d[i,j]**2 >= (p[i,0] - p[j,0])**2 + (p[i,1] - p[j,1])**2 for i, j in self.aux_arcs)
            aux_model.setObjective(sum(d[i,j] for i, j in self.aux_arcs), GRB.MINIMIZE)

        aux_model.optimize()

        if not self.decomposition or (self.decomposition and not self.extended and self.norm == 2):
            self.ub = aux_model.objVal
            self.aux_runtime = aux_model.Runtime
            self.p = {}
            for i in range(self.n):
                self.p[i] = (p[i,0].x, p[i,1].x)

        elif self.decomposition and self.extended:
            self.sub_value = aux_model.objVal
            if self.sub_value < self.ub:
                self.ub = self.sub_value
                self.arcs = self.aux_arcs
                for i in range(self.n):
                    self.p[i] = (p[i,0].x, p[i,1].x)

    def __compute_distance_estimations(self):

        self.estimation = np.zeros((self.n, self.n))
        # compute the minimum possible distance between two neighborhoods, considering the type of norm used
        if self.norm == 2:
            for i in range(self.n):
                for j in range(self.n):
                    self.estimation[i,j] = max(np.sqrt((self.centros[i][0] - self.centros[j][0])**2 + (self.centros[i][1] - self.centros[j][1])**2) - self.radios[i] - self.radios[j], 0)
        elif self.norm == 1:
            for i in range(self.n):
                for j in range(self.n):
                    self.estimation[i,j] = max(abs(self.centros[i][0] - self.centros[j][0]) + abs(self.centros[i][1] - self.centros[j][1]) - self.radios[i] - self.radios[j], 0)
        elif self.norm == GRB.INFINITY:
            for i in range(self.n):
                for j in range(self.n):
                    self.estimation[i,j] = max(max(abs(self.centros[i][0] - self.centros[j][0]), abs(self.centros[i][1] - self.centros[j][1])) - self.radios[i] - self.radios[j], 0)

############# DECOMPOSITION METHODS #############
    def __arc_subproblem(self):
            
        # if self.norm == 2 and not self.extended:
        #     self.__arc_dual()
        #     self.sub_value = self.dual.ObjVal

        #     if self.sub_value < self.ub:
        #         self.ub = self.sub_value
        #         self.arcs = self.soporte


        self.sub = Model()
        self.sub.Params.OutputFlag = 0

        self.p_x = self.sub.addVars(self.n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="p_x")
        self.p_y = self.sub.addVars(self.n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="p_y")

        self.d = self.sub.addVars(self.soporte, vtype=GRB.CONTINUOUS, name="d")

        self.__create_neigh_constraints()
        self.__create_distance_constraints()

        self.sub.setObjective(sum(self.d[i,j] for i, j in self.soporte), GRB.MINIMIZE)

        self.sub.optimize()

        self.sub_value = self.sub.objVal

        if self.sub_value < self.ub:
            self.ub = self.sub.objVal 
            self.arcs = self.soporte
            for i in range(self.n):
                self.p[i] = (self.p_x[i].x, self.p_y[i].x)

    def __seq_subproblem(self):

        self.sub = Model()
        self.sub.Params.OutputFlag = 0
        self.sub.Params.QCPDual = 0

        self.p_x = self.sub.addVars(self.n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="p_x")
        self.p_y = self.sub.addVars(self.n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="p_y")

        self.d = self.sub.addVars(self.n, vtype=GRB.CONTINUOUS, lb =0, name="d")

        self.sub_flag = True

        self.__create_neigh_constraints()
        self.__create_distance_constraints()

        self.sub.setObjective(sum(self.d[k] for k in range(self.n)), GRB.MINIMIZE)

        self.sub.optimize()

        if self.norm != 2:
            self.sub_value = self.sub.ObjVal

            if self.sub_value < self.ub:
                self.ub = self.sub.objVal
                for i in range(self.n):
                    self.p[i] = (self.p_x[i].x, self.p_y[i].x)

        elif self.norm == 2 and self.extended and self.decomposition:
            self.sub_value = self.sub.objVal

            if self.sub_value < self.ub:
                self.ub = self.sub.objVal
                self.best_sol = self.x_sol
        
        elif self.norm == 2:
            for k in range(self.n):
                self.p[k] = (self.p_x[k].x, self.p_y[k].x)

    def __optcut_arc(self, model, where):

        if where == GRB.Callback.MIPSOL:

            self.x_sol = {}
            self.soporte = []
            delta = LinExpr()
            delta_sim = LinExpr()

            for i in range(self.n):
                for j in range(self.n):
                    if model.cbGetSolution(self.x[i,j]) > 0.5:
                        self.x_sol[i,j] = 1
                        self.soporte.append((i,j))
                        delta = delta + (1 - self.x[i,j])
                        delta_sim = delta_sim + (1 - self.x[j,i])
                    elif model.cbGetSolution(self.x[i,j]) < 0.5:
                        self.x_sol[i,j] = 0

            if not self.extended:
                self.current_objective = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                self.current_estimation = sum(self.estimation[i,j] for i,j in self.soporte)
                self.__arc_subproblem()
                Q = self.sub_value - self.current_estimation

            elif self.extended:
                self.aux_arcs = self.soporte
                self.__compute_optimal_points()
                self.current_objective = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                Q = self.sub_value - self.current_objective
            
            if model.cbGetSolution(self.theta) < Q - 0.01:
                # print(f"\nValor de theta: {model.cbGetSolution(self.theta)}, Valor de Q: {Q}")
                # self.cut_count += 1
                if self.norm == 2 and not self.extended:

                    # FEASIBILITY PROBLEM ==================================
                    # print("\n===============Feasibility problem==================\n")
                    # self.iis = []
                    # self.sub_feas = Model()
                    # self.sub_feas.Params.OutputFlag = 0
                    # self.sub_feas.Params.TimeLimit = 1

                    # self.sub_feas.setObjective(0, GRB.MINIMIZE)

                    # p_x = self.sub_feas.addVars(self.n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="p_x")
                    # p_y = self.sub_feas.addVars(self.n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="p_y")

                    # d = self.sub_feas.addVars(self.soporte, vtype=GRB.CONTINUOUS, name="d")
                    # d_a = self.sub_feas.addVars(self.soporte, vtype=GRB.CONTINUOUS, name="d_a")

                    # aux_var = self.sub_feas.addVar(vtype=GRB.CONTINUOUS, name="aux_var")

                    # aux_con_1 = self.sub_feas.addConstr(aux_var <= 0)
                    # aux_con_2 = self.sub_feas.addConstr(aux_var >= 1)
                    # aux_con_1.setAttr(GRB.Attr.IISConstrForce, 0)
                    # aux_con_2.setAttr(GRB.Attr.IISConstrForce, 0)

                    # # x = self.sub_feas.addVars(self.n, self.n, vtype=GRB.CONTINUOUS, name="x")

                    # # Neighborhood constraints
                    # neigh = self.sub_feas.addConstrs(((p_x[i]-self.centros[i][0])**2 + (p_y[i]-self.centros[i][1])**2 <= self.radios[i]**2 for i in range(self.n)), name="neigh")
                    # # Distance constraints
                    # dist = self.sub_feas.addConstrs((d[i,j]**2 >= (p_x[i] - p_x[j])**2 + (p_y[i] - p_y[j])**2 for i,j in self.soporte), name="dist")

                    # # aux x constraint
                    # # aux_con = self.sub_feas.addConstrs((x[i,j] == self.x_sol[i,j] for i in range(self.n) for j in range(self.n)), name="aux_con")

                    # # big M constraints
                    # big_M = self.sub_feas.addConstrs((d[i,j] - self.M[i,j]*(1 - self.x_sol[i,j]) <= d_a[i,j] for i,j in self.soporte), name="big_M")

                    # # feasibility constraint
                    # feas = self.sub_feas.addConstr(sum(d_a[i,j] for i,j in self.soporte) <= self.current_objective, name="feasibility")

                    # self.sub_feas.optimize()

                    # # print(f"Feasibility problem status: {self.sub_feas.status}")

                    # # for i,j in self.soporte:
                    # #     aux_con[i,j].setAttr(GRB.Attr.IISConstrForce, 1)
                    
                    # feas.setAttr(GRB.Attr.IISConstrForce, 1)

                    # if self.sub_feas.status == GRB.INFEASIBLE:

                    #     self.sub_feas.computeIIS()
                    #     # print("IIS found")
                    #     # print all the constraints in the IIS
                    #     for c in self.sub_feas.getConstrs():
                    #         if c.IISConstr:
                    #             # print(f"Constr {c.constrName} is in the IIS")
                    #             if "big_M" not in c.constrName and "feasibility" not in c.constrName:
                    #                 raise Exception(f"Constr {c.constrName} is in the IIS")
                    #     for i,j in self.soporte:
                    #         if big_M[i,j].IISConstr:
                    #             self.iis.append((i,j))

                    # delta = LinExpr()
                    # delta_sim = LinExpr()
                    # for i,j in self.iis:
                    #     delta += (1 - self.x[i,j])
                    #     delta_sim += (1 - self.x[j,i])

                    # END OF FEASIBILITY PROBLEM =========================
                    # model.cbLazy(sum(-self.lambda_c[i].x * self.radios[i] + self.nu_c[i,0].x*self.centros[i][0] + self.nu_c[i,1].x*self.centros[i][1] for i in range(self.n)) - sum(self.lambda_a[i,j].x*self.M[i,j]*(1-self.x[i,j]) for i in range(self.n) for j in range(self.n)) - self.current_estimation <= self.theta)
                    # self.cut_count += 1
                    model.cbLazy(-(Q/2 - self.L)*delta + Q <= self.theta)
                    self.cut_count += 1
                    model.cbLazy(-(Q/2 - self.L)*delta_sim + Q <= self.theta)
                    self.cut_count += 1
                else:
                    model.cbLazy(-(Q/2 - self.L)*delta + Q <= self.theta)
                    self.cut_count += 1
                    model.cbLazy(-(Q/2 - self.L)*delta_sim + Q <= self.theta)
                    self.cut_count += 1


    def __optcut_seq(self, model, where):

        if where == GRB.Callback.MIPSOL:

            self.x_sol = {}
            self.soporte = []
            delta = LinExpr()
            delta_sim = LinExpr()

            for i in range(self.n):
                for k in range(self.n):
                    if model.cbGetSolution(self.x[i,k]) > 0.5:
                        self.x_sol[i,k] = 1
                        self.soporte.append((i,k))
                        delta = delta + (1 - self.x[i,k])
                        if k != 0:
                            delta_sim = delta_sim + (1 - self.x[i, self.n - k])
                        else:
                            delta_sim = delta_sim + (1 - self.x[i,k])
                    elif model.cbGetSolution(self.x[i,k]) < 0.5:
                        self.x_sol[i,k] = 0

            if not self.extended:

                if self.norm != 2:
                    self.__seq_subproblem()
                    eta_1 = {k : self.neigh_con_1[k].pi for k in range(self.n)}
                    eta_2 = {k : self.neigh_con_2[k].pi for k in range(self.n)}
                    eta_3 = {k : self.neigh_con_3[k].pi for k in range(self.n)}
                    eta_4 = {k : self.neigh_con_4[k].pi for k in range(self.n)}

                elif self.norm == 2:
                    self.__seq_dual()

                dual_cut = LinExpr()

                if self.norm == GRB.INFINITY:
                    for k in range(self.n):
                        dual_cut += sum(self.x[i,k]*(eta_1[k]*(self.centros[i][0] - self.radios[i]) + eta_2[k]*(self.centros[i][0] + self.radios[i]) + eta_3[k]*(self.centros[i][1] - self.radios[i]) + eta_4[k]*(self.centros[i][1] + self.radios[i])) for i in range(self.n))
                elif self.norm == 1:
                    for k in range(self.n):
                        dual_cut += sum(self.x[i,k]*(eta_1[k]*(self.centros[i][0] + self.centros[i][1] + self.radios[i]) + eta_2[k]*(self.centros[i][0] - self.centros[i][1] + self.radios[i]) + eta_3[k]*(-self.centros[i][0] + self.centros[i][1] + self.radios[i]) + eta_4[k]*(-self.centros[i][0] - self.centros[i][1] + self.radios[i])) for i in range(self.n))

                print(f"\n Current objective value: {model.cbGet(GRB.Callback.MIPSOL_OBJ)}")
                print(f"Subproblem objective value: {self.sub_value}")

                if model.cbGetSolution(self.theta) < self.sub_value - 0.01:

                    self.cut_count += 1
                    if self.norm == 2:
                        for i in range(self.n):
                            for k in range(self.n):
                                dual_cut += self.nu_1[i,k].x*self.x[i,k]
                        
                        # model.cbLazy(-(self.sub_value/2 - self.L)*delta + self.sub_value <= self.theta)
                        # model.cbLazy(-(self.sub_value/2 - self.L)*delta_sim + self.sub_value <= self.theta)

                    model.cbLazy(dual_cut <= self.theta)

            elif self.extended:

                # solve primal subproblem approach
                # self.aux_arcs = []
                # for k in range(self.n-1):
                #     for i in range(self.n):
                #         for j in range(self.n):
                #             if self.x_sol[i,k] == 1 and self.x_sol[j,k+1] == 1:
                #                 self.aux_arcs.append((i,j))
                # for i in range(self.n):
                #     for j in range(self.n):
                #         if self.x_sol[i,self.n-1] == 1 and self.x_sol[j,0] == 1:
                #             self.aux_arcs.append((i,j))
                # self.__compute_optimal_points()
                self.current_objective = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                self.current_estimation = self.current_objective - model.cbGetSolution(self.theta)

                # self.__seq_subproblem()
                # print(f"Subproblem objective value: {self.sub_value}")

                # solve dual subproblem approach
                self.__seq_dual()
                dual_cut = LinExpr()
                for i in range(self.n):
                    for k in range(self.n):
                        dual_cut += self.nu_1[i,k].x*self.x[i,k]
                dual_cut -= self.current_estimation

                # self.current_objective = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                # print(f"Current objective value: {self.current_objective}")
                Q = self.sub_value - self.current_estimation

                if model.cbGetSolution(self.theta) < Q - 0.01:
                    model.cbLazy(-(Q/2 - self.L)*delta + Q <= self.theta)
                    model.cbLazy(-(Q/2 - self.L)*delta_sim + Q <= self.theta)
                    model.cbLazy(dual_cut <= self.theta)
                    self.cut_count += 3

    def __seq_dual(self):

        self.dual = Model()

        self.dual.Params.OutputFlag = 0

        self.lambda_n = self.dual.addVars(self.n, vtype=GRB.CONTINUOUS, name="lambda_n")
        self.lambda_d = self.dual.addVars(self.n, vtype=GRB.CONTINUOUS, name="lambda_d")
        self.lambda_c = self.dual.addVars(self.n, vtype=GRB.CONTINUOUS, name="lambda_c")

        self.nu_1 = self.dual.addVars(self.n, self.n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="nu_1")
        self.nu_c = self.dual.addVars(self.n, 2, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="nu_c")
        self.nu_d = self.dual.addVars(self.n, 2, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="nu_d")

        self.dual.addConstrs(1 - self.lambda_n[k] - self.lambda_d[k] == 0 for k in range(self.n))
        self.dual.addConstrs(self.nu_1[i,k] + self.centros[i][0]*self.nu_c[k,0] + self.centros[i][1]*self.nu_c[k,1] + self.radios[i]*self.lambda_c[k] == 0 for i in range(self.n) for k in range(self.n))
        self.dual.addConstrs(self.nu_d[k,0] - self.nu_d[k-1,0] + self.nu_c[k,0] == 0 for k in range(1, self.n))
        self.dual.addConstr(self.nu_d[0,0] - self.nu_d[self.n-1,0] + self.nu_c[0,0] == 0)
        self.dual.addConstrs(self.nu_d[k,1] - self.nu_d[k-1,1] + self.nu_c[k,1] == 0 for k in range(1, self.n))
        self.dual.addConstr(self.nu_d[0,1] - self.nu_d[self.n-1,1] + self.nu_c[0,1] == 0)

        self.dual.addConstrs(self.nu_d[k,0]**2 + self.nu_d[k,1]**2 <= self.lambda_d[k]**2 for k in range(self.n))
        self.dual.addConstrs(self.nu_c[k,0]**2 + self.nu_c[k,1]**2 <= self.lambda_c[k]**2 for k in range(self.n))

        self.dual.setObjective(sum(self.nu_1[i,k]*self.x_sol[i,k] for i in range(self.n) for k in range(self.n)), GRB.MAXIMIZE)

        self.dual.optimize()
        #print(f"Dual Objective value: {self.dual.objVal}")

        self.sub_value = self.dual.objVal

        if self.sub_value < self.ub:
            self.ub = self.sub_value
            self.best_sol = self.x_sol
                       
    def __arc_dual(self):

        self.__compute_big_M()

        self.dual = Model()

        self.dual.Params.OutputFlag = 0

        self.lambda_n = self.dual.addVars(self.n, self.n)
        self.lambda_d = self.dual.addVars(self.n, self.n)
        self.lambda_c = self.dual.addVars(self.n)
        self.lambda_a = self.dual.addVars(self.n, self.n)
        self.lambda_na = self.dual.addVars(self.n, self.n)

        self.nu_c = self.dual.addVars(self.n, 2, lb=-GRB.INFINITY)
        self.nu_d = self.dual.addVars(self.n, self.n, 2, lb=-GRB.INFINITY)

        self.dual.addConstrs(self.nu_c[i,0]**2 + self.nu_c[i,1]**2 <= self.lambda_c[i]**2 for i in range(self.n))
        self.dual.addConstrs(self.nu_d[i,j,0]**2 + self.nu_d[i,j,1]**2 <= self.lambda_d[i,j]**2 for i in range(self.n) for j in range(self.n))
        self.dual.addConstrs(self.lambda_a[i,j] - self.lambda_d[i,j] - self.lambda_n[i,j] == 0 for i in range(self.n) for j in range(self.n))
        self.dual.addConstrs(self.lambda_a[i,j] + self.lambda_na[i,j] == 1 for i in range(self.n) for j in range(self.n))
        self.dual.addConstrs(-self.nu_c[i,0] + sum(self.nu_d[j,i,0] - self.nu_d[i,j,0] for j in range(self.n)) == 0 for i in range(self.n))
        self.dual.addConstrs(-self.nu_c[i,1] + sum(self.nu_d[j,i,1] - self.nu_d[i,j,1] for j in range(self.n)) == 0 for i in range(self.n))

        self.dual.setObjective(sum(-self.lambda_c[i]*self.radios[i] + self.nu_c[i,0]*self.centros[i][0] + self.nu_c[i,1]*self.centros[i][1] for i in range(self.n)) - sum(self.lambda_a[i,j]*self.M[i,j]*(1 - self.x_sol[i,j]) for i in range(self.n) for j in range(self.n)), GRB.MAXIMIZE)
                               
        self.dual.optimize()

        #print(f"Dual Objective value: {self.dual.objVal}\n")

if __name__ == "__main__":

    np.random.seed(50)

    n = 13
    c = {}
    r = {}
    for i in range(n-2):
        c[i] = (np.random.uniform(0,10), np.random.uniform(0,10))
        r[i] = np.random.uniform(0,1)

    c[n-2] = (8,8)
    r[n-2] = 1
    c[n-1] = (8 + 1/2, 8 + 1/2)
    r[n-1] = 1/2
    
    data = [n, c, r]

    norma = "L_inf"
    type = "seq"

    modelo = ModelCETSP(data)
    modelo.compute_overlap_ratio()
    modelo.build_model(norma, type)  
    modelo.optimize(600)
    if modelo.modelo.status == GRB.INFEASIBLE:
        print("Modelo infactible")
        modelo.modelo.write("model.lp")
        modelo.modelo.computeIIS()
        modelo.modelo.write("model.ilp")
    modelo.get_solution()
    modelo.plot_solution()
