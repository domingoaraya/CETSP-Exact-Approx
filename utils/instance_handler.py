import numpy as np
import os

def generate_instance(n, r_min, r_max):
    """
    Generates a new CETSP instance with random values.
    """
    x_max = 10
    y_max = 10

    centros_x = np.random.uniform(0, x_max, n)
    centros_y = np.random.uniform(0, y_max, n)
    centros = {i: (centros_x[i], centros_y[i]) for i in range(n)}
    radios = np.random.uniform(r_min, r_max, n)

    radios[0] = 0

    return centros, radios

def create_and_save_instance(n, r_min, r_max, i):
    """
    Creates and saves a single CETSP instance to a text file.
    """
    np.random.seed(i)

    centros, radios = generate_instance(n, r_min, r_max)

    folder = f"Instances/{n}_{r_min}_{r_max}"
    os.makedirs(folder, exist_ok=True)
    title = f"instance_{i}"

    data = []
    for i in range(n):
        city_data = [i, centros[i][0], centros[i][1], radios[i]]
        data.append(city_data)

    output_path = os.path.join(folder, f"{title}.txt")
    with open(output_path, "w") as f:
        for row in data:
            row_str = " ".join(map(str, row))
            f.write(row_str + "\n")

def read_instance(file_path):
    """
    Reads a CETSP instance from a text file.
    """
    with open(file_path, "r") as f:
        lines = [line.split() for line in f]

    n = len(lines)
    centers = {int(parts[0]): (float(parts[1]), float(parts[2])) for parts in lines}
    radii = {int(parts[0]): float(parts[3]) for parts in lines}

    return n, centers, radii
