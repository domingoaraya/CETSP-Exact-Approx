from classes import *
import os 
import tqdm.auto as tqdm

def generate_instance(n, r_min, r_max):

    x_max = 10
    y_max = 10

    centros_x = np.random.uniform(0, x_max, n)
    centros_y = np.random.uniform(0, y_max, n)
    centros = {i: (centros_x[i], centros_y[i]) for i in range(n)}
    radios = np.random.uniform(r_min, r_max, n)

    radios[0] = 0

    return centros, radios

def create_and_save_instance(n, r_min, r_max, i):

    np.random.seed(i)

    data = generate_instance(n, r_min, r_max)
    centros = data[0]
    radios = data[1]

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

    with open(file_path, "r") as f:
        lines = [line.split() for line in f]

    centros = {int(parts[0]): (float(parts[1]), float(parts[2])) for parts in lines}
    radios = {int(parts[0]): float(parts[3]) for parts in lines}

    return centros, radios

def run_experiment(repetitions, n, r_min, r_max, norma, tipo, time_limit = 600, extended = False, decomposition = False, nu = None):

    print(f"\nRunning {repetitions} instances of {n} cities with radii between {r_min} and {r_max}")
    print(f"Using {norma} norm, {tipo}-based formulation, time limit of {time_limit} seconds")
    if extended:
        print("Using extended formulation with parameter nu =", nu)
    if decomposition:
        print("Using decomposition")

    folder = f"Instances/{n}_{r_min}_{r_max}"
    os.makedirs(folder, exist_ok=True)
    instances = os.listdir(folder)
    
    if len(instances) < repetitions:
        for i in range(len(instances), repetitions):
            create_and_save_instance(n, r_min, r_max, i)

    instances = os.listdir(f"Instances/{n}_{r_min}_{r_max}")
    instances = instances[:repetitions]

    gaps = []
    runtimes = []
    overlaps = []
    opt = 0
    subopt = 0
    cuts = []
    status = []

    pbar = tqdm.tqdm(total=repetitions, desc="Solving instances")

    for instance in instances:

        i = int(instance.split("_")[1].split(".")[0])
        # print(f"Solving instance {i+1} of {repetitions}")

        centros, radios = read_instance(f"Instances/{n}_{r_min}_{r_max}/{instance}")

        data = [n, centros, radios]

        modelo = ModelCETSP(data)
        modelo.build_model(norma, tipo, decomposition, extended, nu)
        modelo.modelo.setParam("OutputFlag", 0)
        modelo.optimize(time_limit)
        modelo.get_solution()
        # modelo.plot_solution()
        # if modelo.extended:
        #     print(f"Value: {round(modelo.ub, 2)}")
        # else:
        #     print(f"Value: {round(modelo.modelo.objVal, 2)}")

        if modelo.modelo.Status == GRB.OPTIMAL:
            opt += 1
            status.append("Optimal")
            # print(f"Optimal solution found with value {modelo.modelo.objVal} in time {modelo.runtime} seconds")
        elif modelo.modelo.Status == GRB.SUBOPTIMAL:
            subopt += 1
            # print(f"\n Suboptimal solution found with gap {modelo.gap} \n")
            status.append("Suboptimal")
        elif modelo.modelo.Status == GRB.TIME_LIMIT:
            status.append("Time_limit")

        if getattr(modelo, "cut_count", None) is not None:
            cuts.append(modelo.cut_count)
        else:
            cuts.append(0)

        gaps.append(round(modelo.gap, 8))
        runtimes.append(round(modelo.runtime, 2))
        modelo.compute_overlap_ratio()
        overlaps.append(round(modelo.overlap, 4))

        pbar.update(1)

    pbar.close()

    if len(cuts) == 0:
        mean_cuts = 0
    else:
        mean_cuts = round((np.mean(cuts)), 2)

    return round(np.mean(gaps), 8), round(np.mean(runtimes), 2), round(np.mean(overlaps), 4), opt, subopt, mean_cuts, status, gaps, runtimes, overlaps, cuts