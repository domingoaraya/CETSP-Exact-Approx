import pandas as pd
from instance_handler import *
from itertools import product

time_limit = 60*10
ammount_of_instances = 5
nu = 3
norma = "L_2"
# tipo = "seq"

results = []

for tipo in ["seq"]:   
    # for ex, dec in product([False, True], repeat=2):
    for ex, dec in [(True, True)]:
        for n in [20]:

            if tipo == "arc":
                t = "ABF"

            elif tipo == "seq":
                t = "SBF"

            if ex:
                t += "-A"
            if dec:
                t += "-D"

            for r in [0.25, 0.5, 1]:

                for sigma in [0, 0.2, 0.5]:

                    r_min = r*(1-sigma)
                    r_max = r*(1+sigma)

                    sol = run_experiment(ammount_of_instances, n, r_min, r_max, norma, tipo, time_limit, extended=ex, decomposition=dec, nu=nu)

                    for i in range(ammount_of_instances):

                        row_data = {
                            "Formulation": t,
                            "n": n,
                            "r_min": r_min,
                            "r_max": r_max,
                            
                            "Instance": i,
                            "Overlap": sol[9][i],
                            "Status": sol[6][i],
                            "Gap": sol[7][i],
                            "Time": sol[8][i],
                            "Cuts": sol[10][i],
                        }

                        results.append(row_data)
                    
                print(f"Finished {t} with n={n}, r={r}, extended={ex}, decomposition={dec}")


results_df = pd.DataFrame(results)

title = f"Resultados/sbf-a-d-ll-and-dual-cuts_20.csv"

results_df.to_csv(title, index=False)
