import numpy as np
import os
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import time

directory = '/Users/alaynabaker/Downloads/ISYE6679/Test Cases/All/'
out_folder = '/Users/alaynabaker/Downloads/ISYE6679/Results/Newer Gurobi/'

test_cases = [file for file in os.listdir(directory) if file.endswith('.csv')]

objective_vals = []
file_names = []
execution_times = []

for file in test_cases:
    start_time = time.time()

    Q = np.genfromtxt(os.path.join(directory, file), delimiter=',')
    n = len(Q)

    # Initialize minimum objective value
    min_obj_val = float('inf')

    # Create a new model
    model = gp.Model("QUBO")

    # Add binary decision variables
    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    # Set objective
    obj = gp.QuadExpr()
    for i in range(n):
        for j in range(n):
            obj += Q[i][j] * x[i] * x[j]
    model.setObjective(obj, GRB.MINIMIZE)

    # termination condition = 2500 iterations
    model.Params.IterationLimit = 2500

    # Optimization loop
    while True:
        model.optimize()
        obj_val = model.getObjective().getValue()

        # Check convergence condition
        if obj_val >= min_obj_val:
            break

        min_obj_val = obj_val

    objective_vals.append(min_obj_val)

    case = os.path.splitext(file)[0][-2:]
    file_names.append(case)

    end_time = time.time()
    execution_time = round(end_time - start_time, 3)
    execution_times.append(execution_time)

    print(f"Minimum objective value for {file}: {min_obj_val}")
    print(f"Execution time: {execution_time} seconds")

# Create DataFrame
data = {'Test Case': file_names, 'Objective Value': objective_vals, 'Execution Time (s)': execution_times}
df = pd.DataFrame(data)
df_sorted = df.sort_values(by='Test Case')

# Write results to file
out_file = os.path.join(out_folder, 'Gurobi_objvals.csv')
df_sorted.to_csv(out_file, index=False)
