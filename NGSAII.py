import numpy as np 
from pymoo.core.problem import Problem

class QUBO(Problem):
    def __init__(self, Q, **kwargs):
    	n = len(Q)
    	super().__init__(n_var=n,
				         n_obj=2,
				         n_ieq_constr=0,
				         xl=0.0,
				         xu=1.0,
				         **kwargs)
    	self.Q = Q
    	self.n = n

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0, 1, n_pareto_points)
        return np.array([x, 1 - np.sqrt(x)]).T

    def _evaluate(self, x, out, *args, **kwargs):
        n_sol = x.shape[0]
        f = np.zeros((n_sol, self.n_obj))

        for i in range(n_sol):
            f1 = x[i].T @ self.Q @ x[i]
            f2 = np.sum(x[i] * (1 - x[i]))
            f[i] = [f1, f2]

        out["F"] = f

from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization

n_threads = 8
pool = ThreadPool(n_threads)
runner = StarmapParallelization(pool.starmap)

import os
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize

directory = '/Users/alaynabaker/Downloads/ISYE6679/Test Cases/All/'
out_folder = '/Users/alaynabaker/Downloads/ISYE6679/Results/NGSAII'

test_cases = [file for file in os.listdir(directory) if file.endswith('.csv')]
objective_vals = []
file_names = []
execution_times = []
termination = get_termination("n_gen", 2500)

for file in test_cases: 
	print("STARTING")
	Q = np.genfromtxt(os.path.join(directory, file), delimiter=',')
	problem = MyProblem(Q)
	alg = NSGA2(pop_size = 100, eliminate_duplicated=True)
	res = minimize(problem,
					alg,
					termination,
					pf=problem.pareto_front(),
					seed=1, 
					verbose = False)
	f_min = np.min(res.F)
	objective_vals.append(round(f_min))
	case = os.path.splitext(file)[0][-2:]
	file_names.append(case)
	execution_times.append(round(res.exec_time,3))
	print("DONE")
	print(case)
	print(f_min)
	print(round(res.exec_time,3))
	from pymoo.visualization.scatter import Scatter
	# Scatter().add(res.F).show()

import pandas as pd

print("DONE")
data = {'Test Case': file_names, 'Objective Value': objective_vals, 'Execution Time (s)': execution_times}
df = pd.DataFrame(data)
df_sorted = df.sort_values(by='Test Case')

out_file = os.path.join(out_folder, 'NGSAII_objvals.csv')
df_sorted.to_csv(out_file, index=False)
