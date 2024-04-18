import numpy as np 
from pymoo.core.problem import Problem
import os


class QUBO(Problem):

    def __init__(self, Q, **kwargs):

    	n = len(Q)
    	super().__init__(n_var=n,
				         n_obj=1,
				         n_ieq_constr=0,
				         xl=0.0,
				         xu=1.0,
				         **kwargs)
    	self.Q = Q
    	self.n = n

    def _evaluate(self, x, out, *args, **kwargs):
    	n_sol = x.shape[0]
    	out["F"] = np.zeros(n_sol)

    	for i in range(n_sol): 
	    	out["F"][i] = x[i].T @ self.Q @ x[i]


from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization


# initialize the thread pool and create the runner
n_threads = 6
pool = ThreadPool(n_threads)
runner = StarmapParallelization(pool.starmap)

directory = '/Users/alaynabaker/Downloads/ISYE6679/Test Cases/All/'
out_folder = '/Users/alaynabaker/Downloads/ISYE6679/Results/PyMoo PSO/'

test_cases = [file for file in os.listdir(directory) if file.endswith('.csv')]
objective_vals = []
file_names = []
execution_times = []
from pymoo.visualization.scatter import Scatter
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination import get_termination

termination = get_termination("n_gen", 2500)
ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)


for file in test_cases: 
	print("STARTING")
	Q = np.genfromtxt(os.path.join(directory, file), delimiter=',')
	problem = QUBO(Q)
	pso_alg = PSO(w = 0.729,
				c1 = 1.4595,
				c2 = 1.4595, 
				adaptive=False)
	res = minimize(problem, pso_alg, termination, seed=1, verbose = False)
	f_min = np.min(res.F)
	objective_vals.append(round(f_min))
	case = os.path.splitext(file)[0][-2:]
	file_names.append(case)
	execution_times.append(round(res.exec_time,3))
	print("DONE")
	print(case)
	print(round(f_min,3))
	print(round(res.exec_time,3))

import pandas as pd

print("DONE")
data = {'Test Case': file_names, 'Objective Value': objective_vals, 'Execution Time (s)': execution_times}
df = pd.DataFrame(data)
df_sorted = df.sort_values(by='Test Case')

out_file = os.path.join(out_folder, 'PSO_objvals.csv')
df_sorted.to_csv(out_file, index=False)
