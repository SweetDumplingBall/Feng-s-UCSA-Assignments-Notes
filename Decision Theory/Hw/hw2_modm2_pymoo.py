import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.decomposition.asf import ASF

class MyProblem(Problem):

    def __init__(self):
        super().__init__(
            n_var=2,
            n_obj=2,
            n_ieq_constr=3,
            xl=np.array([0, 0]),
            xu=np.array([6, 5])  # Updated upper bounds
        )

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = - (3 * x[:, 0] + x[:, 1])   # Negative for maximization
        f2 = - (x[:, 0] + 2 * x[:, 1])   # Negative for maximization
        out["F"] = np.column_stack([f1, f2])

        # Inequality constraints (<= 0)
        g1 = x[:, 0] + x[:, 1] - 7
        g2 = 2 * x[:, 0] + x[:, 1] - 12
        g3 = x[:, 1] - 5
        out["G"] = np.column_stack([g1, g2, g3])

problem = MyProblem()

algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               seed=1,
               verbose=False)

# Extract feasible solutions
is_feasible = np.all(res.G <= 0, axis=1)
feasible_solutions = res.X[is_feasible]
feasible_objectives = -res.F[is_feasible]  # Multiply by -1 to get back to maximization

# Plot Feasible Region X
plt.figure(figsize=(8, 6))
plt.scatter(feasible_solutions[:, 0], feasible_solutions[:, 1], c='blue', s=10, label='Feasible Solutions')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Feasible Region $X$')
plt.legend()
plt.grid(True)
plt.show()

# Plot Objective Space Y
plt.figure(figsize=(8, 6))
plt.scatter(feasible_objectives[:, 0], feasible_objectives[:, 1], c='green', s=10, label='Objective Values')
plt.xlabel('$f_1$')
plt.ylabel('$f_2$')
plt.title('Objective Space $Y$')
plt.legend()
plt.grid(True)
plt.show()

# Identify Noninferior Solutions
pareto_mask = res.F[:, 0].argsort()
pareto_solutions = feasible_solutions[pareto_mask]
pareto_objectives = feasible_objectives[pareto_mask]

# Plot Pareto Front
plt.figure(figsize=(8, 6))
plt.scatter(feasible_objectives[:, 0], feasible_objectives[:, 1], c='blue', s=10, label='Feasible Solutions')
plt.scatter(pareto_objectives[:, 0], pareto_objectives[:, 1], c='red', s=20, label='Pareto Optimal Solutions')
plt.xlabel('$f_1$')
plt.ylabel('$f_2$')
plt.title('Pareto Front')
plt.legend()
plt.grid(True)
plt.show()

# Mark the Ideal Solution
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead

def maximize_f1():
    problem = MyProblem()
    problem.n_obj = 1
    problem.n_ieq_constr = 3
    problem.xu = np.array([6, 5])  # Ensure bounds are consistent

    def _evaluate(x, out, *args, **kwargs):
        f1 = - (3 * x[:, 0] + x[:, 1])   # Negative for maximization
        out["F"] = f1
        g1 = x[:, 0] + x[:, 1] - 7
        g2 = 2 * x[:, 0] + x[:, 1] - 12
        g3 = x[:, 1] - 5
        out["G"] = np.column_stack([g1, g2, g3])

    problem._evaluate = _evaluate

    res = minimize(problem,
                   NelderMead(),
                   x0=np.array([0, 0]),
                   verbose=False)
    return res

res_f1 = maximize_f1()
ideal_f1 = -res_f1.F[0]
x_f1 = res_f1.X

def maximize_f2():
    problem = MyProblem()
    problem.n_obj = 1
    problem.n_ieq_constr = 3
    problem.xu = np.array([6, 5])  # Ensure bounds are consistent

    def _evaluate(x, out, *args, **kwargs):
        f2 = - (x[:, 0] + 2 * x[:, 1])   # Negative for maximization
        out["F"] = f2
        g1 = x[:, 0] + x[:, 1] - 7
        g2 = 2 * x[:, 0] + x[:, 1] - 12
        g3 = x[:, 1] - 5
        out["G"] = np.column_stack([g1, g2, g3])

    problem._evaluate = _evaluate

    res = minimize(problem,
                   NelderMead(),
                   x0=np.array([0, 0]),
                   verbose=False)
    return res

res_f2 = maximize_f2()
ideal_f2 = -res_f2.F[0]
x_f2 = res_f2.X

# Plotting the Ideal Solution in Objective Space
plt.figure(figsize=(8, 6))
plt.scatter(feasible_objectives[:, 0], feasible_objectives[:, 1], c='blue', s=10, label='Feasible Solutions')
plt.scatter(pareto_objectives[:, 0], pareto_objectives[:, 1], c='red', s=20, label='Pareto Optimal Solutions')
plt.scatter(ideal_f1, (x_f1[0] + 2 * x_f1[1]), c='gold', s=50, label='Ideal f1 Solution')
plt.scatter((3 * x_f2[0] + x_f2[1]), ideal_f2, c='purple', s=50, label='Ideal f2 Solution')
plt.xlabel('$f_1$')
plt.ylabel('$f_2$')
plt.title('Objective Space with Ideal Solutions')
plt.legend()
plt.grid(True)
plt.show()

# Find Best Compromise Solution
weights = np.array([0.5, 0.5])

decomp = ASF()

I = decomp.do(-feasible_objectives, weights).argmin()
best_compromise_solution = feasible_solutions[I]
best_compromise_objectives = feasible_objectives[I]

print("Best Compromise Solution:")
print(f"x1 = {best_compromise_solution[0]:.4f}, x2 = {best_compromise_solution[1]:.4f}")
print(f"f1 = {best_compromise_objectives[0]:.4f}, f2 = {best_compromise_objectives[1]:.4f}")

# Plot Best Compromise Solution
plt.figure(figsize=(8, 6))
plt.scatter(feasible_objectives[:, 0], feasible_objectives[:, 1], c='blue', s=10, label='Feasible Solutions')
plt.scatter(pareto_objectives[:, 0], pareto_objectives[:, 1], c='red', s=20, label='Pareto Optimal Solutions')
plt.scatter(best_compromise_objectives[0], best_compromise_objectives[1], c='green', s=50, label='Best Compromise Solution')
plt.xlabel('$f_1$')
plt.ylabel('$f_2$')
plt.title('Best Compromise Solution in Objective Space')
plt.legend()
plt.grid(True)
plt.show()
