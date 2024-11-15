import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.optimize import linprog

# 定义约束条件
x1 = np.linspace(0, 7, 400)
x2_1 = 7 - x1         # 来自约束 x1 + x2 <= 7
x2_2 = 12 - 2 * x1    # 来自约束 2x1 + x2 <= 12
x2_3 = np.full_like(x1, 5)  # 来自约束 x2 <= 5

# 取三个约束的最小值，得到可行区域的上边界
x2_upper = np.minimum(np.minimum(x2_1, x2_2), x2_3)

# 绘制可行区域
plt.figure(figsize=(8, 6))
plt.plot(x1, x2_1, label='$x_1 + x_2 = 7$')
plt.plot(x1, x2_2, label='$2x_1 + x_2 = 12$')
plt.plot(x1, x2_3, label='$x_2 = 5$')
plt.fill_between(x1, 0, x2_upper, where=(x2_upper > 0), alpha=0.3)
plt.xlim(0, 7)
plt.ylim(0, 7)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Feasible Region $X$')
plt.legend()
plt.grid(True)
plt.show()

# 定义可行域的顶点
vertices = np.array([
    [0, 0],
    [0, 5],
    [2, 5],
    [3.5, 5],
    [5, 2],
    [6, 0]
])

# 计算对应的目标函数值
f1 = 3 * vertices[:, 0] + vertices[:, 1]
f2 = (-4) * vertices[:, 0] + vertices[:, 1]

# 绘制目标空间中的可行区域
plt.figure(figsize=(8, 6))
plt.plot(f1, f2, 'bo-', label='Mapping in Objective Space')
plt.fill(f1, f2, alpha=0.3)
plt.xlabel('$f_1$')
plt.ylabel('$f_2$')
plt.title('Feasible Region in Objective Space $Y$')
plt.legend()
plt.grid(True)
plt.show()

# 按照 f1 的值进行排序
indices = np.argsort(f1)
sorted_f1 = f1[indices]
sorted_f2 = f2[indices]

# 选择非劣解
pareto_f1 = []
pareto_f2 = []
current_max_f2 = -np.inf
for i in range(len(sorted_f1)):
    if sorted_f2[i] > current_max_f2:
        pareto_f1.append(sorted_f1[i])
        pareto_f2.append(sorted_f2[i])
        current_max_f2 = sorted_f2[i]

# 绘制非劣前沿
plt.figure(figsize=(8, 6))
plt.plot(f1, f2, 'bo', label='All Solutions')
plt.plot(pareto_f1, pareto_f2, 'ro-', label='Non-inferior Solutions')
plt.xlabel('$f_1$')
plt.ylabel('$f_2$')
plt.title('Pareto Frontier')
plt.legend()
plt.grid(True)
plt.show()

# 求解单目标最大化问题，找到理想解
# 最大化 f1
c1 = [-3, -1]  # linprog 进行最小化，所以取负号
A = [
    [1, 1],
    [2, 1],
    [0, 1]
]
b = [7, 12, 5]
x1_bounds = (0, None)
x2_bounds = (0, None)
res1 = linprog(c1, A_ub=A, b_ub=b, bounds=[x1_bounds, x2_bounds], method='highs')

# 最大化 f2
c2 = [-1, -2]
res2 = linprog(c2, A_ub=A, b_ub=b, bounds=[x1_bounds, x2_bounds], method='highs')

# 理想解
f1_star = -res1.fun
f2_star = -res2.fun

# 绘制理想解
plt.figure(figsize=(8, 6))
plt.plot(f1, f2, 'bo', label='All Solutions')
plt.plot(pareto_f1, pareto_f2, 'ro-', label='Non-inferior Solutions')
plt.plot(f1_star, f2_star, 'g*', markersize=15, label='Ideal Solution')
plt.xlabel('$f_1$')
plt.ylabel('$f_2$')
plt.title('Ideal Solution in Objective Space')
plt.legend()
plt.grid(True)
plt.show()

# 使用调和解法，构建目标函数
from scipy.optimize import minimize

# 理想点
f1_ideal = f1_star
f2_ideal = f2_star

# 定义目标函数
def harmony_obj(x):
    f1_val = 3 * x[0] + x[1]
    f2_val = x[0] + 2 * x[1]
    return ((f1_ideal - f1_val) / f1_ideal) ** 2 + ((f2_ideal - f2_val) / f2_ideal) ** 2

# 约束条件
cons = [
    {'type': 'ineq', 'fun': lambda x: 7 - x[0] - x[1]},
    {'type': 'ineq', 'fun': lambda x: 12 - 2 * x[0] - x[1]},
    {'type': 'ineq', 'fun': lambda x: x[1] - 0},
    {'type': 'ineq', 'fun': lambda x: x[0] - 0},
    {'type': 'ineq', 'fun': lambda x: 5 - x[1]},
]

# 初始猜测
x0 = [1, 1]

# 求解
result = minimize(harmony_obj, x0, constraints=cons)
x_opt = result.x
f1_opt = 3 * x_opt[0] + x_opt[1]
f2_opt = x_opt[0] + 2 * x_opt[1]

print("Best Compromise Solution:")
print(f"x1 = {x_opt[0]:.4f}, x2 = {x_opt[1]:.4f}")
print(f"f1 = {f1_opt:.4f}, f2 = {f2_opt:.4f}")

# 在目标空间中绘制最佳妥协解
plt.figure(figsize=(8, 6))
plt.plot(f1, f2, 'bo', label='All Solutions')
plt.plot(pareto_f1, pareto_f2, 'ro-', label='Non-inferior Solutions')
plt.plot(f1_star, f2_star, 'g*', markersize=15, label='Ideal Solution')
plt.plot(f1_opt, f2_opt, 'ks', markersize=10, label='Best Compromise Solution')
plt.xlabel('$f_1$')
plt.ylabel('$f_2$')
plt.title('Best Compromise Solution in Objective Space')
plt.legend()
plt.grid(True)
plt.show()

from pymoo.problems import get_problem