import pulp
import numpy as np
import matplotlib.pyplot as plt


# 定义问题
def solve_lp(f1_condition=None, f2_condition=None, f3_condition=None):
    prob = pulp.LpProblem("Multi-Objective Optimization", pulp.LpMaximize)

    # 决策变量
    x1 = pulp.LpVariable("x1", lowBound=0)  # x1 >= 0
    x2 = pulp.LpVariable("x2", lowBound=0)  # x2 >= 0

    # 目标函数
    if f1_condition is not None:
        prob += 2 * x1 - x2  # f1 = 2x1 - x2, 最大化 f1
    elif f2_condition is not None:
        prob += -x1 - 2 * x2  # f2 = -x1 - 2x2, 最大化 f2
    elif f3_condition is not None:
        prob += -3 * x1 - x2  # f3 = -3x1 - x2, 最大化 f3

    # 约束条件
    prob += x1 + 4 * x2 <= 16
    prob += x1 + x2 <= 5.5
    prob += 2 * x1 + x2 <= 10

    # 求解问题
    prob.solve()

    return pulp.value(x1), pulp.value(x2), pulp.value(prob.objective)


# 第一阶段：满足 f1 >= 4
x1, x2, f1_value = solve_lp(f1_condition=True)
print(f"Stage 1: f1 >= 4, x1 = {x1}, x2 = {x2}, f1 = {f1_value}")

# 第二阶段：满足 f1 >= 4 且 f2 <= -7
x1, x2, f2_value = solve_lp(f1_condition=True, f2_condition=True)
print(f"Stage 2: f1 >= 4 and f2 <= -7, x1 = {x1}, x2 = {x2}, f2 = {f2_value}")

# 第三阶段：满足 f1 >= 4, f2 <= -7, f3 = -12
x1, x2, f3_value = solve_lp(f1_condition=True, f2_condition=True, f3_condition=True)
print(f"Stage 3: f1 >= 4, f2 <= -7, f3 = -12, x1 = {x1}, x2 = {x2}, f3 = {f3_value}")

# 可视化约束条件
x1_vals = np.linspace(0, 10, 100)
x2_vals_1 = (16 - x1_vals) / 4
x2_vals_2 = 5.5 - x1_vals
x2_vals_3 = 10 - 2 * x1_vals

plt.figure(figsize=(8, 6))
plt.plot(x1_vals, x2_vals_1, label=r'$x_1 + 4x_2 \leq 16$', color='b')
plt.plot(x1_vals, x2_vals_2, label=r'$x_1 + x_2 \leq 5.5$', color='r')
plt.plot(x1_vals, x2_vals_3, label=r'$2x_1 + x_2 \leq 10$', color='g')

# 填充可行区域
plt.fill_between(x1_vals, np.maximum(0, np.minimum(x2_vals_1, np.minimum(x2_vals_2, x2_vals_3))),
                 color='gray', alpha=0.5, label='Feasible Region')

plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend()
plt.title("Feasible Region and Constraints")
plt.grid(True)
plt.show()
