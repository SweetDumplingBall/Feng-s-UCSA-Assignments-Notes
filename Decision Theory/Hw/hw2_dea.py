from pulp import *

# 创建问题实例
prob = LpProblem("Minimize_Theta", LpMinimize)

# 定义决策变量
lambda1 = LpVariable("lambda1", lowBound=0)
lambda2 = LpVariable("lambda2", lowBound=0)
lambda3 = LpVariable("lambda3", lowBound=0)
lambda4 = LpVariable("lambda4", lowBound=0)
s1_plus = LpVariable("s1_plus", lowBound=0)
s2_plus = LpVariable("s2_plus", lowBound=0)
s3_plus = LpVariable("s3_plus", lowBound=0)
s1_prime = LpVariable("s1_prime", lowBound=0)
s2_prime = LpVariable("s2_prime", lowBound=0)
theta = LpVariable("theta")

# 添加约束条件
prob += 2*lambda1 + 5*lambda2 + 3*lambda3 + 6*lambda4 + s1_plus == 6*theta
prob += 3*lambda1 + lambda2 + 4*lambda3 + 7*lambda4 + s2_plus == 7*theta
prob += 4*lambda1 + 3*lambda2 + lambda3 + 2*lambda4 + s3_plus == 2*theta
prob += 2*lambda1 + lambda2 + 3*lambda3 + lambda4 - s1_prime == 1
prob += lambda1 + 3*lambda2 + 3*lambda3 + 2*lambda4 - s2_prime == 2

# 设置目标函数
prob += theta

# 解决问题
prob.solve()

# 提取解决方案
print("Optimal value of theta:", value(theta))
print("Optimal values of lambdas and slacks:")
for v in prob.variables():
    print(v.name, "=", v.varValue)