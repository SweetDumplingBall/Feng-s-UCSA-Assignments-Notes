import numpy as np
import matplotlib.pyplot as plt

# 设置参数 a 和 b
a = 10  # 您可以根据需要调整 a 的值
b = 2   # b > 1，确保决策者是风险规避的

# 确保 b > 1，否则决策者不是风险规避的
if b <= 1:
    raise ValueError("为了确保风险规避，参数 b 必须大于 1。")

# 定义财富 w 的范围，w < a
w = np.linspace(0, a - 0.01, 500)  # 取值略小于 a，防止除以零

# 计算效用函数 U(w)
U_w = - (a - w) ** b

# 计算一阶导数 U'(w)
U_prime = b * (a - w) ** (b - 1)

# 计算二阶导数 U''(w)
U_double_prime = -b * (b - 1) * (a - w) ** (b - 2)

# 计算绝对风险规避系数 A(w)
A_w = (b - 1) / (a - w)

# 绘制效用函数 U(w)
plt.figure(figsize=(10, 6))
plt.plot(w, U_w, label='Utility Function U(w)')
plt.xlabel('Wealth w')
plt.ylabel('Utility U(w)')
plt.title('Utility Function U(w) = - (a - w)^b')
plt.legend()
plt.grid(True)
plt.show()

# 绘制绝对风险规避系数 A(w)
plt.figure(figsize=(10, 6))
plt.plot(w, A_w, label='Absolute Risk Aversion A(w)', color='red')
plt.xlabel('Wealth w')
plt.ylabel('Absolute Risk Aversion A(w)')
plt.title('Absolute Risk Aversion A(w) = (b - 1) / (a - w)')
plt.legend()
plt.grid(True)
plt.show()

# 输出部分结果
print(f"参数 a = {a}, b = {b}")
print(f"在财富范围 w ∈ [0, {a}) 内，决策者是风险规避的。")
print(f"绝对风险规避系数 A(w) = (b - 1) / (a - w)")
