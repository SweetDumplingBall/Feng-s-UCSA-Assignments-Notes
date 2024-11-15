import numpy as np
import matplotlib.pyplot as plt

# 定义收益值
x_values = [0, 1, 2, 3]

# 定义区间长度
intervals = np.diff([0] + x_values)  # [0, 1, 1, 1]

# 方案 A 的概率分布
prob_A = [0, 0.2, 0, 0.8]

# 方案 B 的概率分布
prob_B = [0.1, 0, 0.4, 0.5]

# 计算方案 A 的 CDF
cdf_A = np.cumsum(prob_A)

# 计算方案 B 的 CDF
cdf_B = np.cumsum(prob_B)

# 计算方案 A 的累积面积 S_F(x)
S_F = np.cumsum(cdf_A * intervals)

# 计算方案 B 的累积面积 S_G(x)
S_G = np.cumsum(cdf_B * intervals)

# 输出累积面积的计算结果
print("收益 x\t方案 A 的 CDF F_A(x)\t方案 A 的累积面积 S_F(x)\t方案 B 的 CDF F_B(x)\t方案 B 的累积面积 S_G(x)")
for i in range(len(x_values)):
    print("{:.0f}\t\t{:.4f}\t\t\t\t{:.4f}\t\t\t\t{:.4f}\t\t\t\t{:.4f}".format(
        x_values[i], cdf_A[i], S_F[i], cdf_B[i], S_G[i]))

# 计算面积差值 ΔS(x) = S_F(x) - S_G(x)
delta_S = S_F - S_G

print("\n收益 x\tS_F(x)\tS_G(x)\tΔS(x) = S_F(x) - S_G(x)")
for i in range(len(x_values)):
    print("{:.0f}\t\t{:.4f}\t{:.4f}\t{:.4f}".format(
        x_values[i], S_F[i], S_G[i], delta_S[i]))

# 绘制累积分布函数（CDF）
plt.figure(figsize=(10, 6))
plt.step(x_values, cdf_A, where='post', label='CDF of Plan A')
plt.step(x_values, cdf_B, where='post', label='CDF of Plan B')
plt.xlabel('Return (x)')
plt.ylabel('Cumulative Distribution Function F(x)')
plt.title('CDFs of Plan A and Plan B')
plt.xticks(x_values)
plt.legend()
plt.grid(True)
plt.show()

# 绘制二阶累积分布函数
cdf2_A = np.cumsum(cdf_A)
cdf2_B = np.cumsum(cdf_B)

plt.figure(figsize=(10, 6))
plt.step(x_values, cdf2_A, where='post', label='Second-order CDF of Plan A')
plt.step(x_values, cdf2_B, where='post', label='Second-order CDF of Plan B')
plt.xlabel('Return (x)')
plt.ylabel('Second-order Cumulative Distribution Function F^(2)(x)')
plt.title('Second-order CDFs of Plan A and Plan B')
plt.xticks(x_values)
plt.legend()
plt.grid(True)
plt.show()
