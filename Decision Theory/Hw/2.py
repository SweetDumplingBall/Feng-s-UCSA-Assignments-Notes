import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

from hw2_modm2 import vertices

x1 = np.linspace(0, 20, 400)
x2_1 = 1 + x1
x2_2 = 3.5 + 0.5 * x1
x2_3 = 2 * x1 - 8
x2_4 = 3 - (2 / 3) * x1

# 绘制可行区域
plt.figure(figsize=(8, 6))
plt.plot(x1, x2_1, label='1')
plt.plot(x1, x2_2, label='2')
plt.plot(x1, x2_3, label='3')
plt.plot(x1, x2_4, label='4')
plt.xlim(0, 15)
plt.ylim(0, 15)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.grid(True)
plt.show()

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