from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from deap import base, creator, tools, algorithms
from sklearn.svm import SVR
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import time
import random

# 从文件加载数据
data = pd.read_excel(r'E:\02_Postgraduate_UCAS\01_Class Tasks\01_Courses\02_预测理论与方法_2024秋_汤玲老师\03_Data\AQ.xlsx')
# # 获取特征和目标变量
X= data.drop("CO(GT)",axis=1)
y = data['CO(GT)']

# data = pd.read_excel(r'E:\02_Postgraduate_UCAS\01_Class Tasks\01_Courses\02_预测理论与方法_2024秋_汤玲老师\03_Data\weatherdata.xlsx')
# X = data
# y = data['y']


# 数据归一化处理
scaler = StandardScaler()
X = scaler.fit_transform(X)


# 准备数据集，多步预测
def create_multistep_dataset(X, y, steps):
    X_multi, y_multi = [], []
    for i in range(len(y) - steps):
        X_multi.append(X[i:i + steps])  # 多步输入特征
        y_multi.append(y[i + steps])  # 单步输出目标值
    return np.array(X_multi), np.array(y_multi)
steps = 6

# 创建多步回归数据集
X_multi, y_multi = create_multistep_dataset(X, y, steps)
# 调整数据形状以适应随机森林模型
X_multi = X_multi.reshape(X_multi.shape[0], -1)
# 根据时间序列划分训练集和测试集
n_samples_total = len(y_multi)
train_size = int(n_samples_total * 0.8)
X_train_multi, X_test_multi = X_multi[:train_size], X_multi[train_size:]
y_train_multi, y_test_multi = y_multi[:train_size], y_multi[train_size:]

# %% 粒子群优化算法
# 计时器开始
start_time = time.time()


# 创建随机森林回归模型
def evaluate_model(individual):
    '''
    # individual为遗传算法生成的参数集合
    n_estimators, max_depth, min_samples_split, min_samples_leaf = individual
    # 创建随机森林回归模型
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        random_state=42
    )'''
    #需要改**************
    #创建SVR模型
    C, epsilon = individual
    model = SVR(C=C,epsilon=epsilon,kernel='linear')

    model.fit(X_train_multi, y_train_multi)
    y_pred = model.predict(X_test_multi)

    # 计算MSE作为适应度评价
    mse = mean_squared_error(y_test_multi, y_pred)
    return mse,


# 定义适应度函数
def fitness(params, X_train, X_test, y_train, y_test):
    """
    粒子群优化中的适应度函数，用于评估随机森林模型在给定超参数下的性能。

    参数:
        params: list
            包含随机森林的超参数 [n_estimators, max_depth, min_samples_split, min_samples_leaf]。
        X_train, X_test: ndarray
            训练集和测试集的特征数据。
        y_train, y_test: ndarray
            训练集和测试集的目标数据。

    返回:
        float
            随机森林模型在测试集上的均方误差 (MSE)。
    """

    # 强制将粒子位置的浮点数参数转换为整数
    n_estimators, max_depth, min_samples_split, min_samples_leaf = map(int, params)
    # 创建随机森林回归模型并设置超参数
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42  # 固定随机种子，保证结果可重复
    )
    '''
    # 需要改**************
    # 创建SVR模型
    C, epsilon = params[:2]
    model = SVR(C=C, epsilon=epsilon, kernel='linear')'''

    # 训练模型
    model.fit(X_train, y_train)

    # 在测试集上预测
    y_pred = model.predict(X_test)

    # 计算均方误差 (MSE)
    mse = mean_squared_error(y_test, y_pred)

    return mse


def particle_swarm_optimization(X_train, X_test, y_train, y_test, population_size=20, iterations=50, w=0.5, c1=1, c2=2):
    # 初始化粒子群
    population = np.random.uniform(2, 200, (population_size, 4)) #可进行修改拟合
    velocities = np.zeros((population_size, 4))
    personal_best = population.copy()
    personal_best_fitness = np.array([fitness(params, X_train, X_test, y_train, y_test) for params in population])
    global_best_index = np.argmin(personal_best_fitness)
    global_best = population[global_best_index]

    for _ in range(iterations):
        for i in range(population_size):
            r1, r2 = random.uniform(0, 1), random.uniform(0, 1)
            velocities[i] = w * velocities[i] + c1 * r1 * (personal_best[i] - population[i]) + c2 * r2 * (
                        global_best - population[i])
            population[i] += velocities[i]
            population[i] = np.clip(population[i], 2, 200)

            current_fitness = fitness(population[i], X_train, X_test, y_train, y_test)
            if current_fitness < personal_best_fitness[i]:
                personal_best[i] = population[i]
                personal_best_fitness[i] = current_fitness
                if current_fitness < fitness(global_best, X_train, X_test, y_train, y_test):
                    global_best = population[i]

    return global_best


# PSO超参数设置
population_size_pso = 10
iterations_pso = 20
w_pso = 0.5
c1_pso = 1
c2_pso = 2

# 使用PSO算法获取最佳参数
best_params_pso = particle_swarm_optimization(X_train_multi, X_test_multi, y_train_multi, y_test_multi,
                                              population_size_pso, iterations_pso, w_pso, c1_pso, c2_pso)
print("Best Parameters (PSO):", best_params_pso)

# %%best model
# 使用最佳参数的模型进行预测

best_model = RandomForestRegressor(
    n_estimators=int(best_params_pso[0]),
    max_depth=int(best_params_pso[1]),
    min_samples_split=int(best_params_pso[2]),
    min_samples_leaf=int(best_params_pso[3]),
    random_state=42
)

#best_model = SVR(C=best_params_pso[0], epsilon=best_params_pso[1],kernel='linear') #需要改**************
best_model.fit(X_train_multi, y_train_multi)

y_pred = best_model.predict(X_test_multi)
# 计时器结束
end_time = time.time()
# 计算MSE
mse = mean_squared_error(y_test_multi, y_pred)
print(f"Mean Squared Error with Best Parameters: {mse:.2f}")

# 计算 MAE
mae = mean_absolute_error(y_test_multi, y_pred)

# 计算 MSE
mse = mean_squared_error(y_test_multi, y_pred)

# 计算 RMSE
rmse = np.sqrt(mse)

# 计算 R²
r2 = r2_score(y_test_multi, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")


# 方向准确性
#方向准确性
def compute_Dstat(actual_prices, predicted_prices):
    N = len(actual_prices)
    correct_predictions = 0
    for t in range(N - 1):
        if (actual_prices[t + 1] - actual_prices[t]) * (predicted_prices[t + 1] - predicted_prices[t]) >= 0:
            correct_predictions += 1
    Dstat = (correct_predictions / (N - 1)) * 100
    return Dstat


dstat = compute_Dstat(y_test_multi, y_pred)
print("Dstat: ", dstat)

execution_time = end_time - start_time
print(f"GA-RF运行时间: {execution_time:.2f} seconds")

# 可视化结果
plt.figure(figsize=(10, 6))

plt.plot(y_test_multi, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('GA-RF Regression Prediction')
plt.xlabel('Time')
plt.ylabel('CO')
plt.legend()
plt.show()

