from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from deap import base, creator, tools, algorithms
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import time

# 从文件加载数据
data = pd.read_excel(r'E:\02_Postgraduate_UCAS\01_Class Tasks\01_Courses\02_预测理论与方法_2024秋_汤玲老师\03_Data\AQ.xlsx')
# # 获取特征和目标变量
X= data.drop("CO(GT)",axis=1)
y = data['CO(GT)']

#data = pd.read_excel(r'E:\02_Postgraduate_UCAS\01_Class Tasks\01_Courses\02_预测理论与方法_2024秋_汤玲老师\03_Data\weatherdata.xlsx')
#X = data
#y = data['y']

# 数据归一化处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 准备数据集，多步预测
def create_multistep_dataset(X, y, steps):
    X_multi, y_multi = [], []
    for i in range(len(y)-steps):
        X_multi.append(X[i:i+steps])  # 多步输入特征
        y_multi.append(y[i+steps])    # 单步输出目标值
    return np.array(X_multi), np.array(y_multi)

# 设置步长为6
steps = 6

# 创建多步回归数据集
X_multi, y_multi = create_multistep_dataset(X, y, steps)

# 调整数据形状以适应随机森林模型
X_multi = X_multi.reshape(X_multi.shape[0], -1)
# 根据时间序列划分训练集和测试集
n_samples_total = len(y_multi)
train_size = int(n_samples_total * 0.8)
X_train_multi, X_test_multi = X_multi [:train_size], X_multi [train_size:]
y_train_multi, y_test_multi = y_multi[:train_size],y_multi[train_size:]

# 计时器开始
start_time = time.time()
# 创建随机森林回归模型
def evaluate_model(individual):
    # individual为遗传算法生成的参数集合

    n_estimators, max_depth, min_samples_split, min_samples_leaf = individual
    # 创建随机森林回归模型
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        random_state=42
    )
    '''需要改**************
    #创建SVR模型
    C, epsilon = individual
    model = SVR(C=C,epsilon=epsilon,kernel='linear')'''

    # 训练模型
    model.fit(X_train_multi, y_train_multi)
    # 进行预测
    y_pred = model.predict(X_test_multi)
    # 计算MSE作为适应度评价
    mse = mean_squared_error(y_test_multi, y_pred)
    return mse,

# 定义遗传算法参数和适应度函数
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_int", np.random.randint, 1, 200) #变量的范围，需要改****
#toolbox.register("attr_int", np.random.uniform, 1, 10)
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_int,) * 4, n=1) #这个2要变成变量的个数，attr_int需要浮点数要改成浮点数
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_model)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=1, up=200, indpb=0.05) #需要改*******
#toolbox.register("mutate", tools.mutUniformInt, low=1, up=10, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# 设置遗传算法参数
population_size = 20
generations = 5
crossover_probability = 0.6
mutation_probability = 0.2

# 生成初始种群
population = toolbox.population(n=population_size)

# 运行遗传算法进行优化
for gen in range(generations):
    offspring = algorithms.varAnd(population, toolbox, cxpb=crossover_probability, mutpb=mutation_probability)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit

    # 交叉和变异
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if np.random.rand() < crossover_probability:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if np.random.rand() < mutation_probability:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    population = toolbox.select(offspring, k=len(population))

# 输出结果
best_ind = tools.selBest(population, k=1)[0]
best_params = [int(param) for param in best_ind]
print("Best Parameters:", best_params)

# 使用最佳参数的模型进行预测
best_model = RandomForestRegressor(n_estimators=best_params[0], max_depth=best_params[1],
                                   min_samples_split=best_params[2], min_samples_leaf=best_params[3],
                                   random_state=42)
#best_model = SVR(C=best_params[0], epsilon=best_params[1],kernel='linear') 需要改**************
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


#方向准确性
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