import pandas as pd
import numpy as np
from PyEMD import EMD
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


# 从文件加载数据
data = pd.read_excel(r'E:\02_Postgraduate_UCAS\01_Class Tasks\01_Courses\02_预测理论与方法_2024秋_汤玲老师\03_Data\AQ.xlsx')
# 获取特征和目标变量
co_gt = data['CO(GT)'].values  # y 数据

# EMD 分解，集成预测
emd = EMD()
eIMFs = emd.emd(co_gt)
print(len(eIMFs))

# IMF 的图
fig, axes = plt.subplots(len(eIMFs) + 1, 1, figsize=(10, 10))
# 存储结果
predicted_results = np.zeros_like(co_gt)

# 划分训练集和测试集，例如 80% 作为训练集，20% 作为测试集
train_size = int(len(co_gt) * 0.8)
X_train_total = np.arange(len(co_gt)).reshape(-1, 1)


# 对每个 IMF 预测，结果加回 predicted_results
for i, imf in enumerate(eIMFs):
    # 准备训练数据
    X_train = np.arange(len(imf)).reshape(-1, 1)
    y_train = imf.reshape(-1, 1)

    # 训练随机森林模型
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train[:train_size], y_train[:train_size])  # 只使用训练集进行训练

    # 预测
    predicted_imf_train = rf_model.predict(X_train[:train_size]).flatten()  # 训练集预测
    predicted_imf_test = rf_model.predict(X_train[train_size:]).flatten()  # 测试集预测

    # 将训练集的 IMF 预测结果加回
    predicted_results[:train_size] += predicted_imf_train
    # 将测试集的 IMF 预测结果加回
    predicted_results[train_size:] += predicted_imf_test

    # 绘制 IMF 子图
    axes[i].plot(imf)
    axes[i].plot(predicted_imf_train)
    axes[i].legend()


# 调整子图布局
plt.tight_layout()
plt.show()


# 最终结果的图
plt.figure(figsize=(10, 5))
# 可视化原始数据和最终预测结果
plt.plot(co_gt, label='Original Data')
plt.plot(predicted_results, label='Predicted Results')  # 绘制预测值
plt.title('Decision Tree Regression Prediction')  # 设置图形标题
plt.xlabel('Time')  # 设置 X 轴标签
plt.ylabel('CO')  # 设置 Y 轴标签
plt.legend()  # 显示图例
plt.show()


# 计算评估指标
y_train = co_gt[:train_size]
y_test = co_gt[train_size:]
y_pred_train = predicted_results[:train_size]
y_pred_test = predicted_results[train_size:]


# 计算 MSE
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)


# 计算 MAPE
mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
mape_test = mean_absolute_percentage_error(y_test, y_pred_test)


# 计算 MAE
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)


# 计算 R2
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)


print(f"Training MSE: {mse_train}")
print(f"Testing MSE: {mse_test}")
print(f"Training MAPE: {mape_train}")
print(f"Testing MAPE: {mape_test}")
print(f"Training MAE: {mae_train}")
print(f"Testing MAE: {mae_test}")
print(f"Training R2: {r2_train}")
print(f"Testing R2: {r2_test}")


# 计算预测准确度
def compute_Dstat(actual_prices, predicted_prices):
    N = len(actual_prices)
    correct_predictions = 0
    for t in range(N - 1):
        if (actual_prices[t + 1] - actual_prices[t]) * (predicted_prices[t + 1] - predicted_prices[t]) >= 0:
            correct_predictions += 1
    Dstat = (correct_predictions / (N - 1)) * 100
    return Dstat


acc_train = compute_Dstat(y_train, y_pred_train)
acc_test = compute_Dstat(y_test, y_pred_test)


print(f"Training Accuracy: {acc_train}")
print(f"Testing Accuracy: {acc_test}")